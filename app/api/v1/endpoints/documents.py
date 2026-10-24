"""Document management and RAG search API endpoints."""

import logging
import uuid
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.rag_config import get_rag_config
from app.models.document import Document, DocumentStatus
from app.schemas.document import (
    DeleteDocumentResponse,
    DocumentItemResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    RAGSearchChunkResponse,
    RAGSearchRequest,
    RAGSearchResponse,
)
from app.services.document_loader import convert_to_markdown, load_document
from app.services.r2_storage import r2_storage_service
from app.services.retriever import hybrid_search
from app.services.text_splitter import split_markdown_document
from app.services.vector_store import delete_documents_by_metadata, index_documents

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents & RAG"])

# Max allowed upload size: 25MB
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document",
    description="Uploads a document to Cloudflare R2, parses to Markdown via MarkItDown, chunks with tiktoken, and indexes into pgvector and FTS.",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file (PDF, DOCX, XLSX, HTML, TXT, MD)"),
    user_id: str = Form(default="default_user", description="Owner user ID"),
    preset: str = Form(default="balanced", description="Chunking preset (high_precision, balanced, high_recall)"),
    db: AsyncSession = Depends(get_db),
) -> DocumentUploadResponse:
    """Processes document upload, object storage, conversion, and dual-index insertion."""
    file_name = file.filename or "uploaded_file"
    content_type = file.content_type or "application/octet-stream"

    # Read and validate file content
    contents = await file.read()
    file_size = len(contents)
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB.",
        )

    doc_id = uuid.uuid4()
    r2_key = f"documents/{user_id}/{doc_id}_{file_name}"

    # Step 1: Upload raw file to Cloudflare R2
    try:
        await r2_storage_service.upload_file(
            file_data=contents,
            file_key=r2_key,
            content_type=content_type,
        )
    except Exception as exc:
        logger.error("Failed to upload file to Cloudflare R2: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Storage upload failed: {exc}",
        )

    # Step 2: Record document in PostgreSQL with PENDING status
    doc_record = Document(
        id=doc_id,
        user_id=user_id,
        file_name=file_name,
        r2_key=r2_key,
        mime_type=content_type,
        file_size=file_size,
        status=DocumentStatus.PARSING,
    )
    db.add(doc_record)
    await db.commit()

    # Step 3: Parse to Markdown with MarkItDown and chunk with tiktoken
    try:
        raw_doc = load_document(
            source=contents,
            filename=file_name,
            extra_metadata={
                "document_id": str(doc_id),
                "user_id": user_id,
                "file_name": file_name,
            },
        )

        rag_cfg = get_rag_config(preset)
        chunks = split_markdown_document(raw_doc, config=rag_cfg)

        # Step 4: Index chunks into PGVector and PostgreSQL FTS
        chunk_ids = await index_documents(chunks)

        # Update document status to INDEXED
        doc_record.status = DocumentStatus.INDEXED
        await db.commit()

        return DocumentUploadResponse(
            id=str(doc_id),
            file_name=file_name,
            file_size=file_size,
            status=doc_record.status.value,
            chunks_count=len(chunk_ids),
            r2_key=r2_key,
            message=f"Document '{file_name}' successfully parsed and indexed into {len(chunk_ids)} chunks.",
        )

    except Exception as exc:
        logger.error("Document ingestion pipeline failed for %s: %s", doc_id, exc, exc_info=True)
        doc_record.status = DocumentStatus.FAILED
        doc_record.error_message = str(exc)
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document parsing/indexing failed: {exc}",
        )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List uploaded documents",
    description="Retrieves all documents uploaded by a user with status and presigned download links.",
)
async def list_documents(
    user_id: str = "default_user",
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    """Returns all document records for a given user."""
    stmt = (
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.created_at.desc())
    )
    result = await db.execute(stmt)
    documents = result.scalars().all()

    items: list[DocumentItemResponse] = []
    for doc in documents:
        try:
            download_url = await r2_storage_service.generate_presigned_url(doc.r2_key)
        except Exception:
            download_url = None

        items.append(
            DocumentItemResponse(
                id=str(doc.id),
                file_name=doc.file_name,
                mime_type=doc.mime_type,
                file_size=doc.file_size,
                status=doc.status.value,
                created_at=doc.created_at,
                download_url=download_url,
            )
        )

    return DocumentListResponse(total=len(items), documents=items)


@router.delete(
    "/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete a document and purge chunks",
    description="Deletes raw file from Cloudflare R2, purges all vector/FTS chunks, and deletes database record.",
)
async def delete_document(
    document_id: str,
    user_id: str = "default_user",
    db: AsyncSession = Depends(get_db),
) -> DeleteDocumentResponse:
    """Cascade deletes document from storage and database."""
    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid UUID format.")

    stmt = select(Document).where(Document.id == parsed_id, Document.user_id == user_id)
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    # 1. Purge chunks from vector store
    chunks_removed = await delete_documents_by_metadata("document_id", str(doc.id))

    # 2. Delete raw object from Cloudflare R2
    try:
        await r2_storage_service.delete_file(doc.r2_key)
    except Exception as exc:
        logger.warning("Failed to delete R2 object '%s': %s", doc.r2_key, exc)

    # 3. Delete database record
    await db.delete(doc)
    await db.commit()

    return DeleteDocumentResponse(
        id=str(doc.id),
        deleted=True,
        chunks_removed=chunks_removed,
        message=f"Document '{doc.file_name}' and all associated vector chunks were successfully deleted.",
    )


@router.post(
    "/search",
    response_model=RAGSearchResponse,
    summary="Hybrid RAG Search with Reciprocal Rank Fusion",
    description="Executes hybrid search across pgvector (dense) and PostgreSQL FTS (sparse), fused via RRF.",
)
async def search_documents(
    request: RAGSearchRequest,
) -> RAGSearchResponse:
    """Performs hybrid retrieval combining semantic and exact-match search."""
    rag_cfg = get_rag_config(
        preset=request.preset,
        top_k_final=request.top_k or 5,
    )

    filter_dict = {}
    if request.user_id:
        filter_dict["user_id"] = request.user_id

    fused_docs = await hybrid_search(
        query=request.query,
        config=rag_cfg,
        filter_dict=filter_dict if filter_dict else None,
    )

    results: list[RAGSearchChunkResponse] = []
    for doc in fused_docs:
        results.append(
            RAGSearchChunkResponse(
                content=doc.page_content,
                source=doc.metadata.get("source", doc.metadata.get("file_name", "unknown")),
                chunk_index=doc.metadata.get("chunk_index"),
                rrf_score=doc.metadata.get("rrf_score"),
                dense_rank=doc.metadata.get("dense_rank"),
                sparse_rank=doc.metadata.get("sparse_rank"),
                metadata=doc.metadata,
            )
        )

    return RAGSearchResponse(
        query=request.query,
        preset=rag_cfg.preset.value,
        total_results=len(results),
        results=results,
    )
