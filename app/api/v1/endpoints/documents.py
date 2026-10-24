"""Knowledge Base & Document Management API endpoints with RAG search."""

import io
import logging
import uuid
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.rag_config import get_rag_config
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.schemas.document import (
    DeleteDocumentResponse,
    DocumentChunkItem,
    DocumentChunksListResponse,
    DocumentDetailResponse,
    DocumentItemResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    KnowledgeBaseStatsResponse,
    RAGSearchChunkResponse,
    RAGSearchRequest,
    RAGSearchResponse,
    ReindexDocumentRequest,
    ReindexDocumentResponse,
)
from app.schemas.storage import format_bytes
from app.services.document_loader import convert_to_markdown, load_document
from app.services.r2_storage import r2_storage_service
from app.services.retriever import hybrid_search
from app.services.text_splitter import split_markdown_document
from app.services.vector_store import delete_documents_by_metadata, index_documents

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Knowledge Base Management"])

# Max allowed upload size: 25MB
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


@router.get(
    "/stats",
    response_model=KnowledgeBaseStatsResponse,
    summary="Get knowledge base statistics",
    description="Returns aggregated metrics for user's knowledge base including document counts, chunk counts, and storage usage.",
)
async def get_knowledge_base_stats(
    user_id: str = "default_user",
    db: AsyncSession = Depends(get_db),
) -> KnowledgeBaseStatsResponse:
    """Calculates overall knowledge base metrics for the specified user."""
    doc_stmt = select(Document).where(Document.user_id == user_id)
    doc_res = await db.execute(doc_stmt)
    docs = doc_res.scalars().all()

    total_docs = len(docs)
    total_bytes = sum(d.file_size for d in docs)
    status_counts: dict[str, int] = {}
    for d in docs:
        status_counts[d.status.value] = status_counts.get(d.status.value, 0) + 1

    chunk_stmt = (
        select(func.count(DocumentChunk.id))
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(Document.user_id == user_id)
    )
    chunk_res = await db.execute(chunk_stmt)
    total_chunks = chunk_res.scalar_one_or_none() or 0

    return KnowledgeBaseStatsResponse(
        total_documents=total_docs,
        total_chunks=total_chunks,
        total_storage_bytes=total_bytes,
        formatted_storage=format_bytes(total_bytes),
        status_breakdown=status_counts,
        active_preset_default="balanced",
    )


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

    # Step 2: Record document in PostgreSQL with PARSING status
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

        # Step 5: Store chunk records in PostgreSQL for granular inspection
        for idx, chunk in enumerate(chunks):
            chunk_rec = DocumentChunk(
                id=uuid.uuid4(),
                document_id=doc_id,
                chunk_index=idx,
                content=chunk.page_content,
                token_count=chunk.metadata.get("token_count", 0),
                metadata_json=chunk.metadata,
            )
            db.add(chunk_rec)

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
    description="Retrieves documents uploaded by a user with status, search filter, and presigned download links.",
)
async def list_documents(
    user_id: str = "default_user",
    search: str | None = Query(default=None, description="Optional filename filter"),
    status_filter: str | None = Query(default=None, description="Filter by status (INDEXED, FAILED, PARSING)"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    """Returns document records for a given user with optional search and status filters."""
    stmt = select(Document).where(Document.user_id == user_id)

    if search:
        stmt = stmt.where(Document.file_name.ilike(f"%{search}%"))
    if status_filter:
        stmt = stmt.where(Document.status == status_filter.upper())

    stmt = stmt.order_by(Document.created_at.desc()).limit(limit).offset(offset)
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


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details",
    description="Returns detailed metadata, chunk counts, processing status, and presigned download URL for a document.",
)
async def get_document_details(
    document_id: str,
    user_id: str = "default_user",
    db: AsyncSession = Depends(get_db),
) -> DocumentDetailResponse:
    """Fetches details for a single document."""
    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid UUID format.")

    stmt = (
        select(Document)
        .where(Document.id == parsed_id, Document.user_id == user_id)
        .options(selectinload(Document.chunks))
    )
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    try:
        download_url = await r2_storage_service.generate_presigned_url(doc.r2_key)
    except Exception:
        download_url = None

    return DocumentDetailResponse(
        id=str(doc.id),
        file_name=doc.file_name,
        mime_type=doc.mime_type,
        file_size=doc.file_size,
        formatted_size=format_bytes(doc.file_size),
        status=doc.status.value,
        error_message=doc.error_message,
        chunks_count=len(doc.chunks),
        r2_key=doc.r2_key,
        download_url=download_url,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.get(
    "/{document_id}/chunks",
    response_model=DocumentChunksListResponse,
    summary="Inspect document chunks",
    description="Returns individual text chunks, section headers, and token counts generated for a document.",
)
async def get_document_chunks(
    document_id: str,
    user_id: str = "default_user",
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> DocumentChunksListResponse:
    """Lists granular chunks for inspection."""
    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid UUID format.")

    doc_stmt = select(Document).where(Document.id == parsed_id, Document.user_id == user_id)
    doc_res = await db.execute(doc_stmt)
    doc = doc_res.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    chunk_stmt = (
        select(DocumentChunk)
        .where(DocumentChunk.document_id == parsed_id)
        .order_by(DocumentChunk.chunk_index.asc())
        .limit(limit)
        .offset(offset)
    )
    chunks_res = await db.execute(chunk_stmt)
    chunk_records = chunks_res.scalars().all()

    items = []
    for c in chunk_records:
        headers = [v for k, v in c.metadata_json.items() if k.startswith("header_")]
        section_str = " > ".join(headers) if headers else None
        items.append(
            DocumentChunkItem(
                id=str(c.id),
                chunk_index=c.chunk_index,
                token_count=c.token_count,
                section_header=section_str,
                content=c.content,
                preview=c.content[:150].strip() + "..." if len(c.content) > 150 else c.content,
                metadata=c.metadata_json,
            )
        )

    count_stmt = select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == parsed_id)
    count_res = await db.execute(count_stmt)
    total_count = count_res.scalar_one_or_none() or len(items)

    return DocumentChunksListResponse(
        document_id=str(doc.id),
        file_name=doc.file_name,
        total_chunks=total_count,
        chunks=items,
    )


@router.post(
    "/{document_id}/reindex",
    response_model=ReindexDocumentResponse,
    summary="Re-index an existing document",
    description="Re-downloads the raw document from Cloudflare R2, purges old chunks, and re-chunks/re-indexes with the specified preset.",
)
async def reindex_document(
    document_id: str,
    request: ReindexDocumentRequest,
    user_id: str = "default_user",
    db: AsyncSession = Depends(get_db),
) -> ReindexDocumentResponse:
    """Re-chunks and re-indexes an existing document from its R2 stored raw content."""
    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid UUID format.")

    stmt = select(Document).where(Document.id == parsed_id, Document.user_id == user_id)
    res = await db.execute(stmt)
    doc = res.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    # 1. Download raw file from Cloudflare R2
    try:
        raw_bytes = await r2_storage_service.download_file(doc.r2_key)
    except Exception as exc:
        logger.error("Failed to download '%s' from R2: %s", doc.r2_key, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve original document from storage: {exc}",
        )

    # 2. Count and purge previous chunks
    count_stmt = select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == parsed_id)
    prev_count = (await db.execute(count_stmt)).scalar_one_or_none() or 0

    await delete_documents_by_metadata("document_id", str(doc.id))
    del_stmt = delete(DocumentChunk).where(DocumentChunk.document_id == parsed_id)
    await db.execute(del_stmt)

    # 3. Parse and chunk with new preset
    try:
        raw_doc = load_document(
            source=raw_bytes,
            filename=doc.file_name,
            extra_metadata={
                "document_id": str(doc.id),
                "user_id": user_id,
                "file_name": doc.file_name,
            },
        )
        rag_cfg = get_rag_config(request.preset)
        new_chunks = split_markdown_document(raw_doc, config=rag_cfg)

        # 4. Index into PGVector
        await index_documents(new_chunks)

        # 5. Insert new chunk records
        for idx, c in enumerate(new_chunks):
            db.add(
                DocumentChunk(
                    id=uuid.uuid4(),
                    document_id=doc.id,
                    chunk_index=idx,
                    content=c.page_content,
                    token_count=c.metadata.get("token_count", 0),
                    metadata_json=c.metadata,
                )
            )

        doc.status = DocumentStatus.INDEXED
        doc.error_message = None
        await db.commit()

        return ReindexDocumentResponse(
            document_id=str(doc.id),
            file_name=doc.file_name,
            previous_chunks=prev_count,
            new_chunks=len(new_chunks),
            preset=request.preset,
            status=doc.status.value,
            message=f"Document '{doc.file_name}' re-indexed successfully into {len(new_chunks)} chunks using '{request.preset}' preset.",
        )
    except Exception as exc:
        logger.error("Re-indexing failed for %s: %s", doc.id, exc, exc_info=True)
        doc.status = DocumentStatus.FAILED
        doc.error_message = str(exc)
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Re-indexing failed: {exc}",
        )


@router.get(
    "/{document_id}/download",
    summary="Download original document",
    description="Streams or returns the original binary file from Cloudflare R2.",
)
async def download_document(
    document_id: str,
    user_id: str = "default_user",
    db: AsyncSession = Depends(get_db),
):
    """Streams the raw original document file directly from Cloudflare R2."""
    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid UUID format.")

    stmt = select(Document).where(Document.id == parsed_id, Document.user_id == user_id)
    res = await db.execute(stmt)
    doc = res.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    try:
        raw_bytes = await r2_storage_service.download_file(doc.r2_key)
        return StreamingResponse(
            io.BytesIO(raw_bytes),
            media_type=doc.mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="{doc.file_name}"',
                "Content-Length": str(len(raw_bytes)),
            },
        )
    except Exception as exc:
        logger.error("Download failed for doc %s: %s", document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download document from storage: {exc}",
        )


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

    # 3. Delete database record (cascades to document_chunks)
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
