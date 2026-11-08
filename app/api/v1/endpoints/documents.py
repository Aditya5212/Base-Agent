"""Knowledge Base & Document Management API endpoints with RAG search."""

import io
import logging
import uuid
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.auth import UserContext, get_current_user
from app.core.database import get_db
from app.core.rag_config import get_rag_config
from app.core.rate_limit import limiter
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.schemas.document import (
    DeleteDocumentResponse,
    DocumentChunkItem,
    DocumentChunksListResponse,
    DocumentDetailResponse,
    DocumentItemResponse,
    DocumentListResponse,
    DocumentRetryResponse,
    DocumentUploadResponse,
    KnowledgeBaseStatsResponse,
    RAGSearchChunkResponse,
    RAGSearchRequest,
    RAGSearchResponse,
    ReindexDocumentRequest,
    ReindexDocumentResponse,
    RetryDocumentRequest,
)
from app.schemas.storage import format_bytes
from app.services.document_loader import convert_to_markdown, load_document
from app.services.document_processor import process_document_pipeline
from app.services.r2_storage import r2_storage_service
from app.services.retriever import hybrid_search
from app.services.text_splitter import split_markdown_document
from app.services.vector_store import delete_documents_by_metadata, index_documents
from app.services.document_lock import document_lock

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
    user_id: str | None = None,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeBaseStatsResponse:
    """Calculates overall knowledge base metrics for the specified user."""
    effective_user_id = user_id if (current_user.is_admin and user_id and user_id != "root") else current_user.user_id
    doc_stmt = select(Document).where(Document.user_id == effective_user_id)
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
        .where(Document.user_id == effective_user_id)
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
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and index a document in background",
    description="Uploads raw document to Cloudflare R2, records status as PARSING, and dispatches background processing (MarkItDown, tiktoken chunking, PGVector + FTS indexing).",
)
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file (PDF, DOCX, XLSX, HTML, TXT, MD)"),
    user_id: str | None = Form(default=None, description="Owner user ID (admin only)"),
    preset: str = Form(default="balanced", description="Chunking preset (high_precision, balanced, high_recall)"),
    index_graph: bool = Form(default=False, description="Also index the document in LightRAG"),
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentUploadResponse:
    """Processes document upload, object storage, and queues background parsing & dual-indexing."""
    effective_user_id = user_id if (current_user.is_admin and user_id and user_id != "root") else current_user.user_id
    if preset not in {"balanced", "high_precision", "high_recall"}:
        raise HTTPException(status_code=422, detail="Unsupported chunking preset")
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
    r2_key = f"documents/{effective_user_id}/{doc_id}_{file_name}"

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
            detail="Storage upload failed. Please try again.",
        )

    # Step 2: Record document in PostgreSQL with PARSING status
    doc_record = Document(
        id=doc_id,
        user_id=effective_user_id,
        file_name=file_name,
        r2_key=r2_key,
        mime_type=content_type,
        file_size=file_size,
        status=DocumentStatus.PARSING,
        indexing_config={"preset": preset, "index_graph": index_graph},
        provider_status={"hybrid": "PARSING", **({"lightrag": "PARSING"} if index_graph else {})},
    )
    db.add(doc_record)
    try:
        await db.commit()
    except Exception as db_exc:
        logger.error(
            "Database commit failed after R2 upload; rolling back and cleaning up orphaned R2 file: %s",
            db_exc,
            exc_info=True,
        )
        await db.rollback()
        try:
            await r2_storage_service.delete_file(r2_key)
        except Exception as r2_cleanup_exc:
            logger.warning("Failed to clean up orphaned R2 file '%s': %s", r2_key, r2_cleanup_exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist document metadata in database.",
        )

    # Step 3: Dispatch asynchronous pipeline via Taskiq (with BackgroundTasks fallback)
    dispatched = False
    try:
        from app.tasks.document_tasks import process_document_task
        await process_document_task.kiq(
            document_id=str(doc_id),
            r2_key=r2_key,
            user_id=effective_user_id,
            file_name=file_name,
            preset=preset,
            is_reindex=False,
        )
        dispatched = True
        logger.info("Dispatched document %s to Taskiq worker queue.", doc_id)
    except Exception as tq_exc:
        logger.debug(
            "Taskiq task dispatch unavailable (%s); using in-process BackgroundTasks fallback.",
            tq_exc,
        )

    if not dispatched:
        background_tasks.add_task(
            process_document_pipeline,
            document_id=str(doc_id),
            r2_key=r2_key,
            user_id=effective_user_id,
            file_name=file_name,
            preset=preset,
            file_bytes=contents,
            is_reindex=False,
        )

    return DocumentUploadResponse(
        id=str(doc_id),
        file_name=file_name,
        file_size=file_size,
        status=doc_record.status.value,
        chunks_count=0,
        r2_key=r2_key,
        message=f"Document '{file_name}' uploaded successfully. Background ingestion and indexing pipeline started.",
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List uploaded documents",
    description="Retrieves documents uploaded by a user with status, search filter, and presigned download links.",
)
async def list_documents(
    user_id: str | None = None,
    search: str | None = Query(default=None, description="Optional filename filter"),
    status_filter: str | None = Query(default=None, description="Filter by status (INDEXED, FAILED, PARSING)"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    """Returns document records for a given user with optional search and status filters."""
    effective_user_id = user_id if (current_user.is_admin and user_id and user_id != "root") else current_user.user_id
    # Build base filter predicate (reused for COUNT and paginated fetch)
    base_stmt = select(Document).where(Document.user_id == effective_user_id)
    if search:
        base_stmt = base_stmt.where(Document.file_name.ilike(f"%{search}%"))
    if status_filter:
        base_stmt = base_stmt.where(Document.status == status_filter.upper())

    # Run total COUNT before applying pagination
    count_stmt = select(func.count()).select_from(base_stmt.subquery())
    total: int = (await db.execute(count_stmt)).scalar_one() or 0

    # Paginated fetch
    paged_stmt = base_stmt.order_by(Document.created_at.desc()).limit(limit).offset(offset)
    result = await db.execute(paged_stmt)
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
                error_message=doc.error_message,
                created_at=doc.created_at,
                download_url=download_url,
            )
        )

    return DocumentListResponse(total=total, documents=items)


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details",
    description="Returns detailed metadata, chunk counts, processing status, and presigned download URL for a document.",
)
async def get_document_details(
    document_id: str,
    user_id: str | None = None,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentDetailResponse:
    """Fetches details for a single document."""
    effective_user_id = user_id if (current_user.is_admin and user_id and user_id != "root") else current_user.user_id
    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid UUID format.")

    stmt = select(Document).where(Document.id == parsed_id).options(selectinload(Document.chunks))
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    if doc.user_id != effective_user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You do not have permission to view this document.",
        )

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
        indexing_config=doc.indexing_config,
        provider_status=doc.provider_status,
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
    user_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentChunksListResponse:
    """Lists granular chunks for inspection."""
    effective_user_id = user_id if (current_user.is_admin and user_id and user_id != "root") else current_user.user_id
    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid UUID format.")

    doc_stmt = select(Document).where(Document.id == parsed_id)
    doc_res = await db.execute(doc_stmt)
    doc = doc_res.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    if doc.user_id != effective_user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You do not have permission to inspect this document's chunks.",
        )

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
    status_code=status.HTTP_202_ACCEPTED,
    summary="Re-index an existing document in background",
    description="Dispatches background re-indexing: purges previous chunks and re-splits/re-indexes from Cloudflare R2.",
)
async def reindex_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    request: ReindexDocumentRequest = ReindexDocumentRequest(),
    user_id: str | None = None,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ReindexDocumentResponse:
    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid UUID format") from exc
    async with document_lock(str(parsed_id)) as acquired:
        if not acquired:
            raise HTTPException(status_code=409, detail="Document is currently being processed")
        return await _reindex_document_locked(document_id, request, background_tasks, user_id, current_user, db)


async def _reindex_document_locked(document_id, request, background_tasks, user_id, current_user, db):
    """Re-chunks and re-indexes an existing document from its R2 stored raw content in the background."""
    effective_user_id = user_id if (current_user.is_admin and user_id and user_id != "root") else current_user.user_id
    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid UUID format.")

    stmt = select(Document).where(Document.id == parsed_id)
    res = await db.execute(stmt)
    doc = res.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    if doc.user_id != effective_user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You do not have permission to re-index this document.",
        )

    count_stmt = select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == parsed_id)
    prev_count = (await db.execute(count_stmt)).scalar_one_or_none() or 0

    # Capture prior failure error and reset lifecycle for re-indexing
    prior_error = doc.error_message
    doc.status = DocumentStatus.PARSING
    doc.error_message = None
    doc.retry_count = 0
    config = dict(doc.indexing_config or {})
    config["purge_graph"] = config.get("index_graph", False)
    config["preset"] = request.preset
    if request.index_graph is not None:
        config["index_graph"] = request.index_graph
    doc.indexing_config = config
    await db.commit()

    # Dispatch asynchronous re-indexing pipeline via Taskiq (with BackgroundTasks fallback)
    dispatched = False
    try:
        from app.tasks.document_tasks import process_document_task
        await process_document_task.kiq(
            document_id=str(doc.id),
            r2_key=doc.r2_key,
            user_id=doc.user_id,
            file_name=doc.file_name,
            preset=request.preset,
            is_reindex=True,
        )
        dispatched = True
        logger.info("Dispatched document %s re-indexing to Taskiq worker queue.", doc.id)
    except Exception as tq_exc:
        logger.debug(
            "Taskiq task dispatch unavailable (%s); using in-process BackgroundTasks fallback.",
            tq_exc,
        )

    if not dispatched:
        background_tasks.add_task(
            process_document_pipeline,
            document_id=str(doc.id),
            r2_key=doc.r2_key,
            user_id=doc.user_id,
            file_name=doc.file_name,
            preset=request.preset,
            file_bytes=None,
            is_reindex=True,
        )

    return ReindexDocumentResponse(
        document_id=str(doc.id),
        file_name=doc.file_name,
        previous_chunks=prev_count,
        new_chunks=0,
        preset=request.preset,
        status=doc.status.value,
        error_message=prior_error,
        message=f"Document '{doc.file_name}' re-indexing pipeline started in background using '{request.preset}' preset.",
    )


@router.post(
    "/{document_id}/retry",
    response_model=DocumentRetryResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Retry failed or stuck document indexing in background",
    description="Restarts the background parsing and indexing pipeline for a failed or stuck document. Resets error state and retry count.",
)
async def retry_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    request: RetryDocumentRequest = RetryDocumentRequest(),
    user_id: str | None = None,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentRetryResponse:
    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid UUID format") from exc
    async with document_lock(str(parsed_id)) as acquired:
        if not acquired:
            raise HTTPException(status_code=409, detail="Document is currently being processed")
        return await _retry_document_locked(document_id, request, background_tasks, user_id, current_user, db)


async def _retry_document_locked(document_id, request, background_tasks, user_id, current_user, db):
    """Restarts indexing pipeline for a failed or stuck document."""
    effective_user_id = user_id if (current_user.is_admin and user_id and user_id != "root") else current_user.user_id
    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid UUID format.")

    stmt = select(Document).where(Document.id == parsed_id)
    res = await db.execute(stmt)
    doc = res.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    if doc.user_id != effective_user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You do not have permission to retry indexing for this document.",
        )

    config = dict(doc.indexing_config or {})
    target_preset = request.preset or config.get("preset", "balanced")
    target_index_graph = request.index_graph if request.index_graph is not None else config.get("index_graph", False)

    prior_error = doc.error_message

    # Transition to PARSING, clear error, and reset retry counter
    doc.status = DocumentStatus.PARSING
    doc.error_message = None
    doc.retry_count = 0
    config["preset"] = target_preset
    config["index_graph"] = target_index_graph
    config["purge_graph"] = target_index_graph
    doc.indexing_config = config
    doc.provider_status = {
        "hybrid": "PARSING",
        **({"lightrag": "PARSING"} if target_index_graph else {}),
    }
    await db.commit()

    dispatched = False
    try:
        from app.tasks.document_tasks import process_document_task
        await process_document_task.kiq(
            document_id=str(doc.id),
            r2_key=doc.r2_key,
            user_id=doc.user_id,
            file_name=doc.file_name,
            preset=target_preset,
            is_reindex=True,
        )
        dispatched = True
        logger.info("Dispatched document %s retry to Taskiq worker queue.", doc.id)
    except Exception as tq_exc:
        logger.debug(
            "Taskiq task dispatch unavailable (%s); using in-process BackgroundTasks fallback.",
            tq_exc,
        )

    if not dispatched:
        background_tasks.add_task(
            process_document_pipeline,
            document_id=str(doc.id),
            r2_key=doc.r2_key,
            user_id=doc.user_id,
            file_name=doc.file_name,
            preset=target_preset,
            file_bytes=None,
            is_reindex=True,
        )

    return DocumentRetryResponse(
        document_id=str(doc.id),
        file_name=doc.file_name,
        status=doc.status.value,
        preset=target_preset,
        previous_error=prior_error,
        message=f"Document '{doc.file_name}' indexing retry initiated in background using '{target_preset}' preset.",
    )


@router.get(
    "/{document_id}/download",
    summary="Download original document",
    description="Streams or returns the original binary file from Cloudflare R2 with optional browser inline preview.",
)
async def download_document(
    document_id: str,
    user_id: str | None = None,
    inline: bool = Query(default=False, description="Display inline in browser if true, otherwise trigger download"),
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Streams the raw original document file directly from Cloudflare R2."""
    effective_user_id = user_id if (current_user.is_admin and user_id and user_id != "root") else current_user.user_id
    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid UUID format.")

    stmt = select(Document).where(Document.id == parsed_id)
    res = await db.execute(stmt)
    doc = res.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    if doc.user_id != effective_user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You do not have permission to view or download this document.",
        )

    disposition_type = "inline" if inline else "attachment"

    try:
        raw_bytes = await r2_storage_service.download_file(doc.r2_key)
        # Sanitize filename: strip quotes/newlines that could break the header
        safe_filename = doc.file_name.replace('"', '').replace('\n', '').replace('\r', '')
        return StreamingResponse(
            io.BytesIO(raw_bytes),
            media_type=doc.mime_type,
            headers={
                "Content-Disposition": f'{disposition_type}; filename="{safe_filename}"',
                "Content-Length": str(len(raw_bytes)),
            },
        )
    except Exception as exc:
        logger.error("Download failed for doc %s: %s", document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download document from storage. Please try again.",
        )


@router.delete(
    "/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete a document and purge chunks",
    description="Deletes raw file from Cloudflare R2, purges all vector/FTS chunks, and deletes database record.",
)
async def delete_document(
    document_id: str,
    user_id: str | None = None,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DeleteDocumentResponse:
    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid UUID format") from exc
    async with document_lock(str(parsed_id)) as acquired:
        if not acquired:
            raise HTTPException(status_code=409, detail="Document is currently being processed")
        return await _delete_document_locked(document_id, user_id, current_user, db)


async def _delete_document_locked(document_id, user_id, current_user, db):
    """Cascade deletes document from storage and database."""
    effective_user_id = user_id if (current_user.is_admin and user_id and user_id != "root") else current_user.user_id
    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid UUID format.")

    stmt = select(Document).where(Document.id == parsed_id)
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    if doc.user_id != effective_user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You do not have permission to delete this document.",
        )

    # Snapshot fields before any mutation so we can still reference them after deletion
    doc_id_str = str(doc.id)
    doc_file_name = doc.file_name
    doc_r2_key = doc.r2_key
    doc_user_id = doc.user_id

    # Step 1: Purge vector-store chunks (reversible — re-index restores these)
    chunks_removed = await delete_documents_by_metadata("document_id", doc_id_str)

    # Step 2: Purge LightRAG knowledge-graph entries (best-effort; not fatal)
    try:
        from app.core.rag.factory import get_provider
        from app.core.rag.workspace import user_workspace
        lightrag_provider = await get_provider("lightrag", user_workspace(doc_user_id))
        await lightrag_provider.delete_document(doc_id_str)
    except Exception as g_exc:
        logger.debug("LightRAG knowledge graph purge skipped/failed for doc %s: %s", doc_id_str, g_exc)

    # Step 3: Delete raw object from Cloudflare R2 — MUST succeed before we touch the DB row.
    # If this fails the DB record is still intact and the operator can retry or clean up R2 manually.
    try:
        await r2_storage_service.delete_file(doc_r2_key)
    except Exception as exc:
        logger.error(
            "Failed to delete R2 object '%s' for document %s; aborting DB deletion to prevent orphan row: %s",
            doc_r2_key, doc_id_str, exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Storage deletion failed — database record was NOT removed to prevent data loss. Retry or contact support.",
        )

    # Step 4: Delete database record (cascades to document_chunks)
    await db.delete(doc)
    await db.commit()

    return DeleteDocumentResponse(
        id=doc_id_str,
        deleted=True,
        chunks_removed=chunks_removed,
        message=f"Document '{doc_file_name}' and all associated vector chunks were successfully deleted.",
    )


@router.post(
    "/search",
    response_model=RAGSearchResponse,
    summary="Hybrid RAG Search with Reciprocal Rank Fusion",
    description="Executes hybrid search across pgvector (dense) and PostgreSQL FTS (sparse), fused via RRF.",
)
@limiter.limit("60/minute")
async def search_documents(
    request: Request,
    search_request: RAGSearchRequest,
    current_user: UserContext = Depends(get_current_user),
) -> RAGSearchResponse:
    """Performs hybrid retrieval combining semantic and exact-match search."""
    effective_user_id = search_request.user_id if (current_user.is_admin and search_request.user_id and search_request.user_id != "root") else current_user.user_id
    rag_cfg = get_rag_config(
        preset=search_request.preset,
        top_k_final=search_request.top_k or 5,
    )

    filter_dict = {}
    if effective_user_id:
        filter_dict["user_id"] = effective_user_id

    fused_docs = await hybrid_search(
        query=search_request.query,
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
        query=search_request.query,
        preset=rag_cfg.preset.value,
        total_results=len(results),
        results=results,
    )
