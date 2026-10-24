"""Asynchronous document processing and RAG ingestion pipeline."""

import logging
from typing import Any
import uuid

from sqlalchemy import delete, select

from app.core.database import async_session_factory
from app.core.rag_config import get_rag_config
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.services.document_loader import load_document
from app.services.r2_storage import r2_storage_service
from app.services.text_splitter import split_markdown_document
from app.services.vector_store import delete_documents_by_metadata, index_documents

logger = logging.getLogger(__name__)


async def process_document_pipeline(
    document_id: str,
    r2_key: str,
    user_id: str,
    file_name: str,
    preset: str = "balanced",
    file_bytes: bytes | None = None,
    is_reindex: bool = False,
) -> int:
    """Processes document conversion, token chunking, and dual-indexing into PGVector and PostgreSQL FTS.

    Designed to execute asynchronously out-of-band:
    - Currently invoked via FastAPI `BackgroundTasks` (zero infrastructure overhead).
    - In future multi-node distributed deployments, can be dispatched via `await broker.kick(process_document_pipeline, ...)`
      using Taskiq with a PostgreSQL or Redis broker without altering the core pipeline logic.

    Args:
        document_id: UUID string of the document record.
        r2_key: Object storage key in Cloudflare R2.
        user_id: Owner identifier for multi-tenant scoping.
        file_name: Original file name.
        preset: RAG retrieval preset ('high_precision', 'balanced', 'high_recall').
        file_bytes: Optional in-memory raw binary content (avoids extra R2 roundtrip on upload).
        is_reindex: If True, purges prior embeddings and chunks before indexing.

    Returns:
        int: Number of chunks successfully generated and indexed.
    """
    logger.info(
        "Starting document processing pipeline: doc_id=%s, file=%s, preset=%s, is_reindex=%s",
        document_id,
        file_name,
        preset,
        is_reindex,
    )

    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError as exc:
        logger.error("Invalid UUID format for document processing: %s", document_id)
        raise exc

    # Step 1: Retrieve raw file bytes if not supplied in-memory
    if file_bytes is None:
        try:
            logger.info("Downloading raw file from Cloudflare R2: %s", r2_key)
            raw_bytes = await r2_storage_service.download_file(r2_key)
        except Exception as exc:
            logger.error("Failed to download '%s' from R2: %s", r2_key, exc, exc_info=True)
            await _update_document_status(parsed_id, DocumentStatus.FAILED, f"Storage retrieval failed: {exc}")
            raise
    else:
        raw_bytes = file_bytes

    # Step 2: If re-indexing, purge previous chunks and PGVector embeddings
    if is_reindex:
        logger.info("Purging existing chunks for document %s", document_id)
        try:
            await delete_documents_by_metadata("document_id", document_id)
            async with async_session_factory() as session:
                del_stmt = delete(DocumentChunk).where(DocumentChunk.document_id == parsed_id)
                await session.execute(del_stmt)
                await session.commit()
        except Exception as exc:
            logger.error("Failed to purge prior chunks for %s: %s", document_id, exc, exc_info=True)
            await _update_document_status(parsed_id, DocumentStatus.FAILED, f"Chunk purge failed: {exc}")
            raise

    # Step 3: Parse raw document into Markdown via MarkItDown
    try:
        raw_doc = load_document(
            source=raw_bytes,
            filename=file_name,
            extra_metadata={
                "document_id": document_id,
                "user_id": user_id,
                "file_name": file_name,
            },
        )
    except Exception as exc:
        logger.error("Document parsing failed for %s: %s", document_id, exc, exc_info=True)
        await _update_document_status(parsed_id, DocumentStatus.FAILED, f"Parsing failed: {exc}")
        raise

    # Step 4: Split Markdown into structure-aware token chunks
    try:
        rag_cfg = get_rag_config(preset)
        chunks = split_markdown_document(raw_doc, config=rag_cfg)
        logger.info("Generated %d chunks for document %s", len(chunks), document_id)
    except Exception as exc:
        logger.error("Document chunking failed for %s: %s", document_id, exc, exc_info=True)
        await _update_document_status(parsed_id, DocumentStatus.FAILED, f"Chunking failed: {exc}")
        raise

    # Step 5: Index chunks into PGVector HNSW and PostgreSQL FTS GIN
    try:
        chunk_ids = await index_documents(chunks)
        logger.info("Indexed %d vector embeddings for document %s", len(chunk_ids), document_id)
    except Exception as exc:
        logger.error("Vector store indexing failed for %s: %s", document_id, exc, exc_info=True)
        await _update_document_status(parsed_id, DocumentStatus.FAILED, f"Vector indexing failed: {exc}")
        raise

    # Step 6: Insert chunk inspection records and update Document status to INDEXED
    try:
        async with async_session_factory() as session:
            for idx, chunk in enumerate(chunks):
                session.add(
                    DocumentChunk(
                        id=uuid.uuid4(),
                        document_id=parsed_id,
                        chunk_index=idx,
                        content=chunk.page_content,
                        token_count=chunk.metadata.get("token_count", 0),
                        metadata_json=chunk.metadata,
                    )
                )

            doc_stmt = select(Document).where(Document.id == parsed_id)
            doc_res = await session.execute(doc_stmt)
            doc_record = doc_res.scalar_one_or_none()
            if doc_record:
                doc_record.status = DocumentStatus.INDEXED
                doc_record.error_message = None

            await session.commit()

        logger.info(
            "Document processing completed successfully: doc_id=%s, chunks=%d, status=INDEXED",
            document_id,
            len(chunks),
        )
        return len(chunks)

    except Exception as exc:
        logger.error("Failed to commit chunk records for %s: %s", document_id, exc, exc_info=True)
        await _update_document_status(parsed_id, DocumentStatus.FAILED, f"Persistence failed: {exc}")
        raise


async def _update_document_status(
    document_id: uuid.UUID,
    status: DocumentStatus,
    error_message: str | None = None,
) -> None:
    """Updates document status and optional error message in an isolated database session."""
    try:
        async with async_session_factory() as session:
            stmt = select(Document).where(Document.id == document_id)
            res = await session.execute(stmt)
            doc = res.scalar_one_or_none()
            if doc:
                doc.status = status
                doc.error_message = error_message
                await session.commit()
    except Exception as exc:
        logger.error("Failed to update status (%s) for document %s: %s", status, document_id, exc)
