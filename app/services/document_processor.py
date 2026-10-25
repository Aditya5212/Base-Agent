"""Asynchronous document processing and RAG ingestion pipeline."""

import asyncio
from datetime import datetime, timedelta, timezone
import logging
from typing import Any
import uuid

from sqlalchemy import delete, select, text

from app.core.database import async_session_factory
from app.core.rag_config import get_rag_config
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.services.document_loader import load_document
from app.services.r2_storage import r2_storage_service
from app.services.text_splitter import split_markdown_document
from app.services.vector_store import delete_documents_by_metadata, index_documents

logger = logging.getLogger(__name__)

STUCK_DOCUMENT_LOCK_ID = 847291


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
        is_reindex: If True, purges prior embeddings, chunks, and graph entities before indexing.

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

    try:
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

        # Step 2: If re-indexing, purge previous chunks, PGVector embeddings, and graph entities
        if is_reindex:
            logger.info("Purging existing chunks for document %s", document_id)
            try:
                await delete_documents_by_metadata("document_id", document_id)
                async with async_session_factory() as session:
                    del_stmt = delete(DocumentChunk).where(DocumentChunk.document_id == parsed_id)
                    await session.execute(del_stmt)
                    await session.commit()

                # Purge LightRAG knowledge graph entities if active
                try:
                    from app.core.rag.factory import get_provider
                    from app.core.rag.workspace import user_workspace

                    lightrag_provider = await get_provider("lightrag", user_workspace(user_id))
                    await lightrag_provider.delete_document(document_id)
                except Exception as g_exc:
                    logger.debug("LightRAG knowledge graph purge skipped/failed for doc %s: %s", document_id, g_exc)
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

    except BaseException as unhandled:
        logger.critical(
            "Unhandled failure in document pipeline for %s: %s",
            document_id,
            unhandled,
            exc_info=True,
        )
        try:
            await _update_document_status(
                parsed_id,
                DocumentStatus.FAILED,
                f"Unhandled pipeline error: {unhandled}",
            )
        except Exception:
            pass
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


async def recover_stuck_documents(stuck_threshold_seconds: int = 300) -> list[str]:
    """Recovers documents stuck in PARSING status using PostgreSQL advisory locks.

    Coordinates single-worker execution across multiple API instances.
    - If file exists in Cloudflare R2 and retry_count < 3:
      Increments retry_count and automatically re-triggers ingestion.
    - If retry_count >= 3 or file missing in R2:
      Marks status as FAILED with a clear error message, enabling 1-click re-indexing.

    Args:
        stuck_threshold_seconds: Age in seconds beyond which a PARSING doc is considered stuck.

    Returns:
        list[str]: Document IDs that were recovered or marked as FAILED.
    """
    recovered_ids: list[str] = []

    try:
        async with async_session_factory() as session:
            # Try to acquire PostgreSQL advisory lock to ensure only one worker performs recovery
            lock_res = await session.execute(text(f"SELECT pg_try_advisory_lock({STUCK_DOCUMENT_LOCK_ID});"))
            acquired = lock_res.scalar()
            if not acquired:
                logger.debug("Another worker holds the stuck document recovery lock. Skipping.")
                return []

            try:
                cutoff = datetime.now(timezone.utc) - timedelta(seconds=stuck_threshold_seconds)
                stmt = select(Document).where(
                    Document.status == DocumentStatus.PARSING,
                    Document.updated_at <= cutoff,
                )
                res = await session.execute(stmt)
                stuck_docs = list(res.scalars().all())

                if not stuck_docs:
                    return []

                logger.info("Found %d stuck document(s) in PARSING state", len(stuck_docs))

                for doc in stuck_docs:
                    doc_id_str = str(doc.id)
                    exists = await r2_storage_service.file_exists(doc.r2_key)

                    if exists and doc.retry_count < 3:
                        doc.retry_count += 1
                        logger.info(
                            "Recovering stuck doc %s ('%s'): retry_count=%d/3. Re-launching pipeline.",
                            doc_id_str,
                            doc.file_name,
                            doc.retry_count,
                        )
                        await session.commit()

                        # Re-run pipeline asynchronously
                        asyncio.create_task(
                            process_document_pipeline(
                                document_id=doc_id_str,
                                r2_key=doc.r2_key,
                                user_id=doc.user_id,
                                file_name=doc.file_name,
                                preset="balanced",
                                file_bytes=None,
                                is_reindex=True,
                            )
                        )
                        recovered_ids.append(doc_id_str)
                    else:
                        doc.status = DocumentStatus.FAILED
                        if not exists:
                            doc.error_message = f"Recovery failed: source file '{doc.r2_key}' not found in R2."
                        else:
                            doc.error_message = (
                                f"Ingestion timed out and exceeded maximum retries ({doc.retry_count}/3). "
                                "Use the re-index endpoint to retry."
                            )
                        logger.warning(
                            "Marking stuck doc %s ('%s') as FAILED: %s",
                            doc_id_str,
                            doc.file_name,
                            doc.error_message,
                        )
                        await session.commit()
                        recovered_ids.append(doc_id_str)

            finally:
                await session.execute(text(f"SELECT pg_advisory_unlock({STUCK_DOCUMENT_LOCK_ID});"))
                await session.commit()

    except Exception as exc:
        logger.error("Error executing stuck document recovery: %s", exc, exc_info=True)

    return recovered_ids


async def start_document_sweeper(interval_seconds: int = 300) -> None:
    """Background task loop that periodically executes stuck document recovery."""
    logger.info("Starting background stuck-document sweeper loop (interval=%ds)", interval_seconds)
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            recovered = await recover_stuck_documents(stuck_threshold_seconds=interval_seconds)
            if recovered:
                logger.info("Sweeper processed %d stuck document(s): %s", len(recovered), recovered)
        except asyncio.CancelledError:
            logger.info("Stuck-document sweeper task cancelled.")
            break
        except Exception as exc:
            logger.error("Error in background stuck-document sweeper: %s", exc, exc_info=True)
