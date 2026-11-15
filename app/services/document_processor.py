"""Serialized, retry-safe document ingestion and abandoned-job recovery."""
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import logging
import uuid
from typing import TypedDict

from sqlalchemy import delete, func, select, text

from app.core.database import async_session_factory
from app.core.rag_config import get_rag_config
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.services.document_loader import load_document
from app.services.document_lock import document_lock
from app.services.r2_storage import r2_storage_service
from app.services.text_splitter import split_markdown_document
from app.services.vector_store import delete_documents_by_metadata, index_documents

logger = logging.getLogger(__name__)
_recovery_tasks: set[asyncio.Task] = set()


class DocumentTaskArguments(TypedDict, total=False):
    document_id: str
    r2_key: str
    user_id: str
    file_name: str
    preset: str
    is_reindex: bool
    thread_id: str | None


async def process_document_pipeline(
    document_id: str,
    r2_key: str,
    user_id: str,
    file_name: str,
    preset: str = "balanced",
    file_bytes: bytes | None = None,
    is_reindex: bool = False,
    thread_id: str | None = None,
) -> int:
    """Use a dedicated connection lock across indexing and concurrent delete/reindex."""
    parsed_id = uuid.UUID(document_id)
    async with document_lock(str(parsed_id)) as acquired:
        if not acquired:
            return 0
        async with async_session_factory() as session:
            doc = await session.get(Document, parsed_id)
            if doc is None:
                return 0
            if doc.user_id != user_id or doc.r2_key != r2_key:
                raise ValueError("Document task does not match stored owner/source")
            effective_thread_id = thread_id if thread_id is not None else getattr(doc, "thread_id", None)
            if effective_thread_id:
                from app.models.thread import Thread
                th = await session.get(Thread, effective_thread_id)
                if th is None or (getattr(th, "metadata_json", None) or {}).get("status") == "DELETING":
                    logger.info("Aborting ingestion for document %s: thread %s is deleted or deleting", parsed_id, effective_thread_id)
                    doc.status = DocumentStatus.FAILED
                    doc.error_message = "Aborted: Parent thread is being deleted."
                    await session.commit()
                    return 0

            config = dict(doc.indexing_config or {})
            config.setdefault("preset", preset)
            config.setdefault("index_graph", False)
            graph_ready = not config["index_graph"] or (doc.provider_status or {}).get("lightrag") == "INDEXED"
            if doc.status == DocumentStatus.INDEXED and graph_ready:
                if not config.get("pending_generation") or config.get("pending_generation") == config.get("active_generation"):
                    return (await session.execute(select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == parsed_id))).scalar_one()
            if not config.get("pending_generation") or config.get("pending_generation") == config.get("active_generation"):
                config["pending_generation"] = str(uuid.uuid4())
            doc.indexing_config = config
            doc.status = DocumentStatus.PARSING
            if config["index_graph"]:
                doc.provider_status = {**(doc.provider_status or {}), "lightrag": "PARSING"}
            await session.commit()
        try:
            return await _index_document(
                parsed_id,
                r2_key,
                user_id,
                file_name,
                config,
                file_bytes,
                is_reindex,
                thread_id=effective_thread_id,
            )
        except BaseException as exc:
            await _update_document_status(parsed_id, DocumentStatus.FAILED, f"Ingestion failed: {exc}")
            raise


async def _index_document(
    parsed_id,
    r2_key,
    user_id,
    file_name,
    config,
    file_bytes,
    is_reindex,
    thread_id: str | None = None,
):
    document_id = str(parsed_id)
    raw_bytes = file_bytes if file_bytes is not None else await r2_storage_service.download_file(r2_key)
    extra_meta = {"document_id": document_id, "user_id": user_id, "file_name": file_name}
    if thread_id:
        extra_meta["thread_id"] = thread_id
    raw_doc = await asyncio.to_thread(
        load_document,
        source=raw_bytes,
        filename=file_name,
        extra_metadata=extra_meta,
    )
    chunks = await asyncio.to_thread(split_markdown_document, raw_doc, config=get_rag_config(config["preset"]))
    for idx, chunk in enumerate(chunks):
        fingerprint = hashlib.sha256(chunk.page_content.encode()).hexdigest()
        chunk.metadata["chunk_id"] = str(uuid.uuid5(parsed_id, f"{config['pending_generation']}:{idx}:{fingerprint}"))
        chunk.metadata["_base_generation"] = config["pending_generation"]
        chunk.metadata["_base_active"] = False
        if thread_id:
            chunk.metadata["thread_id"] = thread_id

    # Parse and chunk successfully before removing the previous index.
    if is_reindex:
        if config["index_graph"] or config.get("purge_graph"):
            from app.core.rag.factory import get_provider
            from app.core.rag.workspace import thread_workspace, user_workspace
            ws = thread_workspace(user_id, thread_id) if thread_id else user_workspace(user_id)
            provider = await get_provider("lightrag", ws)
            await provider.delete_document(document_id)

    await index_documents(chunks)
    statuses = {"hybrid": "INDEXED"}
    if config["index_graph"]:
        statuses["lightrag"] = "PARSING"
    async with async_session_factory() as session:
        if thread_id:
            from app.models.thread import Thread
            th = await session.get(Thread, thread_id)
            if th is None or (getattr(th, "metadata_json", None) or {}).get("status") == "DELETING":
                logger.info("Aborting chunk activation for document %s: thread %s is deleted or deleting", parsed_id, thread_id)
                doc = await session.get(Document, parsed_id)
                if doc:
                    doc.status = DocumentStatus.FAILED
                    doc.error_message = "Aborted: Parent thread is being deleted."
                    await session.commit()
                return 0

        await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == parsed_id))
        for idx, chunk in enumerate(chunks):
            session.add(DocumentChunk(id=uuid.UUID(chunk.metadata["chunk_id"]), document_id=parsed_id,
                chunk_index=idx, content=chunk.page_content, token_count=chunk.metadata.get("token_count", 0),
                metadata_json={**chunk.metadata, "_base_active": True}))
        doc = await session.get(Document, parsed_id)
        if doc is None:
            raise ValueError("Document disappeared during ingestion")
        doc.status = DocumentStatus.INDEXED
        doc.error_message = None
        doc.provider_status = dict(statuses)
        config["active_generation"] = config["pending_generation"]
        doc.indexing_config = dict(config)
        await _activate_generation(session, document_id, config["pending_generation"])
        await session.commit()

    if config["index_graph"]:
        try:
            from app.core.rag.factory import get_provider
            from app.core.rag.protocols import IngestDocument
            from app.core.rag.workspace import thread_workspace, user_workspace
            ws = thread_workspace(user_id, thread_id) if thread_id else user_workspace(user_id)
            provider = await get_provider("lightrag", ws)
            result = await provider.ingest([IngestDocument(doc_id=document_id, content=raw_doc.page_content,
                metadata=raw_doc.metadata, file_path=file_name)])
            if result.failed:
                raise ValueError("; ".join(result.errors))
            statuses["lightrag"] = "INDEXED"
        except Exception:
            logger.exception("Optional graph indexing failed for %s", document_id)
            statuses["lightrag"] = "FAILED"
        async with async_session_factory() as session:
            doc = await session.get(Document, parsed_id)
            if doc:
                doc.provider_status = statuses
                await session.commit()
    return len(chunks)


async def _activate_generation(session, document_id: str, generation: str):
    """Must share the metadata transaction: rollback preserves the old index."""
    params = {"doc_id": document_id, "generation": generation}
    await session.execute(text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'document_id' = :doc_id AND cmetadata->>'_base_generation' IS DISTINCT FROM :generation"), params)
    await session.execute(text("UPDATE langchain_pg_embedding SET cmetadata = jsonb_set(cmetadata, '{_base_active}', 'true'::jsonb) WHERE cmetadata->>'document_id' = :doc_id AND cmetadata->>'_base_generation' = :generation"), params)


async def _update_document_status(document_id, status, error_message=None):
    try:
        async with async_session_factory() as session:
            doc = await session.get(Document, document_id)
            if doc:
                doc.status = status
                doc.error_message = error_message
                if status == DocumentStatus.FAILED:
                    doc.provider_status = {name: "FAILED" if value == "PARSING" else value for name, value in (doc.provider_status or {}).items()}
                await session.commit()
    except Exception:
        logger.exception("Could not persist document failure status for %s", document_id)


def _recovery_done(task):
    _recovery_tasks.discard(task)
    if not task.cancelled() and task.exception():
        logger.error("Recovered document task failed: %s", task.exception())


async def recover_stuck_documents(stuck_threshold_seconds: int = 300) -> list[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=stuck_threshold_seconds)
    recovered = []
    async with async_session_factory() as session:
        ids = list((await session.execute(select(Document.id).where(
            Document.status == DocumentStatus.PARSING, Document.updated_at <= cutoff))).scalars().all())
    for doc_id in ids:
        try:
            # An active ingestion holds this lock, regardless of its age.
            async with document_lock(str(doc_id)) as acquired:
                if not acquired:
                    continue
                async with async_session_factory() as session:
                    doc = await session.get(Document, doc_id)
                    if not doc or doc.status != DocumentStatus.PARSING or doc.updated_at > cutoff:
                        continue
                    exists = await r2_storage_service.file_exists(doc.r2_key)
                    if not exists or doc.retry_count >= 3:
                        doc.status = DocumentStatus.FAILED
                        doc.error_message = "Source missing or recovery retry limit reached; reindex to retry"
                        await session.commit()
                        recovered.append(str(doc_id))
                        continue
                    doc.retry_count += 1
                    doc.updated_at = datetime.now(timezone.utc)
                    args: DocumentTaskArguments = {
                        "document_id": str(doc.id),
                        "r2_key": doc.r2_key,
                        "user_id": doc.user_id,
                        "file_name": doc.file_name,
                        "preset": (doc.indexing_config or {}).get("preset", "balanced"),
                        "is_reindex": True,
                        "thread_id": doc.thread_id,
                    }
                    await session.commit()
            try:
                from app.tasks.document_tasks import process_document_task
                await process_document_task.kiq(**args)
            except Exception:
                task = asyncio.create_task(process_document_pipeline(**args))
                _recovery_tasks.add(task)
                task.add_done_callback(_recovery_done)
            recovered.append(str(doc_id))
        except Exception:
            logger.exception("Recovery failed for document %s", doc_id)
    return recovered


async def start_document_sweeper(interval_seconds: int = 300) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await recover_stuck_documents(interval_seconds)
        except Exception:
            logger.exception("Document recovery sweep failed")


async def stop_document_recovery_tasks():
    tasks = list(_recovery_tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
