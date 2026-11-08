"""Document ingestion and indexing background tasks executed by Taskiq workers."""

from __future__ import annotations

import logging

from app.core.taskiq import broker
from app.services.document_processor import process_document_pipeline

logger = logging.getLogger(__name__)


@broker.task(task_name="process_document_task")
async def process_document_task(
    document_id: str,
    r2_key: str,
    user_id: str,
    file_name: str,
    preset: str = "balanced",
    is_reindex: bool = False,
    thread_id: str | None = None,
) -> bool:
    """Taskiq worker task for asynchronous document parsing, chunking, and indexing.

    Delegates directly to process_document_pipeline, fetching raw binary bytes
    from Cloudflare R2 storage in the worker process.
    """
    logger.info(
        "Taskiq worker executing process_document_task for doc %s (user=%s, preset=%s, thread=%s)",
        document_id,
        user_id,
        preset,
        thread_id,
    )
    try:
        await process_document_pipeline(
            document_id=document_id,
            r2_key=r2_key,
            user_id=user_id,
            file_name=file_name,
            preset=preset,
            file_bytes=None,  # Worker downloads directly from R2 via r2_key
            is_reindex=is_reindex,
            thread_id=thread_id,
        )
        logger.info("Taskiq process_document_task succeeded for doc %s", document_id)
        return True
    except Exception as exc:
        logger.error(
            "Taskiq process_document_task failed for doc %s: %s",
            document_id,
            exc,
            exc_info=True,
        )
        raise
