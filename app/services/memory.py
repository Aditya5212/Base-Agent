"""PostgreSQL checkpointer service for persistent short-term agent memory.

Uses psycopg3's AsyncConnectionPool and LangGraph's native AsyncPostgresSaver
to persist conversation state across requests and thread sessions.
"""

from collections.abc import Sequence
import logging
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global connection pool and checkpointer instances
_pool: AsyncConnectionPool | None = None
_checkpointer: AsyncPostgresSaver | None = None


async def get_checkpointer_pool() -> AsyncConnectionPool:
    """Returns the active AsyncConnectionPool for the checkpointer."""
    global _pool
    if _pool is None or _pool.closed:
        logger.info("Initializing PostgreSQL checkpointer connection pool...")
        _pool = AsyncConnectionPool(
            conninfo=settings.raw_database_url,
            max_size=settings.DATABASE_POOL_SIZE,
            min_size=1,
            open=False,
            # Required by AsyncPostgresSaver: setup() runs CREATE INDEX CONCURRENTLY,
            # which fails inside a transaction block on a fresh database.
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        await _pool.open()
    return _pool


async def get_checkpointer() -> AsyncPostgresSaver:
    """Returns the AsyncPostgresSaver instance backed by the connection pool."""
    global _checkpointer
    pool = await get_checkpointer_pool()
    if _checkpointer is None:
        _checkpointer = AsyncPostgresSaver(pool)
    return _checkpointer


async def setup_checkpointer() -> None:
    """Initializes checkpointer tables (checkpoints, checkpoint_blobs, checkpoint_writes).

    Should be called during application lifespan startup.
    """
    try:
        checkpointer = await get_checkpointer()
        await checkpointer.setup()
        logger.info("PostgreSQL checkpointer tables verified and ready.")
    except Exception as exc:
        logger.error("Failed to setup PostgreSQL checkpointer: %s", exc)
        raise


async def close_checkpointer_pool() -> None:
    """Closes the checkpointer connection pool gracefully on shutdown."""
    global _pool, _checkpointer
    if _pool is not None and not _pool.closed:
        logger.info("Closing PostgreSQL checkpointer connection pool...")
        await _pool.close()
        logger.info("Checkpointer connection pool closed.")
    _pool = None
    _checkpointer = None


async def get_thread_messages(thread_id: str) -> list[BaseMessage]:
    """Retrieves all persisted messages for a given thread_id."""
    checkpointer = await get_checkpointer()
    config = {"configurable": {"thread_id": thread_id}}
    tuple_result = await checkpointer.aget_tuple(config)
    if not tuple_result or not tuple_result.checkpoint:
        return []

    channel_values = tuple_result.checkpoint.get("channel_values", {})
    messages = channel_values.get("messages", [])
    if isinstance(messages, list):
        return messages
    return []


async def delete_thread_memory(thread_id: str) -> bool:
    """Deletes all checkpoint memory records for a given thread_id."""
    try:
        checkpointer = await get_checkpointer()
        await checkpointer.adelete_thread(thread_id)
        logger.info("Deleted checkpoint memory for thread_id=%s", thread_id)
        return True
    except Exception as exc:
        logger.error("Error deleting thread memory for thread_id=%s: %s", thread_id, exc)
        return False
