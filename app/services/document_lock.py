"""Serialize ingestion/reindex/delete on a dedicated PostgreSQL connection."""
from contextlib import asynccontextmanager
import hashlib

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings

# Never hold a pooled application connection while indexing needs another one.
_lock_engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)


@asynccontextmanager
async def document_lock(document_id: str):
    key = int.from_bytes(hashlib.sha256(f"document:{document_id}".encode()).digest()[:8], "big", signed=True)
    async with _lock_engine.connect() as conn:
        acquired = bool((await conn.execute(text("SELECT pg_try_advisory_lock(:key)"), {"key": key})).scalar())
        try:
            yield acquired
        finally:
            if acquired:
                # The same physical connection owns and releases the session lock.
                await conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})
                await conn.commit()
