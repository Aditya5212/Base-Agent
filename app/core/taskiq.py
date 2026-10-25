"""Taskiq distributed task broker configuration with PostgreSQL broker.

Uses PostgreSQL LISTEN/NOTIFY and persistent task storage via taskiq-pg.
Allows distributed asynchronous task processing across separate worker containers
without introducing Celery or Redis infrastructure requirements.
"""

from __future__ import annotations

import logging
from typing import Any

from taskiq import AsyncBroker
from taskiq_pg import AsyncpgBroker, AsyncpgResultBackend

from app.core.config import settings

logger = logging.getLogger(__name__)


def _get_asyncpg_dsn() -> str:
    """Normalizes the application DATABASE_URL to an asyncpg-compatible DSN."""
    dsn = str(settings.DATABASE_URL)
    # Replace psycopg driver prefix with standard postgresql://
    if dsn.startswith("postgresql+psycopg://"):
        dsn = "postgresql://" + dsn[len("postgresql+psycopg://"):]
    elif dsn.startswith("postgresql+asyncpg://"):
        dsn = "postgresql://" + dsn[len("postgresql+asyncpg://"):]
    return dsn


# Broker and result backend singletons
_dsn = _get_asyncpg_dsn()
result_backend = AsyncpgResultBackend(dsn=_dsn)
broker: AsyncBroker = AsyncpgBroker(dsn=_dsn).with_result_backend(result_backend)


async def init_taskiq_broker() -> None:
    """Initializes the Taskiq broker connection and ensures database tables exist."""
    logger.info("Initializing Taskiq PostgreSQL broker...")
    try:
        await broker.startup()
        logger.info("Taskiq PostgreSQL broker started successfully.")
    except Exception as exc:
        logger.warning("Could not initialize Taskiq broker: %s", exc)


async def shutdown_taskiq_broker() -> None:
    """Closes all Taskiq broker connections gracefully."""
    logger.info("Shutting down Taskiq broker...")
    try:
        await broker.shutdown()
        logger.info("Taskiq broker shutdown complete.")
    except Exception as exc:
        logger.warning("Error during Taskiq broker shutdown: %s", exc)
