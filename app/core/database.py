"""Database engine, session factory, and startup initialization.

Startup sequence
────────────────
1. `run_migrations()` — runs `alembic upgrade head` programmatically.
   This is idempotent: if the schema is already at HEAD it's a no-op.
   Tables managed by external libs (LightRAG, LangGraph checkpoints,
   langchain_pg_embedding) are NOT touched by Alembic — they are still
   created by their respective library initializers in the lifespan hook.
2. `seed_root_user()` — inserts the default root user if it doesn't exist.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator

from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Async engine & session factory
# ---------------------------------------------------------------------------
engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_timeout=settings.DATABASE_POOL_TIMEOUT,
    pool_recycle=settings.DATABASE_POOL_RECYCLE,
    pool_pre_ping=True,
)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

# Alias for generic factory naming conventions
async_session_factory = AsyncSessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that provides an async SQLAlchemy session.

    Ensures proper cleanup and rollback on unhandled errors.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_db_connection() -> tuple[bool, str | None]:
    """Check database connectivity with a lightweight ping query.

    Returns:
        (True, None) if connected successfully.
        (False, error_message) if connection failed.
    """
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True, None
    except Exception as exc:
        logger.warning("Database connectivity check failed: %s", exc)
        return False, str(exc)


# ---------------------------------------------------------------------------
# Alembic migration runner
# ---------------------------------------------------------------------------
def _run_alembic_upgrade() -> None:
    """Runs `alembic upgrade head` synchronously (called from asyncio.to_thread).

    Uses a temporary sync engine with NullPool so it doesn't interfere with
    the async engine's connection pool.
    """
    import pathlib

    # Locate alembic.ini relative to this file's project root
    project_root = pathlib.Path(__file__).resolve().parents[2]
    alembic_cfg = AlembicConfig(str(project_root / "alembic.ini"))
    alembic_cfg.attributes["configure_logger"] = False
    # Override sqlalchemy.url so env.py doesn't need to re-read settings
    alembic_cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
    alembic_command.upgrade(alembic_cfg, "head")



async def run_migrations() -> None:
    """Run Alembic migrations asynchronously (non-blocking)."""
    logger.info("Running Alembic migrations (upgrade head)...")
    try:
        await asyncio.to_thread(_run_alembic_upgrade)
        logger.info("Alembic migrations complete.")
    except Exception as exc:
        logger.error("Alembic migration failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Root user seeding
# ---------------------------------------------------------------------------
async def seed_root_user() -> None:
    """Insert the default root user if it does not already exist."""
    from app.models.user import User

    try:
        async with AsyncSessionLocal() as session:
            root_user = await session.get(User, "root")
            if not root_user:
                root_user = User(
                    id="root",
                    email="root@baseagent.ai",
                    name="Root Administrator",
                    role="root",
                    metadata_json={"description": "Default root user for Base Agent"},
                )
                session.add(root_user)
                await session.commit()
                logger.info("Default root user seeded in database.")

            # Ensure root user has default agent seeded
            from app.services.agent_service import get_or_create_default_agent
            await get_or_create_default_agent(session, "root")
    except Exception as exc:
        logger.warning(
            "Could not auto-seed root user or default agent (may already exist or DB not ready): %s",
            exc,
        )


async def init_db() -> None:
    """Full startup sequence: run migrations, then seed the root user."""
    await run_migrations()
    await seed_root_user()


async def close_db_connection() -> None:
    """Closes all pooled database connections gracefully."""
    logger.info("Disposing database engine connection pool...")
    await engine.dispose()
    logger.info("Database engine disposed.")
