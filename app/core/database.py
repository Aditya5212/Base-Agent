from collections.abc import AsyncGenerator
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings

logger = logging.getLogger(__name__)

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


async def init_db() -> None:
    """Creates database tables defined in SQLAlchemy models if not already present, and seeds the root user."""
    from app.models.base import Base
    import app.models  # noqa: F401 - Ensure all models are registered
    from app.models.user import User

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Ensure migration safe columns
        await conn.execute(
            text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS retry_count INTEGER DEFAULT 0;")
        )

    # Seed default root user for production readiness if not exists
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
    except Exception as exc:
        logger.warning("Could not auto-seed root user (may already exist or DB not ready): %s", exc)


async def close_db_connection() -> None:
    """Closes all pooled database connections gracefully."""
    logger.info("Disposing database engine connection pool...")
    await engine.dispose()
    logger.info("Database engine disposed.")
