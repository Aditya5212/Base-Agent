"""Alembic migration environment — async-compatible with psycopg3.

Reuses the application's DATABASE_URL from pydantic-settings so there is
a single source of truth for the connection string.  All ORM models are
imported here so autogenerate can diff against the live schema.
"""
from __future__ import annotations

import asyncio
import logging
from logging.config import fileConfig

from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# ---------------------------------------------------------------------------
# Load application settings & models
# ---------------------------------------------------------------------------
from app.core.config import settings  # noqa: E402 – must come after sys-path is set
from app.models.base import Base  # noqa: E402
import app.models  # noqa: F401, E402 – registers all ORM models with Base.metadata

# ---------------------------------------------------------------------------
# Alembic Config object (gives access to alembic.ini values)
# ---------------------------------------------------------------------------
config = context.config

# Wire up Python logging from alembic.ini only if not running programmatically from app
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)


logger = logging.getLogger("alembic.env")

# ---------------------------------------------------------------------------
# Inject the live DATABASE_URL so alembic.ini doesn't need to duplicate it
# ---------------------------------------------------------------------------
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# ---------------------------------------------------------------------------
# Target metadata — required for --autogenerate
# ---------------------------------------------------------------------------
target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Table inclusion filter
# Only tables whose names are registered in Base.metadata are managed by
# Alembic.  Everything else (LightRAG, LangGraph checkpoints, langchain_pg_*)
# is created and managed by those libraries — Alembic must never touch them.
# ---------------------------------------------------------------------------
_MANAGED_TABLES: frozenset[str] = frozenset(Base.metadata.tables.keys()) | {"alembic_version"}


def include_name(name, type_, parent_names):  # noqa: ARG001
    """Only include tables that are defined in our ORM Base metadata or alembic_version."""
    if type_ == "table":
        return name in _MANAGED_TABLES
    return True  # always include schemas, columns, indexes for managed tables


def include_object(object, name, type_, reflected, compare_to):  # noqa: A002
    """Exclude reflected objects whose parent table is not managed by us."""
    return True


def compare_type(context, inspected_column, metadata_column, inspected_type, metadata_type):  # noqa: ARG001
    """Return False (= no change) for vector/tsvector columns to suppress noise."""
    inspected_str = str(inspected_type).upper()
    metadata_str = str(metadata_type).upper()
    # Suppress noisy diffs for vector and tsvector columns
    if "VECTOR" in inspected_str or "TSVECTOR" in inspected_str:
        return False
    if "VECTOR" in metadata_str or "TSVECTOR" in metadata_str:
        return False
    return None  # let Alembic decide for everything else


# ---------------------------------------------------------------------------
# Offline migration (generates SQL without a live DB connection)
# ---------------------------------------------------------------------------
def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with just a URL without creating an Engine.
    Useful for generating a migration SQL script.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_name=include_name,
        include_object=include_object,
        compare_type=compare_type,
    )
    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migration (runs against a live async DB connection)
# ---------------------------------------------------------------------------
def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_name=include_name,
        include_object=include_object,
        compare_type=compare_type,
        render_as_batch=False,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations inside a sync wrapper."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # Never reuse connections for migrations
    )

    async with connectable.connect() as connection:
        # Ensure pgvector extension exists before running migrations
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await connection.run_sync(do_run_migrations)
        await connection.commit()

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode with an async engine."""
    asyncio.run(run_async_migrations())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
