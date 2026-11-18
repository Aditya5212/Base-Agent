"""Read-only database query tool executor.

Executes parameterized SQL queries against external PostgreSQL or SQLite databases
with strict safety verification, parameter binding, row-count limits, and timeouts.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from urllib.parse import quote
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.crypto import decrypt_secret
from app.schemas.tool import DatabaseToolConfig

logger = logging.getLogger(__name__)


class DatabaseToolExecutionError(Exception):
    """Raised when database tool query execution fails."""


FORBIDDEN_SQL_KEYWORDS: tuple[str, ...] = (
    "insert ", "update ", "delete ", "drop ", "alter ",
    "truncate ", "grant ", "revoke ", "create ", "execute ", "exec "
)


def validate_readonly_sql(query: str) -> None:
    """Verifies that the SQL query is strictly read-only."""
    cleaned = query.strip().lower()
    if not re.match(r"^(select|with)\b", cleaned):
        raise DatabaseToolExecutionError(
            "Security restriction: Only read-only queries ('SELECT' or 'WITH ... SELECT') are permitted."
        )

    for word in FORBIDDEN_SQL_KEYWORDS:
        if re.search(r"\b" + word.strip() + r"\b", cleaned):
            raise DatabaseToolExecutionError(
                f"Security restriction: Forbidden SQL command '{word.strip().upper()}' detected."
            )


class DatabaseToolExecutor:
    """Executes read-only parameterized queries against target databases."""

    def __init__(self, config: DatabaseToolConfig):
        self.config = config
        validate_readonly_sql(self.config.query_template)

    async def execute(self, arguments: dict[str, Any], metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Executes the query with parameterized inputs and returns rows as dictionaries."""
        raw_conn = self.config.connection_string
        resolved_connection = decrypt_secret(raw_conn) if raw_conn.startswith("gAAAAA") else raw_conn
        if not resolved_connection:
            raise DatabaseToolExecutionError("Database connection string is empty")
        connection_url: str = resolved_connection

        validate_readonly_sql(self.config.query_template)

        max_rows = max(1, min(self.config.max_rows, 1000))
        args = dict(arguments or {})

        # 1. Standard SQLite execution via standard library
        if connection_url.startswith(("sqlite:", "sqlite+aiosqlite:")):
            import sqlite3

            db_path = (
                connection_url.replace("sqlite+aiosqlite:///", "")
                .replace("sqlite:///", "")
                .replace("sqlite://", "")
            )

            def _run_sqlite() -> list[dict[str, Any]]:
                target_path = ":memory:" if db_path in (":memory:", "") else db_path
                conn = sqlite3.connect(":memory:") if target_path == ":memory:" else sqlite3.connect(f"file:{quote(target_path, safe='/!:')}?mode=ro", uri=True)
                conn.row_factory = sqlite3.Row
                try:
                    conn.execute("PRAGMA query_only = ON")
                    deadline = time.monotonic() + self.config.timeout_seconds
                    conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
                    cur = conn.cursor()
                    cur.execute(self.config.query_template, args)
                    rows = cur.fetchmany(max_rows)
                    return [dict(r) for r in rows]
                finally:
                    conn.close()

            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(_run_sqlite),
                    timeout=self.config.timeout_seconds,
                )
            except asyncio.TimeoutError as err:
                raise DatabaseToolExecutionError(
                    f"Query execution timed out after {self.config.timeout_seconds} seconds."
                ) from err
            except Exception as exc:
                raise DatabaseToolExecutionError(f"Database query failed: {exc}") from exc

        # 2. Async SQLAlchemy execution for PostgreSQL and other SQL engines
        if connection_url.startswith("postgresql://"):
            connection_url = connection_url.replace("postgresql://", "postgresql+psycopg://", 1)

        async def _run_query() -> list[dict[str, Any]]:
            temp_engine = create_async_engine(
                connection_url,
                pool_pre_ping=True,
                echo=False,
            )
            try:
                async with temp_engine.begin() as conn:
                    await conn.execute(text("SET TRANSACTION READ ONLY"))
                    await conn.execute(text("SELECT set_config('statement_timeout', :timeout, true)"), {"timeout": str(int(self.config.timeout_seconds * 1000))})
                    stmt = text(self.config.query_template)
                    result = await conn.execute(stmt, args)
                    rows = result.fetchmany(max_rows)
                    return [dict(row._mapping) for row in rows]
            finally:
                await temp_engine.dispose()

        try:
            return await asyncio.wait_for(
                _run_query(),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError as err:
            logger.warning("Database query execution timed out after %s seconds", self.config.timeout_seconds)
            raise DatabaseToolExecutionError(
                f"Query execution timed out after {self.config.timeout_seconds} seconds."
            ) from err
        except Exception as exc:
            logger.error("Database Tool execution error: %s", exc)
            raise DatabaseToolExecutionError(f"Database query failed: {exc}") from exc

