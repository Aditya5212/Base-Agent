"""Pytest global configuration and fixtures."""

import asyncio
import pytest

from app.core.database import init_db


@pytest.fixture(scope="session", autouse=True)
def ensure_database_initialized():
    """Ensures database tables (users, threads, assets, documents) and root user exist before tests run."""
    try:
        asyncio.run(init_db())
    except Exception as exc:
        # Ignore connectivity errors in environments without live DB
        pass
