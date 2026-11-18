import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from alembic import command
from alembic.config import Config
from fastapi import Response


def test_all_migrations_render_offline_with_execution_and_indexing_columns():
    sql = io.StringIO()
    cfg = Config("alembic.ini", output_buffer=sql)
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "head", sql=True)
    ddl = sql.getvalue()
    for column in ("checkpoint_id", "definition_snapshot", "indexing_config", "provider_status"):
        assert column in ddl
    assert "audit_execution_identity" in ddl
    assert "workflow_run_events" in ddl
    assert "control_request" in ddl
    assert "budget_seconds" in ddl
    assert "lease_expires_at" in ddl
    assert "generation" in ddl
    assert "workflow_dispatches" in ddl
    assert "workflow_tool_invocations" in ddl
    assert "resume_payload" in ddl
    assert "uq_tool_invocation_occurrence" in ddl


@pytest.mark.asyncio
async def test_production_startup_rejects_failed_migrations():
    from app.main import lifespan
    app = SimpleNamespace(state=SimpleNamespace())
    with patch("app.main.settings.ENVIRONMENT", "production"), patch("app.main.check_db_connection", AsyncMock(return_value=(True, None))), patch("app.core.database.init_db", AsyncMock(side_effect=RuntimeError("migration failed"))):
        with pytest.raises(RuntimeError, match="migration failed"):
            async with lifespan(app):
                pytest.fail("Startup must not serve requests after migration failure")
    assert app.state.startup_ready is False


@pytest.mark.asyncio
async def test_readiness_requires_initialization_not_just_database_connectivity():
    from app.api.v1.endpoints.health import health_check
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(startup_ready=False)))
    response = Response()
    with patch("app.api.v1.endpoints.health.check_db_connection", AsyncMock(return_value=(True, None))):
        health = await health_check(response, request)
    assert health.status == "degraded"
    assert response.status_code == 503


def test_model_and_embedding_mocks_are_disabled_in_production():
    from app.core.llm import get_llm
    from app.core.embedding import get_embeddings
    get_embeddings.cache_clear()
    with patch("app.core.llm.settings.ENVIRONMENT", "production"):
        with pytest.raises(ValueError, match="credentials"):
            get_llm(api_key="")
        with pytest.raises(ValueError, match="credentials"):
            get_embeddings()
    get_embeddings.cache_clear()
