from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
import uuid

from httpx import ASGITransport, AsyncClient
import pytest
from botocore.exceptions import ClientError

from app.main import app
from app.models.agent import Agent
from app.models.structured_schema import StructuredSchema
from app.schemas.agent import AgentConfig
from app.services.r2_storage import R2StorageService, _is_transient_r2_error


@pytest.mark.asyncio
async def test_agent_endpoints_preserve_response_contract_for_orm_records():
    now = datetime.now(timezone.utc)
    record = Agent(id=uuid.uuid4(), user_id="response_contract_user", name="Contract agent",
                   description=None, config=AgentConfig().model_dump(), is_default=False,
                   is_published=False, version=1, created_at=now, updated_at=now)
    headers = {"X-User-ID": record.user_id}
    with patch("app.api.v1.endpoints.agents.create_agent", AsyncMock(return_value=record)), patch("app.api.v1.endpoints.agents.list_agents", AsyncMock(return_value=([record], 1))):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            created = await client.post("/api/v1/agents", json={"name": "Contract agent"}, headers=headers)
            listed = await client.get("/api/v1/agents", headers=headers)
    assert created.status_code == 201
    assert listed.status_code == 200
    assert created.json()["id"] == str(record.id)
    assert listed.json()["items"] == [created.json()]
    assert created.json()["config"]["model"]["provider"] == "deepseek"


@pytest.mark.asyncio
async def test_schema_endpoints_preserve_response_contract_for_orm_records():
    now = datetime.now(timezone.utc)
    record = StructuredSchema(id=uuid.uuid4(), user_id="schema_response_user", name="Result schema",
                              description=None, schema_definition={"type": "object"}, version=1,
                              created_at=now, updated_at=now)
    headers = {"X-User-ID": record.user_id}
    with patch("app.api.v1.endpoints.structured_schemas.create_structured_schema", AsyncMock(return_value=record)), patch("app.api.v1.endpoints.structured_schemas.list_structured_schemas", AsyncMock(return_value=([record], 1))):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            created = await client.post("/api/v1/structured-schemas", json={"name": record.name, "schema_definition": record.schema_definition}, headers=headers)
            listed = await client.get("/api/v1/structured-schemas", headers=headers)
    assert created.status_code == 201
    assert listed.status_code == 200
    assert listed.json()["items"] == [created.json()]
    assert created.json()["schema_definition"] == {"type": "object"}


@pytest.mark.asyncio
async def test_storage_sdk_provides_supported_async_client_context_without_requests():
    storage = R2StorageService()
    storage.access_key = "synthetic-key"
    storage.secret_key = "synthetic-secret"
    storage.endpoint_url = "http://127.0.0.1:1"
    storage.region_name = "us-east-1"
    # Creating and closing the actual SDK client loads local models only;
    # no service operation is invoked and credentials are explicitly synthetic.
    async with storage.get_client() as client:
        assert client.meta.service_model.service_name == "s3"
        assert callable(client.put_object)


def test_storage_error_classification_handles_sdk_and_generic_exceptions():
    transient = ClientError({"ResponseMetadata": {"HTTPStatusCode": 503}, "Error": {"Code": "ServiceUnavailable"}}, "GetObject")
    permanent = ClientError({"ResponseMetadata": {"HTTPStatusCode": 403}, "Error": {"Code": "AccessDenied"}}, "GetObject")
    assert _is_transient_r2_error(transient)
    assert not _is_transient_r2_error(permanent)
    assert not _is_transient_r2_error(ValueError("no SDK response"))
