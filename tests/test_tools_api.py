"""Comprehensive unit and integration tests for the Tool Builder platform.

Validates:
- Cryptographic encryption, decryption, and secret masking
- HTTP, Sandboxed Code, and Database tool executors
- Dynamic ToolRegistry compilation, LRU caching, and argument schema generation
- Tool CRUD API endpoints (/api/v1/tools) with tenant isolation and secret masking
- Direct test-execution endpoint (/api/v1/tools/{id}/execute)
- Knowledge attachment API (/api/v1/workflows/{id}/nodes/{node_id}/knowledge)
- Workflow ToolCallNode execution of registered custom tools
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from httpx import ASGITransport, AsyncClient, Response

from app.core.crypto import (
    CryptoError,
    decrypt_secret,
    encrypt_secret,
    mask_secret,
)
from app.main import app
from app.models.tool import Tool
from app.schemas.tool import (
    CodeToolConfig,
    DatabaseToolConfig,
    HttpAuthType,
    HttpToolConfig,
    ToolCreate,
    ToolType,
    ToolUpdate,
)
from app.services.tool_executors.code_executor import (
    CodeExecutionError,
    CodeSecurityError,
    CodeToolExecutor,
)
from app.services.tool_executors.database_executor import (
    DatabaseToolExecutionError,
    DatabaseToolExecutor,
)
from app.services.tool_executors.http_executor import (
    HttpToolExecutionError,
    HttpToolExecutor,
)
from app.services.tool_registry import ToolRegistry, json_schema_to_pydantic_model
from app.services.tool_service import ToolService
from app.services.workflow.nodes.tool_call import build_tool_call_node


# ---------------------------------------------------------------------------
# 1. Secret Encryption & Masking Tests
# ---------------------------------------------------------------------------

def test_secret_encryption_and_decryption():
    """Verifies round-trip Fernet encryption and decryption."""
    raw_secret = "sk-live-deepseek-1234567890abcdef"
    encrypted = encrypt_secret(raw_secret)
    assert encrypted is not None
    assert encrypted != raw_secret
    assert encrypted.startswith("gAAAAA")

    decrypted = decrypt_secret(encrypted)
    assert decrypted == raw_secret

    # Empty handling
    assert encrypt_secret(None) is None
    assert decrypt_secret(None) is None


def test_secret_masking():
    """Verifies secret masking protects sensitive tokens."""
    token = "sk-ant-api03-abcdef123456"
    masked = mask_secret(token, prefix_len=4, suffix_len=4)
    assert masked == "sk-a***3456"
    assert "abcdef" not in masked

    short_token = "abc"
    assert mask_secret(short_token) == "***"
    assert mask_secret(None) == ""


def test_corrupted_token_decryption_error():
    """Verifies decrypting invalid ciphertext raises CryptoError."""
    with pytest.raises(CryptoError):
        decrypt_secret("not-a-valid-fernet-token")


# ---------------------------------------------------------------------------
# 2. HTTP Tool Executor Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_http_tool_executor_get_with_params():
    """Verifies HTTP executor handles GET with URL interpolation and query params."""
    cfg = HttpToolConfig(
        method="GET",
        url_template="https://api.example.com/v1/weather/{city}",
        headers_template={"X-Client": "BaseAgent"},
        response_path="forecast.temp",
    )
    executor = HttpToolExecutor(cfg)

    mock_resp = Response(
        200,
        json={"forecast": {"temp": 72.5, "condition": "Sunny"}},
        headers={"content-type": "application/json"},
    )

    with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp

        result = await executor.execute({"city": "San Francisco", "units": "imperial"})

        assert result == 72.5
        mock_req.assert_called_once()
        call_kwargs = mock_req.call_args.kwargs
        assert "https://api.example.com/v1/weather/San%20Francisco" in call_kwargs["url"]
        assert call_kwargs["params"] == {"units": "imperial"}
        assert call_kwargs["headers"]["X-Client"] == "BaseAgent"


@pytest.mark.asyncio
async def test_http_tool_executor_post_with_bearer_auth():
    """Verifies HTTP executor injects bearer auth token and handles POST JSON payload."""
    raw_token = "secret-token-xyz"
    enc_token = encrypt_secret(raw_token)

    cfg = HttpToolConfig(
        method="POST",
        url_template="https://api.example.com/v1/orders",
        auth_type=HttpAuthType.BEARER,
        auth_token=enc_token,
        body_template={"product_id": "{item}", "quantity": "{qty}"},
    )
    executor = HttpToolExecutor(cfg)

    mock_resp = Response(
        201,
        json={"order_id": "ord_999", "status": "confirmed"},
        headers={"content-type": "application/json"},
    )

    with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp

        result = await executor.execute({"item": "sku_101", "qty": 3})

        assert result == {"order_id": "ord_999", "status": "confirmed"}
        call_kwargs = mock_req.call_args.kwargs
        assert call_kwargs["headers"]["Authorization"] == f"Bearer {raw_token}"
        assert call_kwargs["json"] == {"product_id": "sku_101", "quantity": "3"}


# ---------------------------------------------------------------------------
# 3. Sandboxed Code Tool Executor Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_code_tool_executor_valid_execution():
    """Verifies sandboxed Python execution for permitted math and string operations."""
    code = """
import math

def execute(radius: float) -> dict:
    area = math.pi * (radius ** 2)
    circumference = 2 * math.pi * radius
    return {
        "radius": radius,
        "area": round(area, 2),
        "circumference": round(circumference, 2),
    }
"""
    cfg = CodeToolConfig(code=code, entrypoint="execute")
    executor = CodeToolExecutor(cfg)

    result = await executor.execute({"radius": 5.0})
    assert result["radius"] == 5.0
    assert result["area"] == 78.54
    assert result["circumference"] == 31.42


def test_code_tool_executor_blocks_forbidden_import():
    """Verifies importing non-whitelisted modules (os, sys, subprocess) is blocked at AST inspection."""
    bad_code = """
import os

def execute():
    return os.listdir(".")
"""
    with pytest.raises(CodeSecurityError, match="Import of module 'os' is prohibited"):
        CodeToolExecutor(CodeToolConfig(code=bad_code))


def test_code_tool_executor_blocks_eval_and_exec():
    """Verifies calls to eval() or exec() are blocked at AST inspection."""
    bad_code = """
def execute(cmd: str):
    return eval(cmd)
"""
    with pytest.raises(CodeSecurityError, match="Call to built-in function 'eval' is prohibited"):
        CodeToolExecutor(CodeToolConfig(code=bad_code))


@pytest.mark.asyncio
async def test_code_tool_executor_timeout_guard():
    """Verifies long execution is halted when exceeding execution timeout."""
    loop_code = """
def execute():
    total = 0
    for i in range(100000000):
        total += i
    return total
"""
    cfg = CodeToolConfig(code=loop_code, timeout_seconds=0.5)
    executor = CodeToolExecutor(cfg)

    with pytest.raises(CodeExecutionError, match="Execution timed out"):
        await executor.execute({})


# ---------------------------------------------------------------------------
# 4. Database Tool Executor Tests
# ---------------------------------------------------------------------------

def test_database_tool_executor_blocks_forbidden_sql():
    """Verifies non-SELECT statements (DROP, INSERT, DELETE) are rejected."""
    from app.services.tool_executors.database_executor import validate_readonly_sql

    with pytest.raises(DatabaseToolExecutionError, match="Only read-only queries"):
        validate_readonly_sql("DELETE FROM users WHERE id = 1")

    with pytest.raises(DatabaseToolExecutionError, match="Forbidden SQL command 'DROP'"):
        validate_readonly_sql("SELECT 1; DROP TABLE users;")


@pytest.mark.asyncio
async def test_database_tool_executor_parameterized_query():
    """Verifies read-only parameterized query execution using in-memory SQLite."""
    cfg = DatabaseToolConfig(
        connection_string="sqlite:///:memory:",
        query_template="SELECT :val * 2 AS doubled, 'ok' AS status",
    )
    executor = DatabaseToolExecutor(cfg)

    rows = await executor.execute({"val": 21})
    assert len(rows) == 1
    assert rows[0]["doubled"] == 42
    assert rows[0]["status"] == "ok"


# ---------------------------------------------------------------------------
# 5. Dynamic Tool Registry Tests
# ---------------------------------------------------------------------------

def test_json_schema_to_pydantic_model():
    """Verifies conversion of JSON Schema properties to a typed Pydantic model."""
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max items", "default": 10},
        },
        "required": ["query"],
    }
    model = json_schema_to_pydantic_model(schema, "SearchArgs")
    instance = model(query="machine learning")
    assert instance.query == "machine learning"
    assert instance.limit == 10


def test_tool_registry_compiles_code_tool_with_cache():
    """Verifies ToolRegistry compiles Tool ORM record into BaseTool and caches it."""
    code = "def execute(x: int) -> int: return x * 3"
    tool_orm = Tool(
        user_id="alice",
        name="triple_number",
        description="Triples the input number",
        type="code",
        input_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
        config={"code": code, "entrypoint": "execute"},
        version=1,
    )

    base_tool = ToolRegistry.compile_tool(tool_orm)
    assert base_tool.name == "triple_number"
    assert base_tool.description == "Triples the input number"

    # Second compilation hits cache
    cached_tool = ToolRegistry.compile_tool(tool_orm)
    assert cached_tool is base_tool

    # Invalidation removes from cache
    ToolRegistry.invalidate_tool("alice", str(tool_orm.id))


@pytest.mark.asyncio
async def test_tool_registry_resolves_built_ins_and_rag():
    """Verifies ToolRegistry resolves standard calculator and RAG research tools."""
    resolved = await ToolRegistry.resolve_tools(
        user_id="root",
        tool_names=["calculator", "hybrid_document_search"],
    )
    assert len(resolved) == 2
    tool_names = [t.name for t in resolved]
    assert "calculator" in tool_names
    assert "hybrid_document_search" in tool_names


# ---------------------------------------------------------------------------
# 6. Tool CRUD & Execution API Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_get_and_delete_tool_api():
    """Verifies POST /api/v1/tools, GET /api/v1/tools/{id}, and DELETE /api/v1/tools/{id}."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        headers = {"X-User-Id": "test_user_tools_crud"}

        # 1. Create Code Tool
        create_payload = {
            "name": "celsius_to_fahrenheit",
            "description": "Converts temperature in Celsius to Fahrenheit",
            "type": "code",
            "input_schema": {
                "type": "object",
                "properties": {"celsius": {"type": "number", "description": "Temperature in Celsius"}},
                "required": ["celsius"],
            },
            "config": {
                "code": "def execute(celsius: float) -> float:\n    return (celsius * 9/5) + 32\n",
                "entrypoint": "execute",
            },
        }

        resp = await ac.post("/api/v1/tools", json=create_payload, headers=headers)
        assert resp.status_code == 201, resp.text
        data = resp.json()
        tool_id = data["id"]
        assert data["name"] == "celsius_to_fahrenheit"
        assert data["version"] == 1

        # 2. Get Tool details
        get_resp = await ac.get(f"/api/v1/tools/{tool_id}", headers=headers)
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == tool_id

        # 3. Direct Test-Execute Tool
        exec_resp = await ac.post(
            f"/api/v1/tools/{tool_id}/execute",
            json={"arguments": {"celsius": 100.0}},
            headers=headers,
        )
        assert exec_resp.status_code == 200
        exec_data = exec_resp.json()
        assert exec_data["success"] is True
        assert "212" in str(exec_data["output"])

        # 4. Update Tool description
        update_resp = await ac.put(
            f"/api/v1/tools/{tool_id}",
            json={"description": "Updated conversion description"},
            headers=headers,
        )
        assert update_resp.status_code == 200
        assert update_resp.json()["version"] == 2
        assert update_resp.json()["description"] == "Updated conversion description"

        # 5. List Tools
        list_resp = await ac.get("/api/v1/tools", headers=headers)
        assert list_resp.status_code == 200
        assert list_resp.json()["total"] >= 1

        # 6. Delete Tool
        del_resp = await ac.delete(f"/api/v1/tools/{tool_id}", headers=headers)
        assert del_resp.status_code == 204

        # 7. Verify 404 after deletion
        not_found = await ac.get(f"/api/v1/tools/{tool_id}", headers=headers)
        assert not_found.status_code == 404


@pytest.mark.asyncio
async def test_tool_api_secret_masking():
    """Verifies that sensitive credentials (auth_token) are never leaked in API responses."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        headers = {"X-User-Id": "test_secret_mask_user"}

        create_payload = {
            "name": "crm_fetch_customer",
            "description": "Fetches customer data from remote CRM",
            "type": "api",
            "input_schema": {"type": "object", "properties": {"cust_id": {"type": "string"}}},
            "config": {
                "method": "GET",
                "url_template": "https://crm.internal.net/customers/{cust_id}",
                "auth_type": "bearer",
                "auth_token": "sk-super-secret-production-token-123456",
            },
        }

        resp = await ac.post("/api/v1/tools", json=create_payload, headers=headers)
        assert resp.status_code == 201
        data = resp.json()
        tool_id = data["id"]

        # Sensitive token must be masked in response
        masked_token = data["config"]["auth_token"]
        assert "super-secret" not in masked_token
        assert "***" in masked_token

        # Clean up
        await ac.delete(f"/api/v1/tools/{tool_id}", headers=headers)


@pytest.mark.asyncio
async def test_tool_tenant_isolation():
    """Verifies user Alice cannot read or update user Bob's private tools."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        alice_headers = {"X-User-Id": "alice_tenant"}
        bob_headers = {"X-User-Id": "bob_tenant"}

        # Alice creates a tool
        create_payload = {
            "name": "alice_private_tool",
            "description": "Alice's proprietary analysis tool",
            "type": "code",
            "config": {
                "code": "def execute(): return 'alice secret'",
                "entrypoint": "execute",
            },
        }

        resp = await ac.post("/api/v1/tools", json=create_payload, headers=alice_headers)
        assert resp.status_code == 201
        alice_tool_id = resp.json()["id"]

        # Bob attempts to read Alice's tool -> 404
        bob_get = await ac.get(f"/api/v1/tools/{alice_tool_id}", headers=bob_headers)
        assert bob_get.status_code == 404

        # Bob attempts to update Alice's tool -> 404
        bob_put = await ac.put(
            f"/api/v1/tools/{alice_tool_id}",
            json={"description": "Bob trying to hijack"},
            headers=bob_headers,
        )
        assert bob_put.status_code == 404

        # Clean up
        await ac.delete(f"/api/v1/tools/{alice_tool_id}", headers=alice_headers)


# ---------------------------------------------------------------------------
# 7. Workflow ToolCallNode & Knowledge Attachment Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_workflow_tool_call_node_executes_custom_db_tool():
    """Verifies that ToolCallNode dynamically resolves and executes a custom registered tool."""
    from app.core.database import AsyncSessionLocal

    user_id = "wf_tool_test_user"
    async with AsyncSessionLocal() as session:
        prior = await ToolService.get_tool_by_name(session, user_id=user_id, name="multiply_calculator")
        if prior:
            await ToolService.delete_tool(session, user_id=user_id, tool_id=prior.id)

        tool_in = ToolCreate(
            name="multiply_calculator",
            description="Multiplies two numbers",
            type=ToolType.CODE,
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
            config={
                "code": "def execute(a: float, b: float) -> float:\n    return a * b\n",
                "entrypoint": "execute",
            },
        )
        tool_orm = await ToolService.create_tool(session, user_id=user_id, tool_in=tool_in)

        # Build workflow tool_call node referencing the newly created tool by name
        node_handler = build_tool_call_node("calc_node", {
            "tool_name": "multiply_calculator",
            "input_mapping": {"a": "{{variables.num1}}", "b": 7},
            "output_variable": "calc_result",
        })

        state = {
            "user_id": user_id,
            "variables": {"num1": 6},
            "node_history": [],
        }

        result = await node_handler(state)
        assert "42" in str(result["variables"]["calc_result"])

        # Clean up
        await ToolService.delete_tool(session, user_id=user_id, tool_id=tool_orm.id)


@pytest.mark.asyncio
async def test_workflow_attach_node_knowledge_endpoint():
    """Verifies POST /api/v1/workflows/{id}/nodes/{node_id}/knowledge endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        headers = {"X-User-Id": "wf_knowledge_user"}

        # 1. Create a simple workflow with a retrieve node
        wf_payload = {
            "name": "Knowledge Test Workflow",
            "description": "Workflow for testing knowledge configuration endpoint",
            "nodes": [
                {"id": "start", "type": "start", "name": "Start", "config": {}},
                {
                    "id": "ret_1",
                    "type": "retrieve",
                    "name": "Retrieve",
                    "config": {"provider_name": "hybrid", "top_k": 3},
                },
                {"id": "end", "type": "end", "name": "End", "config": {}},
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "ret_1"},
                {"id": "e2", "source": "ret_1", "target": "end"},
            ],
        }

        create_resp = await ac.post("/api/v1/workflows", json=wf_payload, headers=headers)
        assert create_resp.status_code == 201
        wf_data = create_resp.json()
        wf_id = wf_data["id"]

        # 2. Attach / update knowledge on the retrieve node to multi_query provider
        knowledge_payload = {
            "provider_name": "multi_query",
            "workspace_type": "thread",
            "top_k": 8,
            "query_template": "{{variables.search_queries}}",
            "output_variable": "multi_context",
        }

        attach_resp = await ac.post(
            f"/api/v1/workflows/{wf_id}/nodes/ret_1/knowledge",
            json=knowledge_payload,
            headers=headers,
        )
        assert attach_resp.status_code == 422  # Thread ingestion/scoping is not implemented.
        knowledge_payload["workspace_type"] = "user"
        attach_resp = await ac.post(
            f"/api/v1/workflows/{wf_id}/nodes/ret_1/knowledge",
            json=knowledge_payload,
            headers=headers,
        )
        assert attach_resp.status_code == 200
        updated_wf = attach_resp.json()
        assert updated_wf["version"] == 2

        # Verify the node was updated
        ret_node = next(n for n in updated_wf["nodes"] if n["id"] == "ret_1")
        assert ret_node["config"]["provider_name"] == "multi_query"
        assert ret_node["config"]["top_k"] == 8
        assert ret_node["config"]["workspace_type"] == "user"
        assert ret_node["config"]["output_variable"] == "multi_context"

        # Clean up
        await ac.delete(f"/api/v1/workflows/{wf_id}", headers=headers)

