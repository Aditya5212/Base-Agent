"""Tests for Phase 1: Configurable Agent Core, CRUD APIs, LRU Caching, and Multi-Tenant Isolation."""

import uuid
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.schemas.agent import (
    AgentConfig,
    IdentityConfig,
    ModelConfig,
    ProviderSpec,
    ToolSpec,
    MemoryConfig,
)
from app.services.agent_service import (
    get_cached_agent_config,
    set_cached_agent_config,
    invalidate_agent_cache,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Pydantic AgentConfig Schema Validation Tests
# ---------------------------------------------------------------------------

def test_agent_config_defaults():
    """Default AgentConfig should initialize with valid defaults matching Section 1.2."""
    cfg = AgentConfig()
    assert cfg.identity.locale == "English"
    assert cfg.identity.response_format == "markdown"
    assert cfg.model.model_id == "deepseek-flash"
    assert cfg.model.temperature == 0.7
    assert cfg.model.max_tokens == 4096
    assert len(cfg.knowledge_providers) == 1
    assert cfg.knowledge_providers[0].name == "hybrid"
    assert cfg.knowledge_providers[0].mode == "balanced"
    assert cfg.memory.context_window_messages == 20
    assert cfg.advanced.enable_streaming is True


def test_agent_config_custom_values():
    """Custom AgentConfig should accept valid specialized configurations."""
    cfg = AgentConfig(
        identity=IdentityConfig(
            system_prompt="You are a specialized financial analyst agent.",
            response_format="json",
            locale="en-US",
        ),
        model=ModelConfig(
            model_id="deepseek-v4-pro",
            temperature=0.2,
            max_tokens=8192,
            enable_thinking=True,
        ),
        knowledge_providers=[
            ProviderSpec(name="hybrid", mode="high_precision", weight=0.8, top_k=8, enabled=True),
            ProviderSpec(name="lightrag", mode="mix", weight=0.6, top_k=5, enabled=True),
        ],
        tools=[
            ToolSpec(tool_id="search_documents", enabled=True, config={"max_results": 5}),
        ],
        memory=MemoryConfig(context_window_messages=30, enable_long_term=True),
    )
    assert cfg.identity.response_format == "json"
    assert cfg.model.model_id == "deepseek-v4-pro"
    assert cfg.model.temperature == 0.2
    assert len(cfg.knowledge_providers) == 2
    assert len(cfg.tools) == 1


def test_agent_config_validation_errors():
    """Invalid parameters must raise Pydantic ValidationError."""
    with pytest.raises(ValidationError):
        ModelConfig(temperature=3.5)  # exceeds 2.0

    with pytest.raises(ValidationError):
        ModelConfig(max_tokens=0)  # less than 1

    with pytest.raises(ValidationError):
        IdentityConfig(response_format="xml")  # not in literal text/markdown/json

    with pytest.raises(ValidationError):
        IdentityConfig(system_prompt="too short")  # min_length 10


# ---------------------------------------------------------------------------
# 2. In-Memory LRU Config Caching Tests
# ---------------------------------------------------------------------------

def test_agent_config_lru_cache():
    """LRU cache should cache, return, and invalidate configs correctly."""
    agent_id = str(uuid.uuid4())
    cfg_v1 = AgentConfig(model=ModelConfig(temperature=0.5))
    cfg_v2 = AgentConfig(model=ModelConfig(temperature=0.9))

    # Initial miss
    assert get_cached_agent_config(agent_id, 1) is None

    # Cache hit
    set_cached_agent_config(agent_id, 1, cfg_v1)
    hit = get_cached_agent_config(agent_id, 1)
    assert hit is not None
    assert hit.model.temperature == 0.5

    # Version mismatch returns None
    assert get_cached_agent_config(agent_id, 2) is None

    # Invalidate cache
    invalidate_agent_cache(agent_id)
    assert get_cached_agent_config(agent_id, 1) is None


# ---------------------------------------------------------------------------
# 3. Agent CRUD API Endpoints Tests
# ---------------------------------------------------------------------------

def test_default_agent_auto_provisioning():
    """GET /api/v1/agents/default should auto-provision a default agent if none exists."""
    headers = {"X-User-ID": "user_auto_seed", "X-User-Role": "user"}
    resp = client.get("/api/v1/agents/default", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "user_auto_seed"
    assert data["is_default"] is True
    assert data["version"] == 1
    assert data["name"] == "Default Assistant"

    # Calling again should return the exact same default agent
    resp2 = client.get("/api/v1/agents/default", headers=headers)
    assert resp2.status_code == 200
    assert resp2.json()["id"] == data["id"]


def test_create_and_get_agent():
    """POST /api/v1/agents should create a custom agent with version 1."""
    headers = {"X-User-ID": "user_dev", "X-User-Role": "user"}
    # Pre-seed default agent so the next one is not forced default
    client.get("/api/v1/agents/default", headers=headers)

    payload = {
        "name": "Finance Bot",
        "description": "Financial analysis assistant",
        "config": {
            "identity": {
                "system_prompt": "You are a professional investment and financial advisor.",
                "response_format": "markdown",
            },
            "model": {
                "model_id": "deepseek-flash",
                "temperature": 0.3,
                "max_tokens": 2048,
            },
            "knowledge_providers": [
                {"name": "hybrid", "mode": "high_precision", "weight": 1.0, "top_k": 6, "enabled": True}
            ],
        },
    }
    create_resp = client.post("/api/v1/agents", json=payload, headers=headers)
    assert create_resp.status_code == 201
    created = create_resp.json()
    agent_id = created["id"]
    assert created["name"] == "Finance Bot"
    assert created["version"] == 1
    assert created["is_default"] is False
    assert created["config"]["model"]["temperature"] == 0.3

    # Fetch agent by ID
    get_resp = client.get(f"/api/v1/agents/{agent_id}", headers=headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == agent_id
    assert get_resp.json()["name"] == "Finance Bot"


def test_list_agents_and_filtering():
    """GET /api/v1/agents should return paginated list of user's agents."""
    headers = {"X-User-ID": "user_list_test", "X-User-Role": "user"}
    # Auto-seed default
    client.get("/api/v1/agents/default", headers=headers)

    # Create two more agents
    client.post("/api/v1/agents", json={"name": "Alpha Agent"}, headers=headers)
    client.post("/api/v1/agents", json={"name": "Beta Agent"}, headers=headers)

    # List all
    list_resp = client.get("/api/v1/agents", headers=headers)
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert data["total"] >= 3
    assert len(data["items"]) >= 3

    # Filter with search
    search_resp = client.get("/api/v1/agents?search=Alpha", headers=headers)
    assert search_resp.status_code == 200
    s_data = search_resp.json()
    assert any(a["name"] == "Alpha Agent" for a in s_data["items"])


def test_update_agent_increments_version():
    """PUT /api/v1/agents/{id} should update fields and increment version."""
    headers = {"X-User-ID": "user_update_test", "X-User-Role": "user"}
    create_resp = client.post(
        "/api/v1/agents",
        json={"name": "Version Test Bot", "description": "v1"},
        headers=headers,
    )
    agent_id = create_resp.json()["id"]
    assert create_resp.json()["version"] == 1

    # Update description and prompt
    update_resp = client.put(
        f"/api/v1/agents/{agent_id}",
        json={
            "description": "v2 updated",
            "config": {
                "identity": {"system_prompt": "You are an updated system prompt v2."},
                "model": {"temperature": 0.1},
            },
        },
        headers=headers,
    )
    assert update_resp.status_code == 200
    updated = update_resp.json()
    assert updated["version"] == 2
    assert updated["description"] == "v2 updated"
    assert updated["config"]["model"]["temperature"] == 0.1


def test_clone_agent():
    """POST /api/v1/agents/{id}/clone should clone config into a new non-default agent."""
    headers = {"X-User-ID": "user_clone_test", "X-User-Role": "user"}
    create_resp = client.post(
        "/api/v1/agents",
        json={
            "name": "Base Worker",
            "description": "Template worker",
            "config": {"model": {"temperature": 0.4}},
        },
        headers=headers,
    )
    src_id = create_resp.json()["id"]

    clone_resp = client.post(
        f"/api/v1/agents/{src_id}/clone",
        json={"name": "Cloned Worker"},
        headers=headers,
    )
    assert clone_resp.status_code == 201
    cloned = clone_resp.json()
    assert cloned["id"] != src_id
    assert cloned["name"] == "Cloned Worker"
    assert cloned["version"] == 1
    assert cloned["is_default"] is False
    assert cloned["config"]["model"]["temperature"] == 0.4


def test_delete_agent_and_prevent_default_deletion():
    """DELETE /api/v1/agents/{id} should remove agent, but reject deleting default agent."""
    headers = {"X-User-ID": "user_del_test", "X-User-Role": "user"}
    # Auto-seed default agent
    def_resp = client.get("/api/v1/agents/default", headers=headers)
    def_id = def_resp.json()["id"]

    # Attempting to delete default agent must fail with 400
    del_def_resp = client.delete(f"/api/v1/agents/{def_id}", headers=headers)
    assert del_def_resp.status_code == 400
    assert "Cannot delete the default agent" in del_def_resp.json()["detail"]

    # Create and delete a regular agent
    custom_resp = client.post("/api/v1/agents", json={"name": "Ephemeral Bot"}, headers=headers)
    custom_id = custom_resp.json()["id"]

    del_resp = client.delete(f"/api/v1/agents/{custom_id}", headers=headers)
    assert del_resp.status_code == 204

    # Verify 404 on subsequent get
    assert client.get(f"/api/v1/agents/{custom_id}", headers=headers).status_code == 404


# ---------------------------------------------------------------------------
# 4. Multi-Tenant Isolation Tests
# ---------------------------------------------------------------------------

def test_tenant_isolation_cross_user_access_blocked():
    """User Alice cannot view, update, clone, or delete User Bob's agents."""
    bob_headers = {"X-User-ID": "user_bob", "X-User-Role": "user"}
    alice_headers = {"X-User-ID": "user_alice", "X-User-Role": "user"}

    # Bob creates an agent
    bob_agent = client.post(
        "/api/v1/agents",
        json={"name": "Bob's Secret Agent"},
        headers=bob_headers,
    ).json()
    bob_agent_id = bob_agent["id"]

    # Alice tries to read Bob's agent -> 403 Forbidden
    get_res = client.get(f"/api/v1/agents/{bob_agent_id}", headers=alice_headers)
    assert get_res.status_code == 403

    # Alice tries to update Bob's agent -> 403 Forbidden
    put_res = client.put(
        f"/api/v1/agents/{bob_agent_id}",
        json={"name": "Alice Hijack"},
        headers=alice_headers,
    )
    assert put_res.status_code == 403

    # Alice tries to clone Bob's agent -> 403 Forbidden
    clone_res = client.post(
        f"/api/v1/agents/{bob_agent_id}/clone",
        json={"name": "Alice Stolen Clone"},
        headers=alice_headers,
    )
    assert clone_res.status_code == 403

    # Alice tries to delete Bob's agent -> 403 Forbidden
    del_res = client.delete(f"/api/v1/agents/{bob_agent_id}", headers=alice_headers)
    assert del_res.status_code == 403

    # Admin (root) CAN view Bob's agent
    admin_headers = {"X-User-ID": "root", "X-User-Role": "root"}
    admin_get = client.get(f"/api/v1/agents/{bob_agent_id}", headers=admin_headers)
    assert admin_get.status_code == 200
    assert admin_get.json()["id"] == bob_agent_id


def test_admin_list_agents_scoping_and_single_default():
    """Admin calling GET /api/v1/agents only receives root's agents unless all_tenants=True."""
    root_headers = {"X-User-ID": "root", "X-User-Role": "root"}
    bob_headers = {"X-User-ID": "user_bob_scoping", "X-User-Role": "user"}

    # Seed bob agent
    client.post("/api/v1/agents", json={"name": "Bob Agent"}, headers=bob_headers)

    # Root lists agents without all_tenants -> should only return root's agents
    r = client.get("/api/v1/agents", headers=root_headers)
    assert r.status_code == 200
    data = r.json()
    for item in data["items"]:
        assert item["user_id"] == "root"
    defaults = [item for item in data["items"] if item["is_default"]]
    assert len(defaults) <= 1

    # Root lists agents with all_tenants=True -> returns cross-tenant agents
    r_all = client.get("/api/v1/agents?all_tenants=true", headers=root_headers)
    assert r_all.status_code == 200
    data_all = r_all.json()
    assert any(item["user_id"] == "user_bob_scoping" for item in data_all["items"])


# ---------------------------------------------------------------------------
# 5. Chat Endpoint Dynamic Agent Resolution Tests
# ---------------------------------------------------------------------------

def test_chat_endpoint_resolves_default_agent_transparently():
    """POST /api/v1/chat without agent_id resolves user's default agent and includes agent_id in response."""
    from langchain_core.messages import AIMessage
    headers = {"X-User-ID": "user_chat_default", "X-User-Role": "user"}
    with patch("app.api.v1.endpoints.chat.execute_agent", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = ({"messages": [AIMessage(content="Hello!")]}, "deepseek-flash")
        resp = client.post(
            "/api/v1/chat",
            json={"message": "Hello without explicit agent"},
            headers=headers,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["agent_id"] is not None
    assert data["user_id"] == "user_chat_default"
    assert mock_exec.call_args.kwargs["agent_id"] == data["agent_id"]


def test_chat_endpoint_resolves_custom_agent_id():
    """POST /api/v1/chat with explicit agent_id uses custom agent."""
    from langchain_core.messages import AIMessage
    headers = {"X-User-ID": "user_chat_custom", "X-User-Role": "user"}
    # Create custom agent
    agent_resp = client.post(
        "/api/v1/agents",
        json={
            "name": "Custom Math Bot",
            "config": {
                "identity": {"system_prompt": "You are an expert mathematician."},
                "model": {"temperature": 0.0},
            },
        },
        headers=headers,
    )
    custom_agent_id = agent_resp.json()["id"]

    # Send chat with agent_id
    with patch("app.api.v1.endpoints.chat.execute_agent", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = ({"messages": [AIMessage(content="4")]}, "deepseek-flash")
        chat_resp = client.post(
            "/api/v1/chat",
            json={
                "message": "Calculate 2 + 2",
                "agent_id": custom_agent_id,
            },
            headers=headers,
        )
    assert chat_resp.status_code == 200
    data = chat_resp.json()
    assert data["agent_id"] == custom_agent_id
    assert mock_exec.call_args.kwargs["agent_id"] == custom_agent_id


def test_chat_endpoint_rejects_unauthorized_agent_id():
    """POST /api/v1/chat with another user's agent_id is rejected with 403."""
    user1_headers = {"X-User-ID": "user_one", "X-User-Role": "user"}
    user2_headers = {"X-User-ID": "user_two", "X-User-Role": "user"}

    agent_resp = client.post(
        "/api/v1/agents",
        json={"name": "User One Private Bot"},
        headers=user1_headers,
    )
    agent_id = agent_resp.json()["id"]

    # User Two tries to invoke User One's agent
    chat_resp = client.post(
        "/api/v1/chat",
        json={
            "message": "Attempting unauthorized turn",
            "agent_id": agent_id,
        },
        headers=user2_headers,
    )
    assert chat_resp.status_code == 403
