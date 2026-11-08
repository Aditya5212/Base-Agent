"""Tests for User Management, Schemas, RBAC Guards, and Cascade Deletion."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.core.auth import UserContext
from app.models.user import User
from app.schemas.user import (
    UserCreate,
    UserDeleteResponse,
    UserListResponse,
    UserResponse,
    UserStats,
    UserUpdate,
)
from app.services import user_service


# ---------------------------------------------------------------------------
# 1. Pydantic User Schema Validation Unit Tests (Offline)
# ---------------------------------------------------------------------------

def test_user_create_schema_valid():
    """Valid UserCreate payload passes validation with defaults."""
    payload = UserCreate(email="alice@example.com", name="Alice Smith")
    assert payload.email == "alice@example.com"
    assert payload.name == "Alice Smith"
    assert payload.role == "user"
    assert payload.is_active is True
    assert payload.id is None
    assert payload.metadata_json == {}


def test_user_create_schema_custom_id():
    """UserCreate accepts valid custom ID format."""
    payload = UserCreate(
        id="alice_dev.01",
        email="alice@example.com",
        role="admin",
    )
    assert payload.id == "alice_dev.01"
    assert payload.role == "admin"


def test_user_create_schema_invalid_id():
    """UserCreate rejects invalid characters or malformed identifiers."""
    with pytest.raises(ValidationError):
        UserCreate(id="alice smith", email="alice@example.com")

    with pytest.raises(ValidationError):
        UserCreate(id="alice@work", email="alice@example.com")

    with pytest.raises(ValidationError):
        UserCreate(id="a", email="alice@example.com")


def test_user_create_schema_invalid_email():
    """UserCreate rejects empty or malformed emails."""
    with pytest.raises(ValidationError):
        UserCreate(email="")


def test_user_update_schema_partial():
    """UserUpdate allows partial attribute updates."""
    up = UserUpdate(name="New Name", is_active=False)
    assert up.name == "New Name"
    assert up.is_active is False
    assert up.email is None
    assert up.role is None


def test_user_stats_schema_defaults():
    """UserStats initializes all entity counters to 0 by default."""
    stats = UserStats()
    assert stats.agents_count == 0
    assert stats.workflows_count == 0
    assert stats.documents_count == 0
    assert stats.threads_count == 0
    assert stats.tools_count == 0
    assert stats.schemas_count == 0


def test_user_delete_response_schema():
    """UserDeleteResponse serializes deleted breakdown counts."""
    resp = UserDeleteResponse(
        id="usr_test123",
        message="User deleted successfully.",
        deleted_counts={"agents": 2, "documents": 5, "threads": 3},
    )
    assert resp.id == "usr_test123"
    assert resp.deleted_counts["agents"] == 2
    assert resp.deleted_counts["documents"] == 5


# ---------------------------------------------------------------------------
# 2. Service Rule Invariant Unit Tests (Offline)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_root_user_delete_rejected_by_service():
    """delete_user service function strictly forbids deleting 'root'."""
    mock_db = AsyncMock()
    with pytest.raises(HTTPException) as exc_info:
        await user_service.delete_user(session=mock_db, user_id="root")
    assert exc_info.value.status_code == 400
    assert "root administrator user cannot be deleted" in exc_info.value.detail


@pytest.mark.asyncio
async def test_root_user_demotion_rejected_by_service():
    """update_user service function forbids demoting root role or deactivating root."""
    mock_db = AsyncMock()
    mock_db.get.return_value = User(
        id="root",
        email="root@baseagent.ai",
        name="Root Administrator",
        role="root",
        is_active=True,
    )

    # Attempt demoting role
    with pytest.raises(HTTPException) as exc_info:
        await user_service.update_user(
            session=mock_db,
            user_id="root",
            user_update=UserUpdate(role="user"),
            is_admin=True,
        )
    assert exc_info.value.status_code == 400
    assert "The root user role cannot be modified" in exc_info.value.detail

    # Attempt deactivating
    with pytest.raises(HTTPException) as exc_info2:
        asyncio.run(user_service.update_user(
            session=mock_db,
            user_id="root",
            user_update=UserUpdate(is_active=False),
            is_admin=True,
        ))
    assert exc_info2.value.status_code == 400
    assert "The root user cannot be deactivated" in exc_info2.value.detail


# ---------------------------------------------------------------------------
# 3. HTTP API & RBAC Integration Tests (Service-backed)
# ---------------------------------------------------------------------------

def _get_client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_non_admin_cannot_list_all_users():
    """GET /api/v1/users requires admin privileges; rejects regular users with 403."""
    client = _get_client()
    headers = {"X-User-ID": "user_regular", "X-User-Role": "user"}
    resp = client.get("/api/v1/users", headers=headers)
    assert resp.status_code == 403
    assert "lacks administrative privileges" in resp.json()["detail"]


def test_non_admin_cannot_create_elevated_role():
    """POST /api/v1/users prevents regular users from provisioning admin or root accounts."""
    client = _get_client()
    headers = {"X-User-ID": "user_regular", "X-User-Role": "user"}
    resp = client.post(
        "/api/v1/users",
        json={"email": "attacker@evil.com", "role": "admin"},
        headers=headers,
    )
    assert resp.status_code == 403
    assert "Only administrators can create users with elevated roles" in resp.json()["detail"]


def test_cannot_create_or_register_root_id():
    """POST /api/v1/users disallows using 'root' as the account ID."""
    client = _get_client()
    headers = {"X-User-ID": "root", "X-User-Role": "root"}
    resp = client.post(
        "/api/v1/users",
        json={"id": "root", "email": "impostor@root.ai"},
        headers=headers,
    )
    assert resp.status_code == 400
    assert "Cannot register or create the 'root' administrator" in resp.json()["detail"]


def test_non_admin_cannot_inspect_other_user_profile():
    """GET /api/v1/users/{user_id} prevents user B from inspecting user A's profile."""
    client = _get_client()
    headers = {"X-User-ID": "user_bob", "X-User-Role": "user"}
    resp = client.get("/api/v1/users/user_alice", headers=headers)
    assert resp.status_code == 403
    assert "You can only view your own account profile" in resp.json()["detail"]


def test_non_admin_cannot_modify_other_user():
    """PUT /api/v1/users/{user_id} prevents user B from modifying user A's profile."""
    client = _get_client()
    headers = {"X-User-ID": "user_bob", "X-User-Role": "user"}
    resp = client.put(
        "/api/v1/users/user_alice",
        json={"name": "Hacked Name"},
        headers=headers,
    )
    assert resp.status_code == 403
    assert "You can only update your own account profile" in resp.json()["detail"]


def test_non_admin_cannot_delete_other_user():
    """DELETE /api/v1/users/{user_id} prevents user B from deleting user A."""
    client = _get_client()
    headers = {"X-User-ID": "user_bob", "X-User-Role": "user"}
    resp = client.delete("/api/v1/users/user_alice", headers=headers)
    assert resp.status_code == 403
    assert "You can only delete your own account" in resp.json()["detail"]


def test_cannot_delete_root_user():
    """DELETE /api/v1/users/root is rejected with 400 Bad Request."""
    client = _get_client()
    headers = {"X-User-ID": "root", "X-User-Role": "root"}
    resp = client.delete("/api/v1/users/root", headers=headers)
    assert resp.status_code == 400
    assert "root administrator user cannot be deleted" in resp.json()["detail"]


def test_user_crud_and_cascade_lifecycle():
    """Complete lifecycle: create user, verify default agent, update, get stats, cascade delete."""
    client = _get_client()
    admin_headers = {"X-User-ID": "root", "X-User-Role": "root"}
    unique_suffix = uuid.uuid4().hex[:8]
    test_user_id = f"test_usr_{unique_suffix}"
    test_email = f"test_{unique_suffix}@example.com"

    # 1. Create user
    create_resp = client.post(
        "/api/v1/users",
        json={
            "id": test_user_id,
            "email": test_email,
            "name": "Integration Test User",
            "metadata_json": {"tier": "pro"},
        },
        headers=admin_headers,
    )
    assert create_resp.status_code == 201
    user_data = create_resp.json()
    assert user_data["id"] == test_user_id
    assert user_data["email"] == test_email
    assert user_data["role"] == "user"

    # 2. Verify default agent was auto-created for this new user
    user_headers = {"X-User-ID": test_user_id, "X-User-Role": "user"}
    agent_resp = client.get("/api/v1/agents/default", headers=user_headers)
    assert agent_resp.status_code == 200
    default_agent = agent_resp.json()
    assert default_agent["user_id"] == test_user_id
    assert default_agent["is_default"] is True
    assert default_agent["name"] == "Default Assistant"

    # 3. Get profile via /me with stats
    me_resp = client.get("/api/v1/users/me?include_stats=true", headers=user_headers)
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["id"] == test_user_id
    assert me_data["stats"]["agents_count"] == 1
    assert me_data["stats"]["workflows_count"] == 0

    # 4. Update profile
    update_resp = client.put(
        f"/api/v1/users/{test_user_id}",
        json={"name": "Updated Name", "metadata_json": {"updated": True}},
        headers=user_headers,
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["name"] == "Updated Name"
    assert update_resp.json()["metadata_json"]["tier"] == "pro"
    assert update_resp.json()["metadata_json"]["updated"] is True

    # 5. List users with search as admin
    list_resp = client.get(f"/api/v1/users?search={unique_suffix}", headers=admin_headers)
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total"] >= 1
    assert any(u["id"] == test_user_id for u in list_data["items"])

    # 6. Cascade delete user
    del_resp = client.delete(f"/api/v1/users/{test_user_id}", headers=user_headers)
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data["id"] == test_user_id
    assert del_data["deleted_counts"]["agents"] >= 1

    # 7. Confirm user is gone (404)
    get_again = client.get(f"/api/v1/users/{test_user_id}", headers=admin_headers)
    assert get_again.status_code == 404
