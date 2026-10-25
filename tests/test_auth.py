"""Tests for Authentication, UserContext, and Tenant Isolation."""

import pytest
from fastapi.testclient import TestClient

from app.core.auth import UserContext, get_current_user, require_admin
from app.main import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_get_current_user_default_is_root():
    """When no headers are passed, default identity is root administrator."""
    user = await get_current_user()
    assert user.user_id == "root"
    assert user.is_admin is True
    assert user.role == "root"


@pytest.mark.asyncio
async def test_get_current_user_from_headers():
    """X-User-ID header establishes tenant identity."""
    user = await get_current_user(
        x_user_id="tenant_alice",
        x_user_email="alice@tenant.io",
        x_user_role="user",
    )
    assert user.user_id == "tenant_alice"
    assert user.email == "alice@tenant.io"
    assert user.role == "user"
    assert user.is_admin is False


@pytest.mark.asyncio
async def test_get_current_user_from_mock_bearer():
    """Authorization: Bearer mock_<id> extracts user tenant."""
    user = await get_current_user(authorization="Bearer mock_bob")
    assert user.user_id == "bob"
    assert user.is_admin is False


@pytest.mark.asyncio
async def test_require_admin_guard():
    """require_admin permits admins and rejects regular users with 403."""
    admin_ctx = UserContext(user_id="root", is_admin=True)
    res = await require_admin(admin_ctx)
    assert res.user_id == "root"

    regular_ctx = UserContext(user_id="alice", is_admin=False, role="user")
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await require_admin(regular_ctx)
    assert exc_info.value.status_code == 403


def test_cross_tenant_thread_isolation():
    """User B cannot post messages or read history from User A's thread."""
    # Step 1: User Alice creates a thread
    alice_headers = {"X-User-ID": "user_alice", "X-User-Role": "user"}
    res_a = client.get("/api/v1/chat/alice-private-thread/details", headers=alice_headers)
    assert res_a.status_code == 200
    assert res_a.json()["user_id"] == "user_alice"

    # Step 2: User Bob tries to access Alice's thread -> 403 Forbidden
    bob_headers = {"X-User-ID": "user_bob", "X-User-Role": "user"}
    res_b = client.get("/api/v1/chat/alice-private-thread/details", headers=bob_headers)
    assert res_b.status_code == 403
    assert "belongs to user 'user_alice'" in res_b.json()["detail"]

    # Bob tries to get history -> 403
    res_b_hist = client.get("/api/v1/chat/alice-private-thread/history", headers=bob_headers)
    assert res_b_hist.status_code == 403

    # Bob tries to delete Alice's thread -> 403
    res_b_del = client.delete("/api/v1/chat/alice-private-thread", headers=bob_headers)
    assert res_b_del.status_code == 403

    # Step 3: Admin root CAN access Alice's thread
    admin_headers = {"X-User-ID": "root", "X-User-Role": "root"}
    res_admin = client.get("/api/v1/chat/alice-private-thread/details", headers=admin_headers)
    assert res_admin.status_code == 200


def test_admin_storage_forbidden_for_regular_user_header():
    """Header-based regular user receives 403 Forbidden on admin storage endpoints."""
    user_headers = {"X-User-ID": "charlie", "X-User-Role": "user"}
    res = client.get("/api/v1/storage/stats", headers=user_headers)
    assert res.status_code == 403
    assert "Access denied" in res.json()["detail"]
