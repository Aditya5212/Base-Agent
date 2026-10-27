"""Tests for Structured Schema Validation, CRUD APIs, and Multi-Tenant Isolation."""

import uuid
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.services.schema_service import validate_json_schema

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. JSON Schema Validation Unit Tests
# ---------------------------------------------------------------------------

def test_validate_json_schema_valid():
    """Valid Draft 2020-12 JSON Schema should validate without error."""
    valid_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "confidence"],
    }
    # Should not raise
    validate_json_schema(valid_schema)


def test_validate_json_schema_invalid():
    """Invalid schema specification must raise HTTPException with 422 status code."""
    # Not a dictionary
    with pytest.raises(HTTPException) as exc_info:
        validate_json_schema("not-a-dict")  # type: ignore[arg-type]
    assert exc_info.value.status_code == 422

    # Malformed Draft 2020-12 schema type
    invalid_schema = {
        "type": "invalid_primitive_type",
    }
    with pytest.raises(HTTPException) as exc_info:
        validate_json_schema(invalid_schema)
    assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# 2. Structured Schema CRUD API Endpoints Tests
# ---------------------------------------------------------------------------

def test_create_and_get_structured_schema():
    """POST /api/v1/structured-schemas creates schema and GET retrieves it."""
    headers = {"X-User-ID": "user_schema_1", "X-User-Role": "user"}
    schema_def = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "score": {"type": "integer"},
        },
        "required": ["title"],
    }

    payload = {
        "name": "SentimentScoreOutput",
        "description": "Schema for LLM sentiment analysis classification",
        "schema_definition": schema_def,
    }

    create_resp = client.post("/api/v1/structured-schemas", json=payload, headers=headers)
    assert create_resp.status_code == 201
    created = create_resp.json()
    schema_id = created["id"]
    assert created["name"] == "SentimentScoreOutput"
    assert created["version"] == 1
    assert created["user_id"] == "user_schema_1"
    assert created["schema_definition"] == schema_def

    # Retrieve by ID
    get_resp = client.get(f"/api/v1/structured-schemas/{schema_id}", headers=headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == schema_id


def test_list_structured_schemas_and_search():
    """GET /api/v1/structured-schemas lists schemas with pagination and search."""
    headers = {"X-User-ID": "user_schema_list", "X-User-Role": "user"}
    client.post(
        "/api/v1/structured-schemas",
        json={"name": "AlphaSchema", "schema_definition": {"type": "object"}},
        headers=headers,
    )
    client.post(
        "/api/v1/structured-schemas",
        json={"name": "BetaSchema", "schema_definition": {"type": "string"}},
        headers=headers,
    )

    list_resp = client.get("/api/v1/structured-schemas", headers=headers)
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert data["total"] >= 2
    assert len(data["items"]) >= 2

    search_resp = client.get("/api/v1/structured-schemas?search=Alpha", headers=headers)
    assert search_resp.status_code == 200
    s_data = search_resp.json()
    assert any(s["name"] == "AlphaSchema" for s in s_data["items"])


def test_update_structured_schema_increments_version():
    """PUT /api/v1/structured-schemas/{id} updates schema and increments version."""
    headers = {"X-User-ID": "user_schema_update", "X-User-Role": "user"}
    create_resp = client.post(
        "/api/v1/structured-schemas",
        json={"name": "EvolvingSchema", "schema_definition": {"type": "object"}},
        headers=headers,
    )
    schema_id = create_resp.json()["id"]
    assert create_resp.json()["version"] == 1

    new_def = {
        "type": "object",
        "properties": {"new_field": {"type": "boolean"}},
    }
    update_resp = client.put(
        f"/api/v1/structured-schemas/{schema_id}",
        json={
            "description": "Updated v2 description",
            "schema_definition": new_def,
        },
        headers=headers,
    )
    assert update_resp.status_code == 200
    updated = update_resp.json()
    assert updated["version"] == 2
    assert updated["description"] == "Updated v2 description"
    assert updated["schema_definition"] == new_def


def test_delete_structured_schema():
    """DELETE /api/v1/structured-schemas/{id} deletes the schema."""
    headers = {"X-User-ID": "user_schema_del", "X-User-Role": "user"}
    create_resp = client.post(
        "/api/v1/structured-schemas",
        json={"name": "TempSchema", "schema_definition": {"type": "object"}},
        headers=headers,
    )
    schema_id = create_resp.json()["id"]

    del_resp = client.delete(f"/api/v1/structured-schemas/{schema_id}", headers=headers)
    assert del_resp.status_code == 204

    assert client.get(f"/api/v1/structured-schemas/{schema_id}", headers=headers).status_code == 404


# ---------------------------------------------------------------------------
# 3. Multi-Tenant Isolation Tests
# ---------------------------------------------------------------------------

def test_structured_schema_tenant_isolation():
    """User Alice cannot view, update, or delete User Bob's schemas."""
    bob_headers = {"X-User-ID": "user_bob_schema", "X-User-Role": "user"}
    alice_headers = {"X-User-ID": "user_alice_schema", "X-User-Role": "user"}

    bob_schema = client.post(
        "/api/v1/structured-schemas",
        json={"name": "BobSchema", "schema_definition": {"type": "object"}},
        headers=bob_headers,
    ).json()
    schema_id = bob_schema["id"]

    # Alice GET -> 403
    assert client.get(f"/api/v1/structured-schemas/{schema_id}", headers=alice_headers).status_code == 403

    # Alice PUT -> 403
    assert client.put(
        f"/api/v1/structured-schemas/{schema_id}",
        json={"name": "Hijacked"},
        headers=alice_headers,
    ).status_code == 403

    # Alice DELETE -> 403
    assert client.delete(f"/api/v1/structured-schemas/{schema_id}", headers=alice_headers).status_code == 403

    # Admin (root) CAN view
    admin_headers = {"X-User-ID": "root", "X-User-Role": "root"}
    assert client.get(f"/api/v1/structured-schemas/{schema_id}", headers=admin_headers).status_code == 200
