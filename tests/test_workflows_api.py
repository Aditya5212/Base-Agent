"""Tests for Phase 2: Workflow Engine, Graph Validation, CRUD APIs, and Execution Lifecycle."""

import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _valid_workflow_payload():
    return {
        "name": "Customer Inquiry Classifier",
        "description": "Routes inquiry to VIP or standard queue",
        "nodes": [
            {
                "id": "start",
                "type": "start",
                "config": {"initial_variables": {"default_status": "new"}},
            },
            {
                "id": "check_tier",
                "type": "conditional",
                "config": {
                    "branches": [
                        {
                            "target_node_id": "vip_calc",
                            "condition_type": "expression",
                            "expression": "variables.tier == 'vip'",
                        }
                    ],
                    "default_target_node_id": "std_calc",
                },
            },
            {
                "id": "vip_calc",
                "type": "code",
                "config": {"code": "priority = 'high'", "output_variable": "priority"},
            },
            {
                "id": "std_calc",
                "type": "code",
                "config": {"code": "priority = 'normal'", "output_variable": "priority"},
            },
            {
                "id": "end",
                "type": "end",
                "config": {"response_variable": "priority"},
            },
        ],
        "edges": [
            {"source": "start", "target": "check_tier"},
            {"source": "check_tier", "target": "vip_calc"},
            {"source": "check_tier", "target": "std_calc"},
            {"source": "vip_calc", "target": "end"},
            {"source": "std_calc", "target": "end"},
        ],
        "recursion_limit": 25,
    }


def test_validate_workflow_endpoint():
    headers = {"X-User-ID": "wf_user_1", "X-User-Role": "user"}

    # 1. Valid graph dry-run
    valid_resp = client.post(
        "/api/v1/workflows/validate",
        json=_valid_workflow_payload(),
        headers=headers,
    )
    assert valid_resp.status_code == 200
    assert valid_resp.json()["valid"] is True
    assert valid_resp.json()["errors"] == []

    # 2. Invalid graph (missing end node)
    bad_payload = _valid_workflow_payload()
    bad_payload["nodes"] = [n for n in bad_payload["nodes"] if n["id"] != "end"]
    bad_resp = client.post(
        "/api/v1/workflows/validate",
        json=bad_payload,
        headers=headers,
    )
    assert bad_resp.status_code == 200
    assert bad_resp.json()["valid"] is False
    assert len(bad_resp.json()["errors"]) > 0


def test_create_and_get_workflow():
    headers = {"X-User-ID": "wf_user_crud", "X-User-Role": "user"}
    payload = _valid_workflow_payload()

    # Create
    create_resp = client.post("/api/v1/workflows", json=payload, headers=headers)
    assert create_resp.status_code == 201
    created = create_resp.json()
    wf_id = created["id"]
    assert created["name"] == payload["name"]
    assert created["version"] == 1
    assert len(created["nodes"]) == 5

    # Get by ID
    get_resp = client.get(f"/api/v1/workflows/{wf_id}", headers=headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == wf_id


def test_list_workflows_pagination_and_search():
    headers = {"X-User-ID": "wf_user_list", "X-User-Role": "user"}

    for i in range(3):
        p = _valid_workflow_payload()
        p["name"] = f"Workflow Sample {i}"
        client.post("/api/v1/workflows", json=p, headers=headers)

    # List all
    list_resp = client.get("/api/v1/workflows?page=1&page_size=10", headers=headers)
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert data["total"] >= 3

    # Search filter
    search_resp = client.get("/api/v1/workflows?search=Sample%201", headers=headers)
    assert search_resp.status_code == 200
    assert any("Sample 1" in item["name"] for item in search_resp.json()["items"])


def test_update_workflow():
    headers = {"X-User-ID": "wf_user_upd", "X-User-Role": "user"}
    payload = _valid_workflow_payload()
    created = client.post("/api/v1/workflows", json=payload, headers=headers).json()
    wf_id = created["id"]

    # Update name and description
    upd_resp = client.put(
        f"/api/v1/workflows/{wf_id}",
        json={"name": "Updated Classifier Title", "description": "Updated description"},
        headers=headers,
    )
    assert upd_resp.status_code == 200
    updated = upd_resp.json()
    assert updated["name"] == "Updated Classifier Title"
    assert updated["version"] == 2


def test_delete_workflow():
    headers = {"X-User-ID": "wf_user_del", "X-User-Role": "user"}
    created = client.post("/api/v1/workflows", json=_valid_workflow_payload(), headers=headers).json()
    wf_id = created["id"]

    del_resp = client.delete(f"/api/v1/workflows/{wf_id}", headers=headers)
    assert del_resp.status_code == 204

    # Confirm 404 after deletion
    get_resp = client.get(f"/api/v1/workflows/{wf_id}", headers=headers)
    assert get_resp.status_code == 404


def test_workflow_tenant_isolation():
    user1_headers = {"X-User-ID": "wf_user_alpha", "X-User-Role": "user"}
    user2_headers = {"X-User-ID": "wf_user_beta", "X-User-Role": "user"}

    created = client.post("/api/v1/workflows", json=_valid_workflow_payload(), headers=user1_headers).json()
    wf_id = created["id"]

    # User 2 cannot access User 1's workflow
    get_resp = client.get(f"/api/v1/workflows/{wf_id}", headers=user2_headers)
    assert get_resp.status_code == 403

    # User 2 cannot delete User 1's workflow
    del_resp = client.delete(f"/api/v1/workflows/{wf_id}", headers=user2_headers)
    assert del_resp.status_code == 403


def test_execute_workflow_run():
    headers = {"X-User-ID": "wf_exec_user", "X-User-Role": "user"}
    created = client.post("/api/v1/workflows", json=_valid_workflow_payload(), headers=headers).json()
    wf_id = created["id"]

    # Run workflow with input_data (VIP tier)
    run_resp = client.post(
        f"/api/v1/workflows/{wf_id}/run?wait=true",
        json={"input_data": {"tier": "vip"}},
        headers=headers,
    )
    assert run_resp.status_code == 200
    run_data = run_resp.json()
    assert run_data["status"] == "COMPLETED"
    assert run_data["output_data"]["priority"] == "high"

    # Get run details
    run_id = run_data["id"]
    get_run = client.get(f"/api/v1/workflows/runs/{run_id}", headers=headers)
    assert get_run.status_code == 200
    assert get_run.json()["id"] == run_id

    # List runs for workflow
    runs_list = client.get(f"/api/v1/workflows/{wf_id}/runs", headers=headers)
    assert runs_list.status_code == 200
    assert runs_list.json()["total"] >= 1

