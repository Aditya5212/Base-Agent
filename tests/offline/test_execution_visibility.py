from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from fastapi import HTTPException

from app.models.workflow import WorkflowRun, WorkflowRunEvent
from app.schemas.workflow import (
    WorkflowRunEventListResponse,
    WorkflowRunEventResponse,
    WorkflowRunRequest,
    WorkflowRunStateResponse,
)
from app.services.workflow.tracing import (
    redact_error_message,
    redact_value,
    record_run_event,
    wrap_node_with_tracing,
)
from app.services import workflow_service as svc


def test_redaction_rules_mask_secrets_and_signed_urls():
    payload = {
        "api_key": "sk-secret-12345",
        "nested": {
            "auth_token": "Bearer abcdef",
            "password": "supersecretpassword",
            "normal_field": "hello world",
            "download_url": "https://storage.example.com/file.pdf?X-Amz-Signature=abcd1234efgh&token=xyz",
        },
        "items": [1, 2, 3, 4, 5, 6, 7, 8],
        "long_string": "x" * 600,
    }

    redacted = redact_value(payload, max_str_len=100)
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["nested"]["auth_token"] == "[REDACTED]"
    assert redacted["nested"]["password"] == "[REDACTED]"
    assert redacted["nested"]["normal_field"] == "hello world"
    assert "X-Amz-Signature" not in redacted["nested"]["download_url"]
    assert "[REDACTED_URL]" in redacted["nested"]["download_url"]
    assert len(redacted["items"]) == 6
    assert "[+3 items truncated]" in redacted["items"]
    assert redacted["long_string"].endswith("...[truncated]")
    assert len(redacted["long_string"]) <= 120


def test_redaction_error_message_masks_connection_strings():
    raw_error = "Connection failed at postgresql+psycopg://dbuser:supersecretpass@neon.tech:5432/db"
    clean = redact_error_message(raw_error)
    assert "supersecretpass" not in clean
    assert "postgresql+psycopg://dbuser:[REDACTED]@neon.tech:5432/db" in clean


@pytest.mark.asyncio
async def test_node_tracing_records_lifecycle_events_with_monotonic_sequence():
    run_id = uuid.uuid4()
    recorded_events = []

    async def mock_record(**kwargs):
        recorded_events.append(kwargs)
        return None

    async def sample_handler(state):
        return {
            "variables": {"count": 42, "api_key": "secret_in_output"},
            "messages": [],
        }

    traced = wrap_node_with_tracing("sample_node", "code", sample_handler)

    with patch("app.services.workflow.tracing.record_run_event", side_effect=mock_record):
        state = {
            "run_id": str(run_id),
            "event_sequence": 0,
            "variables": {"input_val": "test", "auth_token": "secret_in_input"},
        }
        res = await traced(state)
        assert res["event_sequence"] == 2
        assert len(recorded_events) == 2

        # 1. node_started
        start_ev = recorded_events[0]
        assert start_ev["run_id"] == str(run_id)
        assert start_ev["sequence"] == 1
        assert start_ev["event_type"] == "node_started"
        assert start_ev["node_id"] == "sample_node"
        assert start_ev["input_summary"]["auth_token"] == "[REDACTED]"
        assert start_ev["input_summary"]["input_val"] == "test"

        # 2. node_completed
        done_ev = recorded_events[1]
        assert done_ev["run_id"] == str(run_id)
        assert done_ev["sequence"] == 2
        assert done_ev["event_type"] == "node_completed"
        assert done_ev["status"] == "success"
        assert isinstance(done_ev["duration_ms"], int)
        assert done_ev["output_summary"]["api_key"] == "[REDACTED]"
        assert done_ev["output_summary"]["count"] == 42


@pytest.mark.asyncio
async def test_node_tracing_handles_failures_and_records_node_failed():
    run_id = uuid.uuid4()
    recorded_events = []

    async def mock_record(**kwargs):
        recorded_events.append(kwargs)
        return None

    async def failing_handler(state):
        raise ValueError("Database connection failed at postgresql+psycopg://user:mypass@host/db")

    traced = wrap_node_with_tracing("fail_node", "retrieve", failing_handler)

    with patch("app.services.workflow.tracing.record_run_event", side_effect=mock_record):
        state = {
            "run_id": str(run_id),
            "event_sequence": 4,
            "variables": {},
        }
        with pytest.raises(ValueError, match="Database connection failed"):
            await traced(state)

        assert len(recorded_events) == 2
        start_ev = recorded_events[0]
        assert start_ev["sequence"] == 5
        assert start_ev["event_type"] == "node_started"

        fail_ev = recorded_events[1]
        assert fail_ev["sequence"] == 6
        assert fail_ev["event_type"] == "node_failed"
        assert fail_ev["status"] == "failed"
        assert fail_ev["error_code"] == "ValueError"
        assert "mypass" not in fail_ev["error_message"]
        assert "[REDACTED]" in fail_ev["error_message"]


@pytest.mark.asyncio
async def test_get_workflow_run_events_cursor_pagination():
    run_id = uuid.uuid4()
    mock_run = WorkflowRun(
        id=run_id,
        user_id="alice",
        workflow_id=uuid.uuid4(),
        thread_id="t1",
        status="COMPLETED",
    )

    ev1 = WorkflowRunEvent(
        id=uuid.uuid4(), run_id=run_id, sequence=1, invocation_id=uuid.uuid4(),
        node_id="s", node_type="start", event_type="node_started",
        occurred_at=datetime.now(timezone.utc), metadata_json={},
    )
    ev2 = WorkflowRunEvent(
        id=uuid.uuid4(), run_id=run_id, sequence=2, invocation_id=uuid.uuid4(),
        node_id="s", node_type="start", event_type="node_completed",
        occurred_at=datetime.now(timezone.utc), metadata_json={},
    )

    db = AsyncMock()
    # Mock count query result
    mock_count_res = MagicMock()
    mock_count_res.scalar_one.return_value = 2
    # Mock select events query result
    mock_events_res = MagicMock()
    mock_events_res.scalars.return_value.all.return_value = [ev1, ev2]

    db.execute.side_effect = [mock_count_res, mock_events_res]

    with patch.object(svc, "get_workflow_run", AsyncMock(return_value=mock_run)):
        events, total, next_cursor = await svc.get_workflow_run_events(
            db, run_id=run_id, user_id="alice", after_sequence=None, limit=2
        )
        assert len(events) == 2
        assert total == 2
        assert next_cursor == 2
        assert events[0].sequence == 1
        assert events[1].sequence == 2


@pytest.mark.asyncio
async def test_get_workflow_run_state_redacts_variables_and_reports_controls():
    run_id = uuid.uuid4()
    mock_run = WorkflowRun(
        id=run_id,
        user_id="alice",
        workflow_id=uuid.uuid4(),
        thread_id="t1",
        status="INTERRUPTED",
        checkpoint_id="workflow:owner:123",
        definition_snapshot={"version": 3},
        input_data={"secret_token": "token123", "normal": "val"},
        output_data=None,
        execution_log=[
            {
                "node_history": ["s", "retrieve", "approval_node"],
                "interrupts": [{"node_id": "approval_node", "timeout_seconds": 86400}],
            }
        ],
    )

    db = AsyncMock()
    with patch.object(svc, "get_workflow_run", AsyncMock(return_value=mock_run)), \
         patch.object(svc, "get_checkpointer", AsyncMock(return_value=None)):
        state = await svc.get_workflow_run_state(db, run_id=run_id, user_id="alice")

        assert state["run_id"] == run_id
        assert state["status"] == "INTERRUPTED"
        assert state["current_node"] == "approval_node"
        assert state["pending_approval"]["node_id"] == "approval_node"
        assert state["supported_controls"] == ["resume"]
        assert state["version"] == 3
        # Ensure secret_token is masked in variables
        assert state["variables"]["secret_token"] == "[REDACTED]"
        assert state["variables"]["normal"] == "val"


@pytest.mark.asyncio
async def test_events_and_state_enforce_tenant_isolation():
    run_id = uuid.uuid4()
    mock_run = WorkflowRun(
        id=run_id,
        user_id="alice",
        workflow_id=uuid.uuid4(),
        thread_id="t1",
        status="RUNNING",
    )

    db = AsyncMock()
    # Calling with bob should raise 403 Forbidden
    with patch.object(svc, "get_workflow_run", side_effect=HTTPException(status_code=403, detail="Forbidden")):
        with pytest.raises(HTTPException) as exc:
            await svc.get_workflow_run_events(db, run_id=run_id, user_id="bob")
        assert exc.value.status_code == 403

        with pytest.raises(HTTPException) as exc2:
            await svc.get_workflow_run_state(db, run_id=run_id, user_id="bob")
        assert exc2.value.status_code == 403


@pytest.mark.asyncio
async def test_execution_capabilities_exposes_events_and_state():
    from app.api.v1.endpoints.workflows import execution_capabilities
    from app.core.auth import UserContext

    user = UserContext(user_id="test", role="user")
    caps = await execution_capabilities(current_user=user)
    assert caps["execution_events"] is True
    assert caps["state_inspection"] is True
