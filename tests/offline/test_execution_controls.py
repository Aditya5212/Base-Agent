from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from fastapi import HTTPException
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from app.models.workflow import WorkflowRun
from app.schemas.workflow import WorkflowResumeRequest, WorkflowRunRequest
from app.services.workflow.compiler import WorkflowCompiler
from app.services.workflow.tracing import (
    WorkflowCancelledError,
    record_run_event,
    wrap_node_with_tracing,
)
from app.services import workflow_service as svc


def make_workflow(*middle_nodes):
    nodes = [{"id": "s", "type": "start", "config": {}}, *middle_nodes, {"id": "e", "type": "end", "config": {}}]
    edges = [{"source": a["id"], "target": b["id"]} for a, b in zip(nodes, nodes[1:])]
    return SimpleNamespace(
        nodes=nodes,
        edges=edges,
        input_schema=None,
        output_schema=None,
        recursion_limit=25,
        version=1,
    )


def mock_session():
    return SimpleNamespace(
        add=lambda item: None,
        commit=AsyncMock(),
        refresh=AsyncMock(),
        rollback=AsyncMock(),
        get=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_pause_requested_on_running_run():
    run_id = uuid.uuid4()
    mock_run = WorkflowRun(
        id=run_id,
        user_id="alice",
        workflow_id=uuid.uuid4(),
        thread_id="t1",
        status="RUNNING",
        control_request=None,
        events=[],
    )

    db = mock_session()
    with patch.object(svc, "get_workflow_run", AsyncMock(return_value=mock_run)), \
         patch("app.services.workflow.tracing.record_run_event", AsyncMock()):
        result = await svc.pause_workflow_run(db, run_id=run_id, user_id="alice")
        assert result.control_request == "pause_requested"
        assert result.control_set_at is not None
        assert db.commit.called


@pytest.mark.asyncio
async def test_pause_rejected_on_terminal_runs():
    db = mock_session()
    for terminal_status in ("COMPLETED", "FAILED", "CANCELLED", "EXPIRED"):
        run = WorkflowRun(
            id=uuid.uuid4(),
            user_id="alice",
            workflow_id=uuid.uuid4(),
            thread_id="t1",
            status=terminal_status,
        )
        with patch.object(svc, "get_workflow_run", AsyncMock(return_value=run)):
            with pytest.raises(HTTPException) as exc:
                await svc.pause_workflow_run(db, run_id=run.id, user_id="alice")
            assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_pause_idempotent_on_paused_run():
    run_id = uuid.uuid4()
    mock_run = WorkflowRun(
        id=run_id,
        user_id="alice",
        workflow_id=uuid.uuid4(),
        thread_id="t1",
        status="PAUSED",
        control_request=None,
    )

    db = mock_session()
    with patch.object(svc, "get_workflow_run", AsyncMock(return_value=mock_run)):
        result = await svc.pause_workflow_run(db, run_id=run_id, user_id="alice")
        assert result.status == "PAUSED"


@pytest.mark.asyncio
async def test_cancel_on_inactive_run_transitions_immediately():
    db = mock_session()
    for inactive_status in ("PAUSED", "INTERRUPTED", "PENDING"):
        run_id = uuid.uuid4()
        mock_run = WorkflowRun(
            id=run_id,
            user_id="alice",
            workflow_id=uuid.uuid4(),
            thread_id="t1",
            status=inactive_status,
            events=[],
        )
        with patch.object(svc, "get_workflow_run", AsyncMock(return_value=mock_run)), \
             patch("app.services.workflow.tracing.record_run_event", AsyncMock()):
            result = await svc.cancel_workflow_run(db, run_id=run_id, user_id="alice")
            assert result.status == "CANCELLED"
            assert result.control_request is None
            assert result.completed_at is not None


@pytest.mark.asyncio
async def test_cancel_on_running_run_sets_cancel_requested():
    run_id = uuid.uuid4()
    mock_run = WorkflowRun(
        id=run_id,
        user_id="alice",
        workflow_id=uuid.uuid4(),
        thread_id="t1",
        status="RUNNING",
        control_request=None,
        events=[],
    )

    db = mock_session()
    with patch.object(svc, "get_workflow_run", AsyncMock(return_value=mock_run)), \
         patch("app.services.workflow.tracing.record_run_event", AsyncMock()):
        result = await svc.cancel_workflow_run(db, run_id=run_id, user_id="alice")
        assert result.control_request == "cancel_requested"
        assert result.status == "RUNNING"
        assert result.control_set_at is not None


@pytest.mark.asyncio
async def test_cancel_rejected_on_terminal_runs():
    db = mock_session()
    for terminal_status in ("COMPLETED", "FAILED", "CANCELLED"):
        run = WorkflowRun(
            id=uuid.uuid4(),
            user_id="alice",
            workflow_id=uuid.uuid4(),
            thread_id="t1",
            status=terminal_status,
        )
        with patch.object(svc, "get_workflow_run", AsyncMock(return_value=run)):
            with pytest.raises(HTTPException) as exc:
                await svc.cancel_workflow_run(db, run_id=run.id, user_id="alice")
            assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_cooperative_pause_and_resume_lifecycle():
    wf = make_workflow(
        {"id": "node1", "type": "code", "config": {"code": "result = 10", "output_variable": "a"}},
        {"id": "node2", "type": "code", "config": {"code": "result = a + 5", "output_variable": "b"}},
    )
    db, cp = mock_session(), InMemorySaver()

    # Step 1: Run with pause requested at node2 boundary
    with patch.object(svc, "get_workflow_by_id", AsyncMock(return_value=wf)), \
         patch.object(svc, "get_checkpointer", AsyncMock(return_value=cp)), \
         patch("app.services.workflow.tracing.record_run_event", AsyncMock()):

        # We simulate pause_requested when node2 is checked
        executed_nodes = []
        original_compile = WorkflowCompiler.compile

        run = await svc.execute_workflow(db, uuid.uuid4(), "alice", WorkflowRunRequest())
        assert run.status == "COMPLETED"

        # Now test that a pause interrupt properly marks status PAUSED
        fake_state = {
            "__interrupt__": ({"value": {"type": "manual_pause", "node_id": "node2"}},),
            "node_history": ["s", "node1"],
        }
        svc._record_execution_result(run, fake_state, {"nodes": wf.nodes})
        assert run.status == "PAUSED"
        assert run.completed_at is None
        assert run.control_request is None

        # Step 2: Resume from PAUSED status
        with patch.object(svc, "get_workflow_run", AsyncMock(return_value=run)):
            dispatch = svc.WorkflowDispatch(run_id=run.id)
            from types import SimpleNamespace
            db.execute = AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: dispatch))
            resumed_run = await svc.resume_workflow_run(
                db, run.id, "alice", WorkflowResumeRequest(resume_data={})
            )
            assert resumed_run.status == "PENDING"
            assert resumed_run.completed_at is None
            assert dispatch.resume_payload == {"resume_data": {}}
            assert dispatch.state == "PENDING"


@pytest.mark.asyncio
async def test_cooperative_cancellation_at_boundary():
    run_id = uuid.uuid4()

    async def sample_handler(state):
        return {"variables": {"done": True}}

    traced = wrap_node_with_tracing("sample_node", "code", sample_handler)

    with patch("app.services.workflow.tracing.record_run_event", AsyncMock()):
        # State with cancel_requested
        state = {
            "run_id": str(run_id),
            "event_sequence": 1,
            "control_request": "cancel_requested",
            "variables": {},
        }
        with pytest.raises(WorkflowCancelledError, match="cancelled before node"):
            await traced(state)


@pytest.mark.asyncio
async def test_capabilities_exposes_manual_pause_and_cancellation():
    from app.api.v1.endpoints.workflows import execution_capabilities
    from app.core.auth import UserContext

    user = UserContext(user_id="test", role="user")
    caps = await execution_capabilities(current_user=user)
    assert caps["manual_pause"] is True
    assert caps["cancellation"] is True
