from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from app.models.workflow import WorkflowRun, WorkflowRunEvent
from app.schemas.tool import HttpAuthType, HttpToolConfig
from app.schemas.workflow import WorkflowResumeRequest, WorkflowRunRequest
from app.services.tool_executors.http_executor import HttpToolExecutor
from app.services.workflow.nodes.tool_call import build_tool_call_node
from app.services.workflow.state import WorkflowState
from app.services.workflow.tracing import (
    WorkflowTimeoutError,
    record_run_event,
    set_event_hook,
    reset_event_hook,
    wrap_node_with_tracing,
)
from app.services import workflow_service as svc


def mock_session():
    return SimpleNamespace(
        add=lambda item: None,
        commit=AsyncMock(),
        refresh=AsyncMock(),
        rollback=AsyncMock(),
        get=AsyncMock(),
        execute=AsyncMock(),
    )


def test_active_runtime_budget_accounting_across_pause_and_resume():
    """Verify active execution time is added to consumed_seconds while paused time is ignored."""
    run = WorkflowRun(
        id=uuid.uuid4(),
        user_id="alice",
        workflow_id=uuid.uuid4(),
        thread_id="t1",
        status="RUNNING",
        budget_seconds=30.0,
        consumed_seconds=2.0,
        execution_started_at=datetime.now(timezone.utc) - timedelta(seconds=3),
        execution_log=[],
        events=[],
        input_data={},
    )

    # 1. First execution slice pauses (manual pause interrupt)
    definition = {"nodes": [], "edges": [], "output_schema": None}
    interrupt_state = {
        "__interrupt__": [{"value": {"type": "manual_pause", "node_id": "test"}}],
        "node_history": ["test"],
    }
    svc._record_execution_result(run, interrupt_state, definition)

    assert run.status == "PAUSED"
    assert run.execution_started_at is None
    assert run.consumed_seconds >= 4.9  # 2.0 initial + ~3.0 slice

    # 2. Time passes while paused (simulate 100 seconds in PAUSED status)
    paused_consumed = run.consumed_seconds

    # 3. Resume execution slice
    run.status = "RUNNING"
    run.execution_started_at = datetime.now(timezone.utc) - timedelta(seconds=4)

    completed_state = {
        "variables": {"result": "success"},
        "node_history": ["test", "next"],
    }
    svc._record_execution_result(run, completed_state, definition, resumed=True)

    assert run.status == "COMPLETED"
    assert run.execution_started_at is None
    # Verify consumed_seconds only accumulated active slice (~4s), ignoring the paused period
    assert run.consumed_seconds >= paused_consumed + 3.9
    assert run.consumed_seconds < paused_consumed + 10.0


@pytest.mark.asyncio
async def test_workflow_timeout_transitions_to_timed_out_at_boundary():
    """Verify that exceeding configured budget_seconds triggers WorkflowTimeoutError and timed_out event."""
    recorded_events = []

    async def mock_record(**kwargs):
        recorded_events.append(kwargs)
        return None

    raw_run_id = str(uuid.uuid4())
    state: WorkflowState = {
        "run_id": raw_run_id,
        "event_sequence": 2,
        "budget_seconds": 5.0,
        "consumed_seconds": 4.5,
        # Active slice has run for 2 seconds -> total = 6.5s > 5.0s budget
        "execution_started_at": (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat(),
        "variables": {},
        "messages": [],
        "node_history": [],
    }

    async def dummy_node(s):
        return {"variables": {"done": True}}

    traced = wrap_node_with_tracing("step_2", "code", dummy_node)

    with patch("app.services.workflow.tracing.record_run_event", side_effect=mock_record):
        with pytest.raises(WorkflowTimeoutError) as exc_info:
            await traced(state)

        assert "budget of 5.0s exceeded" in str(exc_info.value)
        assert len(recorded_events) == 1
        timeout_evt = recorded_events[0]
        assert timeout_evt["event_type"] == "timed_out"
        assert timeout_evt["status"] == "timed_out"
        assert timeout_evt["sequence"] == 3
        assert timeout_evt["node_id"] == "step_2"


@pytest.mark.asyncio
async def test_execute_workflow_handles_timeout_error():
    """Verify execute_workflow catches WorkflowTimeoutError, sets status TIMED_OUT, and completes."""
    db = mock_session()
    w_id = uuid.uuid4()
    u_id = "alice"
    payload = WorkflowRunRequest(input_data={"x": 1}, budget_seconds=10)

    mock_wf = SimpleNamespace(
        id=w_id,
        user_id=u_id,
        name="test_wf",
        nodes=[{"id": "s", "type": "start", "config": {}}],
        edges=[],
        input_schema=None,
        output_schema=None,
        recursion_limit=25,
        version=1,
    )

    with patch.object(svc, "get_workflow_by_id", AsyncMock(return_value=mock_wf)), \
         patch.object(svc, "get_checkpointer", AsyncMock(return_value=MagicMock())), \
         patch.object(svc.WorkflowCompiler, "compile") as mock_compile:

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(side_effect=WorkflowTimeoutError("Execution budget of 10s exceeded"))
        mock_compile.return_value = mock_graph

        run = await svc.execute_workflow(db, w_id, u_id, payload)
        assert run.status == "TIMED_OUT"
        assert run.completed_at is not None
        assert "Execution budget of 10s exceeded" in run.error_message
        assert run.execution_started_at is None
        assert db.commit.called


@pytest.mark.asyncio
async def test_tool_idempotency_key_format_and_http_header():
    """Verify deterministic idempotency key format {run_id}:{node_id}:{invocation_id}:{attempt}
    and its injection into HTTP tool execution headers.
    """
    recorded_events = []

    async def mock_record(**kwargs):
        recorded_events.append(kwargs)
        return None

    run_uuid = uuid.uuid4()
    state: WorkflowState = {
        "run_id": str(run_uuid),
        "event_sequence": 0,
        "variables": {"city": "Berlin"},
        "messages": [],
        "node_history": [],
    }

    # 1. Tracing wraps tool node and produces deterministic key
    async def mock_tool_handler(s):
        assert s.get("current_invocation_id") is not None
        return {"variables": {"weather": "sunny"}}

    traced = wrap_node_with_tracing("weather_api", "tool_call", mock_tool_handler)

    with patch("app.services.workflow.tracing.record_run_event", side_effect=mock_record):
        await traced(state)

        # Check emitted tool event
        assert len(recorded_events) >= 2
        start_evt = recorded_events[0]
        complete_evt = recorded_events[1]

        assert start_evt["tool_invocation_id"] is not None
        parts = start_evt["tool_invocation_id"].split(":")
        assert len(parts) == 4
        assert parts[0] == str(run_uuid)
        assert parts[1] == "weather_api"
        assert parts[2] == str(start_evt["invocation_id"])
        assert parts[3] == "1"

        assert complete_evt["tool_invocation_id"] == start_evt["tool_invocation_id"]

        # 2. Verify HttpToolExecutor maps __idempotency_key__ to X-Idempotency-Key header
        config = HttpToolConfig(
            url_template="https://api.example.com/check",
            method="POST",
            auth_type=HttpAuthType.NONE,
        )
        executor = HttpToolExecutor(config)

        test_args = {
            "__idempotency_key__": start_evt["tool_invocation_id"],
            "city": "Berlin",
        }

        with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_req:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/json"}
            mock_resp.json.return_value = {"status": "ok"}
            mock_req.return_value = mock_resp

            await executor.execute(test_args)
            assert mock_req.called
            call_kwargs = mock_req.call_args.kwargs
            assert "X-Idempotency-Key" in call_kwargs["headers"]
            assert call_kwargs["headers"]["X-Idempotency-Key"] == start_evt["tool_invocation_id"]


@pytest.mark.asyncio
async def test_reconcile_stale_workflow_runs_marks_expired_leases_failed():
    """Verify stale run sweeper detects expired leases and transitions them to FAILED with event."""
    stale_run = WorkflowRun(
        id=uuid.uuid4(),
        user_id="bob",
        workflow_id=uuid.uuid4(),
        thread_id="t2",
        status="RUNNING",
        worker_id="worker-crash-123",
        lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=45),
        execution_started_at=datetime.now(timezone.utc) - timedelta(seconds=90),
        consumed_seconds=5.0,
        execution_log=[],
        events=[],
        input_data={},
    )

    db = mock_session()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [stale_run]

    mock_seq_result = MagicMock()
    mock_seq_result.scalar.return_value = 4

    db.execute = AsyncMock(side_effect=[mock_result, mock_seq_result])

    with patch("app.services.workflow.tracing.record_run_event", AsyncMock()) as mock_event:
        recovered = await svc.reconcile_stale_workflow_runs(db, threshold_seconds=0)

        assert recovered == [stale_run.id]
        assert stale_run.status == "FAILED"
        assert stale_run.completed_at is not None
        assert "Worker heartbeat expired" in stale_run.error_message
        assert stale_run.execution_started_at is None
        assert stale_run.consumed_seconds > 5.0
        assert db.commit.called

        assert mock_event.called
        call_kwargs = mock_event.call_args.kwargs
        assert call_kwargs["event_type"] == "lease_expired"
        assert call_kwargs["status"] == "failed"
        assert call_kwargs["sequence"] == 5


@pytest.mark.asyncio
async def test_claim_and_heartbeat_workflow_run():
    """Verify worker lease claim and heartbeat renewal logic."""
    run_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    run = WorkflowRun(
        id=run_id,
        user_id="bob",
        workflow_id=uuid.uuid4(),
        thread_id="t2",
        status="PENDING",
        worker_id=None,
        lease_expires_at=None,
        execution_log=[],
        events=[],
        input_data={},
    )

    db = mock_session()
    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = run
    db.execute = AsyncMock(return_value=mock_res)

    # 1. Claim unassigned run
    claimed = await svc.claim_workflow_run(db, run_id=run_id, worker_id="worker-A", lease_seconds=30)
    assert claimed is True
    assert run.worker_id == "worker-A"
    assert run.lease_expires_at > now
    assert db.commit.called

    # 2. Another worker cannot claim while active lease exists
    db.commit.reset_mock()
    claimed_other = await svc.claim_workflow_run(db, run_id=run_id, worker_id="worker-B", lease_seconds=30)
    assert claimed_other is False
    assert run.worker_id == "worker-A"

    # 3. Original worker heartbeats run
    heartbeat_ok = await svc.heartbeat_workflow_run(db, run_id=run_id, worker_id="worker-A", lease_seconds=60)
    assert heartbeat_ok is True
    assert run.lease_expires_at > now + timedelta(seconds=20)
    assert db.commit.called


@pytest.mark.asyncio
async def test_capabilities_exposes_execution_budgets_and_worker_recovery():
    """Verify GET /capabilities includes durability features."""
    from app.api.v1.endpoints.workflows import execution_capabilities
    from app.core.auth import UserContext

    user = UserContext(user_id="test", role="user")
    caps = await execution_capabilities(current_user=user)
    assert caps["execution_budgets"] is True
    assert caps["worker_recovery"] is True
