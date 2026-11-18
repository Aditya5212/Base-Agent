"""Regression coverage for queued controls, fencing, and replay identity."""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.workflow import WorkflowRun
from app.services import workflow_service as svc
from app.services.workflow import tracing


@pytest.mark.asyncio
async def test_replay_identity_survives_uncheckpointed_attempt_but_changes_for_loop():
    events = []
    initial = {"run_id": str(uuid.uuid4()), "variables": {}}
    handler = tracing.wrap_node_with_tracing("action", "tool_call", AsyncMock(return_value={"variables": {}}))
    token = tracing.set_event_hook(events.append)
    try:
        first = await handler(deepcopy(initial))
        await handler(deepcopy(initial))
        await handler({**initial, **first})
    finally:
        tracing.reset_event_hook(token)
    started = [event for event in events if event.event_type == "node_started"]
    assert started[0].invocation_id == started[1].invocation_id
    assert started[2].invocation_id != started[0].invocation_id


@pytest.mark.asyncio
async def test_event_failure_does_not_fall_back_to_uncommitted_execution_transaction():
    execution_db = AsyncMock(spec=AsyncSession)

    @asynccontextmanager
    async def unavailable():
        raise RuntimeError("database unavailable")
        yield

    token = tracing.set_execution_session(execution_db)
    try:
        with patch.object(tracing, "AsyncSessionLocal", unavailable):
            with pytest.raises(tracing.ExecutionObservabilityDegradedError):
                await tracing.record_run_event(run_id=uuid.uuid4())
    finally:
        tracing.reset_execution_session(token)
    execution_db.add.assert_not_called()


@pytest.mark.asyncio
async def test_control_read_failure_fences_worker_before_handler():
    execution_db = AsyncMock(spec=AsyncSession)
    handler = AsyncMock()
    node = tracing.wrap_node_with_tracing("action", "code", handler)

    @asynccontextmanager
    async def unavailable():
        raise RuntimeError("database unavailable")
        yield

    token = tracing.set_execution_session(execution_db)
    try:
        with patch.object(tracing, "AsyncSessionLocal", unavailable):
            with pytest.raises(tracing.WorkerFencedError):
                await node({"run_id": str(uuid.uuid4()), "worker_id": "worker", "generation": 1})
    finally:
        tracing.reset_execution_session(token)
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_finalization_rejects_ownership_change_without_committing():
    now = datetime.now(timezone.utc)
    run = WorkflowRun(
        id=uuid.uuid4(), workflow_id=uuid.uuid4(), user_id="alice", thread_id="test",
        checkpoint_id="workflow:test", definition_snapshot={"nodes": [], "edges": [], "recursion_limit": 25},
        input_data={}, execution_log=[], status="RUNNING", worker_id="first", generation=1,
        execution_started_at=now, consumed_seconds=0.0, lease_expires_at=now + timedelta(seconds=30),
    )
    db = AsyncMock(spec=AsyncSession)
    no_payload = MagicMock()
    no_payload.scalar_one_or_none.return_value = None
    rejected_update = MagicMock()
    rejected_update.scalar_one_or_none.return_value = None
    db.execute.side_effect = [no_payload, rejected_update]
    graph = SimpleNamespace(ainvoke=AsyncMock(return_value={"variables": {}, "node_history": []}))
    with patch.object(svc, "get_checkpointer", AsyncMock(return_value=object())), \
         patch.object(svc.WorkflowCompiler, "compile", return_value=graph), \
         patch.object(svc, "_run_heartbeat_loop", AsyncMock()):
        with pytest.raises(tracing.WorkerFencedError, match="before finalization"):
            await svc.execute_claimed_run(db, run.id, "first", run_instance=run)
    db.commit.assert_not_called()
    sql = str(db.execute.call_args_list[-1].args[0])
    for predicate in ("worker_id", "generation", "status", "lease_expires_at"):
        assert predicate in sql.split("WHERE")[1]


@pytest.mark.asyncio
async def test_periodic_maintenance_runs_both_recovery_operations_and_cancels():
    recovered = asyncio.Event()

    @asynccontextmanager
    async def database():
        yield object()

    retry = AsyncMock()

    async def reconcile(session):
        recovered.set()

    with patch("app.core.database.AsyncSessionLocal", database), \
         patch.object(svc, "retry_pending_dispatches", retry), \
         patch.object(svc, "reconcile_stale_workflow_runs", reconcile):
        task = asyncio.create_task(svc.start_workflow_maintenance(interval_seconds=60))
        try:
            await asyncio.wait_for(recovered.wait(), timeout=1)
            retry.assert_awaited_once()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def test_global_maintenance_routes_require_admin():
    from app.api.v1.endpoints.workflows import router
    from app.core.auth import require_admin

    for path in ("/workflows/reconcile-stale", "/workflows/dispatches/retry"):
        route = next(route for route in router.routes if route.path == path)
        assert any(dependency.call is require_admin for dependency in route.dependant.dependencies)
