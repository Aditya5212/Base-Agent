"""Integration tests for durable execution, worker ownership, transactional outbox dispatch, and invocation recovery.

Requires --run-integration and an explicitly disposable TEST_DATABASE_URL.
"""
import asyncio
from datetime import datetime, timedelta, timezone
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select, text

from app.core.database import AsyncSessionLocal
from app.models.workflow import (
    Workflow,
    WorkflowRun,
    WorkflowRunEvent,
    WorkflowDispatch,
    WorkflowToolInvocation,
)
from app.schemas.workflow import WorkflowCreate, WorkflowRunRequest, WorkflowResumeRequest
from app.services import workflow_service as svc
from app.services.memory import close_checkpointer_pool, setup_checkpointer, get_checkpointer
from app.services.thread_service import get_or_create_user
from app.services.workflow.tracing import record_run_event, wrap_node_with_tracing, WorkerFencedError


@pytest.mark.asyncio
async def test_integration_duplicate_delivery_admits_only_one_executor():
    """Proves that two workers concurrently attempting to claim the same pending run result in exactly one winner."""
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    wf_id = uuid.uuid4()

    async with AsyncSessionLocal() as db:
        await get_or_create_user(db, user_id=user_id)
        wf = Workflow(
            id=wf_id,
            user_id=user_id,
            name="Concurrent Claim Test",
            nodes=[{"id": "s", "type": "start"}, {"id": "e", "type": "end"}],
            edges=[{"source": "s", "target": "e"}],
            version=1,
        )
        db.add(wf)
        await db.commit()

        run, dispatch = await svc.create_workflow_run(
            db, wf_id, user_id, WorkflowRunRequest(input_data={})
        )
        run_id = run.id

    # Concurrently attempt to claim run with two different workers
    async def try_claim(worker_name: str) -> bool:
        async with AsyncSessionLocal() as claim_sess:
            return await svc.claim_workflow_run(claim_sess, run_id, worker_id=worker_name, lease_seconds=30)

    outcomes = await asyncio.gather(
        try_claim("worker-alpha"),
        try_claim("worker-beta"),
    )

    # Exactly one worker must succeed
    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 1

    # Verify database state
    async with AsyncSessionLocal() as db:
        claimed_run = await db.get(WorkflowRun, run_id)
        assert claimed_run is not None
        assert claimed_run.status == "RUNNING"
        assert claimed_run.worker_id in ("worker-alpha", "worker-beta")
        assert claimed_run.generation == 1
        assert claimed_run.lease_expires_at > datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_integration_pause_and_queued_resume_preserve_progress():
    """Pause from another session after a node, then execute its queued continuation."""
    from app.services.workflow.nodes.code import build_code_node
    from app.services.workflow import compiler

    await close_checkpointer_pool()
    await setup_checkpointer()
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    payload = WorkflowCreate.model_validate({
        "name": "Pause continuation",
        "nodes": [
            {"id": "s", "type": "start"},
            {"id": "step1", "type": "code", "config": {"code": "result = 100", "output_variable": "first"}},
            {"id": "step2", "type": "code", "config": {"code": "result = first + 50", "output_variable": "second"}},
            {"id": "e", "type": "end", "config": {"response_variable": "second"}},
        ],
        "edges": [{"source": "s", "target": "step1"}, {"source": "step1", "target": "step2"}, {"source": "step2", "target": "e"}],
    })
    run = None
    calls = []

    def observed_code(node_id, config):
        handler = build_code_node(node_id, config)

        async def observed(state):
            calls.append(node_id)
            result = await handler(state)
            if node_id == "step1":
                async with AsyncSessionLocal() as control_db:
                    await svc.pause_workflow_run(control_db, run.id, user_id)
            return result
        return observed

    try:
        async with AsyncSessionLocal() as db:
            wf = await svc.create_workflow(db, user_id, payload)
            run, _ = await svc.create_workflow_run(db, wf.id, user_id, WorkflowRunRequest())
        with patch.object(compiler, "build_code_node", observed_code):
            async with AsyncSessionLocal() as db:
                assert await svc.claim_workflow_run(db, run.id, "first-worker")
                paused = await svc.execute_claimed_run(db, run.id, "first-worker")
                assert paused.status == "PAUSED"
            async with AsyncSessionLocal() as db:
                queued = await svc.resume_workflow_run(db, run.id, user_id, WorkflowResumeRequest())
                assert queued.status == "PENDING"
            async with AsyncSessionLocal() as db:
                assert await svc.claim_workflow_run(db, run.id, "second-worker")
                completed = await svc.execute_claimed_run(db, run.id, "second-worker")
                assert completed.status == "COMPLETED"
                assert completed.output_data == {"second": 150}
            assert calls == ["step1", "step2"]
    finally:
        if run is not None:
            await (await get_checkpointer()).adelete_thread(run.checkpoint_id)
        await close_checkpointer_pool()


@pytest.mark.asyncio
async def test_integration_events_readable_while_execution_is_active():
    """Proves execution events are committed in short autonomous transactions and immediately visible."""
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    run_id = uuid.uuid4()

    async with AsyncSessionLocal() as db:
        await get_or_create_user(db, user_id=user_id)
        wf = Workflow(user_id=user_id, name="Recovery fixture", nodes=[], edges=[])
        db.add(wf)
        await db.flush()
        run = WorkflowRun(
            id=run_id,
            user_id=user_id,
            workflow_id=wf.id,
            thread_id=f"thread_{uuid.uuid4().hex[:8]}",
            status="RUNNING",
            input_data={},
            next_event_sequence=1,
        )
        db.add(run)
        await db.commit()

    # Record event in an autonomous transaction
    ev = await record_run_event(
        run_id=run_id,
        invocation_id=uuid.uuid4(),
        node_id="active_step",
        node_type="code",
        event_type="node_started",
        input_summary={"x": 1},
    )
    assert ev is not None
    assert ev.sequence == 1

    # In a separate concurrent read session, events must be immediately readable
    async with AsyncSessionLocal() as read_session:
        events, total, cursor = await svc.get_workflow_run_events(read_session, run_id, user_id)
        assert len(events) == 1
        assert events[0].node_id == "active_step"
        assert events[0].event_type == "node_started"
        assert events[0].sequence == 1


@pytest.mark.asyncio
async def test_integration_concurrent_event_writers_produce_unique_ordered_sequences():
    """Proves concurrent event writers atomically increment next_event_sequence without duplicate sequences."""
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    run_id = uuid.uuid4()

    async with AsyncSessionLocal() as db:
        await get_or_create_user(db, user_id=user_id)
        wf = Workflow(user_id=user_id, name="Recovery fixture", nodes=[], edges=[])
        db.add(wf)
        await db.flush()
        run = WorkflowRun(
            id=run_id,
            user_id=user_id,
            workflow_id=wf.id,
            thread_id=f"thread_{uuid.uuid4().hex[:8]}",
            status="RUNNING",
            input_data={},
            next_event_sequence=1,
        )
        db.add(run)
        await db.commit()

    # 10 writers insert events concurrently
    async def write_event(idx: int):
        return await record_run_event(
            run_id=run_id,
            invocation_id=uuid.uuid4(),
            node_id=f"step_{idx}",
            node_type="code",
            event_type="node_completed",
            output_summary={"idx": idx},
        )

    tasks = [write_event(i) for i in range(10)]
    written_events = await asyncio.gather(*tasks)

    assert len(written_events) == 10
    assert all(e is not None for e in written_events)

    # Check distinct and ordered sequences
    sequences = sorted([e.sequence for e in written_events])
    assert sequences == list(range(1, 11))

    # Verify run.next_event_sequence reflects all increments
    async with AsyncSessionLocal() as verify_sess:
        r = await verify_sess.get(WorkflowRun, run_id)
        assert r.next_event_sequence == 11


@pytest.mark.asyncio
async def test_integration_heartbeat_expiry_and_fencing_prevents_stale_workers():
    """Proves lease expiry fences stale workers when reclaimed, halting stale execution."""
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    run_id = uuid.uuid4()
    past_time = datetime.now(timezone.utc) - timedelta(seconds=60)

    async with AsyncSessionLocal() as db:
        await get_or_create_user(db, user_id=user_id)
        wf = Workflow(user_id=user_id, name="Recovery fixture", nodes=[], edges=[])
        db.add(wf)
        await db.flush()
        run = WorkflowRun(
            id=run_id,
            user_id=user_id,
            workflow_id=wf.id,
            thread_id=f"thread_{uuid.uuid4().hex[:8]}",
            status="RUNNING",
            worker_id="worker-stale",
            lease_expires_at=past_time,
            execution_started_at=past_time,
            generation=1,
            input_data={},
        )
        db.add(run)
        await db.commit()

    # Reconciler sweeps and fences stale run
    async with AsyncSessionLocal() as recon_sess:
        recovered = await svc.reconcile_stale_workflow_runs(recon_sess, threshold_seconds=0)
        assert run_id in recovered

    # Stale worker tries to heartbeat with old generation -> must be rejected
    async with AsyncSessionLocal() as hb_sess:
        renewed = await svc.heartbeat_workflow_run(
            hb_sess, run_id, worker_id="worker-stale", generation=1, lease_seconds=30
        )
        assert renewed is False

    # Stale worker executing node boundary detects fencing
    state = {
        "run_id": str(run_id),
        "worker_id": "worker-stale",
        "generation": 1,
        "variables": {},
    }

    async def dummy_handler(s):
        return {"variables": {"ok": True}}

    traced_node = wrap_node_with_tracing("step_fenced", "code", dummy_handler)

    from app.services.workflow.tracing import _CURRENT_DB_SESSION
    async with AsyncSessionLocal() as live_session:
        tok = _CURRENT_DB_SESSION.set(live_session)
        try:
            with pytest.raises(WorkerFencedError) as exc_info:
                await traced_node(state)
            assert "Worker fenced" in str(exc_info.value)
        finally:
            _CURRENT_DB_SESSION.reset(tok)


@pytest.mark.asyncio
async def test_integration_replaying_tool_retains_invocation_key_and_reuses_succeeded_result():
    """Proves persisted tool invocation records retain idempotency keys and prevent duplicate external calls."""
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    run_id = uuid.uuid4()
    node_id = "calc_step"
    inv_id = str(uuid.uuid4())
    occurrence = 1
    idempotency_key = f"{run_id}:{node_id}:{inv_id}:{occurrence}"

    async with AsyncSessionLocal() as db:
        await get_or_create_user(db, user_id=user_id)
        wf = Workflow(user_id=user_id, name="Invocation fixture", nodes=[], edges=[])
        db.add(wf)
        await db.flush()
        db.add(WorkflowRun(id=run_id, user_id=user_id, workflow_id=wf.id, thread_id="tool-replay", status="RUNNING", input_data={}))
        await db.flush()
        # Pre-seed a succeeded tool invocation record
        inv_record = WorkflowToolInvocation(
            id=uuid.UUID(inv_id),
            run_id=run_id,
            node_id=node_id,
            logical_occurrence=occurrence,
            idempotency_key=idempotency_key,
            status="succeeded",
            tool_name="calculator",
            arguments_json={"expression": "10*10"},
            result_data={"result": 100},
        )
        db.add(inv_record)
        await db.commit()

    from app.services.workflow.nodes.tool_call import build_tool_call_node
    from app.schemas.workflow import ToolCallNodeConfig

    config = ToolCallNodeConfig(
        tool_name="calculator",
        input_mapping={"expression": "10*10"},
        output_variable="calc_res",
    )
    handler = build_tool_call_node(node_id, config.model_dump())

    state = {
        "run_id": str(run_id),
        "user_id": user_id,
        "current_invocation_id": inv_id,
        "node_invocations": {f"{node_id}:{occurrence}": inv_id},
        "node_occurrences": {node_id: occurrence},
        "variables": {},
        "node_history": [],
    }

    # Execute tool call handler: must reuse cached result without invoking tool engine
    with patch("app.services.workflow.nodes.tool_call._execute_builtin_tool") as mock_builtin:
        result = await handler(state)
        # Ensure external execution was skipped
        assert not mock_builtin.called
        assert result["variables"]["calc_res"] == 100


@pytest.mark.asyncio
async def test_integration_queue_dispatch_recovery_after_failure():
    """Proves transactional outbox dispatches in FAILED state are retried and recovered."""
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    run_id = uuid.uuid4()

    async with AsyncSessionLocal() as db:
        await get_or_create_user(db, user_id=user_id)
        wf = Workflow(user_id=user_id, name="Recovery fixture", nodes=[], edges=[])
        db.add(wf)
        await db.flush()
        run = WorkflowRun(
            id=run_id,
            user_id=user_id,
            workflow_id=wf.id,
            thread_id=f"thread_{uuid.uuid4().hex[:8]}",
            status="PENDING",
            input_data={},
        )
        db.add(run)
        dispatch = WorkflowDispatch(
            run_id=run_id,
            state="FAILED",
            attempt_count=1,
            next_retry_at=datetime.now(timezone.utc) - timedelta(minutes=5),
            last_error="Temporary broker connection refused",
        )
        db.add(dispatch)
        await db.commit()

    # Retry pending dispatches with mock broker enqueue
    mock_task = AsyncMock()
    with patch("app.tasks.workflow_tasks.execute_workflow_task.kiq", mock_task):
        async with AsyncSessionLocal() as retry_sess:
            retried = await svc.retry_pending_dispatches(retry_sess, max_retries=3)
            assert run_id in retried

        # Verify dispatch record transitions to DISPATCHED
        async with AsyncSessionLocal() as verify_sess:
            stmt = select(WorkflowDispatch).where(WorkflowDispatch.run_id == run_id)
            res = await verify_sess.execute(stmt)
            updated_dispatch = res.scalar_one()
            assert updated_dispatch.state == "DISPATCHED"
            assert updated_dispatch.attempt_count == 2
            assert updated_dispatch.updated_at is not None
            assert mock_task.called
