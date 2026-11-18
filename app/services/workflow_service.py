from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import logging
from typing import Any
import uuid

from fastapi import HTTPException, status
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from sqlalchemy import func, select, update
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.workflow import Workflow, WorkflowRun, WorkflowRunEvent, WorkflowDispatch, WorkflowToolInvocation
from app.schemas.workflow import (
    WorkflowCreate,
    WorkflowNodeKnowledgeConfig,
    WorkflowResumeRequest,
    WorkflowRunRequest,
    WorkflowUpdate,
)
from app.services.memory import get_checkpointer
from app.core.json_validation import validate_payload
from app.services.schema_service import validate_json_schema
from app.services.thread_service import get_or_create_user
from app.services.workflow.compiler import WorkflowCompiler
from app.services.workflow.tracing import (
    WorkerFencedError,
    WorkflowCancelledError,
    WorkflowTimeoutError,
    record_run_event,
    reset_execution_session,
    set_execution_session,
)
from app.services.workflow.validator import GraphValidationError, validate_workflow_graph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workflow CRUD Service Functions
# ---------------------------------------------------------------------------

async def create_workflow(
    session: AsyncSession,
    user_id: str,
    payload: WorkflowCreate,
) -> Workflow:
    """Creates and persists a new workflow for a user after graph validation."""
    await get_or_create_user(session, user_id=user_id)

    # Convert nodes and edges to plain dicts for storage and validation
    raw_nodes = [node.model_dump(mode="json") for node in payload.nodes]
    raw_edges = [edge.model_dump(mode="json") for edge in payload.edges]

    # Topological graph validation
    is_valid, errors, _ = validate_workflow_graph(raw_nodes, raw_edges)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Graph validation failed: {'; '.join(errors)}",
        )

    # Validate input and output schemas if present
    if payload.input_schema is not None:
        validate_json_schema(payload.input_schema)
    if payload.output_schema is not None:
        validate_json_schema(payload.output_schema)

    try:
        WorkflowCompiler.compile(raw_nodes, raw_edges)
    except (ValueError, PermissionError, GraphValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    workflow = Workflow(
        user_id=user_id,
        name=payload.name,
        description=payload.description,
        nodes=raw_nodes,
        edges=raw_edges,
        input_schema=payload.input_schema,
        output_schema=payload.output_schema,
        recursion_limit=payload.recursion_limit,
        is_published=payload.is_published,
        version=1,
    )
    session.add(workflow)
    await session.commit()
    await session.refresh(workflow)
    logger.info("Created workflow '%s' (ID: %s) for user '%s'", workflow.name, workflow.id, user_id)
    return workflow


async def get_workflow_by_id(
    session: AsyncSession,
    workflow_id: uuid.UUID,
    user_id: str,
) -> Workflow:
    """Retrieves a workflow by ID enforcing tenant isolation."""
    stmt = select(Workflow).where(Workflow.id == workflow_id)
    result = await session.execute(stmt)
    workflow = result.scalar_one_or_none()

    if workflow is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow with ID '{workflow_id}' not found.",
        )
    if workflow.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this workflow.",
        )
    return workflow


async def list_workflows(
    session: AsyncSession,
    user_id: str,
    page: int = 1,
    page_size: int = 20,
    search: str | None = None,
) -> tuple[list[Workflow], int]:
    """Lists workflows owned by user with pagination and search."""
    base_query = select(Workflow).where(Workflow.user_id == user_id)
    count_query = select(func.count(Workflow.id)).where(Workflow.user_id == user_id)

    if search and search.strip():
        search_filter = f"%{search.strip()}%"
        condition = Workflow.name.ilike(search_filter) | Workflow.description.ilike(search_filter)
        base_query = base_query.where(condition)
        count_query = count_query.where(condition)

    total_result = await session.execute(count_query)
    total = total_result.scalar_one() or 0

    offset = (page - 1) * page_size
    query = base_query.order_by(Workflow.updated_at.desc()).offset(offset).limit(page_size)
    result = await session.execute(query)
    items = list(result.scalars().all())

    return items, total


async def update_workflow(
    session: AsyncSession,
    workflow_id: uuid.UUID,
    user_id: str,
    payload: WorkflowUpdate,
) -> Workflow:
    """Updates workflow configuration, re-validating graph topology and incrementing version."""
    workflow = await get_workflow_by_id(session, workflow_id, user_id)

    if payload.name is not None:
        workflow.name = payload.name
    if payload.description is not None:
        workflow.description = payload.description
    if payload.recursion_limit is not None:
        workflow.recursion_limit = payload.recursion_limit
    if payload.is_published is not None:
        workflow.is_published = payload.is_published

    nodes_changed = payload.nodes is not None
    edges_changed = payload.edges is not None

    if nodes_changed:
        workflow.nodes = [n.model_dump(mode="json") for n in payload.nodes]  # type: ignore[union-attr]
    if edges_changed:
        workflow.edges = [e.model_dump(mode="json") for e in payload.edges]  # type: ignore[union-attr]

    # If topology was updated, re-validate
    if nodes_changed or edges_changed:
        is_valid, errors, _ = validate_workflow_graph(workflow.nodes, workflow.edges)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Updated graph validation failed: {'; '.join(errors)}",
            )

    if payload.input_schema is not None:
        validate_json_schema(payload.input_schema)
        workflow.input_schema = payload.input_schema
    if payload.output_schema is not None:
        validate_json_schema(payload.output_schema)
        workflow.output_schema = payload.output_schema

    try:
        WorkflowCompiler.compile(workflow.nodes, workflow.edges)
    except (ValueError, PermissionError, GraphValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    workflow.version += 1
    await session.commit()
    await session.refresh(workflow)
    logger.info("Updated workflow '%s' to version %d", workflow.name, workflow.version)
    return workflow


async def attach_node_knowledge(
    session: AsyncSession,
    workflow_id: uuid.UUID,
    user_id: str,
    node_id: str,
    knowledge_config: WorkflowNodeKnowledgeConfig,
) -> Workflow:
    """Configures or attaches knowledge retrieval parameters to a specific workflow node."""
    workflow = await get_workflow_by_id(session, workflow_id, user_id)

    if knowledge_config.workspace_type != "user" or knowledge_config.provider_name not in {"hybrid", "lightrag", "multi_query"}:
        raise HTTPException(status_code=422, detail="Select a supported provider with user knowledge scope")
    nodes = list(workflow.nodes or [])
    node_found = False
    updated_nodes = []

    for node in nodes:
        if node.get("id") == node_id:
            node_found = True
            n_type = node.get("type")
            node_cfg = dict(node.get("config") or {})

            if n_type == "retrieve":
                node_cfg["provider_name"] = knowledge_config.provider_name
                node_cfg["workspace_type"] = knowledge_config.workspace_type
                node_cfg["top_k"] = knowledge_config.top_k
                if knowledge_config.query_template:
                    node_cfg["query_template"] = knowledge_config.query_template
                if knowledge_config.output_variable:
                    node_cfg["output_variable"] = knowledge_config.output_variable
                if knowledge_config.sources_variable:
                    node_cfg["sources_variable"] = knowledge_config.sources_variable
            else:
                raise HTTPException(status_code=422, detail="Knowledge attachment is supported only on retrieve nodes")

            node["config"] = node_cfg
        updated_nodes.append(node)

    if not node_found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node '{node_id}' not found in workflow '{workflow_id}'.",
        )

    workflow.nodes = updated_nodes
    flag_modified(workflow, "nodes")
    workflow.version += 1
    await session.commit()
    await session.refresh(workflow)
    logger.info("Attached knowledge config to node '%s' in workflow '%s'", node_id, workflow.name)
    return workflow


async def delete_workflow(
    session: AsyncSession,
    workflow_id: uuid.UUID,
    user_id: str,
) -> None:
    """Deletes workflow and all associated runs."""
    workflow = await get_workflow_by_id(session, workflow_id, user_id)
    await session.delete(workflow)
    await session.commit()
    logger.info("Deleted workflow ID: %s", workflow_id)


# ---------------------------------------------------------------------------
# Workflow Execution Service Functions
# ---------------------------------------------------------------------------

def _record_execution_result(run: WorkflowRun, state: dict[str, Any], definition: dict[str, Any], resumed: bool = False) -> None:
    """Interpret LangGraph's returned interrupts before declaring completion."""
    now = datetime.now(timezone.utc)
    if run.execution_started_at:
        start_time = run.execution_started_at
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        elapsed = (now - start_time).total_seconds()
        run.consumed_seconds = (run.consumed_seconds or 0.0) + max(0.0, elapsed)
        run.execution_started_at = None

    entry: dict[str, Any] = {"node_history": state.get("node_history", [])}
    if resumed:
        entry["resumed"] = True
    interrupts = state.get("__interrupt__") or ()
    if interrupts:
        now = datetime.now(timezone.utc)
        entry["interrupts"] = []

        def _get_val(it: Any) -> Any:
            if hasattr(it, "value"):
                return it.value
            if isinstance(it, dict) and "value" in it:
                return it["value"]
            return it

        for item in interrupts:
            value = _get_val(item)
            timeout = value.get("timeout_seconds", 86400) if isinstance(value, dict) else 86400
            entry["interrupts"].append({
                "id": getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None),
                "value": value,
                "expires_at": (now + timedelta(seconds=timeout)).isoformat(),
            })

        # Differentiate manual pause from approval requirements
        is_manual_pause = any(
            isinstance(i, dict) and isinstance(i.get("value"), dict) and i["value"].get("type") == "manual_pause"
            for i in entry["interrupts"]
        )
        if is_manual_pause:
            run.status = "PAUSED"
            run.control_request = None
        else:
            run.status = "INTERRUPTED"

        run.output_data = None
        run.completed_at = None
    else:
        variables = dict(state.get("variables") or {})
        output: dict[str, Any] = {}
        executed = set(state.get("node_history") or [])
        for node in definition["nodes"]:
            if node.get("type") == "end" and node["id"] in executed:
                key = (node.get("config") or {}).get("response_variable")
                if key:
                    output[key] = variables[key]
        if not output:
            output = variables
        validate_payload(output, definition.get("output_schema"), "Workflow output")
        run.status = "COMPLETED"
        run.output_data = output
        run.completed_at = datetime.now(timezone.utc)
    run.error_message = None
    run.execution_log = list(run.execution_log or []) + [entry]

async def create_workflow_run(
    session: AsyncSession,
    workflow_id: uuid.UUID,
    user_id: str,
    payload: WorkflowRunRequest,
    queue_execution: bool = True,
) -> tuple[WorkflowRun, WorkflowDispatch]:
    """Creates a PENDING workflow execution run and transactional dispatch outbox record."""
    workflow = await get_workflow_by_id(session, workflow_id, user_id)
    definition = deepcopy({
        "nodes": workflow.nodes, "edges": workflow.edges,
        "input_schema": workflow.input_schema, "output_schema": workflow.output_schema,
        "recursion_limit": workflow.recursion_limit, "version": workflow.version,
    })
    try:
        validate_payload(payload.input_data, definition["input_schema"], "Workflow input")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        checkpointer = await get_checkpointer()
        if checkpointer is None:
            raise RuntimeError("Persistence is unavailable")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Workflow checkpointing is unavailable") from exc

    if payload.thread_id and hasattr(session, "get"):
        from app.services.thread_service import verify_thread_ownership
        await verify_thread_ownership(session, payload.thread_id, user_id=user_id, allow_missing=True)

    run_id = uuid.uuid4()
    thread_id = payload.thread_id or f"wf_run_{run_id}"
    owner_key = hashlib.sha256(f"{user_id}:{workflow_id}".encode()).hexdigest()
    checkpoint_id = f"workflow:{owner_key}:{run_id}"

    now = datetime.now(timezone.utc)

    run = WorkflowRun(
        id=run_id,
        workflow_id=workflow_id,
        user_id=user_id,
        thread_id=thread_id,
        checkpoint_id=checkpoint_id,
        definition_snapshot=definition,
        execution_log=[],
        status="PENDING",
        generation=0,
        next_event_sequence=1,
        input_data=payload.input_data,
        started_at=None,
        budget_seconds=payload.budget_seconds,
        consumed_seconds=0.0,
        execution_started_at=None,
    )
    dispatch = WorkflowDispatch(
        id=uuid.uuid4(),
        run_id=run_id,
        state="PENDING" if queue_execution else "DISPATCHED",
        attempt_count=0,
        next_retry_at=now,
    )
    session.add(run)
    session.add(dispatch)
    await session.commit()
    await session.refresh(run)
    await session.refresh(dispatch)
    return run, dispatch


async def dispatch_workflow_run(
    session: AsyncSession,
    run_id: uuid.UUID,
) -> bool:
    """Submits a workflow run to the background worker queue via transactional outbox."""
    from app.tasks.workflow_tasks import execute_workflow_task

    stmt = select(WorkflowDispatch).where(WorkflowDispatch.run_id == run_id).with_for_update()
    res = await session.execute(stmt)
    dispatch = res.scalar_one_or_none()
    now = datetime.now(timezone.utc)

    if not dispatch:
        dispatch = WorkflowDispatch(
            id=uuid.uuid4(),
            run_id=run_id,
            state="PENDING",
            attempt_count=0,
            next_retry_at=now,
        )
        session.add(dispatch)

    try:
        await execute_workflow_task.kiq(run_id=str(run_id))
        dispatch.state = "DISPATCHED"
        dispatch.attempt_count += 1
        dispatch.last_error = None
        await session.commit()
        return True
    except Exception as exc:
        logger.warning("Failed to dispatch workflow run %s to Taskiq: %s", run_id, exc)
        dispatch.state = "FAILED"
        dispatch.attempt_count += 1
        dispatch.last_error = str(exc)
        backoff = min(300, 5 * (2 ** min(dispatch.attempt_count, 6)))
        dispatch.next_retry_at = now + timedelta(seconds=backoff)
        await session.commit()
        return False


async def retry_pending_dispatches(
    session: AsyncSession,
    limit: int = 50,
    max_retries: int = 5,
) -> list[uuid.UUID]:
    """Retries unsent dispatch records whose next_retry_at has arrived."""
    now = datetime.now(timezone.utc)
    stmt = (
        select(WorkflowDispatch)
        .where(
            WorkflowDispatch.state.in_(["PENDING", "FAILED"]),
            WorkflowDispatch.next_retry_at <= now,
            WorkflowDispatch.attempt_count < max_retries,
        )
        .order_by(WorkflowDispatch.next_retry_at.asc())
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    res = await session.execute(stmt)
    dispatches = list(res.scalars().all())
    retried_ids: list[uuid.UUID] = []
    from app.tasks.workflow_tasks import execute_workflow_task

    for d in dispatches:
        try:
            await execute_workflow_task.kiq(run_id=str(d.run_id))
            d.state = "DISPATCHED"
            d.attempt_count += 1
            d.last_error = None
            retried_ids.append(d.run_id)
        except Exception as exc:
            d.state = "FAILED"
            d.attempt_count += 1
            d.last_error = str(exc)
            backoff = min(300, 5 * (2 ** min(d.attempt_count, 6)))
            d.next_retry_at = now + timedelta(seconds=backoff)

    if dispatches:
        await session.commit()
    return retried_ids


async def _run_heartbeat_loop(
    run_id: uuid.UUID,
    worker_id: str,
    generation: int,
    stop_event: asyncio.Event,
    fenced_event: asyncio.Event,
    interval_seconds: float = 10.0,
    lease_seconds: int = 30,
):
    """Background task renewing worker lease periodically until stopped or fenced."""
    from app.core.database import AsyncSessionLocal

    try:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
                break
            except asyncio.TimeoutError:
                pass
            if stop_event.is_set():
                break
            try:
                async with AsyncSessionLocal() as hb_session:
                    success = await heartbeat_workflow_run(
                        session=hb_session,
                        run_id=run_id,
                        worker_id=worker_id,
                        generation=generation,
                        lease_seconds=lease_seconds,
                    )
                    if not success:
                        logger.warning(
                            "Worker '%s' generation %d heartbeat failed for run '%s' (lease lost or fenced)",
                            worker_id,
                            generation,
                            run_id,
                        )
                        fenced_event.set()
                        break
            except Exception as hb_err:
                logger.error("Heartbeat error for run '%s': %s", run_id, hb_err)
    except asyncio.CancelledError:
        pass


async def execute_claimed_run(
    session: AsyncSession,
    run_id: uuid.UUID,
    worker_id: str,
    run_instance: WorkflowRun | None = None,
    resume_payload: dict[str, Any] | None = None,
) -> WorkflowRun:
    """Executes a previously claimed workflow execution run under active heartbeat and fencing."""
    if run_instance is not None:
        run = run_instance
    else:
        stmt = select(WorkflowRun).where(WorkflowRun.id == run_id)
        res = await session.execute(stmt)
        run = res.scalar_one_or_none()
        if not run:
            raise HTTPException(status_code=404, detail=f"Workflow run '{run_id}' not found")
        if run.worker_id != worker_id or run.status != "RUNNING":
            raise HTTPException(status_code=409, detail=f"Workflow run is not owned by worker '{worker_id}' (status={run.status})")

    generation = run.generation
    definition = run.definition_snapshot
    if not definition:
        raise HTTPException(status_code=500, detail="Run definition snapshot is missing")

    checkpointer = await get_checkpointer()
    if checkpointer is None:
        raise HTTPException(status_code=503, detail="Workflow checkpointing is unavailable")

    compiled_graph = WorkflowCompiler.compile(
        nodes=definition["nodes"],
        edges=definition["edges"],
        checkpointer=checkpointer,
        validate=True,
    )

    now = datetime.now(timezone.utc)
    initial_state = {
        "messages": [],
        "variables": dict(run.input_data or {}),
        "user_id": run.user_id,
        "workflow_id": str(run.workflow_id),
        "thread_id": run.thread_id,
        "run_id": str(run_id),
        "worker_id": worker_id,
        "generation": generation,
        "event_sequence": 0,
        "node_history": [],
        "error": None,
        "budget_seconds": run.budget_seconds,
        "consumed_seconds": run.consumed_seconds or 0.0,
        "execution_started_at": now.isoformat(),
    }

    exec_config: RunnableConfig = {
        "configurable": {"thread_id": run.checkpoint_id},
        "recursion_limit": definition["recursion_limit"],
    }

    stop_heartbeat = asyncio.Event()
    fenced_event = asyncio.Event()
    heartbeat_task = None
    if hasattr(session, "execute"):
        heartbeat_task = asyncio.create_task(
            _run_heartbeat_loop(
                run_id=run_id,
                worker_id=worker_id,
                generation=generation,
                stop_event=stop_heartbeat,
                fenced_event=fenced_event,
                interval_seconds=10.0,
                lease_seconds=30,
            )
        )

    if isinstance(session, AsyncSession):
        dispatch_result = await session.execute(
            select(WorkflowDispatch.resume_payload).where(WorkflowDispatch.run_id == run_id)
            .order_by(WorkflowDispatch.created_at.desc()).limit(1)
        )
        resume_payload = dispatch_result.scalar_one_or_none()
        # Release the read transaction while the graph performs external work.
        session.expunge(run)
        await session.rollback()

    session_token = set_execution_session(session)
    try:
        if resume_payload is not None:
            await compiled_graph.aupdate_state(exec_config, {
                "worker_id": worker_id, "generation": generation,
                "control_request": None, "consumed_seconds": initial_state["consumed_seconds"],
                "execution_started_at": initial_state["execution_started_at"],
            })
            final_state = await compiled_graph.ainvoke(
                Command(resume=resume_payload["resume_data"]), config=exec_config,
            )
        else:
            final_state = await compiled_graph.ainvoke(initial_state, config=exec_config)
        if fenced_event.is_set():
            raise WorkerFencedError(f"Worker '{worker_id}' generation {generation} was fenced during execution")
        _record_execution_result(run, final_state, definition)
    except GraphInterrupt as interrupt_exc:
        logger.info("Workflow run '%s' was paused by interrupt", run_id)
        _record_execution_result(run, {"__interrupt__": interrupt_exc.args[0]}, definition)
    except WorkflowCancelledError as cancel_exc:
        logger.info("Workflow run '%s' was cancelled: %s", run_id, cancel_exc)
        run.status = "CANCELLED"
        run.control_request = None
        now_c = datetime.now(timezone.utc)
        run.completed_at = now_c
        if run.execution_started_at:
            start_time = run.execution_started_at
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            elapsed = (now_c - start_time).total_seconds()
            run.consumed_seconds = (run.consumed_seconds or 0.0) + max(0.0, elapsed)
            run.execution_started_at = None
        run.error_message = str(cancel_exc)
    except WorkflowTimeoutError as timeout_exc:
        logger.info("Workflow run '%s' timed out: %s", run_id, timeout_exc)
        run.status = "TIMED_OUT"
        run.control_request = None
        now_t = datetime.now(timezone.utc)
        run.completed_at = now_t
        if run.execution_started_at:
            start_time = run.execution_started_at
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            elapsed = (now_t - start_time).total_seconds()
            run.consumed_seconds = (run.consumed_seconds or 0.0) + max(0.0, elapsed)
            run.execution_started_at = None
        run.error_message = str(timeout_exc)
    except WorkerFencedError as fenced_exc:
        logger.warning("Workflow run '%s' execution halted by fencing: %s", run_id, fenced_exc)
        raise
    except Exception as exc:
        logger.error("Workflow run '%s' execution failed: %s", run_id, exc, exc_info=True)
        run.status = "FAILED"
        run.error_message = str(exc)
        now_e = datetime.now(timezone.utc)
        run.completed_at = now_e
        if run.execution_started_at:
            start_time = run.execution_started_at
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            elapsed = (now_e - start_time).total_seconds()
            run.consumed_seconds = (run.consumed_seconds or 0.0) + max(0.0, elapsed)
            run.execution_started_at = None
    finally:
        stop_heartbeat.set()
        if heartbeat_task:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        reset_execution_session(session_token)

    if fenced_event.is_set():
        raise WorkerFencedError(f"Worker '{worker_id}' generation {generation} was fenced; status update rejected")

    if isinstance(session, AsyncSession):
        values = {
            key: getattr(run, key) for key in (
                "status", "control_request", "completed_at", "consumed_seconds",
                "execution_started_at", "error_message", "output_data", "execution_log",
            )
        }
        values["lease_expires_at"] = None
        finalized = await session.execute(
            update(WorkflowRun).where(
                WorkflowRun.id == run_id, WorkflowRun.worker_id == worker_id,
                WorkflowRun.generation == generation, WorkflowRun.status == "RUNNING",
                WorkflowRun.lease_expires_at > datetime.now(timezone.utc),
            ).values(**values).returning(WorkflowRun.id)
        )
        if finalized.scalar_one_or_none() is None:
            await session.rollback()
            raise WorkerFencedError("Execution ownership changed before finalization")
        await session.commit()
        return (await session.execute(select(WorkflowRun).where(WorkflowRun.id == run_id))).scalar_one()
    await session.commit()
    await session.refresh(run)
    return run


async def execute_workflow(
    session: AsyncSession,
    workflow_id: uuid.UUID,
    user_id: str,
    payload: WorkflowRunRequest,
    worker_id: str | None = None,
) -> WorkflowRun:
    """Executes a workflow state graph by creating, claiming, and executing the run."""
    run, _ = await create_workflow_run(session, workflow_id, user_id, payload, queue_execution=False)
    assigned_worker = worker_id or f"direct-{uuid.uuid4().hex[:8]}"
    if isinstance(session, AsyncSession):
        claimed = await claim_workflow_run(session, run.id, worker_id=assigned_worker, lease_seconds=60)
        if not claimed:
            raise HTTPException(status_code=409, detail="Could not claim newly created workflow run")
    else:
        run.worker_id = assigned_worker
        run.status = "RUNNING"
        run.generation = 1
        now = datetime.now(timezone.utc)
        run.started_at = now
        run.execution_started_at = now
    return await execute_claimed_run(session, run.id, worker_id=assigned_worker, run_instance=run)


async def resume_workflow_run(
    session: AsyncSession,
    run_id: uuid.UUID,
    user_id: str,
    payload: WorkflowResumeRequest,
) -> WorkflowRun:
    """Queues continuation of an interrupted or paused run without executing in the request."""
    run = await get_workflow_run(session, run_id, user_id, for_update=True)
    if run.status not in ("INTERRUPTED", "PAUSED"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Workflow run is in status '{run.status}', only 'INTERRUPTED' or 'PAUSED' runs can be resumed.",
        )

    if not run.definition_snapshot or not run.checkpoint_id:
        raise HTTPException(status_code=409, detail="This legacy run has no pinned definition or isolated checkpoint. Start a new run.")
    if run.status == "INTERRUPTED":
        pending = (run.execution_log or [{}])[-1].get("interrupts", [])
        if any(datetime.fromisoformat(item["expires_at"]) <= datetime.now(timezone.utc) for item in pending):
            run.status = "EXPIRED"
            run.completed_at = datetime.now(timezone.utc)
            run.error_message = "Approval deadline expired"
            await session.commit()
            return run

    # Queue the continuation and its decision in the same transaction.
    dispatch_result = await session.execute(
        select(WorkflowDispatch).where(WorkflowDispatch.run_id == run.id).with_for_update()
    )
    dispatch = dispatch_result.scalar_one_or_none()
    if dispatch is None:
        dispatch = WorkflowDispatch(run_id=run.id)
        session.add(dispatch)
    dispatch.resume_payload = {"resume_data": payload.resume_data}
    dispatch.state = "PENDING"
    dispatch.attempt_count = 0
    dispatch.next_retry_at = datetime.now(timezone.utc)
    dispatch.last_error = None
    run.status = "PENDING"
    run.control_request = None
    run.worker_id = None
    run.lease_expires_at = None
    await session.commit()
    await session.refresh(run)
    return run


async def get_workflow_run(
    session: AsyncSession,
    run_id: uuid.UUID,
    user_id: str,
    for_update: bool = False,
) -> WorkflowRun:
    """Retrieves a workflow execution run by ID enforcing tenant isolation."""
    stmt = select(WorkflowRun).where(WorkflowRun.id == run_id)
    if for_update:
        stmt = stmt.with_for_update().execution_options(populate_existing=True)
    result = await session.execute(stmt)
    run = result.scalar_one_or_none()

    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow run with ID '{run_id}' not found.",
        )
    if run.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this workflow run.",
        )
    return run


async def list_workflow_runs(
    session: AsyncSession,
    workflow_id: uuid.UUID,
    user_id: str,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[WorkflowRun], int]:
    """Lists execution runs for a given workflow."""
    # Enforce workflow ownership first
    await get_workflow_by_id(session, workflow_id, user_id)

    count_query = select(func.count(WorkflowRun.id)).where(
        WorkflowRun.workflow_id == workflow_id,
        WorkflowRun.user_id == user_id,
    )
    total_result = await session.execute(count_query)
    total = total_result.scalar_one() or 0

    offset = (page - 1) * page_size
    query = (
        select(WorkflowRun)
        .where(
            WorkflowRun.workflow_id == workflow_id,
            WorkflowRun.user_id == user_id,
        )
        .order_by(WorkflowRun.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    result = await session.execute(query)
    items = list(result.scalars().all())

    return items, total


async def get_workflow_run_events(
    session: AsyncSession,
    run_id: uuid.UUID,
    user_id: str,
    after_sequence: int | None = None,
    limit: int = 50,
) -> tuple[list[WorkflowRunEvent], int, int | None]:
    """Retrieves cursor-paginated execution trace events for a workflow run enforcing tenant ownership."""
    run = await get_workflow_run(session, run_id, user_id)

    total_query = select(func.count(WorkflowRunEvent.id)).where(WorkflowRunEvent.run_id == run.id)
    total_res = await session.execute(total_query)
    total = total_res.scalar_one() or 0

    query = select(WorkflowRunEvent).where(WorkflowRunEvent.run_id == run.id)
    if after_sequence is not None:
        query = query.where(WorkflowRunEvent.sequence > after_sequence)

    query = query.order_by(WorkflowRunEvent.sequence.asc()).limit(limit)
    result = await session.execute(query)
    events = list(result.scalars().all())

    next_cursor = events[-1].sequence if events and len(events) == limit else None
    return events, total, next_cursor


async def get_workflow_run_state(
    session: AsyncSession,
    run_id: uuid.UUID,
    user_id: str,
) -> dict[str, Any]:
    """Inspects redacted state, current node, pending approval, and supported controls for a workflow run."""
    run = await get_workflow_run(session, run_id, user_id)

    checkpoint_state = None
    if run.checkpoint_id:
        try:
            checkpointer = await get_checkpointer()
            if checkpointer:
                config: RunnableConfig = {"configurable": {"thread_id": run.checkpoint_id}}
                checkpoint_tuple = await checkpointer.aget_tuple(config)
                if checkpoint_tuple and checkpoint_tuple.checkpoint:
                    checkpoint_state = checkpoint_tuple.checkpoint.get("channel_values", {})
        except Exception as exc:
            logger.warning("Could not read checkpoint state for run %s: %s", run_id, exc)

    raw_variables = {}
    node_history = []
    if checkpoint_state:
        raw_variables = checkpoint_state.get("variables") or {}
        node_history = checkpoint_state.get("node_history") or []
    else:
        raw_variables = run.output_data or run.input_data or {}
        if run.execution_log:
            node_history = (run.execution_log[-1] or {}).get("node_history") or []

    from app.services.workflow.tracing import redact_value
    redacted_variables = redact_value(raw_variables, max_depth=5, max_str_len=500)

    pending_approval = None
    if run.status == "INTERRUPTED" and run.execution_log:
        last_entry = run.execution_log[-1]
        interrupts = last_entry.get("interrupts") or []
        if interrupts:
            pending_approval = interrupts[0]

    supported_controls = []
    if run.status == "INTERRUPTED":
        supported_controls = ["resume"]
    elif run.status == "RUNNING":
        supported_controls = ["pause", "cancel"]
    elif run.status == "PAUSED":
        supported_controls = ["resume", "cancel"]

    current_node = node_history[-1] if node_history else None

    return {
        "run_id": run.id,
        "workflow_id": run.workflow_id,
        "status": run.status,
        "current_node": current_node,
        "pending_approval": pending_approval,
        "variables": redacted_variables if isinstance(redacted_variables, dict) else {},
        "node_history": node_history,
        "supported_controls": supported_controls,
        "version": (run.definition_snapshot or {}).get("version") if run.definition_snapshot else None,
    }


async def pause_workflow_run(
    session: AsyncSession,
    run_id: uuid.UUID,
    user_id: str,
) -> WorkflowRun:
    """Requests execution of a running workflow to pause at the next node boundary."""
    run = await get_workflow_run(session, run_id, user_id, for_update=True)
    if run.status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT", "EXPIRED"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot pause workflow run in terminal status '{run.status}'.",
        )
    if run.status == "PAUSED" or run.control_request == "pause_requested":
        return run
    if run.status != "RUNNING":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot pause workflow run in status '{run.status}', only 'RUNNING' runs can be paused.",
        )

    run.control_request = "pause_requested"
    run.control_set_at = datetime.now(timezone.utc)
    from app.services.workflow.tracing import record_run_event
    await record_run_event(
            event_session=session,
        run_id=run.id,
        invocation_id=uuid.uuid4(),
        node_id="system",
        node_type="system",
        event_type="pause_requested",
        status="pause_requested",
    )
    await session.commit()
    await session.refresh(run)
    logger.info("Pause requested for workflow run '%s'", run_id)
    return run


async def cancel_workflow_run(
    session: AsyncSession,
    run_id: uuid.UUID,
    user_id: str,
) -> WorkflowRun:
    """Requests cancellation of a workflow run. Inactive runs cancel immediately."""
    run = await get_workflow_run(session, run_id, user_id, for_update=True)
    if run.status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT", "EXPIRED"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot cancel workflow run in terminal status '{run.status}'.",
        )

    from app.services.workflow.tracing import record_run_event

    if run.status in ("PAUSED", "INTERRUPTED", "PENDING"):
        # Inactive run cancels immediately
        run.status = "CANCELLED"
        run.control_request = None
        run.completed_at = datetime.now(timezone.utc)
        await record_run_event(
            event_session=session,
            run_id=run.id,
            invocation_id=uuid.uuid4(),
            node_id="system",
            node_type="system",
            event_type="cancelled",
            status="cancelled",
        )
        await session.commit()
        await session.refresh(run)
        logger.info("Cancelled inactive workflow run '%s'", run_id)
        return run

    if run.status == "RUNNING":
        run.control_request = "cancel_requested"
        run.control_set_at = datetime.now(timezone.utc)
        await record_run_event(
            event_session=session,
            run_id=run.id,
            invocation_id=uuid.uuid4(),
            node_id="system",
            node_type="system",
            event_type="cancel_requested",
            status="cancel_requested",
        )
        await session.commit()
        await session.refresh(run)
        logger.info("Cancellation requested for running workflow run '%s'", run_id)
        return run

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Cannot cancel workflow run in status '{run.status}'.",
    )


# ---------------------------------------------------------------------------
# Worker Lease & Recovery Functions
# ---------------------------------------------------------------------------

async def claim_workflow_run(
    session: AsyncSession,
    run_id: uuid.UUID,
    worker_id: str,
    lease_seconds: int = 30,
) -> bool:
    """Claims a pending or unassigned running workflow execution for a worker."""
    if not hasattr(session, "execute"):
        return True
    stmt = (
        select(WorkflowRun)
        .where(WorkflowRun.id == run_id)
        .with_for_update()
    )
    res = await session.execute(stmt)
    run = res.scalar_one_or_none()
    if not run:
        return False
    if not isinstance(run, WorkflowRun):
        return True
    # Expired active runs require reconciliation, not blind graph replay.
    if run.status != "PENDING":
        return False
    now = datetime.now(timezone.utc)
    if run.worker_id and run.worker_id != worker_id:
        if run.lease_expires_at and run.lease_expires_at > now:
            return False
    run.generation = (run.generation or 0) + 1
    run.worker_id = worker_id
    run.status = "RUNNING"
    run.lease_expires_at = now + timedelta(seconds=lease_seconds)
    if not run.started_at:
        run.started_at = now
    run.execution_started_at = now
    await session.commit()
    return True


async def heartbeat_workflow_run(
    session: AsyncSession,
    run_id: uuid.UUID,
    worker_id: str,
    generation: int | None = None,
    lease_seconds: int = 30,
) -> bool:
    """Renews worker lease on a running workflow execution with fencing check."""
    if not hasattr(session, "execute"):
        return True
    stmt = (
        select(WorkflowRun)
        .where(WorkflowRun.id == run_id, WorkflowRun.status == "RUNNING")
        .with_for_update()
    )
    res = await session.execute(stmt)
    run = res.scalar_one_or_none()
    if not run:
        return False
    if not isinstance(run, WorkflowRun):
        return True
    if run.worker_id and run.worker_id != worker_id:
        return False
    if generation is not None and run.generation != generation:
        return False
    if run.lease_expires_at is None or run.lease_expires_at <= datetime.now(timezone.utc):
        return False
    run.worker_id = worker_id
    run.lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
    await session.commit()
    return True


async def reconcile_stale_workflow_runs(
    session: AsyncSession,
    threshold_seconds: int = 0,
) -> list[uuid.UUID]:
    """Sweeps and recovers stalled workflow runs whose worker lease has expired."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=threshold_seconds)

    stmt = (
        select(WorkflowRun)
        .where(
            WorkflowRun.status == "RUNNING",
            WorkflowRun.lease_expires_at.is_not(None),
            WorkflowRun.lease_expires_at < cutoff,
        )
        .with_for_update(skip_locked=True)
    )
    result = await session.execute(stmt)
    stale_runs = list(result.scalars().all())

    from app.services.workflow.tracing import record_run_event
    from app.models.workflow import WorkflowRunEvent
    from sqlalchemy import func

    recovered_ids = []
    for run in stale_runs:
        expired_at_str = run.lease_expires_at.isoformat() if run.lease_expires_at else "unknown"
        run.generation = (run.generation or 0) + 1
        run.status = "FAILED"
        run.completed_at = now
        run.error_message = f"Worker heartbeat expired (lease expired at {expired_at_str})"

        if run.execution_started_at:
            start_time = run.execution_started_at
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            if run.lease_expires_at:
                l_exp = run.lease_expires_at
                if l_exp.tzinfo is None:
                    l_exp = l_exp.replace(tzinfo=timezone.utc)
                elapsed = (l_exp - start_time).total_seconds()
            else:
                elapsed = (now - start_time).total_seconds()
            run.consumed_seconds = (run.consumed_seconds or 0.0) + max(0.0, elapsed)
            run.execution_started_at = None

        seq_stmt = select(func.max(WorkflowRunEvent.sequence)).where(WorkflowRunEvent.run_id == run.id)
        seq_res = await session.execute(seq_stmt)
        max_seq = seq_res.scalar() or 0
        allocated_seq = max_seq + 1

        await record_run_event(
            event_session=session,
            run_id=run.id,
            sequence=allocated_seq,
            invocation_id=uuid.uuid4(),
            node_id="system",
            node_type="system",
            event_type="lease_expired",
            status="failed",
            error_message=run.error_message,
        )
        recovered_ids.append(run.id)

    if recovered_ids:
        await session.commit()
        logger.warning("Reconciled %d stale workflow runs: %s", len(recovered_ids), recovered_ids)

    return recovered_ids




async def start_workflow_maintenance(interval_seconds: float = 10.0) -> None:
    """Retry dispatches and reconcile expired leases using short locked transactions."""
    from app.core.database import AsyncSessionLocal

    while True:
        try:
            async with AsyncSessionLocal() as session:
                await retry_pending_dispatches(session)
            async with AsyncSessionLocal() as session:
                await reconcile_stale_workflow_runs(session)
        except Exception:
            logger.exception("Workflow maintenance failed; retrying on the next cycle")
        await asyncio.sleep(interval_seconds)
