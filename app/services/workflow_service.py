from __future__ import annotations

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
from sqlalchemy import func, select
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.workflow import Workflow, WorkflowRun
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
    entry: dict[str, Any] = {"node_history": state.get("node_history", [])}
    if resumed:
        entry["resumed"] = True
    interrupts = state.get("__interrupt__") or ()
    if interrupts:
        now = datetime.now(timezone.utc)
        entry["interrupts"] = []
        for item in interrupts:
            value = getattr(item, "value", item)
            timeout = value.get("timeout_seconds", 86400) if isinstance(value, dict) else 86400
            entry["interrupts"].append({
                "id": getattr(item, "id", None), "value": value,
                "expires_at": (now + timedelta(seconds=timeout)).isoformat(),
            })
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

async def execute_workflow(
    session: AsyncSession,
    workflow_id: uuid.UUID,
    user_id: str,
    payload: WorkflowRunRequest,
) -> WorkflowRun:
    """Executes a workflow state graph with persistent checkpointing."""
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
    # Do not run a supposedly resumable workflow without durable checkpointing.
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

    # 1. Initialize run record
    run = WorkflowRun(
        id=run_id,
        workflow_id=workflow_id,
        user_id=user_id,
        thread_id=thread_id,
        checkpoint_id=checkpoint_id,
        definition_snapshot=definition,
        execution_log=[],
        status="RUNNING",
        input_data=payload.input_data,
        started_at=datetime.now(timezone.utc),
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    # 2. Compile graph with checkpointer
    try:
        compiled_graph = WorkflowCompiler.compile(
            nodes=definition["nodes"],
            edges=definition["edges"],
            checkpointer=checkpointer,
            validate=True,
        )

        initial_state = {
            "messages": [],
            "variables": dict(payload.input_data or {}),
            "user_id": user_id,
            "workflow_id": str(workflow_id),
            "thread_id": thread_id,
            "node_history": [],
            "error": None,
        }

        exec_config: RunnableConfig = {
            "configurable": {"thread_id": checkpoint_id},
            "recursion_limit": definition["recursion_limit"],
        }

        # Execute StateGraph
        final_state = await compiled_graph.ainvoke(initial_state, config=exec_config)

        _record_execution_result(run, final_state, definition)

    except GraphInterrupt as interrupt_exc:
        logger.info("Workflow run '%s' was paused by human-in-the-loop interrupt", run_id)
        _record_execution_result(run, {"__interrupt__": interrupt_exc.args[0]}, definition)
    except Exception as exc:
        logger.error("Workflow run '%s' execution failed: %s", run_id, exc, exc_info=True)
        run.status = "FAILED"
        run.error_message = str(exc)
        run.completed_at = datetime.now(timezone.utc)

    await session.commit()
    await session.refresh(run)
    return run


async def resume_workflow_run(
    session: AsyncSession,
    run_id: uuid.UUID,
    user_id: str,
    payload: WorkflowResumeRequest,
) -> WorkflowRun:
    """Resumes an interrupted workflow execution run."""
    run = await get_workflow_run(session, run_id, user_id, for_update=True)
    if run.status != "INTERRUPTED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Workflow run is in status '{run.status}', only 'INTERRUPTED' runs can be resumed.",
        )

    if not run.definition_snapshot or not run.checkpoint_id:
        raise HTTPException(status_code=409, detail="This legacy run has no pinned definition or isolated checkpoint. Start a new run.")
    definition = run.definition_snapshot
    pending = (run.execution_log or [{}])[-1].get("interrupts", [])
    if any(datetime.fromisoformat(item["expires_at"]) <= datetime.now(timezone.utc) for item in pending):
        run.status = "EXPIRED"
        run.completed_at = datetime.now(timezone.utc)
        run.error_message = "Approval deadline expired"
        await session.commit()
        return run

    try:
        checkpointer = await get_checkpointer()
    except Exception as cp_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Checkpointer unavailable to resume run: {cp_err}",
        )

    compiled_graph = WorkflowCompiler.compile(
        nodes=definition["nodes"],
        edges=definition["edges"],
        checkpointer=checkpointer,
        validate=True,
    )

    exec_config: RunnableConfig = {
        "configurable": {"thread_id": run.checkpoint_id},
        "recursion_limit": definition["recursion_limit"],
    }

    run.status = "RUNNING"
    await session.commit()

    try:
        # Resume LangGraph execution with human decision payload
        final_state = await compiled_graph.ainvoke(
            Command(resume=payload.resume_data),
            config=exec_config,
        )

        _record_execution_result(run, final_state, definition, resumed=True)
    except GraphInterrupt as interrupt_exc:
        logger.info("Workflow run '%s' interrupted again", run_id)
        _record_execution_result(run, {"__interrupt__": interrupt_exc.args[0]}, definition, resumed=True)
    except Exception as exc:
        logger.error("Error resuming workflow run '%s': %s", run_id, exc, exc_info=True)
        run.status = "FAILED"
        run.error_message = str(exc)
        run.completed_at = datetime.now(timezone.utc)

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

