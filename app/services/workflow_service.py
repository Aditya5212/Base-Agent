from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any
import uuid

from fastapi import HTTPException, status
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.workflow import Workflow, WorkflowRun
from app.schemas.workflow import (
    WorkflowCreate,
    WorkflowResumeRequest,
    WorkflowRunRequest,
    WorkflowUpdate,
)
from app.services.memory import get_checkpointer
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

    workflow.version += 1
    await session.commit()
    await session.refresh(workflow)
    logger.info("Updated workflow '%s' to version %d", workflow.name, workflow.version)
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

async def execute_workflow(
    session: AsyncSession,
    workflow_id: uuid.UUID,
    user_id: str,
    payload: WorkflowRunRequest,
) -> WorkflowRun:
    """Executes a workflow state graph with persistent checkpointing."""
    workflow = await get_workflow_by_id(session, workflow_id, user_id)
    run_id = uuid.uuid4()
    thread_id = payload.thread_id or f"wf_run_{run_id}"

    # 1. Initialize run record
    run = WorkflowRun(
        id=run_id,
        workflow_id=workflow_id,
        user_id=user_id,
        thread_id=thread_id,
        status="RUNNING",
        input_data=payload.input_data,
        started_at=datetime.now(timezone.utc),
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    # 2. Compile graph with checkpointer
    try:
        checkpointer = await get_checkpointer()
    except Exception as cp_err:
        logger.warning("Checkpointer unavailable, compiling without persistence: %s", cp_err)
        checkpointer = None

    try:
        compiled_graph = WorkflowCompiler.compile(
            nodes=workflow.nodes,
            edges=workflow.edges,
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

        exec_config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": workflow.recursion_limit,
        }

        # Execute StateGraph
        final_state = await compiled_graph.ainvoke(initial_state, config=exec_config)

        # 3. Determine output data
        # Check if an END node designated a specific response_variable
        output_data: dict[str, Any] = {}
        for n in workflow.nodes:
            if str(n.get("type", "")).lower() == "end":
                resp_var = n.get("config", {}).get("response_variable")
                if resp_var and resp_var in final_state.get("variables", {}):
                    output_data[resp_var] = final_state["variables"][resp_var]

        if not output_data:
            output_data = dict(final_state.get("variables") or {})

        run.status = "COMPLETED"
        run.output_data = output_data
        run.execution_log = [{"node_history": final_state.get("node_history", [])}]
        run.completed_at = datetime.now(timezone.utc)

    except GraphInterrupt as interrupt_exc:
        logger.info("Workflow run '%s' was paused by human-in-the-loop interrupt", run_id)
        run.status = "INTERRUPTED"
        run.execution_log = [{"interrupt": str(interrupt_exc)}]
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
    run = await get_workflow_run(session, run_id, user_id)
    if run.status != "INTERRUPTED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Workflow run is in status '{run.status}', only 'INTERRUPTED' runs can be resumed.",
        )

    workflow = await get_workflow_by_id(session, run.workflow_id, user_id)

    try:
        checkpointer = await get_checkpointer()
    except Exception as cp_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Checkpointer unavailable to resume run: {cp_err}",
        )

    compiled_graph = WorkflowCompiler.compile(
        nodes=workflow.nodes,
        edges=workflow.edges,
        checkpointer=checkpointer,
        validate=False,
    )

    exec_config = {
        "configurable": {"thread_id": run.thread_id},
        "recursion_limit": workflow.recursion_limit,
    }

    try:
        # Resume LangGraph execution with human decision payload
        final_state = await compiled_graph.ainvoke(
            Command(resume=payload.resume_data),
            config=exec_config,
        )

        run.status = "COMPLETED"
        run.output_data = dict(final_state.get("variables") or {})
        run.execution_log = run.execution_log + [{"resumed": True, "node_history": final_state.get("node_history", [])}]
        run.completed_at = datetime.now(timezone.utc)
    except GraphInterrupt as interrupt_exc:
        logger.info("Workflow run '%s' interrupted again", run_id)
        run.status = "INTERRUPTED"
        run.execution_log = run.execution_log + [{"interrupt": str(interrupt_exc)}]
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
) -> WorkflowRun:
    """Retrieves a workflow execution run by ID enforcing tenant isolation."""
    stmt = select(WorkflowRun).where(WorkflowRun.id == run_id)
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

