from __future__ import annotations

import logging
from typing import Any
import uuid

from fastapi import APIRouter, Depends, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import UserContext, get_current_user
from app.core.database import get_db
from app.core.rate_limit import limiter
from app.schemas.workflow import (
    GraphValidationResponse,
    WorkflowCreate,
    WorkflowListResponse,
    WorkflowNodeKnowledgeConfig,
    WorkflowResponse,
    WorkflowResumeRequest,
    WorkflowRunControlResponse,
    WorkflowRunEventListResponse,
    WorkflowRunEventResponse,
    WorkflowRunListResponse,
    WorkflowRunRequest,
    WorkflowRunResponse,
    WorkflowRunStateResponse,
    WorkflowUpdate,
)
from app.services.workflow.validator import validate_workflow_graph
from app.services.workflow_service import (
    attach_node_knowledge,
    cancel_workflow_run,
    create_workflow,
    delete_workflow,
    execute_workflow,
    get_workflow_by_id,
    get_workflow_run,
    get_workflow_run_events,
    get_workflow_run_state,
    list_workflow_runs,
    list_workflows,
    pause_workflow_run,
    resume_workflow_run,
    update_workflow,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["Workflow Engine"])


@router.get("/capabilities", summary="Supported execution capabilities")
async def execution_capabilities(current_user: UserContext = Depends(get_current_user)) -> dict[str, Any]:
    from app.services.workflow.validator import SUPPORTED_NODE_TYPES
    return {
        "node_types": sorted(SUPPORTED_NODE_TYPES),
        "knowledge_providers": ["hybrid", "lightrag", "multi_query"],
        "knowledge_scopes": ["user", "thread", "global"],
        "knowledge_attachment_node_types": ["retrieve"],
        "parallel_execution": False, "subgraph_execution": False,
        "agent_tools": True, "structured_output_validation": True,
        "long_term_memory": False, "automatic_summarization": False,
        "execution_events": True, "state_inspection": True,
        "manual_pause": True, "cancellation": True,
        "streaming": {"text": "tokens", "tools_or_output_validation": "buffered"},
    }


@router.post(
    "",
    response_model=WorkflowResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create workflow",
    description="Registers a new workflow with graph nodes and directed edges after validating topology.",
)
@limiter.limit("20/minute")
async def create_workflow_endpoint(
    request: Request,
    workflow_in: WorkflowCreate,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WorkflowResponse:
    workflow = await create_workflow(
        session=db,
        user_id=current_user.user_id,
        payload=workflow_in,
    )
    return WorkflowResponse.model_validate(workflow)


@router.get(
    "",
    response_model=WorkflowListResponse,
    summary="List workflows",
    description="Retrieves a paginated list of workflows created by the authenticated user.",
)
@limiter.limit("60/minute")
async def list_workflows_endpoint(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    search: str | None = Query(None, description="Filter by workflow name or description"),
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WorkflowListResponse:
    items, total = await list_workflows(
        session=db,
        user_id=current_user.user_id,
        page=page,
        page_size=page_size,
        search=search,
    )
    return WorkflowListResponse(
        items=[WorkflowResponse.model_validate(w) for w in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post(
    "/validate",
    response_model=GraphValidationResponse,
    summary="Validate workflow graph topology",
    description="Performs dry-run topological validation of nodes and edges without persisting.",
)
@limiter.limit("30/minute")
async def validate_workflow_endpoint(
    request: Request,
    workflow_in: WorkflowCreate,
    current_user: UserContext = Depends(get_current_user),
) -> GraphValidationResponse:
    raw_nodes = [node.model_dump(mode="json") for node in workflow_in.nodes]
    raw_edges = [edge.model_dump(mode="json") for edge in workflow_in.edges]
    is_valid, errors, warnings = validate_workflow_graph(raw_nodes, raw_edges)

    return GraphValidationResponse(
        valid=is_valid,
        errors=errors,
        warnings=warnings,
        node_count=len(raw_nodes),
        edge_count=len(raw_edges),
    )


@router.get(
    "/runs/{run_id}",
    response_model=WorkflowRunResponse,
    summary="Get workflow execution run details",
    description="Retrieves execution status, output data, and traces for a specific run.",
)
@limiter.limit("60/minute")
async def get_workflow_run_endpoint(
    request: Request,
    run_id: uuid.UUID,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WorkflowRunResponse:
    run = await get_workflow_run(session=db, run_id=run_id, user_id=current_user.user_id)
    return WorkflowRunResponse.model_validate(run)


@router.get(
    "/runs/{run_id}/events",
    response_model=WorkflowRunEventListResponse,
    summary="Get workflow execution run trace events",
    description="Retrieves cursor-paginated granular node and lifecycle events for a specific run.",
)
@limiter.limit("60/minute")
async def get_workflow_run_events_endpoint(
    request: Request,
    run_id: uuid.UUID,
    after_sequence: int | None = Query(None, ge=0, description="Fetch events with sequence greater than this cursor"),
    limit: int = Query(50, ge=1, le=200, description="Maximum events to return"),
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WorkflowRunEventListResponse:
    events, total, next_cursor = await get_workflow_run_events(
        session=db,
        run_id=run_id,
        user_id=current_user.user_id,
        after_sequence=after_sequence,
        limit=limit,
    )
    return WorkflowRunEventListResponse(
        items=[WorkflowRunEventResponse.model_validate(e) for e in events],
        total=total,
        next_cursor=next_cursor,
    )


@router.get(
    "/runs/{run_id}/state",
    response_model=WorkflowRunStateResponse,
    summary="Get workflow execution run state",
    description="Inspects redacted state variables, current node, pending approval, and supported controls.",
)
@limiter.limit("60/minute")
async def get_workflow_run_state_endpoint(
    request: Request,
    run_id: uuid.UUID,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WorkflowRunStateResponse:
    state_data = await get_workflow_run_state(
        session=db,
        run_id=run_id,
        user_id=current_user.user_id,
    )
    return WorkflowRunStateResponse(**state_data)


@router.post(
    "/runs/{run_id}/resume",
    response_model=WorkflowRunResponse,
    summary="Resume interrupted or paused workflow run",
    description="Resumes execution of an INTERRUPTED or PAUSED workflow run.",
)
@limiter.limit("20/minute")
async def resume_workflow_run_endpoint(
    request: Request,
    run_id: uuid.UUID,
    payload: WorkflowResumeRequest,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WorkflowRunResponse:
    run = await resume_workflow_run(
        session=db,
        run_id=run_id,
        user_id=current_user.user_id,
        payload=payload,
    )
    return WorkflowRunResponse.model_validate(run)


@router.post(
    "/runs/{run_id}/pause",
    response_model=WorkflowRunControlResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Pause active workflow run",
    description="Requests execution of a RUNNING workflow to pause at the next safe node boundary.",
)
@limiter.limit("20/minute")
async def pause_workflow_run_endpoint(
    request: Request,
    run_id: uuid.UUID,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WorkflowRunControlResponse:
    run = await pause_workflow_run(session=db, run_id=run_id, user_id=current_user.user_id)
    return WorkflowRunControlResponse(
        run_id=run.id,
        status=run.status,
        control_request=run.control_request,
        message="Workflow pause requested; execution will suspend at next node boundary.",
    )


@router.post(
    "/runs/{run_id}/cancel",
    response_model=WorkflowRunControlResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Cancel workflow run",
    description="Requests cancellation of a workflow run. Inactive paused/interrupted runs cancel immediately.",
)
@limiter.limit("20/minute")
async def cancel_workflow_run_endpoint(
    request: Request,
    run_id: uuid.UUID,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WorkflowRunControlResponse:
    run = await cancel_workflow_run(session=db, run_id=run_id, user_id=current_user.user_id)
    msg = (
        "Workflow run cancelled immediately."
        if run.status == "CANCELLED"
        else "Workflow cancellation requested; execution will stop at next boundary."
    )
    return WorkflowRunControlResponse(
        run_id=run.id,
        status=run.status,
        control_request=run.control_request,
        message=msg,
    )


@router.get(
    "/{workflow_id}",
    response_model=WorkflowResponse,
    summary="Get workflow details",
    description="Retrieves workflow configuration, nodes, and edges by ID.",
)
@limiter.limit("60/minute")
async def get_workflow_endpoint(
    request: Request,
    workflow_id: uuid.UUID,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WorkflowResponse:
    workflow = await get_workflow_by_id(session=db, workflow_id=workflow_id, user_id=current_user.user_id)
    return WorkflowResponse.model_validate(workflow)


@router.put(
    "/{workflow_id}",
    response_model=WorkflowResponse,
    summary="Update workflow",
    description="Updates workflow configuration, re-validates topology, and auto-increments version.",
)
@limiter.limit("20/minute")
async def update_workflow_endpoint(
    request: Request,
    workflow_id: uuid.UUID,
    workflow_in: WorkflowUpdate,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WorkflowResponse:
    workflow = await update_workflow(
        session=db,
        workflow_id=workflow_id,
        user_id=current_user.user_id,
        payload=workflow_in,
    )
    return WorkflowResponse.model_validate(workflow)


@router.delete(
    "/{workflow_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete workflow",
    description="Deletes a workflow and cascades deletion to all execution run records.",
)
@limiter.limit("20/minute")
async def delete_workflow_endpoint(
    request: Request,
    workflow_id: uuid.UUID,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    await delete_workflow(session=db, workflow_id=workflow_id, user_id=current_user.user_id)


@router.post(
    "/{workflow_id}/run",
    response_model=WorkflowRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute workflow",
    description="Compiles and executes the workflow with input variables, returning execution output.",
)
@limiter.limit("20/minute")
async def run_workflow_endpoint(
    request: Request,
    workflow_id: uuid.UUID,
    payload: WorkflowRunRequest,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WorkflowRunResponse:
    run = await execute_workflow(
        session=db,
        workflow_id=workflow_id,
        user_id=current_user.user_id,
        payload=payload,
    )
    return WorkflowRunResponse.model_validate(run)


@router.get(
    "/{workflow_id}/runs",
    response_model=WorkflowRunListResponse,
    summary="List workflow execution runs",
    description="Lists previous execution runs and outcomes for this workflow.",
)
@limiter.limit("60/minute")
async def list_workflow_runs_endpoint(
    request: Request,
    workflow_id: uuid.UUID,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WorkflowRunListResponse:
    items, total = await list_workflow_runs(
        session=db,
        workflow_id=workflow_id,
        user_id=current_user.user_id,
        page=page,
        page_size=page_size,
    )
    return WorkflowRunListResponse(
        items=[WorkflowRunResponse.model_validate(r) for r in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post(
    "/{workflow_id}/nodes/{node_id}/knowledge",
    response_model=WorkflowResponse,
    summary="Attach or configure knowledge for a workflow node",
    description="Configures knowledge provider (hybrid, lightrag, multi_query) and retrieval parameters for a node.",
)
@limiter.limit("30/minute")
async def attach_node_knowledge_endpoint(
    request: Request,
    workflow_id: uuid.UUID,
    node_id: str,
    knowledge_config: WorkflowNodeKnowledgeConfig,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WorkflowResponse:
    """Configures knowledge provider and retrieval settings for a specific workflow node."""
    workflow = await attach_node_knowledge(
        session=db,
        workflow_id=workflow_id,
        user_id=current_user.user_id,
        node_id=node_id,
        knowledge_config=knowledge_config,
    )
    return WorkflowResponse.model_validate(workflow)


