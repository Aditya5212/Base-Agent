"""API endpoints for the Tool Builder platform.

Supports registration, management, inspection, and execution of custom reusable tools
(HTTP REST APIs, Sandboxed Python scripts, and Read-Only Database Queries).
"""

from __future__ import annotations

import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import UserContext, get_current_user
from app.core.database import get_db
from app.core.rate_limit import limiter
from app.schemas.tool import (
    ToolCreate,
    ToolExecutionRequest,
    ToolExecutionResponse,
    ToolListResponse,
    ToolResponse,
    ToolType,
    ToolUpdate,
)
from app.services.rag_tools import get_user_rag_tool_map
from app.services.tool_service import ToolService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools", tags=["Tools"])


@router.post(
    "",
    response_model=ToolResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new tool",
    description="Registers a new custom tool (HTTP REST API, Python code, or read-only SQL database query).",
)
@limiter.limit("30/minute")
async def create_tool_endpoint(
    request: Request,
    tool_in: ToolCreate,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ToolResponse:
    """Registers a new custom tool in the user's workspace."""
    tool = await ToolService.create_tool(db, user_id=current_user.user_id, tool_in=tool_in)
    return ToolResponse.from_orm_model(tool)


@router.get(
    "",
    response_model=ToolListResponse,
    summary="List available tools",
    description="Lists all custom and system tools accessible to the authenticated user.",
)
async def list_tools_endpoint(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    tool_type: ToolType | None = Query(None, description="Optional tool type filter"),
    search: str | None = Query(None, description="Search filter for tool name or description"),
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ToolListResponse:
    """Lists accessible tools with pagination and optional search."""
    skip = (page - 1) * page_size
    type_str = tool_type.value if tool_type else None
    total, items = await ToolService.list_tools(
        db,
        user_id=current_user.user_id,
        skip=skip,
        limit=page_size,
        tool_type=type_str,
        search=search,
    )
    return ToolListResponse(
        total=total,
        items=[ToolResponse.from_orm_model(t) for t in items],
    )


@router.get(
    "/built-in",
    summary="List built-in system tools",
    description="Lists all built-in mathematical, utility, and modular RAG research tools available for workflows.",
)
async def list_builtin_tools_endpoint(
    request: Request,
    current_user: UserContext = Depends(get_current_user),
) -> list[dict[str, str]]:
    """Lists all built-in tools (calculator, RAG research tools) available to the user."""
    rag_tools = get_user_rag_tool_map(current_user.user_id)
    built_ins = [
        {
            "name": "calculator",
            "type": "built_in",
            "description": "Evaluates basic arithmetic expressions safely (e.g. '128 * 4 + 15').",
        }
    ]
    for name, tool in rag_tools.items():
        if name in ("hybrid_document_search", "graph_rag_search", "multi_query_search"):
            built_ins.append({
                "name": name,
                "type": "built_in_rag",
                "description": tool.description,
            })
    return built_ins


@router.get(
    "/{tool_id}",
    response_model=ToolResponse,
    summary="Get tool details",
    description="Retrieves a specific tool definition by UUID with secrets masked.",
)
async def get_tool_endpoint(
    tool_id: uuid.UUID,
    request: Request,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ToolResponse:
    """Fetches details of a specific tool."""
    tool = await ToolService.get_tool(db, user_id=current_user.user_id, tool_id=tool_id)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool with ID '{tool_id}' not found.",
        )
    return ToolResponse.from_orm_model(tool)


@router.put(
    "/{tool_id}",
    response_model=ToolResponse,
    summary="Update a tool",
    description="Updates an existing tool definition and increments its version.",
)
@limiter.limit("30/minute")
async def update_tool_endpoint(
    tool_id: uuid.UUID,
    request: Request,
    tool_in: ToolUpdate,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ToolResponse:
    """Updates an existing custom tool."""
    tool = await ToolService.update_tool(db, user_id=current_user.user_id, tool_id=tool_id, tool_in=tool_in)
    return ToolResponse.from_orm_model(tool)


@router.delete(
    "/{tool_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a tool",
    description="Deletes a tool owned by the authenticated user.",
)
@limiter.limit("20/minute")
async def delete_tool_endpoint(
    tool_id: uuid.UUID,
    request: Request,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Deletes a custom tool from the workspace."""
    await ToolService.delete_tool(db, user_id=current_user.user_id, tool_id=tool_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{tool_id}/execute",
    response_model=ToolExecutionResponse,
    summary="Directly test or execute a tool",
    description="Executes a tool directly with test arguments and returns execution output and latency metrics.",
)
@limiter.limit("30/minute")
async def execute_tool_endpoint(
    tool_id: uuid.UUID,
    request: Request,
    payload: ToolExecutionRequest,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ToolExecutionResponse:
    """Executes a tool with arguments for testing or ad-hoc invocation."""
    return await ToolService.execute_tool(
        db,
        user_id=current_user.user_id,
        tool_id=tool_id,
        arguments=payload.arguments,
    )

