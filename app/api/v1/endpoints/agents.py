"""API endpoints for Agent configuration, management, and customization."""

from __future__ import annotations

import logging
import uuid
from fastapi import APIRouter, Depends, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import UserContext, get_current_user
from app.core.database import get_db
from app.core.rate_limit import limiter
from app.schemas.agent import (
    AgentCloneRequest,
    AgentCreate,
    AgentListResponse,
    AgentResponse,
    AgentUpdate,
)
from app.services.agent_service import (
    clone_agent,
    create_agent,
    delete_agent,
    get_agent_by_id,
    get_or_create_default_agent,
    list_agents,
    set_default_agent,
    update_agent,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["Agent Configuration"])


@router.post(
    "",
    response_model=AgentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new agent",
    description="Registers a new custom agent with identity, LLM provider, knowledge, and memory parameters.",
)
@limiter.limit("20/minute")
async def create_agent_endpoint(
    request: Request,
    agent_in: AgentCreate,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AgentResponse:
    """Creates a new agent owned by the authenticated user."""
    agent = await create_agent(db, user_id=current_user.user_id, agent_in=agent_in)
    return AgentResponse.model_validate(agent)


@router.get(
    "",
    response_model=AgentListResponse,
    summary="List agents",
    description="Retrieves a paginated list of agents owned by the current user.",
)
async def list_agents_endpoint(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    search: str | None = Query(None, description="Search filter for agent name or description"),
    is_default: bool | None = Query(None, description="Filter by default agent status"),
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AgentListResponse:
    """Lists agents for the current user."""
    items, total = await list_agents(
        db,
        user_id=current_user.user_id,
        is_admin=current_user.is_admin,
        page=page,
        page_size=page_size,
        search=search,
        is_default=is_default,
    )
    return AgentListResponse(items=[AgentResponse.model_validate(item) for item in items], total=total, page=page, page_size=page_size)


@router.get(
    "/default",
    response_model=AgentResponse,
    summary="Get default agent",
    description="Retrieves or automatically provisions the primary default agent for the current user.",
)
async def get_default_agent_endpoint(
    request: Request,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AgentResponse:
    """Returns the user's default agent, creating one if not yet provisioned."""
    agent = await get_or_create_default_agent(db, user_id=current_user.user_id)
    return AgentResponse.model_validate(agent)


@router.get(
    "/{agent_id}",
    response_model=AgentResponse,
    summary="Get agent details",
    description="Fetches full metadata and configuration for a specific agent.",
)
async def get_agent_endpoint(
    request: Request,
    agent_id: uuid.UUID,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AgentResponse:
    """Retrieves an agent by ID with ownership verification."""
    agent = await get_agent_by_id(
        db,
        agent_id=agent_id,
        user_id=current_user.user_id,
        is_admin=current_user.is_admin,
    )
    return AgentResponse.model_validate(agent)


@router.put(
    "/{agent_id}",
    response_model=AgentResponse,
    summary="Update agent",
    description="Updates configuration or metadata for an agent, incrementing its version.",
)
@limiter.limit("20/minute")
async def update_agent_endpoint(
    request: Request,
    agent_id: uuid.UUID,
    agent_in: AgentUpdate,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AgentResponse:
    """Updates an agent and increments its version counter."""
    agent = await update_agent(
        db,
        agent_id=agent_id,
        user_id=current_user.user_id,
        agent_in=agent_in,
        is_admin=current_user.is_admin,
    )
    return AgentResponse.model_validate(agent)


@router.delete(
    "/{agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete agent",
    description="Deletes an agent record after verifying ownership.",
)
async def delete_agent_endpoint(
    request: Request,
    agent_id: uuid.UUID,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Deletes an agent."""
    await delete_agent(
        db,
        agent_id=agent_id,
        user_id=current_user.user_id,
        is_admin=current_user.is_admin,
    )


@router.post(
    "/{agent_id}/clone",
    response_model=AgentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Clone agent",
    description="Duplicates an existing agent configuration into a new agent for experimentation.",
)
@limiter.limit("20/minute")
async def clone_agent_endpoint(
    request: Request,
    agent_id: uuid.UUID,
    clone_in: AgentCloneRequest,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AgentResponse:
    """Duplicates an agent config."""
    agent = await clone_agent(
        db,
        agent_id=agent_id,
        user_id=current_user.user_id,
        clone_in=clone_in,
        is_admin=current_user.is_admin,
    )
    return AgentResponse.model_validate(agent)


@router.post(
    "/{agent_id}/set-default",
    response_model=AgentResponse,
    summary="Set default agent",
    description="Promotes the specified agent to be the user's primary default agent.",
)
async def set_default_agent_endpoint(
    request: Request,
    agent_id: uuid.UUID,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AgentResponse:
    """Sets an agent as default."""
    agent = await set_default_agent(
        db,
        agent_id=agent_id,
        user_id=current_user.user_id,
        is_admin=current_user.is_admin,
    )
    return AgentResponse.model_validate(agent)
