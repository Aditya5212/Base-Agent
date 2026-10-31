"""Agent service: business logic, lifecycle, caching, and resolution for AI Agents."""

from __future__ import annotations

import logging
from typing import Any
import uuid

from fastapi import HTTPException, status
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rag.registry import PROVIDER_REGISTRY
from app.models.agent import Agent
from app.schemas.agent import AgentCloneRequest, AgentConfig, AgentCreate, AgentUpdate
from app.services.thread_service import get_or_create_user

logger = logging.getLogger(__name__)

# In-memory LRU cache for resolved agent configs: key is (agent_id_str, version_int)
_AGENT_CONFIG_CACHE: dict[tuple[str, int], AgentConfig] = {}
_MAX_CACHE_SIZE = 256


def get_cached_agent_config(agent_id: str, version: int) -> AgentConfig | None:
    """Retrieves an AgentConfig from the in-memory cache if present."""
    return _AGENT_CONFIG_CACHE.get((str(agent_id), version))


def set_cached_agent_config(agent_id: str, version: int, config: AgentConfig) -> None:
    """Stores an AgentConfig in the in-memory cache, evicting the oldest if full."""
    if len(_AGENT_CONFIG_CACHE) >= _MAX_CACHE_SIZE:
        # Simple FIFO/LRU eviction: remove first key
        oldest_key = next(iter(_AGENT_CONFIG_CACHE))
        _AGENT_CONFIG_CACHE.pop(oldest_key, None)
    _AGENT_CONFIG_CACHE[(str(agent_id), version)] = config


def invalidate_agent_cache(agent_id: str) -> None:
    """Invalidates all cached versions for a given agent_id."""
    keys_to_remove = [k for k in _AGENT_CONFIG_CACHE if k[0] == str(agent_id)]
    for k in keys_to_remove:
        _AGENT_CONFIG_CACHE.pop(k, None)


def validate_agent_config(config_data: dict[str, Any] | AgentConfig) -> AgentConfig:
    """Validates raw dictionary or AgentConfig instance against known providers and constraints."""
    if isinstance(config_data, AgentConfig):
        cfg = config_data
    else:
        try:
            cfg = AgentConfig.model_validate(config_data)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Invalid agent configuration: {exc}",
            ) from exc

    # Verify that enabled knowledge providers are registered
    for prov in cfg.knowledge_providers:
        if prov.enabled and prov.name not in PROVIDER_REGISTRY:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Knowledge provider '{prov.name}' is not registered. Available: {list(PROVIDER_REGISTRY.keys())}",
            )

    return cfg


async def get_or_create_default_agent(session: AsyncSession, user_id: str) -> Agent:
    """Ensures a default agent exists for the user.
    
    If no default agent exists, auto-creates one with standard Base Agent settings.
    This guarantees full backward compatibility for users invoking chat without specifying an agent_id.
    """
    await get_or_create_user(session, user_id=user_id)

    stmt = (
        select(Agent)
        .where(Agent.user_id == user_id, Agent.is_default.is_(True))
        .order_by(Agent.created_at.asc())
        .limit(1)
    )
    result = await session.execute(stmt)
    agent = result.scalars().first()

    if agent:
        return agent

    # Check if any agent exists for this user
    stmt_any = (
        select(Agent)
        .where(Agent.user_id == user_id)
        .order_by(Agent.created_at.asc())
        .limit(1)
    )
    res_any = await session.execute(stmt_any)
    existing_agent = res_any.scalars().first()

    if existing_agent:
        existing_agent.is_default = True
        await session.commit()
        await session.refresh(existing_agent)
        logger.info("Marked existing agent '%s' as default for user '%s'", existing_agent.id, user_id)
        return existing_agent

    # Auto-seed initial default agent for user
    default_config = AgentConfig()
    new_agent = Agent(
        id=uuid.uuid4(),
        user_id=user_id,
        name="Default Assistant",
        description="Standard conversational assistant with persistent memory and hybrid search.",
        config=default_config.model_dump(),
        is_default=True,
        is_published=True,
        version=1,
    )
    session.add(new_agent)
    await session.commit()
    await session.refresh(new_agent)
    logger.info("Auto-seeded default agent '%s' for user '%s'", new_agent.id, user_id)
    return new_agent


async def create_agent(session: AsyncSession, user_id: str, agent_in: AgentCreate) -> Agent:
    """Creates a new agent record with validated configuration."""
    await get_or_create_user(session, user_id=user_id)
    validated_config = validate_agent_config(agent_in.config)

    # If this new agent is marked as default, unset any prior default for this user
    if agent_in.is_default:
        await session.execute(
            update(Agent)
            .where(Agent.user_id == user_id, Agent.is_default.is_(True))
            .values(is_default=False)
        )

    # If user has no existing agents, make this first agent default regardless
    count_stmt = select(func.count()).select_from(Agent).where(Agent.user_id == user_id)
    total_existing = await session.scalar(count_stmt) or 0
    is_default = agent_in.is_default or (total_existing == 0)

    agent = Agent(
        id=uuid.uuid4(),
        user_id=user_id,
        name=agent_in.name,
        description=agent_in.description,
        config=validated_config.model_dump(),
        is_default=is_default,
        is_published=agent_in.is_published,
        version=1,
    )
    session.add(agent)
    await session.commit()
    await session.refresh(agent)
    logger.info("Created agent '%s' for user '%s' (is_default=%s)", agent.id, user_id, agent.is_default)
    return agent


async def get_agent_by_id(
    session: AsyncSession,
    agent_id: uuid.UUID | str,
    user_id: str,
    is_admin: bool = False,
) -> Agent:
    """Fetches an agent by ID with multi-tenant ownership verification."""
    parsed_id = agent_id if isinstance(agent_id, uuid.UUID) else uuid.UUID(str(agent_id))
    stmt = select(Agent).where(Agent.id == parsed_id)
    result = await session.execute(stmt)
    agent = result.scalars().first()

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_id}' not found.",
        )

    if agent.user_id != user_id and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this agent.",
        )

    return agent


async def list_agents(
    session: AsyncSession,
    user_id: str,
    is_admin: bool = False,
    page: int = 1,
    page_size: int = 20,
    search: str | None = None,
    is_default: bool | None = None,
) -> tuple[list[Agent], int]:
    """Lists agents belonging to the user with search and pagination."""
    base_query = select(Agent)
    if not is_admin:
        base_query = base_query.where(Agent.user_id == user_id)

    if search:
        search_pattern = f"%{search}%"
        base_query = base_query.where(
            (Agent.name.ilike(search_pattern)) | (Agent.description.ilike(search_pattern))
        )

    if is_default is not None:
        base_query = base_query.where(Agent.is_default.is_(is_default))

    # Total count query
    count_stmt = select(func.count()).select_from(base_query.subquery())
    total = await session.scalar(count_stmt) or 0

    # Paginated results ordered by is_default DESC, updated_at DESC
    items_stmt = (
        base_query
        .order_by(Agent.is_default.desc(), Agent.updated_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    result = await session.execute(items_stmt)
    items = list(result.scalars().all())

    return items, total


async def update_agent(
    session: AsyncSession,
    agent_id: uuid.UUID | str,
    user_id: str,
    agent_in: AgentUpdate,
    is_admin: bool = False,
) -> Agent:
    """Updates an agent's metadata or configuration, incrementing its version."""
    agent = await get_agent_by_id(session, agent_id, user_id, is_admin=is_admin)

    if agent_in.name is not None:
        agent.name = agent_in.name
    if agent_in.description is not None:
        agent.description = agent_in.description
    if agent_in.is_published is not None:
        agent.is_published = agent_in.is_published

    if agent_in.is_default is True and not agent.is_default:
        # Unset any prior default
        await session.execute(
            update(Agent)
            .where(Agent.user_id == agent.user_id, Agent.is_default.is_(True))
            .values(is_default=False)
        )
        agent.is_default = True
    elif agent_in.is_default is False and agent.is_default:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot unset default on this agent directly. Designate another agent as default instead.",
        )

    if agent_in.config is not None:
        validated_cfg = validate_agent_config(agent_in.config)
        agent.config = validated_cfg.model_dump()
        agent.version += 1
        invalidate_agent_cache(str(agent.id))

    await session.commit()
    await session.refresh(agent)
    logger.info("Updated agent '%s' (version=%d)", agent.id, agent.version)
    return agent


async def delete_agent(
    session: AsyncSession,
    agent_id: uuid.UUID | str,
    user_id: str,
    is_admin: bool = False,
) -> None:
    """Deletes an agent record after verifying ownership and default constraints."""
    agent = await get_agent_by_id(session, agent_id, user_id, is_admin=is_admin)

    if agent.is_default:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete the default agent. Please promote another agent as default first.",
        )

    invalidate_agent_cache(str(agent.id))
    await session.delete(agent)
    await session.commit()
    logger.info("Deleted agent '%s'", agent_id)


async def set_default_agent(
    session: AsyncSession,
    agent_id: uuid.UUID | str,
    user_id: str,
    is_admin: bool = False,
) -> Agent:
    """Sets the given agent as the user's primary default agent."""
    agent = await get_agent_by_id(session, agent_id, user_id, is_admin=is_admin)

    if agent.is_default:
        return agent

    # Atomically unset previous default and set new default
    await session.execute(
        update(Agent)
        .where(Agent.user_id == agent.user_id, Agent.is_default.is_(True))
        .values(is_default=False)
    )
    agent.is_default = True
    await session.commit()
    await session.refresh(agent)
    logger.info("Promoted agent '%s' to default for user '%s'", agent.id, agent.user_id)
    return agent


async def clone_agent(
    session: AsyncSession,
    agent_id: uuid.UUID | str,
    user_id: str,
    clone_in: AgentCloneRequest,
    is_admin: bool = False,
) -> Agent:
    """Clones an existing agent configuration into a new agent."""
    await get_or_create_user(session, user_id=user_id)
    source_agent = await get_agent_by_id(session, agent_id, user_id, is_admin=is_admin)

    new_name = clone_in.name or f"{source_agent.name} (Copy)"
    new_desc = clone_in.description if clone_in.description is not None else source_agent.description

    cloned_agent = Agent(
        id=uuid.uuid4(),
        user_id=user_id,  # clone belongs to the calling user
        name=new_name,
        description=new_desc,
        config=dict(source_agent.config),
        is_default=False,
        is_published=False,
        version=1,
    )
    session.add(cloned_agent)
    await session.commit()
    await session.refresh(cloned_agent)
    logger.info("Cloned agent '%s' -> new agent '%s'", source_agent.id, cloned_agent.id)
    return cloned_agent


async def resolve_agent_for_chat(
    session: AsyncSession,
    user_id: str,
    agent_id: str | uuid.UUID | None = None,
    is_admin: bool = False,
) -> tuple[Agent, AgentConfig]:
    """Resolves the target Agent and its validated AgentConfig for an incoming conversation turn.
    
    If agent_id is provided, loads the specific agent.
    If agent_id is None, automatically resolves or provisions the user's default agent.
    Utilizes LRU caching to eliminate database queries on repeated turns.
    """
    if agent_id:
        agent = await get_agent_by_id(session, agent_id, user_id, is_admin=is_admin)
    else:
        agent = await get_or_create_default_agent(session, user_id)

    # Check in-memory cache
    cached_cfg = get_cached_agent_config(str(agent.id), agent.version)
    if cached_cfg:
        return agent, cached_cfg

    validated_cfg = validate_agent_config(agent.config)
    set_cached_agent_config(str(agent.id), agent.version, validated_cfg)
    return agent, validated_cfg


def resolve_agent_tools(agent_config: AgentConfig, user_id: str) -> list[Any]:
    """Resolves and instantiates LangChain tools attached to an agent, bound to *user_id*.

    Supports:
    - RAG research tools: 'hybrid_document_search', 'graph_rag_search', 'multi_query_search', 'deep_research_search'
    - Macro selector: 'all_rag_tools' (attaches full exploitation & exploration suite)
    """
    if not agent_config.tools:
        return []

    from app.services.rag_tools import get_user_rag_tool_map, get_user_rag_tools

    rag_tool_map = get_user_rag_tool_map(user_id=user_id)
    resolved_tools: list[Any] = []
    seen_names: set[str] = set()

    for tool_spec in agent_config.tools:
        if not tool_spec.enabled:
            continue

        tool_id = tool_spec.tool_id.lower().strip()
        if tool_id in ("all_rag_tools", "all_rag", "rag_suite"):
            for t in get_user_rag_tools(user_id=user_id):
                if t.name not in seen_names:
                    resolved_tools.append(t)
                    seen_names.add(t.name)
            continue

        if tool_id in rag_tool_map:
            t = rag_tool_map[tool_id]
            if t.name not in seen_names:
                resolved_tools.append(t)
                seen_names.add(t.name)

    return resolved_tools

