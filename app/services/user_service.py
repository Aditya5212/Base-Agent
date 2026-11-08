"""User service: business logic, lifecycle, cascade deletion, and statistics for User accounts."""

from __future__ import annotations

import logging
from typing import Any
import uuid

from fastapi import HTTPException, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent import Agent
from app.models.document import Document
from app.models.structured_schema import StructuredSchema
from app.models.thread import Thread, ThreadAsset
from app.models.tool import Tool
from app.models.user import User
from app.models.workflow import Workflow, WorkflowRun
from app.schemas.user import UserCreate, UserStats, UserUpdate
from app.services.agent_service import get_or_create_default_agent, invalidate_agent_cache
from app.services.memory import delete_thread_memory, get_checkpointer
from app.services.r2_storage import r2_storage_service
from app.services.vector_store import delete_documents_by_metadata

logger = logging.getLogger(__name__)


async def get_user_stats(session: AsyncSession, user_id: str) -> UserStats:
    """Computes entity counts owned by a specific user."""
    agents_count = await session.scalar(
        select(func.count(Agent.id)).where(Agent.user_id == user_id)
    ) or 0
    workflows_count = await session.scalar(
        select(func.count(Workflow.id)).where(Workflow.user_id == user_id)
    ) or 0
    documents_count = await session.scalar(
        select(func.count(Document.id)).where(Document.user_id == user_id)
    ) or 0
    threads_count = await session.scalar(
        select(func.count(Thread.id)).where(Thread.user_id == user_id)
    ) or 0
    tools_count = await session.scalar(
        select(func.count(Tool.id)).where(Tool.user_id == user_id)
    ) or 0
    schemas_count = await session.scalar(
        select(func.count(StructuredSchema.id)).where(StructuredSchema.user_id == user_id)
    ) or 0

    return UserStats(
        agents_count=agents_count,
        workflows_count=workflows_count,
        documents_count=documents_count,
        threads_count=threads_count,
        tools_count=tools_count,
        schemas_count=schemas_count,
    )


async def create_user(session: AsyncSession, user_in: UserCreate) -> User:
    """Creates a new user and auto-seeds their default assistant agent."""
    user_id = user_in.id or f"usr_{uuid.uuid4().hex[:12]}"

    # Validate ID uniqueness
    existing_by_id = await session.get(User, user_id)
    if existing_by_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with identifier '{user_id}' already exists.",
        )

    # Validate email uniqueness
    existing_by_email = await session.scalar(
        select(User).where(User.email == user_in.email)
    )
    if existing_by_email:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with email '{user_in.email}' already exists.",
        )

    new_user = User(
        id=user_id,
        email=user_in.email,
        name=user_in.name,
        role=user_in.role,
        is_active=user_in.is_active,
        metadata_json=user_in.metadata_json,
    )
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)
    logger.info("Created user '%s' (%s) with role '%s'", new_user.id, new_user.email, new_user.role)

    # Provision standard default assistant for the new user
    await get_or_create_default_agent(session, user_id=new_user.id)

    return new_user


async def get_user_by_id(
    session: AsyncSession,
    user_id: str,
    include_stats: bool = False,
) -> tuple[User, UserStats | None]:
    """Retrieves a user by ID, optionally calculating entity ownership stats."""
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found.",
        )

    stats = await get_user_stats(session, user_id) if include_stats else None
    return user, stats


async def list_users(
    session: AsyncSession,
    page: int = 1,
    page_size: int = 20,
    search: str | None = None,
    role: str | None = None,
    is_active: bool | None = None,
    include_stats: bool = False,
) -> tuple[list[tuple[User, UserStats | None]], int]:
    """Lists users with optional filtering, search, pagination, and ownership metrics."""
    filters = []
    if search:
        pattern = f"%{search.strip()}%"
        filters.append(
            or_(
                User.id.ilike(pattern),
                User.email.ilike(pattern),
                User.name.ilike(pattern),
            )
        )
    if role:
        filters.append(User.role == role)
    if is_active is not None:
        filters.append(User.is_active == is_active)

    total = await session.scalar(
        select(func.count(User.id)).where(*filters)
    ) or 0

    stmt = (
        select(User)
        .where(*filters)
        .order_by(User.created_at.asc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    users = (await session.execute(stmt)).scalars().all()

    items: list[tuple[User, UserStats | None]] = []
    for u in users:
        stats = await get_user_stats(session, u.id) if include_stats else None
        items.append((u, stats))

    return items, total


async def update_user(
    session: AsyncSession,
    user_id: str,
    user_update: UserUpdate,
    is_admin: bool = False,
) -> User:
    """Updates user profile attributes, with strict invariants on root and admin roles."""
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found.",
        )

    # Invariants for root account
    if user_id == "root":
        if user_update.role is not None and user_update.role != "root":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The root user role cannot be modified.",
            )
        if user_update.is_active is False:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The root user cannot be deactivated.",
            )

    # Role change permission check
    if user_update.role is not None and user_update.role != user.role:
        if not is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can modify user roles.",
            )
        user.role = user_update.role

    # Email uniqueness check
    if user_update.email is not None and user_update.email != user.email:
        conflict = await session.scalar(
            select(User).where(User.email == user_update.email, User.id != user_id)
        )
        if conflict:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Email '{user_update.email}' is already in use by another user.",
            )
        user.email = user_update.email

    if user_update.name is not None:
        user.name = user_update.name
    if user_update.is_active is not None:
        user.is_active = user_update.is_active
    if user_update.metadata_json is not None:
        # Shallow merge or replace metadata
        merged_meta = dict(user.metadata_json)
        merged_meta.update(user_update.metadata_json)
        user.metadata_json = merged_meta

    await session.commit()
    await session.refresh(user)
    logger.info("Updated user '%s'", user.id)
    return user


async def delete_user(session: AsyncSession, user_id: str) -> dict[str, int]:
    """Deletes a user and cascades across database, R2 storage, vector store, and memory checkpointers.

    Returns:
        dict[str, int]: Summary count of deleted resources.
    """
    if user_id == "root":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The root administrator user cannot be deleted.",
        )

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found.",
        )

    # Collect entities and counts before deletion
    docs = (await session.execute(
        select(Document).where(Document.user_id == user_id)
    )).scalars().all()

    threads = (await session.execute(
        select(Thread).where(Thread.user_id == user_id)
    )).scalars().all()

    assets = (await session.execute(
        select(ThreadAsset).where(ThreadAsset.user_id == user_id)
    )).scalars().all()

    agents = (await session.execute(
        select(Agent).where(Agent.user_id == user_id)
    )).scalars().all()

    workflows_count = await session.scalar(
        select(func.count(Workflow.id)).where(Workflow.user_id == user_id)
    ) or 0

    runs = (await session.execute(
        select(WorkflowRun).where(WorkflowRun.user_id == user_id)
    )).scalars().all()

    tools_count = await session.scalar(
        select(func.count(Tool.id)).where(Tool.user_id == user_id)
    ) or 0

    schemas_count = await session.scalar(
        select(func.count(StructuredSchema.id)).where(StructuredSchema.user_id == user_id)
    ) or 0

    # 1. Purge R2 Storage objects for documents
    for doc in docs:
        if doc.r2_key:
            try:
                await r2_storage_service.delete_file(doc.r2_key)
            except Exception as exc:
                logger.warning("Failed to delete R2 object '%s' for doc '%s': %s", doc.r2_key, doc.id, exc)

    # 2. Purge R2 Storage objects for thread media assets
    for asset in assets:
        if asset.r2_key:
            try:
                await r2_storage_service.delete_file(asset.r2_key)
            except Exception as exc:
                logger.warning("Failed to delete R2 asset '%s' for thread '%s': %s", asset.r2_key, asset.thread_id, exc)

    # 3. Purge PGVector chunks
    try:
        await delete_documents_by_metadata("user_id", user_id)
    except Exception as exc:
        logger.warning("Failed to purge vector chunks for user '%s': %s", user_id, exc)

    # 4. Purge LightRAG knowledge graph documents (best-effort)
    try:
        from app.core.rag.factory import get_provider
        from app.core.rag.workspace import user_workspace
        lightrag_provider = await get_provider("lightrag", user_workspace(user_id))
        for doc in docs:
            await lightrag_provider.delete_document(str(doc.id))
    except Exception as exc:
        logger.debug("LightRAG cleanup skipped for user '%s': %s", user_id, exc)

    # 5. Purge LangGraph checkpointer memory for user threads
    for thread in threads:
        try:
            await delete_thread_memory(thread.id)
        except Exception as exc:
            logger.warning("Failed to delete thread memory for '%s': %s", thread.id, exc)

    # 6. Purge LangGraph checkpointer memory for workflow runs
    try:
        checkpointer = await get_checkpointer()
        for run in runs:
            if run.thread_id:
                try:
                    await checkpointer.adelete_thread(run.thread_id)
                except Exception as exc:
                    logger.debug("Checkpointer thread delete for run '%s' skipped: %s", run.thread_id, exc)
    except Exception as exc:
        logger.debug("Workflow run checkpointer cleanup skipped: %s", exc)

    # 7. Invalidate in-memory LRU agent configs
    for agent in agents:
        invalidate_agent_cache(str(agent.id))

    # 8. Delete user record from PostgreSQL (cascades to all user tables)
    await session.delete(user)
    await session.commit()
    logger.info("Successfully cascade-deleted user '%s' and all associated resources", user_id)

    return {
        "agents": len(agents),
        "documents": len(docs),
        "threads": len(threads),
        "thread_assets": len(assets),
        "workflows": workflows_count,
        "workflow_runs": len(runs),
        "tools": tools_count,
        "schemas": schemas_count,
    }

