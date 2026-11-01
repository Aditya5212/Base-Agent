"""Business logic service for managing custom tool lifecycle, encryption, and execution.

Provides tenant-scoped CRUD operations, secret encryption/decryption,
cache invalidation, and direct execution testing.
"""

from __future__ import annotations

import logging
import time
from typing import Any
import uuid

from fastapi import HTTPException, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.crypto import encrypt_secret
from app.models.tool import Tool
from app.schemas.tool import (
    HttpToolConfig,
    ToolCreate,
    ToolExecutionResponse,
    ToolType,
    ToolUpdate,
)
from app.services.thread_service import get_or_create_user
from app.services.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


def _encrypt_sensitive_config(tool_type: str, config: dict[str, Any]) -> dict[str, Any]:
    """Encrypts credentials (e.g. auth_token, connection_string) in tool config before DB storage."""
    processed = dict(config or {})

    if tool_type == ToolType.API.value:
        raw_token = processed.get("auth_token")
        if raw_token and not str(raw_token).startswith("gAAAAA"):  # standard Fernet prefix
            encrypted = encrypt_secret(raw_token)
            if encrypted:
                processed["auth_token"] = encrypted

    elif tool_type == ToolType.DATABASE.value:
        raw_conn = processed.get("connection_string")
        if raw_conn and not str(raw_conn).startswith("gAAAAA"):
            encrypted = encrypt_secret(raw_conn)
            if encrypted:
                processed["connection_string"] = encrypted

    return processed


class ToolService:
    """Service managing tool lifecycle, persistence, and execution."""

    @staticmethod
    async def create_tool(db: AsyncSession, user_id: str, tool_in: ToolCreate) -> Tool:
        """Creates and stores a new tool definition for the specified user."""
        await get_or_create_user(db, user_id=user_id)

        # 1. Check for duplicate tool name under the same user
        existing = await ToolService.get_tool_by_name(db, user_id=user_id, name=tool_in.name)
        if existing and existing.user_id == user_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A tool with name '{tool_in.name}' already exists in your workspace.",
            )

        # 2. Encrypt sensitive secrets in config
        encrypted_config = _encrypt_sensitive_config(tool_in.type.value, tool_in.config)

        tool = Tool(
            id=uuid.uuid4(),
            user_id=user_id,
            name=tool_in.name,
            description=tool_in.description,
            type=tool_in.type.value,
            input_schema=tool_in.input_schema,
            output_schema=tool_in.output_schema,
            config=encrypted_config,
            is_system=tool_in.is_system,
            version=1,
        )

        # A tool must be executable before it becomes a persisted usable definition.
        try:
            ToolRegistry.compile_tool(tool, cache=False)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid tool: {exc}") from exc

        db.add(tool)
        await db.commit()
        await db.refresh(tool)

        return tool

    @staticmethod
    async def get_tool(db: AsyncSession, user_id: str, tool_id: uuid.UUID) -> Tool | None:
        """Retrieves a tool by ID if owned by user or marked as a system tool."""
        stmt = select(Tool).where(
            Tool.id == tool_id,
            or_(Tool.user_id == user_id, Tool.is_system.is_(True)),
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_tool_by_name(db: AsyncSession, user_id: str, name: str) -> Tool | None:
        """Retrieves a tool by name if owned by user or marked as a system tool."""
        stmt = select(Tool).where(
            Tool.name == name,
            or_(Tool.user_id == user_id, Tool.is_system.is_(True)),
        ).order_by(Tool.is_system.asc())  # user tool takes precedence over system tool
        result = await db.execute(stmt)
        return result.scalars().first()

    @staticmethod
    async def list_tools(
        db: AsyncSession,
        user_id: str,
        skip: int = 0,
        limit: int = 50,
        tool_type: str | None = None,
        search: str | None = None,
    ) -> tuple[int, list[Tool]]:
        """Lists tools accessible to the user with optional type and search filtering."""
        query = select(Tool).where(or_(Tool.user_id == user_id, Tool.is_system.is_(True)))

        if tool_type:
            query = query.where(Tool.type == tool_type)

        if search and search.strip():
            term = f"%{search.strip()}%"
            query = query.where(
                or_(Tool.name.ilike(term), Tool.description.ilike(term))
            )

        # Count total
        count_stmt = select(func.count()).select_from(query.subquery())
        total_res = await db.execute(count_stmt)
        total = total_res.scalar() or 0

        # Fetch page
        query = query.order_by(Tool.created_at.desc()).offset(skip).limit(limit)
        items_res = await db.execute(query)
        items = list(items_res.scalars().all())

        return total, items

    @staticmethod
    async def update_tool(
        db: AsyncSession,
        user_id: str,
        tool_id: uuid.UUID,
        tool_in: ToolUpdate,
    ) -> Tool:
        """Updates an existing tool definition and increments its version."""
        tool = await ToolService.get_tool(db, user_id=user_id, tool_id=tool_id)
        if not tool:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool with ID '{tool_id}' not found.",
            )

        if tool.is_system and tool.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="System tools cannot be modified.",
            )

        if tool_in.name is not None and tool_in.name != tool.name:
            # Check for conflict
            conflict = await ToolService.get_tool_by_name(db, user_id=user_id, name=tool_in.name)
            if conflict and conflict.id != tool.id:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"A tool with name '{tool_in.name}' already exists in your workspace.",
                )
            tool.name = tool_in.name

        if tool_in.description is not None:
            tool.description = tool_in.description

        if tool_in.input_schema is not None:
            tool.input_schema = tool_in.input_schema

        if tool_in.output_schema is not None:
            tool.output_schema = tool_in.output_schema

        if tool_in.is_system is not None:
            tool.is_system = tool_in.is_system

        if tool_in.config is not None:
            tool.config = _encrypt_sensitive_config(tool.type, tool_in.config)

        tool.version += 1
        try:
            ToolRegistry.compile_tool(tool, cache=False)
        except Exception as exc:
            await db.rollback()
            raise HTTPException(status_code=422, detail=f"Invalid tool update: {exc}") from exc
        await db.commit()
        await db.refresh(tool)

        # Invalidate old cache entries
        ToolRegistry.invalidate_tool(user_id, str(tool_id))

        return tool

    @staticmethod
    async def delete_tool(db: AsyncSession, user_id: str, tool_id: uuid.UUID) -> bool:
        """Deletes a tool owned by the user."""
        tool = await ToolService.get_tool(db, user_id=user_id, tool_id=tool_id)
        if not tool:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool with ID '{tool_id}' not found.",
            )

        if tool.is_system and tool.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="System tools cannot be deleted.",
            )

        ToolRegistry.invalidate_tool(user_id, str(tool_id))
        await db.delete(tool)
        await db.commit()
        return True

    @staticmethod
    async def execute_tool(
        db: AsyncSession,
        user_id: str,
        tool_id: uuid.UUID,
        arguments: dict[str, Any],
    ) -> ToolExecutionResponse:
        """Directly executes a tool for testing or manual execution with performance metrics."""
        tool = await ToolService.get_tool(db, user_id=user_id, tool_id=tool_id)
        if not tool:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool with ID '{tool_id}' not found.",
            )

        start_time = time.perf_counter()
        success = True
        output: Any = None
        error_msg: str | None = None

        try:
            compiled_tool = ToolRegistry.compile_tool(tool)
            output = await compiled_tool.ainvoke(arguments or {})
        except Exception as exc:
            logger.error("Execution of tool '%s' failed: %s", tool.name, exc, exc_info=True)
            success = False
            error_msg = str(exc)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return ToolExecutionResponse(
            tool_id=str(tool.id),
            tool_name=tool.name,
            success=success,
            output=output,
            error=error_msg,
            execution_time_ms=duration_ms,
        )

