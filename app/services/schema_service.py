"""Structured schema service: validation, storage, and retrieval of JSON schemas."""

from __future__ import annotations

import logging
from typing import Any
import uuid

from fastapi import HTTPException, status
from jsonschema import Draft202012Validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.structured_schema import StructuredSchema
from app.schemas.structured_schema import StructuredSchemaCreate, StructuredSchemaUpdate
from app.services.thread_service import get_or_create_user

logger = logging.getLogger(__name__)


def validate_json_schema(schema_dict: dict[str, Any]) -> None:
    """Validates that a dictionary is a syntactically valid JSON Schema specification."""
    if not isinstance(schema_dict, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="JSON Schema definition must be a valid JSON object.",
        )

    try:
        Draft202012Validator.check_schema(schema_dict)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid JSON Schema: {exc}",
        ) from exc


async def create_structured_schema(
    session: AsyncSession,
    user_id: str,
    schema_in: StructuredSchemaCreate,
) -> StructuredSchema:
    """Creates and stores a new structured schema definition."""
    await get_or_create_user(session, user_id=user_id)
    validate_json_schema(schema_in.schema_definition)

    new_schema = StructuredSchema(
        id=uuid.uuid4(),
        user_id=user_id,
        name=schema_in.name,
        description=schema_in.description,
        schema_definition=schema_in.schema_definition,
        version=1,
    )
    session.add(new_schema)
    await session.commit()
    await session.refresh(new_schema)
    logger.info("Created structured schema '%s' for user '%s'", new_schema.id, user_id)
    return new_schema


async def get_structured_schema_by_id(
    session: AsyncSession,
    schema_id: uuid.UUID | str,
    user_id: str,
    is_admin: bool = False,
) -> StructuredSchema:
    """Retrieves a structured schema by ID with tenant access control."""
    parsed_id = schema_id if isinstance(schema_id, uuid.UUID) else uuid.UUID(str(schema_id))
    stmt = select(StructuredSchema).where(StructuredSchema.id == parsed_id)
    result = await session.execute(stmt)
    schema_obj = result.scalars().first()

    if not schema_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Structured schema '{schema_id}' not found.",
        )

    if schema_obj.user_id != user_id and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this structured schema.",
        )

    return schema_obj


async def list_structured_schemas(
    session: AsyncSession,
    user_id: str,
    is_admin: bool = False,
    page: int = 1,
    page_size: int = 20,
    search: str | None = None,
) -> tuple[list[StructuredSchema], int]:
    """Lists structured schemas belonging to the user with search and pagination."""
    base_query = select(StructuredSchema)
    if not is_admin:
        base_query = base_query.where(StructuredSchema.user_id == user_id)

    if search:
        search_pattern = f"%{search}%"
        base_query = base_query.where(
            (StructuredSchema.name.ilike(search_pattern))
            | (StructuredSchema.description.ilike(search_pattern))
        )

    count_stmt = select(func.count()).select_from(base_query.subquery())
    total = await session.scalar(count_stmt) or 0

    items_stmt = (
        base_query
        .order_by(StructuredSchema.updated_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    result = await session.execute(items_stmt)
    items = list(result.scalars().all())

    return items, total


async def update_structured_schema(
    session: AsyncSession,
    schema_id: uuid.UUID | str,
    user_id: str,
    schema_in: StructuredSchemaUpdate,
    is_admin: bool = False,
) -> StructuredSchema:
    """Updates a structured schema's name, description, or JSON Schema definition."""
    schema_obj = await get_structured_schema_by_id(session, schema_id, user_id, is_admin=is_admin)

    if schema_in.name is not None:
        schema_obj.name = schema_in.name
    if schema_in.description is not None:
        schema_obj.description = schema_in.description
    if schema_in.schema_definition is not None:
        validate_json_schema(schema_in.schema_definition)
        schema_obj.schema_definition = schema_in.schema_definition
        schema_obj.version += 1

    await session.commit()
    await session.refresh(schema_obj)
    logger.info("Updated structured schema '%s' (version=%d)", schema_obj.id, schema_obj.version)
    return schema_obj


async def delete_structured_schema(
    session: AsyncSession,
    schema_id: uuid.UUID | str,
    user_id: str,
    is_admin: bool = False,
) -> None:
    """Deletes a structured schema definition."""
    schema_obj = await get_structured_schema_by_id(session, schema_id, user_id, is_admin=is_admin)
    await session.delete(schema_obj)
    await session.commit()
    logger.info("Deleted structured schema '%s'", schema_id)
