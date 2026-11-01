"""API endpoints for managing reusable JSON Schemas for structured LLM output enforcement."""

from __future__ import annotations

import logging
import uuid
from fastapi import APIRouter, Depends, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import UserContext, get_current_user
from app.core.database import get_db
from app.core.rate_limit import limiter
from app.schemas.structured_schema import (
    StructuredSchemaCreate,
    StructuredSchemaListResponse,
    StructuredSchemaResponse,
    StructuredSchemaUpdate,
)
from app.services.schema_service import (
    create_structured_schema,
    delete_structured_schema,
    get_structured_schema_by_id,
    list_structured_schemas,
    update_structured_schema,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/structured-schemas", tags=["Structured Schemas"])


@router.post(
    "",
    response_model=StructuredSchemaResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a structured schema",
    description="Registers a reusable JSON Schema definition for enforcing structured output from LLMs.",
)
@limiter.limit("20/minute")
async def create_schema_endpoint(
    request: Request,
    schema_in: StructuredSchemaCreate,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StructuredSchemaResponse:
    """Creates a new structured schema definition."""
    schema = await create_structured_schema(db, user_id=current_user.user_id, schema_in=schema_in)
    return StructuredSchemaResponse.model_validate(schema)


@router.get(
    "",
    response_model=StructuredSchemaListResponse,
    summary="List structured schemas",
    description="Lists all structured schemas defined by the authenticated user.",
)
async def list_schemas_endpoint(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    search: str | None = Query(None, description="Search filter for schema name or description"),
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StructuredSchemaListResponse:
    """Lists structured schemas for the user."""
    items, total = await list_structured_schemas(
        db,
        user_id=current_user.user_id,
        is_admin=current_user.is_admin,
        page=page,
        page_size=page_size,
        search=search,
    )
    return StructuredSchemaListResponse(items=[StructuredSchemaResponse.model_validate(item) for item in items], total=total, page=page, page_size=page_size)


@router.get(
    "/{schema_id}",
    response_model=StructuredSchemaResponse,
    summary="Get structured schema",
    description="Retrieves a specific structured schema definition.",
)
async def get_schema_endpoint(
    request: Request,
    schema_id: uuid.UUID,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StructuredSchemaResponse:
    """Retrieves a schema by ID."""
    schema = await get_structured_schema_by_id(
        db,
        schema_id=schema_id,
        user_id=current_user.user_id,
        is_admin=current_user.is_admin,
    )
    return StructuredSchemaResponse.model_validate(schema)


@router.put(
    "/{schema_id}",
    response_model=StructuredSchemaResponse,
    summary="Update structured schema",
    description="Updates a structured schema's name, description, or JSON Schema definition.",
)
@limiter.limit("20/minute")
async def update_schema_endpoint(
    request: Request,
    schema_id: uuid.UUID,
    schema_in: StructuredSchemaUpdate,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StructuredSchemaResponse:
    """Updates a schema and increments its version."""
    schema = await update_structured_schema(
        db,
        schema_id=schema_id,
        user_id=current_user.user_id,
        schema_in=schema_in,
        is_admin=current_user.is_admin,
    )
    return StructuredSchemaResponse.model_validate(schema)


@router.delete(
    "/{schema_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete structured schema",
    description="Deletes a structured schema definition.",
)
async def delete_schema_endpoint(
    request: Request,
    schema_id: uuid.UUID,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Deletes a structured schema."""
    await delete_structured_schema(
        db,
        schema_id=schema_id,
        user_id=current_user.user_id,
        is_admin=current_user.is_admin,
    )
