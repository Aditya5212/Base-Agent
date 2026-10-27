from __future__ import annotations

from datetime import datetime
from typing import Any
import uuid

from pydantic import BaseModel, ConfigDict, Field


class StructuredSchemaCreate(BaseModel):
    """Request payload for registering a new structured JSON schema."""

    name: str = Field(..., min_length=1, max_length=255, description="Human-readable schema name.")
    description: str | None = Field(None, max_length=1000, description="Optional description.")
    schema_definition: dict[str, Any] = Field(
        ...,
        description="Standard JSON Schema specification (e.g. {'type': 'object', 'properties': {...}}).",
    )


class StructuredSchemaUpdate(BaseModel):
    """Request payload for updating an existing structured schema."""

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = Field(None, max_length=1000)
    schema_definition: dict[str, Any] | None = None


class StructuredSchemaResponse(BaseModel):
    """Detailed response model for a structured schema."""

    id: uuid.UUID
    user_id: str
    name: str
    description: str | None
    schema_definition: dict[str, Any]
    version: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class StructuredSchemaListResponse(BaseModel):
    """Paginated list of structured schemas."""

    items: list[StructuredSchemaResponse]
    total: int
    page: int
    page_size: int
