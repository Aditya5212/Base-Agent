"""Pydantic schemas and contracts for User management."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
import re
from pydantic import BaseModel, ConfigDict, Field, field_validator


USER_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]{2,128}$")


class UserBase(BaseModel):
    """Base user properties."""

    email: str = Field(..., min_length=3, max_length=255, description="Unique user email address.")
    name: str | None = Field(default=None, max_length=255, description="Full or display name.")
    role: Literal["user", "admin", "root"] = Field(default="user", description="Authorization role.")
    is_active: bool = Field(default=True, description="Whether the user account is active.")
    metadata_json: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary user metadata or preferences.",
    )


class UserCreate(BaseModel):
    """Payload to register or create a new user account."""

    id: str | None = Field(
        default=None,
        min_length=2,
        max_length=128,
        description="Optional custom identifier (e.g. 'alice' or UUID). If omitted, an ID will be generated.",
    )
    email: str = Field(..., min_length=3, max_length=255, description="Unique user email address.")
    name: str | None = Field(default=None, max_length=255, description="User display name.")
    role: Literal["user", "admin", "root"] = Field(default="user", description="User role.")
    is_active: bool = Field(default=True, description="Whether the account is active immediately.")
    metadata_json: dict[str, Any] = Field(default_factory=dict, description="Custom metadata.")

    @field_validator("id")
    @classmethod
    def validate_user_id(cls, v: str | None) -> str | None:
        if v is not None:
            if not USER_ID_REGEX.match(v):
                raise ValueError(
                    "User ID must be 2-128 characters and contain only alphanumeric, hyphen, underscore, or period characters."
                )
        return v


class UserUpdate(BaseModel):
    """Payload to update an existing user account."""

    email: str | None = Field(default=None, min_length=3, max_length=255)
    name: str | None = Field(default=None, max_length=255)
    role: Literal["user", "admin", "root"] | None = Field(default=None)
    is_active: bool | None = Field(default=None)
    metadata_json: dict[str, Any] | None = Field(default=None)


class UserStats(BaseModel):
    """Summary of data entities owned by the user."""

    agents_count: int = Field(default=0, description="Total agents created by the user.")
    workflows_count: int = Field(default=0, description="Total workflows created by the user.")
    documents_count: int = Field(default=0, description="Total documents ingested for the user.")
    threads_count: int = Field(default=0, description="Total conversation threads.")
    tools_count: int = Field(default=0, description="Total tools registered by the user.")
    schemas_count: int = Field(default=0, description="Total structured schemas created by the user.")


class UserResponse(BaseModel):
    """Public representation of a user account."""

    id: str
    email: str
    name: str | None
    role: str
    is_active: bool
    metadata_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    stats: UserStats | None = Field(default=None, description="Optional entity counts.")

    model_config = ConfigDict(from_attributes=True)


class UserListResponse(BaseModel):
    """Paginated list of user accounts."""

    items: list[UserResponse]
    total: int
    page: int
    page_size: int


class UserDeleteResponse(BaseModel):
    """Response confirming cascade deletion of a user and all associated data."""

    id: str
    message: str
    deleted_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown of deleted associated resources.",
    )

