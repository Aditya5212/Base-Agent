"""Pydantic contracts and schemas for the Tool Builder platform.

Defines schemas for:
- Tool execution types: API (HTTP REST), Code (Sandboxed Python), Database (Read-Only SQL), and Built-In
- Executor-specific configuration models
- Tool CRUD operations and validation
- Direct tool execution and testing payloads
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import re
from typing import Any, Literal
import uuid

import jsonschema
from pydantic import BaseModel, Field, field_validator, model_validator


class ToolType(str, Enum):
    """Supported tool execution backends."""

    API = "api"
    CODE = "code"
    DATABASE = "database"
    BUILT_IN = "built_in"


class HttpAuthType(str, Enum):
    """Authentication strategies for HTTP API tools."""

    NONE = "none"
    BEARER = "bearer"
    API_KEY = "api_key"
    BASIC = "basic"


class HttpToolConfig(BaseModel):
    """Configuration for HTTP REST API tool execution."""

    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = Field(
        default="GET",
        description="HTTP method for calling the remote API endpoint",
    )
    url_template: str = Field(
        ...,
        min_length=1,
        description="URL template supporting parameter interpolation (e.g. 'https://api.example.com/items/{item_id}')",
    )
    headers_template: dict[str, str] = Field(
        default_factory=dict,
        description="HTTP headers dictionary supporting template interpolation",
    )
    body_template: dict[str, Any] | str | None = Field(
        default=None,
        description="JSON template or raw string template for request payload",
    )
    auth_type: HttpAuthType = Field(
        default=HttpAuthType.NONE,
        description="Authentication strategy to apply",
    )
    auth_token: str | None = Field(
        default=None,
        description="Secret token, bearer key, or api key (encrypted at rest)",
    )
    auth_header_name: str = Field(
        default="Authorization",
        description="Header name when auth_type is 'api_key' or 'bearer'",
    )
    timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="HTTP request timeout in seconds",
    )
    response_path: str | None = Field(
        default=None,
        description="Optional dot-path or key to extract from response JSON (e.g. 'data.results' or 'weather[0]')",
    )
    retry_count: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum retry attempts on transient network errors (5xx/timeouts)",
    )


class CodeToolConfig(BaseModel):
    """Configuration for sandboxed Python code tool execution."""

    code: str = Field(
        ...,
        min_length=1,
        description="Python code defining an entrypoint function that accepts tool arguments",
    )
    entrypoint: str = Field(
        default="execute",
        min_length=1,
        max_length=64,
        description="Name of the entrypoint function inside code to invoke (defaults to 'execute')",
    )
    timeout_seconds: float = Field(
        default=5.0,
        ge=0.5,
        le=30.0,
        description="Maximum execution timeout in seconds",
    )
    allowed_modules: list[str] = Field(
        default_factory=lambda: ["math", "json", "datetime", "re", "random", "urllib.parse"],
        description="Whitelist of standard Python modules the code is permitted to import",
    )


class DatabaseToolConfig(BaseModel):
    """Configuration for read-only database query execution."""

    connection_string: str = Field(
        ...,
        min_length=1,
        description="Database connection URI (e.g. postgresql://... or sqlite://...; encrypted at rest)",
    )
    query_template: str = Field(
        ...,
        min_length=1,
        description="Read-only parameterized SQL query (e.g. 'SELECT id, name FROM users WHERE role = :role')",
    )
    max_rows: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of rows returned to prevent LLM context exhaustion",
    )
    timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Database execution query timeout in seconds",
    )

    @field_validator("query_template")
    @classmethod
    def validate_readonly_query(cls, v: str) -> str:
        trimmed = v.strip().lower()
        if not (trimmed.startswith("select") or trimmed.startswith("with")):
            raise ValueError("Only read-only queries ('SELECT' or 'WITH ... SELECT') are permitted.")
        
        forbidden = [
            "insert ", "update ", "delete ", "drop ", "alter ",
            "truncate ", "grant ", "revoke ", "create ", "exec ", "execute "
        ]
        for word in forbidden:
            if word in trimmed:
                raise ValueError(f"Dangerous SQL statement detected: '{word.strip().upper()}'. Only read queries are allowed.")
        return v


class BuiltInToolConfig(BaseModel):
    """Configuration for built-in or system-provided tools."""

    handler: str = Field(..., description="Internal handler identifier for the tool")
    params: dict[str, Any] = Field(default_factory=dict, description="Pre-configured static parameters")


def validate_json_schema(schema_dict: dict[str, Any]) -> dict[str, Any]:
    """Validates that a dictionary is a syntactically valid JSON Schema."""
    if not isinstance(schema_dict, dict):
        raise ValueError("input_schema must be a dictionary representing a valid JSON Schema object.")
    try:
        jsonschema.Draft202012Validator.check_schema(schema_dict)
    except jsonschema.SchemaError as err:
        raise ValueError(f"Invalid JSON Schema: {err.message}") from err
    return schema_dict


NAME_REGEX = re.compile(r"^[a-zA-Z0-9_-]{2,64}$")


class ToolCreate(BaseModel):
    """Request payload for registering a new tool."""

    name: str = Field(
        ...,
        min_length=2,
        max_length=64,
        description="Unique tool identifier name (alphanumeric, underscores, hyphens)",
    )
    description: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Clear, detailed description explaining when and how the LLM should use this tool",
    )
    type: ToolType = Field(..., description="Tool execution backend: 'api', 'code', 'database', 'built_in'")
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema describing arguments accepted by the tool",
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON Schema describing expected return data format",
    )
    config: dict[str, Any] = Field(
        ...,
        description="Execution-specific configuration dictionary",
    )
    is_system: bool = Field(default=False, description="Whether tool is globally accessible")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not NAME_REGEX.match(v):
            raise ValueError("Tool name must be 2-64 characters and contain only alphanumeric characters, underscores, or hyphens.")
        return v

    @field_validator("input_schema")
    @classmethod
    def check_input_schema(cls, v: dict[str, Any]) -> dict[str, Any]:
        return validate_json_schema(v)

    @field_validator("output_schema")
    @classmethod
    def check_output_schema(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is not None:
            return validate_json_schema(v)
        return None

    @model_validator(mode="after")
    def validate_type_config(self) -> "ToolCreate":
        # Validate config matches declared type
        t = self.type
        cfg = self.config
        if t == ToolType.API:
            HttpToolConfig.model_validate(cfg)
        elif t == ToolType.CODE:
            CodeToolConfig.model_validate(cfg)
        elif t == ToolType.DATABASE:
            DatabaseToolConfig.model_validate(cfg)
        elif t == ToolType.BUILT_IN:
            BuiltInToolConfig.model_validate(cfg)
        return self


class ToolUpdate(BaseModel):
    """Request payload for updating an existing tool."""

    name: str | None = Field(default=None, min_length=2, max_length=64)
    description: str | None = Field(default=None, min_length=5, max_length=2000)
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    is_system: bool | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        if v is not None and not NAME_REGEX.match(v):
            raise ValueError("Tool name must be 2-64 characters and contain only alphanumeric characters, underscores, or hyphens.")
        return v

    @field_validator("input_schema")
    @classmethod
    def check_input_schema(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is not None:
            return validate_json_schema(v)
        return None

    @field_validator("output_schema")
    @classmethod
    def check_output_schema(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is not None:
            return validate_json_schema(v)
        return None


class ToolResponse(BaseModel):
    """Representation of a persisted Tool record with masked secrets."""

    id: str = Field(..., description="Unique tool UUID")
    user_id: str = Field(..., description="Owner user ID")
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool prompt description")
    type: ToolType = Field(..., description="Tool execution backend")
    input_schema: dict[str, Any] = Field(..., description="JSON Schema for arguments")
    output_schema: dict[str, Any] | None = Field(default=None, description="JSON Schema for output")
    config: dict[str, Any] = Field(..., description="Execution config with sensitive secrets masked")
    is_system: bool = Field(..., description="Whether tool is globally accessible")
    version: int = Field(..., description="Version number")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    @classmethod
    def from_orm_model(cls, tool_orm: Any) -> "ToolResponse":
        """Factory masking sensitive keys in config."""
        from app.core.crypto import mask_secret

        raw_cfg = dict(tool_orm.config or {})
        masked_cfg = dict(raw_cfg)

        # Mask sensitive keys
        if "auth_token" in masked_cfg and masked_cfg["auth_token"]:
            masked_cfg["auth_token"] = mask_secret(masked_cfg["auth_token"])
        if "connection_string" in masked_cfg and masked_cfg["connection_string"]:
            masked_cfg["connection_string"] = mask_secret(masked_cfg["connection_string"], prefix_len=10, suffix_len=4)

        return cls(
            id=str(tool_orm.id),
            user_id=tool_orm.user_id,
            name=tool_orm.name,
            description=tool_orm.description,
            type=ToolType(tool_orm.type),
            input_schema=tool_orm.input_schema,
            output_schema=tool_orm.output_schema,
            config=masked_cfg,
            is_system=tool_orm.is_system,
            version=tool_orm.version,
            created_at=tool_orm.created_at,
            updated_at=tool_orm.updated_at,
        )


class ToolListResponse(BaseModel):
    """Paginated list of tools."""

    total: int = Field(..., description="Total matching tools")
    items: list[ToolResponse] = Field(..., description="List of tools")


class ToolExecutionRequest(BaseModel):
    """Payload for directly testing or executing a tool."""

    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments dictionary passed to the tool invocation",
    )


class ToolExecutionResponse(BaseModel):
    """Response returned from executing a tool."""

    tool_id: str = Field(..., description="UUID of the executed tool")
    tool_name: str = Field(..., description="Name of the tool")
    success: bool = Field(..., description="Whether execution completed without unhandled exceptions")
    output: Any = Field(default=None, description="Result returned from the tool")
    error: str | None = Field(default=None, description="Error message if execution failed")
    execution_time_ms: float = Field(..., description="Execution duration in milliseconds")

