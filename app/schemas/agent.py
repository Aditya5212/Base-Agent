from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Detailed Agent Configuration Sub-schemas (Section 1.2 Specification)
# ---------------------------------------------------------------------------

class IdentityConfig(BaseModel):
    """Identity, system persona, and output style settings."""

    system_prompt: str = Field(
        default="You are a helpful and intelligent AI assistant.",
        min_length=10,
        description="System prompt defining persona, constraints, and instructions.",
    )
    response_format: Literal["text", "markdown", "json"] = Field(
        default="markdown",
        description="Expected format style of generated responses.",
    )
    locale: str = Field(
        default="English",
        description="Primary language / locale for responses.",
    )


class ModelConfig(BaseModel):
    """LLM provider and generation hyperparameter settings."""

    provider: Literal["deepseek", "openai", "anthropic", "custom"] = Field(
        default="deepseek",
        description="LLM provider backend.",
    )
    model_id: str = Field(
        default="deepseek-flash",
        description="Model identifier (e.g. deepseek-flash, deepseek-v4-pro, gpt-4o).",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature between 0.0 (deterministic) and 2.0 (creative).",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=32000,
        description="Maximum tokens to generate in response.",
    )
    enable_thinking: bool = Field(
        default=False,
        description="Enable deep reasoning / thinking mode if supported by model.",
    )
    base_url: str | None = Field(
        default=None,
        description="Optional custom API base URL for OpenAI-compatible endpoints.",
    )


class ProviderSpec(BaseModel):
    """Knowledge provider attachment specification."""

    name: Literal["hybrid", "lightrag"] = Field(
        ...,
        description="Knowledge provider name from registry.",
    )
    mode: str = Field(
        default="balanced",
        description="Retrieval mode (e.g. balanced, high_precision, high_recall, mix, local).",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Weighting coefficient for multi-provider fusion.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of passages/chunks to retrieve.",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this knowledge provider is active.",
    )


class ToolSpec(BaseModel):
    """Tool attachment specification."""

    tool_id: str = Field(
        ...,
        description="Unique identifier of registered tool.",
    )
    enabled: bool = Field(
        default=True,
        description="Whether the tool is available for LLM function calling.",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific configuration parameters.",
    )


class StructuredOutputConfig(BaseModel):
    """Structured JSON schema output enforcement settings."""

    enabled: bool = Field(
        default=False,
        description="Whether structured output enforcement is active.",
    )
    schema_id: str | None = Field(
        default=None,
        description="UUID string of structured_schemas record.",
    )
    enforce: bool = Field(
        default=False,
        description="Strict validation error vs best-effort parsing.",
    )


class GuardrailsConfig(BaseModel):
    """Input and output safety guardrails."""

    input_validation: str | None = Field(
        default=None,
        description="Regex pattern or prompt check for input filtering.",
    )
    output_filtering: str | None = Field(
        default=None,
        description="PII redaction or moderation pattern.",
    )
    blocked_topics: list[str] = Field(
        default_factory=list,
        description="List of prohibited topics or keywords.",
    )


class MemoryConfig(BaseModel):
    """Conversation history and checkpoint memory settings."""

    context_window_messages: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Maximum recent messages to include in LLM context window.",
    )
    enable_long_term: bool = Field(
        default=False,
        description="Whether to enable long-term episodic or semantic memory.",
    )
    summarization_threshold: int = Field(
        default=40,
        ge=5,
        description="Number of messages before triggering automated background summarization.",
    )


class AdvancedConfig(BaseModel):
    """Execution and resilience controls."""

    enable_streaming: bool = Field(
        default=True,
        description="Stream response tokens via SSE by default.",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum transient failure retries on model invocation.",
    )
    timeout_seconds: int = Field(
        default=60,
        ge=5,
        le=300,
        description="Per-turn generation timeout in seconds.",
    )


class AgentConfig(BaseModel):
    """Comprehensive Agent Configuration stored in agents.config JSONB column."""

    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    knowledge_providers: list[ProviderSpec] = Field(
        default_factory=lambda: [
            ProviderSpec(name="hybrid", mode="balanced", weight=1.0, top_k=5, enabled=True)
        ]
    )
    tools: list[ToolSpec] = Field(default_factory=list)
    structured_output: StructuredOutputConfig = Field(default_factory=StructuredOutputConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)
    custom_settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Framework-agnostic custom key-value settings or UI metadata.",
    )

    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# Agent CRUD Request & Response Schemas
# ---------------------------------------------------------------------------

class AgentCreate(BaseModel):
    """Request payload for creating a new agent."""

    name: str = Field(..., min_length=1, max_length=255, description="Name of the agent.")
    description: str | None = Field(None, max_length=1000, description="Optional description.")
    config: AgentConfig = Field(default_factory=AgentConfig, description="Configuration settings.")
    is_default: bool = Field(False, description="Set as the default agent for this user.")
    is_published: bool = Field(False, description="Whether the agent is published.")


class AgentUpdate(BaseModel):
    """Request payload for updating an existing agent."""

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = Field(None, max_length=1000)
    config: AgentConfig | None = None
    is_default: bool | None = None
    is_published: bool | None = None


class AgentResponse(BaseModel):
    """Detailed response model for an agent."""

    id: uuid.UUID
    user_id: str
    name: str
    description: str | None
    config: AgentConfig
    is_default: bool
    is_published: bool
    version: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AgentListResponse(BaseModel):
    """Paginated list of agents."""

    items: list[AgentResponse]
    total: int
    page: int
    page_size: int


class AgentCloneRequest(BaseModel):
    """Request payload for cloning an existing agent."""

    name: str | None = Field(None, min_length=1, max_length=255, description="Name for the clone.")
    description: str | None = Field(None, max_length=1000, description="Optional description for the clone.")
