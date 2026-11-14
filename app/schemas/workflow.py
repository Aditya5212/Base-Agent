from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Union
import uuid

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Node Types Enum
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    START = "start"
    END = "end"
    CHAT = "chat"
    RETRIEVE = "retrieve"
    TOOL_CALL = "tool_call"
    CONDITIONAL = "conditional"
    HUMAN_APPROVAL = "human_approval"
    CODE = "code"
    SUBGRAPH = "subgraph"
    PARALLEL = "parallel"


# ---------------------------------------------------------------------------
# Visual Canvas Layout Helper
# ---------------------------------------------------------------------------

class NodePosition(BaseModel):
    """Position coordinates for visual graph canvas UI."""
    model_config = ConfigDict(extra="ignore")
    x: float = 0.0
    y: float = 0.0


# ---------------------------------------------------------------------------
# Node-Specific Configuration Schemas
# ---------------------------------------------------------------------------

class StartNodeConfig(BaseModel):
    """Configuration for workflow start node."""
    model_config = ConfigDict(extra="ignore")
    input_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON Schema (Draft 2020-12) to validate workflow inputs.",
    )
    initial_variables: dict[str, Any] = Field(
        default_factory=dict,
        description="Default variables injected into workflow state upon execution.",
    )


class EndNodeConfig(BaseModel):
    """Configuration for workflow end node."""
    model_config = ConfigDict(extra="ignore")
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON Schema to validate workflow execution outputs.",
    )
    response_variable: str | None = Field(
        default=None,
        description="Specific variable key to return as workflow output. If None, returns all variables.",
    )


class ChatNodeConfig(BaseModel):
    """Configuration for LLM generation node."""
    model_config = ConfigDict(extra="ignore")
    system_prompt: str = Field(
        ...,
        min_length=1,
        description="System prompt with optional variable interpolation (e.g. {{rag_context}}).",
    )
    user_prompt_template: str | None = Field(
        default=None,
        description="User message template with variable interpolation (e.g. 'Question: {{input}}').",
    )
    model_id: str = Field(
        default="deepseek-flash",
        description="Target model identifier or alias.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Max response tokens.",
    )
    fallback_model_id: str | None = Field(
        default=None,
        description="Secondary fallback model if primary fails (ModelFallbackMiddleware).",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Exponential backoff retries on 429/5xx errors (ModelRetryMiddleware).",
    )
    output_variable: str = Field(
        default="last_response",
        description="Key in workflow state variables to store the generated response text.",
    )
    enable_memory: bool = Field(
        default=True,
        description="Whether to append the LLM response to conversational message history.",
    )


class RetrieveNodeConfig(BaseModel):
    """Configuration for RAG knowledge retrieval node."""
    model_config = ConfigDict(extra="ignore")
    provider_name: str = Field(
        default="hybrid",
        description="Registered knowledge provider name ('hybrid', 'lightrag').",
    )
    query_template: str = Field(
        default="{{input}}",
        description="Query template with variable interpolation (e.g. {{variables.user_query}}).",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of chunks/passages to retrieve.",
    )
    workspace_type: Literal["user", "thread", "agent", "global"] = Field(
        default="user",
        description="Knowledge isolation boundary.",
    )
    scope: Literal["global_only", "thread_only", "thread_with_global", "all_user"] | None = Field(
        default=None,
        description="Explicit retrieval scope. If omitted, resolved based on workspace_type and active thread.",
    )
    output_variable: str = Field(
        default="rag_context",
        description="Variable key to store formatted retrieved context text.",
    )
    sources_variable: str = Field(
        default="rag_sources",
        description="Variable key to store retrieved metadata and citation records.",
    )


class WorkflowNodeKnowledgeConfig(BaseModel):
    """Payload for configuring or attaching knowledge retrieval to a workflow node."""
    model_config = ConfigDict(extra="ignore")
    provider_name: str = Field(
        default="hybrid",
        description="Knowledge retrieval provider ('hybrid', 'lightrag', 'multi_query')",
    )
    workspace_type: Literal["user", "thread", "agent", "global"] = Field(
        default="user",
        description="Knowledge isolation boundary",
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Passages/entities to retrieve")
    query_template: str = Field(default="{{input}}", description="Query template")
    output_variable: str = Field(default="knowledge_context", description="Variable key for context")
    sources_variable: str = Field(default="knowledge_sources", description="Variable key for sources")


class ConditionBranch(BaseModel):
    """A conditional branch rule evaluated by a ConditionalNode."""
    model_config = ConfigDict(extra="ignore")
    target_node_id: str = Field(..., description="Target node ID if condition matches.")
    condition_type: Literal["expression", "variable_match", "regex", "jsonpath"] = Field(
        default="expression",
        description="Evaluation strategy.",
    )
    variable: str | None = Field(
        default=None,
        description="Variable name to inspect for variable_match or regex conditions.",
    )
    operator: Literal["eq", "neq", "gt", "gte", "lt", "lte", "contains", "in"] | None = Field(
        default=None,
        description="Comparison operator for variable_match.",
    )
    value: Any = Field(
        default=None,
        description="Comparison value for variable_match.",
    )
    expression: str | None = Field(
        default=None,
        description="Python-like boolean expression (e.g. 'variables.score > 0.8').",
    )


class ConditionalNodeConfig(BaseModel):
    """Configuration for branching router node."""
    model_config = ConfigDict(extra="ignore")
    branches: list[ConditionBranch] = Field(
        default_factory=list,
        description="Ordered list of branch conditions. First matching branch is followed.",
    )
    default_target_node_id: str = Field(
        ...,
        description="Fallback target node ID if none of the branches match.",
    )


class ToolCallNodeConfig(BaseModel):
    """Configuration for tool execution node."""
    model_config = ConfigDict(extra="ignore")
    tool_name: str = Field(
        ...,
        description="Registered tool name (e.g. 'hybrid_document_search', 'web_search').",
    )
    input_mapping: dict[str, Any] = Field(
        default_factory=dict,
        description="Mapping from tool arguments to static values or template variables (e.g. '{{query}}').",
    )
    output_variable: str = Field(
        default="tool_result",
        description="Variable key to store tool execution result.",
    )
    catch_errors: bool = Field(
        default=True,
        description="Catch tool exceptions and write actionable error to state (ToolErrorMiddleware).",
    )
    retry_count: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Retry attempts on transient tool failures (ToolRetryMiddleware).",
    )
    requires_approval: bool = Field(
        default=False,
        description="Pause execution using HumanInTheLoopMiddleware before invoking tool.",
    )


class HumanApprovalNodeConfig(BaseModel):
    """Configuration for human-in-the-loop review node."""
    model_config = ConfigDict(extra="ignore")
    message_template: str = Field(
        default="Action requires human review and approval to proceed.",
        description="Notification/prompt shown to the reviewer.",
    )
    approval_variable: str = Field(
        default="approval_status",
        description="Variable key populated with 'approved' or 'rejected' upon resume.",
    )
    timeout_seconds: int = Field(
        default=86400,
        gt=0,
        description="Timeout before execution automatically expires.",
    )


class CodeNodeConfig(BaseModel):
    """Configuration for custom Python execution node."""
    model_config = ConfigDict(extra="ignore")
    code: str = Field(
        ...,
        min_length=1,
        description="Python code snippet executed inside a secure restricted sandbox.",
    )
    input_variables: list[str] = Field(
        default_factory=list,
        description="List of state variable keys injected into the code's local execution scope.",
    )
    output_variable: str = Field(
        default="code_result",
        description="Variable key to store result assigned to 'result' in code.",
    )
    timeout_seconds: int = Field(
        default=10,
        gt=0,
        le=60,
        description="Execution timeout in seconds.",
    )


class SubgraphNodeConfig(BaseModel):
    """Configuration for invoking another workflow as a child subgraph."""
    model_config = ConfigDict(extra="ignore")
    subgraph_workflow_id: str = Field(
        ...,
        description="UUID of the child workflow to invoke.",
    )
    input_mapping: dict[str, Any] = Field(
        default_factory=dict,
        description="Mapping from parent variables to child workflow input.",
    )
    output_variable: str = Field(
        default="subgraph_result",
        description="Variable key to store child workflow execution output.",
    )


class ParallelNodeConfig(BaseModel):
    """Configuration for fan-out parallel execution node."""
    model_config = ConfigDict(extra="ignore")
    target_nodes: list[str] = Field(
        ...,
        min_length=1,
        description="Target node IDs to trigger concurrently.",
    )
    merge_strategy: Literal["wait_all", "first_completed"] = Field(
        default="wait_all",
        description="Synchronization strategy when combining branches.",
    )


# ---------------------------------------------------------------------------
# Discriminated Node Models
# ---------------------------------------------------------------------------

class BaseNodeModel(BaseModel):
    """Base attributes shared by all workflow nodes."""
    model_config = ConfigDict(extra="ignore")
    id: str = Field(..., min_length=1, description="Unique identifier for the node within the workflow.")
    name: str = Field(default="", description="Human-readable node label.")
    description: str | None = Field(default=None, description="Optional node documentation.")
    position: NodePosition | None = Field(default=None, description="Visual UI canvas coordinates.")


class StartNode(BaseNodeModel):
    type: Literal[NodeType.START] = NodeType.START
    config: StartNodeConfig = Field(default_factory=StartNodeConfig)


class EndNode(BaseNodeModel):
    type: Literal[NodeType.END] = NodeType.END
    config: EndNodeConfig = Field(default_factory=EndNodeConfig)


class ChatNode(BaseNodeModel):
    type: Literal[NodeType.CHAT] = NodeType.CHAT
    config: ChatNodeConfig


class RetrieveNode(BaseNodeModel):
    type: Literal[NodeType.RETRIEVE] = NodeType.RETRIEVE
    config: RetrieveNodeConfig = Field(default_factory=RetrieveNodeConfig)


class ConditionalNode(BaseNodeModel):
    type: Literal[NodeType.CONDITIONAL] = NodeType.CONDITIONAL
    config: ConditionalNodeConfig


class ToolCallNode(BaseNodeModel):
    type: Literal[NodeType.TOOL_CALL] = NodeType.TOOL_CALL
    config: ToolCallNodeConfig


class HumanApprovalNode(BaseNodeModel):
    type: Literal[NodeType.HUMAN_APPROVAL] = NodeType.HUMAN_APPROVAL
    config: HumanApprovalNodeConfig = Field(default_factory=HumanApprovalNodeConfig)


class CodeNode(BaseNodeModel):
    type: Literal[NodeType.CODE] = NodeType.CODE
    config: CodeNodeConfig


class SubgraphNode(BaseNodeModel):
    type: Literal[NodeType.SUBGRAPH] = NodeType.SUBGRAPH
    config: SubgraphNodeConfig


class ParallelNode(BaseNodeModel):
    type: Literal[NodeType.PARALLEL] = NodeType.PARALLEL
    config: ParallelNodeConfig


# Discriminated Union for polymorphic node deserialization and validation
WorkflowNode = Annotated[
    Union[
        StartNode,
        EndNode,
        ChatNode,
        RetrieveNode,
        ConditionalNode,
        ToolCallNode,
        HumanApprovalNode,
        CodeNode,
        SubgraphNode,
        ParallelNode,
    ],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Workflow Edge Schema
# ---------------------------------------------------------------------------

class WorkflowEdge(BaseModel):
    """Directed connection between two workflow nodes."""
    model_config = ConfigDict(extra="ignore")
    id: str | None = Field(default=None, description="Optional unique edge ID.")
    source: str = Field(..., min_length=1, description="Source node ID.")
    target: str = Field(..., min_length=1, description="Target node ID.")
    condition: str | None = Field(
        default=None,
        description="Optional routing condition or source output handle identifier.",
    )
    label: str | None = Field(default=None, description="Optional display label for visual canvas.")


# ---------------------------------------------------------------------------
# Workflow CRUD & Execution Request / Response Contracts
# ---------------------------------------------------------------------------

class WorkflowCreate(BaseModel):
    """Payload to create a new workflow."""
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(default=None)
    nodes: list[WorkflowNode] = Field(..., min_length=2, description="Workflow nodes (must include START and END).")
    edges: list[WorkflowEdge] = Field(..., min_length=1, description="Directed edges connecting nodes.")
    input_schema: dict[str, Any] | None = Field(default=None)
    output_schema: dict[str, Any] | None = Field(default=None)
    recursion_limit: int = Field(default=25, ge=1, le=100)
    is_published: bool = Field(default=False)


class WorkflowUpdate(BaseModel):
    """Payload to update an existing workflow."""
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None)
    nodes: list[WorkflowNode] | None = Field(default=None, min_length=2)
    edges: list[WorkflowEdge] | None = Field(default=None, min_length=1)
    input_schema: dict[str, Any] | None = Field(default=None)
    output_schema: dict[str, Any] | None = Field(default=None)
    recursion_limit: int | None = Field(default=None, ge=1, le=100)
    is_published: bool | None = Field(default=None)


class WorkflowResponse(BaseModel):
    """Public representation of a workflow entity."""
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    user_id: str
    name: str
    description: str | None
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    input_schema: dict[str, Any] | None
    output_schema: dict[str, Any] | None
    recursion_limit: int
    version: int
    is_published: bool
    created_at: datetime
    updated_at: datetime


class WorkflowListResponse(BaseModel):
    """Paginated list of workflows."""
    items: list[WorkflowResponse]
    total: int
    page: int
    page_size: int


class WorkflowRunRequest(BaseModel):
    """Payload to execute a workflow."""
    input_data: dict[str, Any] = Field(default_factory=dict, description="Input variables passed to workflow START node.")
    thread_id: str | None = Field(default=None, description="Optional thread session ID for memory checkpointing.")


class WorkflowResumeRequest(BaseModel):
    """Payload to resume an interrupted workflow (e.g. human-in-the-loop)."""
    resume_data: dict[str, Any] = Field(default_factory=dict, description="Variables or approval payload to resume with.")


class WorkflowRunResponse(BaseModel):
    """Public representation of a workflow execution run."""
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    workflow_id: uuid.UUID
    user_id: str
    thread_id: str
    checkpoint_id: str | None = None
    definition_snapshot: dict[str, Any] | None = None
    status: str
    input_data: dict[str, Any]
    output_data: dict[str, Any] | None
    error_message: str | None
    execution_log: list[dict[str, Any]]
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class WorkflowRunListResponse(BaseModel):
    """Paginated list of workflow execution runs."""
    items: list[WorkflowRunResponse]
    total: int
    page: int
    page_size: int


class GraphValidationResponse(BaseModel):
    """Result of graph topology validation."""
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    node_count: int
    edge_count: int
