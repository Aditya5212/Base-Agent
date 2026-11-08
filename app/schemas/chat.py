import json
from typing import Any, Literal
import uuid
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Request payload for sending a message to the agent."""

    message: str = Field(..., min_length=1, description="User prompt or message text")
    thread_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Thread ID organizing conversation memory. Provide existing ID to continue conversation.",
    )
    user_id: str = Field(default="root", description="Identifier of the user (defaults to 'root')")
    images: list[str] = Field(
        default_factory=list,
        description="Optional image URLs, presigned URLs, base64 strings, or R2 keys for multimodal vision input",
    )
    model: str | None = Field(
        default=None,
        description="Optional model override (e.g. 'DeepSeek-V4.1-Flash-1017' or 'DeepSeek-V4-Pro-0813')",
    )
    use_rag: bool = Field(
        default=False,
        description="Whether to retrieve relevant passages from user documents via hybrid search",
    )
    rag_preset: Literal["balanced", "high_precision", "high_recall"] | None = Field(
        default=None,
        description="RAG precision/recall preset: high_precision, balanced, high_recall",
    )
    retrieval_mode: Literal["hybrid", "lightrag", "multi_query"] | None = Field(
        default=None,
        description="RAG retrieval strategy: hybrid (dense+sparse), lightrag (graph), multi_query (parallel queries)",
    )
    agent_id: str | None = Field(
        default=None,
        description="Optional agent UUID. If omitted, the user's default agent is resolved.",
    )
    tools: list[str] | None = Field(
        default=None,
        description="Optional list of tools to enable for this turn (e.g. 'hybrid_document_search', 'graph_rag_search', 'multi_query_search', 'all_rag_tools', 'calculator')",
    )

    @field_validator("thread_id")
    @classmethod
    def reject_reserved_namespace(cls, value: str) -> str:
        if value.startswith("workflow:"):
            raise ValueError("Reserved workflow checkpoint namespace")
        return value


class ChatMessage(BaseModel):
    """Representation of an individual message in conversation history."""

    role: str = Field(..., description="Role of the sender (human, ai, system)")
    content: str = Field(..., description="Message text content")
    id: str | None = Field(default=None, description="Optional message identifier")


class ChatResponse(BaseModel):
    """Response returned from a chat execution turn."""

    thread_id: str = Field(..., description="Thread ID of the conversation")
    user_id: str = Field(..., description="User ID associated with the conversation")
    agent_id: str | None = Field(default=None, description="Agent ID configured for this conversation turn")
    response: str = Field(..., description="Agent response message")
    model: str = Field(..., description="Model version used for generating the response")
    history_length: int = Field(..., description="Total number of messages currently stored in thread state")
    sources: list[dict[str, Any]] | None = Field(
        default=None,
        description="Retrieved document sources when RAG is enabled",
    )
    thread_assets: list[dict[str, Any]] | None = Field(
        default=None,
        description="Uploaded assets (images, documents) associated with this thread with fresh signed URLs",
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None,
        description="List of tool calls executed during this conversation turn",
    )



class ThreadAssetItem(BaseModel):
    """Uploaded asset (image or document) associated with a conversation thread."""

    id: str = Field(..., description="Unique asset UUID")
    thread_id: str = Field(..., description="Associated thread ID")
    asset_type: str = Field(..., description="Type of asset ('image' or 'document')")
    file_name: str = Field(..., description="Original file name")
    mime_type: str = Field(..., description="MIME content type")
    file_size: int = Field(..., description="File size in bytes")
    presigned_url: str | None = Field(default=None, description="Active presigned download/view URL")
    created_at: str | None = Field(default=None, description="Creation timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")


class ThreadAssetUploadResponse(BaseModel):
    """Response returned upon uploading an asset to a thread."""

    id: str = Field(..., description="Unique asset UUID")
    thread_id: str = Field(..., description="Associated thread ID")
    file_name: str = Field(..., description="Original file name")
    asset_type: str = Field(..., description="Detected asset type ('image' or 'document')")
    r2_key: str = Field(..., description="Cloudflare R2 storage key")
    presigned_url: str = Field(..., description="Presigned URL for immediate client viewing/downloading")
    message: str = Field(..., description="Status summary")


class ThreadDetailsResponse(BaseModel):
    """Detailed representation of a conversation thread including its uploaded assets."""

    id: str = Field(..., description="Thread ID")
    user_id: str = Field(..., description="Owner user ID")
    title: str = Field(..., description="Conversation title")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Thread metadata dictionary")
    is_pinned: bool = Field(default=False, description="Whether thread is pinned")
    is_archived: bool = Field(default=False, description="Whether thread is archived")
    assets: list[ThreadAssetItem] = Field(default_factory=list, description="Uploaded images and documents in this thread")


class ThreadHistoryResponse(BaseModel):
    """Response containing persisted short-term memory history for a thread."""

    thread_id: str = Field(..., description="Thread ID")
    total_messages: int = Field(..., description="Total number of messages in memory")
    messages: list[ChatMessage] = Field(..., description="List of messages in chronological order")


class DeleteThreadResponse(BaseModel):
    """Response returned after deleting a thread's memory."""

    thread_id: str = Field(..., description="Thread ID that was deleted")
    deleted: bool = Field(..., description="Whether memory was deleted successfully")
    message: str = Field(..., description="Status description")


class StreamEventType:
    """Standard SSE event type constants."""

    TOKEN = "token"
    REASONING = "reasoning"
    END = "end"
    ERROR = "error"


class StreamEvent(BaseModel):
    """Standardized event envelope for Server-Sent Events (SSE) streaming."""

    event: str = Field(..., description="Event type: token, reasoning, end, or error")
    data: dict[str, Any] = Field(..., description="Structured payload dictionary")

    def to_sse(self) -> str:
        """Serializes event to Server-Sent Events wire format."""
        payload = json.dumps(self.data)
        return f"event: {self.event}\ndata: {payload}\n\n"

    @classmethod
    def token(cls, delta: str, node: str = "chat_node") -> "StreamEvent":
        """Factory for text token deltas."""
        return cls(
            event=StreamEventType.TOKEN,
            data={"type": StreamEventType.TOKEN, "delta": delta, "node": node},
        )

    @classmethod
    def reasoning(cls, delta: str, node: str = "chat_node") -> "StreamEvent":
        """Factory for reasoning/thinking token deltas."""
        return cls(
            event=StreamEventType.REASONING,
            data={"type": StreamEventType.REASONING, "delta": delta, "node": node},
        )

    @classmethod
    def end(
        cls,
        thread_id: str,
        user_id: str,
        response: str,
        model: str,
        history_length: int,
        sources: list[dict[str, Any]] | None = None,
        agent_id: str | None = None,
    ) -> "StreamEvent":
        """Factory for final turn completion event."""
        data: dict[str, Any] = {
            "type": StreamEventType.END,
            "thread_id": thread_id,
            "user_id": user_id,
            "response": response,
            "model": model,
            "history_length": history_length,
        }
        if agent_id is not None:
            data["agent_id"] = agent_id
        if sources is not None:
            data["sources"] = sources
        return cls(event=StreamEventType.END, data=data)

    @classmethod
    def error(cls, error: str) -> "StreamEvent":
        """Factory for streaming error event."""
        return cls(
            event=StreamEventType.ERROR,
            data={"type": StreamEventType.ERROR, "error": error},
        )

