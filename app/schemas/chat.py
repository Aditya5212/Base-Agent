import json
from typing import Any
import uuid
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request payload for sending a message to the agent."""

    message: str = Field(..., min_length=1, description="User prompt or message text")
    thread_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Thread ID organizing conversation memory. Provide existing ID to continue conversation.",
    )
    user_id: str = Field(default="default_user", description="Identifier of the user")
    model: str | None = Field(
        default=None,
        description="Optional model override (e.g. 'DeepSeek-V4.1-Flash-1017' or 'DeepSeek-V4-Pro-0813')",
    )
    use_rag: bool = Field(
        default=False,
        description="Whether to retrieve relevant passages from user documents via hybrid search",
    )
    rag_preset: str = Field(
        default="balanced",
        description="RAG precision/recall preset: high_precision, balanced, high_recall",
    )


class ChatMessage(BaseModel):
    """Representation of an individual message in conversation history."""

    role: str = Field(..., description="Role of the sender (human, ai, system)")
    content: str = Field(..., description="Message text content")
    id: str | None = Field(default=None, description="Optional message identifier")


class ChatResponse(BaseModel):
    """Response returned from a chat execution turn."""

    thread_id: str = Field(..., description="Thread ID of the conversation")
    user_id: str = Field(..., description="User ID associated with the conversation")
    response: str = Field(..., description="Agent response message")
    model: str = Field(..., description="Model version used for generating the response")
    history_length: int = Field(..., description="Total number of messages currently stored in thread state")
    sources: list[dict[str, Any]] | None = Field(
        default=None,
        description="Retrieved document sources when RAG is enabled",
    )


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
        import json
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

