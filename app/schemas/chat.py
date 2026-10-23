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
