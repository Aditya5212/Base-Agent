"""Chat endpoints with persistent PostgreSQL short-term memory."""

import logging
from fastapi import APIRouter, HTTPException, status
from langchain_core.messages import BaseMessage

from app.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    DeleteThreadResponse,
    ThreadHistoryResponse,
)
from app.services.agent import execute_agent
from app.services.memory import delete_thread_memory, get_thread_messages

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat & Memory"])


def _map_message_to_schema(msg: BaseMessage) -> ChatMessage:
    """Maps a LangChain BaseMessage to ChatMessage response model."""
    role = getattr(msg, "type", "unknown")
    if role == "human":
        display_role = "human"
    elif role == "ai":
        display_role = "ai"
    elif role == "system":
        display_role = "system"
    else:
        display_role = role

    return ChatMessage(
        role=display_role,
        content=str(msg.content),
        id=getattr(msg, "id", None),
    )


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message to the agent",
    description="Invokes the agent with persistent short-term conversation memory backed by PostgreSQL.",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Executes a conversation turn and returns the agent response with updated thread context."""
    try:
        result, resolved_model = await execute_agent(
            message=request.message,
            thread_id=request.thread_id,
            user_id=request.user_id,
            model=request.model,
        )

        messages = result.get("messages", [])
        if not messages:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Agent execution produced no response messages.",
            )

        last_message = messages[-1]
        response_text = str(last_message.content)

        return ChatResponse(
            thread_id=request.thread_id,
            user_id=request.user_id,
            response=response_text,
            model=resolved_model,
            history_length=len(messages),
        )
    except Exception as exc:
        logger.error("Chat turn failed for thread_id=%s: %s", request.thread_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat execution failed: {exc}",
        )


@router.get(
    "/{thread_id}/history",
    response_model=ThreadHistoryResponse,
    summary="Retrieve thread conversation history",
    description="Fetches all stored short-term memory messages for a given thread from PostgreSQL.",
)
async def get_history(thread_id: str) -> ThreadHistoryResponse:
    """Returns the full chronological message history stored in PostgreSQL for the thread."""
    try:
        raw_messages = await get_thread_messages(thread_id)
        chat_messages = [_map_message_to_schema(m) for m in raw_messages]

        return ThreadHistoryResponse(
            thread_id=thread_id,
            total_messages=len(chat_messages),
            messages=chat_messages,
        )
    except Exception as exc:
        logger.error("Failed to fetch history for thread_id=%s: %s", thread_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve thread history: {exc}",
        )


@router.delete(
    "/{thread_id}",
    response_model=DeleteThreadResponse,
    summary="Clear thread memory",
    description="Deletes all persisted checkpoint memory records for a given thread from PostgreSQL.",
)
async def clear_memory(thread_id: str) -> DeleteThreadResponse:
    """Removes all checkpoint data for the specified thread."""
    success = await delete_thread_memory(thread_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not delete memory for thread {thread_id}",
        )

    return DeleteThreadResponse(
        thread_id=thread_id,
        deleted=True,
        message=f"Memory for thread {thread_id} successfully cleared.",
    )
