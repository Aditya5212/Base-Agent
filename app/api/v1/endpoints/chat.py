"""Chat endpoints with persistent PostgreSQL short-term memory, multimodal image support, and thread assets."""

import logging
import uuid
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import UserContext, get_current_user
from app.core.rate_limit import limiter
from app.core.database import get_db
from app.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    DeleteThreadResponse,
    StreamEvent,
    ThreadAssetItem,
    ThreadAssetUploadResponse,
    ThreadDetailsResponse,
    ThreadHistoryResponse,
)
from app.schemas.agent import ToolSpec
from app.services.agent import execute_agent, stream_agent_events
from app.services.agent_service import resolve_agent_for_chat
from app.services.memory import delete_thread_memory, get_thread_messages
from app.services.r2_storage import r2_storage_service
from app.services.thread_service import (
    add_thread_asset,
    get_or_create_thread,
    get_thread_assets_with_signed_urls,
    verify_thread_ownership,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat & Memory"])

# Maximum allowed thread-asset upload size: 10 MB
MAX_ASSET_SIZE_BYTES = 10 * 1024 * 1024


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

    content = getattr(msg, "text", None) or str(msg.content)

    return ChatMessage(
        role=display_role,
        content=content,
        id=getattr(msg, "id", None),
    )


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message to the agent",
    description="Invokes the agent with persistent conversation memory, multimodal images, and thread assets context.",
)
@limiter.limit("30/minute")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """Executes a conversation turn and returns the agent response with updated thread context."""
    # Enforce authenticated tenant isolation (admin can override user_id if explicitly specified)
    effective_user_id = chat_request.user_id if (current_user.is_admin and chat_request.user_id != "root") else current_user.user_id
    try:
        # Ensure thread entity exists in PostgreSQL and user has access
        await get_or_create_thread(
            db,
            thread_id=chat_request.thread_id,
            user_id=effective_user_id,
            is_admin=current_user.is_admin,
        )

        # Resolve target agent and configuration (falls back to user's default agent if omitted)
        agent, agent_config = await resolve_agent_for_chat(
            db,
            user_id=effective_user_id,
            agent_id=chat_request.agent_id,
            is_admin=current_user.is_admin,
        )

        # Attach/enable tools on agent_config:
        agent_config = agent_config.model_copy(deep=True)
        if chat_request.tools is not None:
            existing_tool_ids = {t.tool_id for t in agent_config.tools}
            for tool_name in chat_request.tools:
                if tool_name not in existing_tool_ids:
                    agent_config.tools.append(ToolSpec(tool_id=tool_name, enabled=True))
                else:
                    for t in agent_config.tools:
                        if t.tool_id == tool_name:
                            t.enabled = True
        elif not any(t.enabled for t in agent_config.tools):
            # Default to providing hybrid_document_search tool so chat agent can search dynamically
            agent_config.tools.append(ToolSpec(tool_id="hybrid_document_search", enabled=True))

        result, resolved_model = await execute_agent(
            message=chat_request.message,
            thread_id=chat_request.thread_id,
            user_id=effective_user_id,
            images=chat_request.images,
            model=chat_request.model,
            use_rag=chat_request.use_rag,
            rag_preset=chat_request.rag_preset,
            retrieval_mode=chat_request.retrieval_mode,
            agent_id=str(agent.id),
            agent_config=agent_config,
        )

        messages = result.get("messages", [])
        if not messages:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Agent execution produced no response messages.",
            )

        last_message = messages[-1]
        response_text = getattr(last_message, "text", None) or str(last_message.content)

        # Extract executed tool calls from message history
        executed_tool_calls: list[dict[str, Any]] = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    executed_tool_calls.append(
                        {
                            "id": tc.get("id"),
                            "name": tc.get("name"),
                            "args": tc.get("args"),
                        }
                    )

        return ChatResponse(
            thread_id=chat_request.thread_id,
            user_id=effective_user_id,
            agent_id=str(agent.id),
            response=response_text,
            model=resolved_model,
            history_length=len(messages),
            sources=result.get("rag_sources"),
            thread_assets=result.get("thread_assets"),
            tool_calls=executed_tool_calls if executed_tool_calls else None,
        )
    except HTTPException:
        raise
    except ValueError as val_err:
        logger.warning("Invalid chat request parameters for thread_id=%s: %s", chat_request.thread_id, val_err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.error("Chat turn failed for thread_id=%s: %s", chat_request.thread_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat execution failed. Please try again.",
        )


@router.post(
    "/stream",
    summary="Stream agent response via Server-Sent Events (SSE)",
    description=(
        "Streams real-time token and reasoning updates from LangGraph event streaming (v3) "
        "with persistent PostgreSQL short-term memory, multimodal image support, and thread assets."
    ),
)
@limiter.limit("20/minute")
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Streams conversation turn tokens and completion metadata using Server-Sent Events (SSE)."""
    # Enforce authenticated tenant isolation
    effective_user_id = chat_request.user_id if (current_user.is_admin and chat_request.user_id != "root") else current_user.user_id
    # Ensure thread entity exists in PostgreSQL before streaming begins
    await get_or_create_thread(
        db,
        thread_id=chat_request.thread_id,
        user_id=effective_user_id,
        is_admin=current_user.is_admin,
    )

    # Resolve target agent and configuration (falls back to user's default agent if omitted)
    agent, agent_config = await resolve_agent_for_chat(
        db,
        user_id=effective_user_id,
        agent_id=chat_request.agent_id,
        is_admin=current_user.is_admin,
    )

    # Attach/enable tools on agent_config:
    agent_config = agent_config.model_copy(deep=True)
    if chat_request.tools is not None:
        existing_tool_ids = {t.tool_id for t in agent_config.tools}
        for tool_name in chat_request.tools:
            if tool_name not in existing_tool_ids:
                agent_config.tools.append(ToolSpec(tool_id=tool_name, enabled=True))
            else:
                for t in agent_config.tools:
                    if t.tool_id == tool_name:
                        t.enabled = True
    elif not any(t.enabled for t in agent_config.tools):
        agent_config.tools.append(ToolSpec(tool_id="hybrid_document_search", enabled=True))

    async def sse_event_generator():
        try:
            async for event in stream_agent_events(
                message=chat_request.message,
                thread_id=chat_request.thread_id,
                user_id=effective_user_id,
                images=chat_request.images,
                model=chat_request.model,
                use_rag=chat_request.use_rag,
                rag_preset=chat_request.rag_preset,
                retrieval_mode=chat_request.retrieval_mode,
                agent_id=str(agent.id),
                agent_config=agent_config,
            ):
                yield event.to_sse()
        except Exception as exc:
            logger.error(
                "Streaming endpoint error for thread_id=%s: %s",
                chat_request.thread_id,
                exc,
                exc_info=True,
            )
            yield StreamEvent.error(error=f"Streaming execution failed: {exc}").to_sse()

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/{thread_id}/assets",
    response_model=ThreadAssetUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an image or document to a conversation thread",
    description="Uploads an asset to Cloudflare R2, associates it with the conversation thread, and returns a signed URL.",
)
@limiter.limit("20/minute")
async def upload_thread_asset(
    request: Request,
    thread_id: str,
    file: UploadFile = File(..., description="Image or document attachment"),
    user_id: str = Form(default="root", description="User ID associated with the thread"),
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ThreadAssetUploadResponse:
    """Uploads an image or document asset scoped directly to a conversation session."""
    effective_user_id = user_id if (current_user.is_admin and user_id != "root") else current_user.user_id
    # Verify thread ownership
    await verify_thread_ownership(
        db,
        thread_id=thread_id,
        user_id=effective_user_id,
        is_admin=current_user.is_admin,
        allow_missing=True,
    )
    # Stream the upload in chunks and enforce size limit before sending to R2,
    # avoiding reading an arbitrarily large file into memory all at once.
    chunks: list[bytes] = []
    total_read = 0
    chunk_size = 64 * 1024  # 64 KB read window

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_read += len(chunk)
        if total_read > MAX_ASSET_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Asset exceeds maximum allowed size of {MAX_ASSET_SIZE_BYTES // (1024 * 1024)} MB.",
            )
        chunks.append(chunk)

    contents = b"".join(chunks)
    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    file_name = file.filename or "attachment"
    content_type = file.content_type or "application/octet-stream"
    asset_id = uuid.uuid4()
    r2_key = f"threads/{thread_id}/{asset_id}_{file_name}"

    try:
        await r2_storage_service.upload_file(
            file_data=contents,
            file_key=r2_key,
            content_type=content_type,
        )
    except Exception as exc:
        logger.error("Failed to upload thread asset to R2: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Storage upload failed. Please try again.",
        )

    asset = await add_thread_asset(
        db=db,
        thread_id=thread_id,
        user_id=effective_user_id,
        file_name=file_name,
        r2_key=r2_key,
        mime_type=content_type,
        file_size=len(contents),
    )

    presigned_url = await r2_storage_service.generate_presigned_url(r2_key)

    return ThreadAssetUploadResponse(
        id=str(asset.id),
        thread_id=thread_id,
        file_name=file_name,
        asset_type=asset.asset_type,
        r2_key=r2_key,
        presigned_url=presigned_url,
        message=f"Asset '{file_name}' uploaded and bound to thread {thread_id}.",
    )


@router.get(
    "/{thread_id}/assets",
    response_model=list[ThreadAssetItem],
    summary="List uploaded assets for a conversation thread",
    description="Retrieves all uploaded images and documents attached to this thread with active presigned URLs.",
)
async def list_thread_assets(
    thread_id: str,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[ThreadAssetItem]:
    """Returns all assets attached to the specified thread with active presigned URLs."""
    await verify_thread_ownership(
        db,
        thread_id=thread_id,
        user_id=current_user.user_id,
        is_admin=current_user.is_admin,
        allow_missing=True,
    )
    raw_assets = await get_thread_assets_with_signed_urls(thread_id, db=db)
    return [ThreadAssetItem(**a) for a in raw_assets]


@router.get(
    "/{thread_id}/details",
    response_model=ThreadDetailsResponse,
    summary="Get conversation thread details and metadata",
    description="Returns thread metadata, title, and all active uploaded assets.",
)
async def get_thread_details(
    thread_id: str,
    user_id: str = "root",
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ThreadDetailsResponse:
    """Returns thread metadata, ownership, and current assets."""
    effective_user_id = user_id if (current_user.is_admin and user_id != "root") else current_user.user_id
    thread = await get_or_create_thread(
        db,
        thread_id=thread_id,
        user_id=effective_user_id,
        is_admin=current_user.is_admin,
    )
    raw_assets = await get_thread_assets_with_signed_urls(thread_id, db=db)
    asset_items = [ThreadAssetItem(**a) for a in raw_assets]

    return ThreadDetailsResponse(
        id=thread.id,
        user_id=thread.user_id,
        title=thread.title,
        metadata=thread.metadata_json or {},
        is_pinned=thread.is_pinned,
        is_archived=thread.is_archived,
        assets=asset_items,
    )


@router.get(
    "/{thread_id}/history",
    response_model=ThreadHistoryResponse,
    summary="Retrieve thread conversation history",
    description="Fetches all stored short-term memory messages for a given thread from PostgreSQL.",
)
async def get_history(
    thread_id: str,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ThreadHistoryResponse:
    """Returns the full chronological message history stored in PostgreSQL for the thread."""
    await verify_thread_ownership(
        db,
        thread_id=thread_id,
        user_id=current_user.user_id,
        is_admin=current_user.is_admin,
        allow_missing=True,
    )
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
            detail="Failed to retrieve thread history. Please try again.",
        )


@router.delete(
    "/{thread_id}",
    response_model=DeleteThreadResponse,
    summary="Clear thread memory",
    description="Deletes all persisted checkpoint memory records for a given thread from PostgreSQL.",
)
async def clear_memory(
    thread_id: str,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DeleteThreadResponse:
    """Removes all checkpoint data for the specified thread."""
    await verify_thread_ownership(
        db,
        thread_id=thread_id,
        user_id=current_user.user_id,
        is_admin=current_user.is_admin,
        allow_missing=True,
    )
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
