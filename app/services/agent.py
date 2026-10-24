"""Single-node LangGraph conversational agent with persistent memory.

Implements the architecture specified in AGENTS.md:
- State schema: AgentState with messages, user_id, thread_id
- Graph topology: START -> chat_node -> END
- Persistence: AsyncPostgresSaver checkpointer
- Model Provider: DeepSeek via OpenAI-compatible endpoint
"""

import asyncio
from collections.abc import AsyncGenerator, Sequence
import logging
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from app.core.config import settings
from app.core.llm import DEFAULT_MODEL, get_llm
from app.core.rag_config import RAGPreset, get_rag_config
from app.schemas.chat import StreamEvent
from app.services.memory import get_checkpointer
from app.services.r2_storage import r2_storage_service
from app.services.retriever import hybrid_search
from app.services.thread_service import format_thread_assets_context, get_thread_assets_with_signed_urls

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are Base Agent, an intelligent and helpful AI assistant powered by DeepSeek. "
    "You have persistent conversation memory and recall previous context within this thread."
)


class AgentState(TypedDict, total=False):
    """LangGraph agent state schema matching AGENTS.md with thread asset context."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str
    thread_id: str
    rag_context: str | None
    rag_sources: list[dict[str, Any]]
    thread_assets: list[dict[str, Any]]
    thread_assets_context: str | None


async def retrieve_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Retrieves relevant document passages via hybrid search and fetches uploaded thread assets."""
    configurable = config.get("configurable", {})
    thread_id = state.get("thread_id") or configurable.get("thread_id")
    user_id = state.get("user_id") or configurable.get("user_id", "root")

    # 1. Fetch thread assets (uploaded images, documents) with signed URLs
    thread_assets = []
    thread_assets_context = ""
    if thread_id:
        try:
            thread_assets = await get_thread_assets_with_signed_urls(thread_id)
            thread_assets_context = format_thread_assets_context(thread_assets)
        except Exception as exc:
            logger.warning("Could not fetch thread assets for thread %s: %s", thread_id, exc)

    use_rag = configurable.get("use_rag", False)
    if not use_rag:
        return {
            "rag_context": None,
            "rag_sources": [],
            "thread_assets": thread_assets,
            "thread_assets_context": thread_assets_context,
        }

    messages = state.get("messages", [])
    last_human_msg = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )
    if not last_human_msg:
        return {
            "rag_context": None,
            "rag_sources": [],
            "thread_assets": thread_assets,
            "thread_assets_context": thread_assets_context,
        }

    # Extract text from content (handling plain string or multimodal list of blocks)
    if isinstance(last_human_msg.content, str):
        query_text = last_human_msg.content
    elif isinstance(last_human_msg.content, list):
        query_text = " ".join(
            item.get("text", "")
            for item in last_human_msg.content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    else:
        query_text = str(last_human_msg.content)

    preset = configurable.get("rag_preset") or RAGPreset.BALANCED
    rag_cfg = get_rag_config(preset)

    try:
        # Search documents indexed under this user's user_id
        fused_docs = await hybrid_search(
            query=query_text,
            config=rag_cfg,
            filter_dict={"user_id": user_id},
        )
    except Exception as exc:
        logger.error("Error during RAG retrieval in retrieve_node: %s", exc, exc_info=True)
        return {
            "rag_context": None,
            "rag_sources": [],
            "thread_assets": thread_assets,
            "thread_assets_context": thread_assets_context,
        }

    if not fused_docs:
        return {
            "rag_context": None,
            "rag_sources": [],
            "thread_assets": thread_assets,
            "thread_assets_context": thread_assets_context,
        }

    context_blocks = []
    sources = []
    for idx, doc in enumerate(fused_docs, 1):
        filename = doc.metadata.get("filename") or doc.metadata.get("source") or "Document"
        headers = [v for k, v in doc.metadata.items() if k.startswith("header_")]
        section_str = f" | Section: {' > '.join(headers)}" if headers else ""
        context_blocks.append(f"[Source {idx}: {filename}{section_str}]\n{doc.page_content}")
        sources.append({
            "source_id": idx,
            "filename": filename,
            "section": " > ".join(headers) if headers else None,
            "preview": doc.page_content[:150],
            "metadata": doc.metadata,
        })

    formatted_context = "\n\n---\n\n".join(context_blocks)
    return {
        "rag_context": formatted_context,
        "rag_sources": sources,
        "thread_assets": thread_assets,
        "thread_assets_context": thread_assets_context,
    }


async def chat_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Chat node: receives message history, thread assets, and RAG context, invokes DeepSeek LLM."""
    messages = list(state.get("messages", []))
    configurable = config.get("configurable", {})
    model_name = configurable.get("model") or settings.LLM_MODEL or DEFAULT_MODEL
    rag_context = state.get("rag_context")
    thread_assets_context = state.get("thread_assets_context")

    llm = get_llm(model_name=model_name)

    if llm is None:
        last_human = "Hello"
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                if isinstance(m.content, str):
                    last_human = m.content
                elif isinstance(m.content, list):
                    last_human = " ".join(
                        item.get("text", "")
                        for item in m.content
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
                break

        # Mock / development fallback when DEEPSEEK_API_KEY is not configured
        rag_note = f"\n[RAG Context Applied: {len(state.get('rag_sources', []))} sources]" if rag_context else ""
        assets_note = f"\n[Thread Assets Active: {len(state.get('thread_assets', []))} assets]" if thread_assets_context else ""
        mock_response = (
            f"[Development Mode - No DEEPSEEK_API_KEY configured (Model: {model_name})]: "
            f"I received your message: '{last_human}'.{rag_note}{assets_note} Conversation history has been saved to PostgreSQL."
        )
        return {"messages": [AIMessage(content=mock_response)]}

    # Prepend system prompt
    system_prompt = DEFAULT_SYSTEM_PROMPT
    if thread_assets_context:
        system_prompt += f"\n\n{thread_assets_context}"

    if rag_context:
        system_prompt += (
            "\n\n=== Retrieved Document Context ===\n"
            f"{rag_context}\n"
            "===================================\n"
            "Use the retrieved document context above to answer the user's inquiry accurately. "
            "Cite relevant sources (e.g. [Source 1]) when referencing facts from the context."
        )

    input_messages: list[BaseMessage] = []
    if not messages or not isinstance(messages[0], SystemMessage):
        input_messages.append(SystemMessage(content=system_prompt))
    else:
        existing_sys = messages[0].content
        combined_sys = str(existing_sys)
        if thread_assets_context and "Uploaded Thread Assets" not in combined_sys:
            combined_sys += f"\n\n{thread_assets_context}"
        if rag_context and "Retrieved Document Context" not in combined_sys:
            combined_sys += f"\n\n=== Retrieved Document Context ===\n{rag_context}\n==================================="
        input_messages.append(SystemMessage(content=combined_sys))

    start_idx = 1 if messages and isinstance(messages[0], SystemMessage) else 0
    input_messages.extend(messages[start_idx:])

    response = await llm.ainvoke(input_messages, config=config)
    return {"messages": [response]}


def build_agent_graph() -> StateGraph:
    """Constructs the StateGraph topology (START -> retrieve_node -> chat_node -> END)."""
    builder = StateGraph(AgentState)
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("chat_node", chat_node)
    builder.add_edge(START, "retrieve_node")
    builder.add_edge("retrieve_node", "chat_node")
    builder.add_edge("chat_node", END)
    return builder


def get_agent_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Compiles the StateGraph with the provided or default checkpointer."""
    builder = build_agent_graph()
    return builder.compile(checkpointer=checkpointer)


async def _build_human_message(message: str, images: list[str] | None = None) -> HumanMessage:
    """Constructs a text or multimodal HumanMessage supporting vision input for DeepSeek."""
    if not images:
        return HumanMessage(content=message)

    content: list[dict[str, Any]] = [{"type": "text", "text": message}]
    for img in images:
        url = img
        if not img.startswith(("http://", "https://", "data:")):
            try:
                url = await r2_storage_service.generate_presigned_url(img)
            except Exception:
                url = r2_storage_service.get_public_url(img) or img
        content.append({"type": "image_url", "image_url": {"url": url}})

    return HumanMessage(content=content)


async def execute_agent(
    message: str,
    thread_id: str,
    user_id: str = "root",
    images: list[str] | None = None,
    model: str | None = None,
    use_rag: bool = False,
    rag_preset: str | RAGPreset | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> tuple[dict[str, Any], str]:
    """Executes a single conversational turn with persistent memory in PostgreSQL.

    Args:
        message: The user's input text.
        thread_id: Unique thread identifier for conversation isolation.
        user_id: Unique user identifier (defaults to 'root').
        images: Optional list of image URLs, presigned URLs, or R2 keys for multimodal vision input.
        model: Optional model override (e.g. DeepSeek-V4.1-Flash-1017 or DeepSeek-V4-Pro-0813).
        use_rag: Whether to execute hybrid retrieval across user documents.
        rag_preset: Preset tuning chunk size and fusion weights.
        checkpointer: Optional explicit checkpointer (defaults to global AsyncPostgresSaver).

    Returns:
        A tuple of (updated_agent_state, resolved_model_name).
    """
    if checkpointer is None:
        checkpointer = await get_checkpointer()

    resolved_model = model or settings.LLM_MODEL or DEFAULT_MODEL
    graph = get_agent_graph(checkpointer=checkpointer)

    thread_config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "model": resolved_model,
            "use_rag": use_rag,
            "rag_preset": rag_preset or RAGPreset.BALANCED,
        }
    }

    human_msg = await _build_human_message(message=message, images=images)

    input_payload = {
        "messages": [human_msg],
        "user_id": user_id,
        "thread_id": thread_id,
    }

    result = await graph.ainvoke(input_payload, config=thread_config)
    return result, resolved_model


async def stream_agent_events(
    message: str,
    thread_id: str,
    user_id: str = "root",
    images: list[str] | None = None,
    model: str | None = None,
    use_rag: bool = False,
    rag_preset: str | RAGPreset | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """Streams a conversational turn using LangGraph v3 event streaming with persistent memory.

    Emits structured StreamEvent instances for token deltas, reasoning content, and the
    final completion turn snapshot. All turns are checkpointed to PostgreSQL.

    Args:
        message: The user's input text.
        thread_id: Unique thread identifier for conversation memory isolation.
        user_id: Unique user identifier (defaults to 'root').
        rag_preset: Preset tuning chunk size and fusion weights.
        checkpointer: Optional checkpointer override (defaults to global AsyncPostgresSaver).

    Yields:
        StreamEvent objects (token, reasoning, end, or error).
    """
    if checkpointer is None:
        checkpointer = await get_checkpointer()

    resolved_model = model or settings.LLM_MODEL or DEFAULT_MODEL
    graph = get_agent_graph(checkpointer=checkpointer)

    thread_config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "model": resolved_model,
            "use_rag": use_rag,
            "rag_preset": rag_preset or RAGPreset.BALANCED,
        }
    }

    human_msg = await _build_human_message(message=message, images=images)

    input_payload = {
        "messages": [human_msg],
        "user_id": user_id,
        "thread_id": thread_id,
    }

    queue: asyncio.Queue[tuple[str, Any, str] | None] = asyncio.Queue()

    async def _pump_stream() -> None:
        try:
            stream = await graph.astream_events(input_payload, config=thread_config, version="v3")
            async for msg in stream.messages:
                node_name = getattr(msg, "node", "chat_node")

                async def read_reasoning() -> None:
                    async for r in msg.reasoning:
                        await queue.put(("reasoning", r, node_name))

                async def read_text() -> None:
                    async for d in msg.text:
                        await queue.put(("token", d, node_name))

                await asyncio.gather(read_reasoning(), read_text())

            final_output = await stream.output()
            await queue.put(("final", final_output, "chat_node"))
        except Exception as exc:
            logger.error(
                "Streaming execution failed for thread_id=%s: %s",
                thread_id,
                exc,
                exc_info=True,
            )
            await queue.put(("error", str(exc), "chat_node"))
        finally:
            await queue.put(None)

    producer_task = asyncio.create_task(_pump_stream())

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            kind, val, node = item
            if kind == "token":
                yield StreamEvent.token(delta=val, node=node)
            elif kind == "reasoning":
                yield StreamEvent.reasoning(delta=val, node=node)
            elif kind == "final":
                messages = val.get("messages", []) if val else []
                sources = val.get("rag_sources") if val else None
                last_msg = messages[-1] if messages else None
                response_text = ""
                if last_msg:
                    response_text = getattr(last_msg, "text", None) or str(last_msg.content)
                yield StreamEvent.end(
                    thread_id=thread_id,
                    user_id=user_id,
                    response=response_text,
                    model=resolved_model,
                    history_length=len(messages),
                    sources=sources,
                )
            elif kind == "error":
                yield StreamEvent.error(error=val)
    finally:
        if not producer_task.done():
            producer_task.cancel()
            try:
                await producer_task
            except (asyncio.CancelledError, Exception):
                pass
        else:
            try:
                await producer_task
            except Exception:
                pass

