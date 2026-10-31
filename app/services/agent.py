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
from app.core.rag.factory import get_provider
from app.core.rag.protocols import RetrievalQuery
from app.core.rag.workspace import agent_workspace, user_workspace
from app.core.rag_config import RAGPreset, get_rag_config
from app.schemas.agent import AgentConfig
from app.schemas.chat import StreamEvent
from app.services.memory import get_checkpointer
from app.services.multi_query import multi_query_hybrid_search
from app.services.r2_storage import r2_storage_service
from app.services.retriever import hybrid_search
from app.services.thread_service import format_thread_assets_context, get_thread_assets_with_signed_urls

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are Base Agent, an intelligent and helpful AI assistant powered by DeepSeek. "
    "You have persistent conversation memory and recall previous context within this thread."
)


def _normalize_agent_config(cfg: Any) -> AgentConfig:
    """Safely normalizes raw dicts or AgentConfig instances into a validated AgentConfig."""
    if isinstance(cfg, AgentConfig):
        return cfg
    if isinstance(cfg, dict):
        try:
            return AgentConfig.model_validate(cfg)
        except Exception:
            pass
    return AgentConfig()


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
    agent_id = configurable.get("agent_id")
    agent_config = _normalize_agent_config(configurable.get("agent_config"))

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

    preset = configurable.get("rag_preset")
    if not preset and agent_config and agent_config.knowledge_providers:
        for p in agent_config.knowledge_providers:
            if p.enabled and p.mode:
                preset = p.mode
                break
    preset = preset or RAGPreset.BALANCED
    rag_cfg = get_rag_config(preset)

    retrieval_mode = configurable.get("retrieval_mode")
    if not retrieval_mode and agent_config and agent_config.knowledge_providers:
        for p in agent_config.knowledge_providers:
            if p.enabled:
                if p.name == "lightrag":
                    retrieval_mode = "lightrag"
                    break
                if p.mode in ("multi_query", "deep_research"):
                    retrieval_mode = p.mode
                    break
    retrieval_mode = (retrieval_mode or "hybrid").lower().strip()

    # 1. Knowledge Graph mode
    if retrieval_mode in ("lightrag", "graph", "knowledge_graph"):
        try:
            workspace = user_workspace(user_id)
            rag_provider = await get_provider("lightrag", workspace=workspace)
            graph_res = await rag_provider.query(
                RetrievalQuery(query=query_text, mode="mix", top_k=5)
            )
            if not graph_res.context:
                return {
                    "rag_context": None,
                    "rag_sources": [],
                    "thread_assets": thread_assets,
                    "thread_assets_context": thread_assets_context,
                }
            graph_sources = [
                {
                    "source_id": idx,
                    "document_id": None,
                    "chunk_id": chunk.chunk_id,
                    "filename": "Knowledge Graph",
                    "section": None,
                    "preview": chunk.content[:150],
                    "content": chunk.content,
                    "view_url": None,
                    "download_url": None,
                    "metadata": chunk.metadata,
                }
                for idx, chunk in enumerate(graph_res.chunks, 1)
            ]
            return {
                "rag_context": graph_res.context,
                "rag_sources": graph_sources,
                "thread_assets": thread_assets,
                "thread_assets_context": thread_assets_context,
            }
        except Exception as exc:
            logger.error("Error during LightRAG retrieval in retrieve_node: %s", exc, exc_info=True)
            return {
                "rag_context": None,
                "rag_sources": [],
                "thread_assets": thread_assets,
                "thread_assets_context": thread_assets_context,
            }

    # 2. Multi-Query or Standard Hybrid Search
    try:
        if retrieval_mode in ("multi_query", "multiquery", "multi_query_search"):
            fused_docs = await multi_query_hybrid_search(
                queries=[query_text],
                user_id=user_id,
                preset=preset,
            )
        else:
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
        doc_id = str(doc.metadata.get("document_id") or "")
        chunk_id = str(doc.metadata.get("id") or doc.metadata.get("chunk_id") or f"c_{idx}")
        r2_key = doc.metadata.get("r2_key") or doc.metadata.get("file_key")
        headers = [v for k, v in doc.metadata.items() if k.startswith("header_")]
        section_str = f" | Section: {' > '.join(headers)}" if headers else ""
        context_blocks.append(f"[Source {idx}: {filename}{section_str}]\n{doc.page_content}")

        view_url = None
        download_url = None
        if doc_id:
            view_url = f"/api/v1/documents/{doc_id}/download?inline=true"
            download_url = f"/api/v1/documents/{doc_id}/download"
        elif r2_key:
            try:
                view_url = await r2_storage_service.generate_presigned_url(r2_key, inline=True, filename=filename)
                download_url = await r2_storage_service.generate_presigned_url(r2_key, inline=False, filename=filename)
            except Exception:
                pass

        sources.append({
            "source_id": idx,
            "document_id": doc_id or None,
            "chunk_id": chunk_id,
            "filename": filename,
            "section": " > ".join(headers) if headers else None,
            "preview": doc.page_content[:150],
            "content": doc.page_content,
            "view_url": view_url,
            "download_url": download_url,
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
    agent_config = _normalize_agent_config(configurable.get("agent_config"))

    # Resolve model: configurable override > agent_config.model > settings
    if configurable.get("model"):
        model_name = configurable["model"]
    elif agent_config.model and agent_config.model.model_id:
        model_name = agent_config.model.model_id
    else:
        model_name = settings.LLM_MODEL or DEFAULT_MODEL

    temperature = agent_config.model.temperature if agent_config.model else None
    max_tokens = agent_config.model.max_tokens if agent_config.model else None
    base_url = agent_config.model.base_url if agent_config.model else None

    # Context window slicing based on agent memory configuration
    context_window = agent_config.memory.context_window_messages if agent_config.memory else 20
    if len(messages) > context_window:
        messages = messages[-context_window:]

    rag_context = state.get("rag_context")
    thread_assets_context = state.get("thread_assets_context")

    llm = get_llm(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
    )

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

    # Dynamic system persona prompt from agent_config
    system_prompt = (
        agent_config.identity.system_prompt
        if agent_config and agent_config.identity and agent_config.identity.system_prompt
        else DEFAULT_SYSTEM_PROMPT
    )
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


# ---------------------------------------------------------------------------
# Module-level graph cache: compile the StateGraph once per checkpointer
# identity and reuse it for all subsequent requests.  LangGraph's compiled
# graph is stateless between invocations (all state lives in the checkpointer),
# so sharing a single compiled instance across concurrent requests is safe.
# ---------------------------------------------------------------------------
_graph_cache: dict[int, CompiledStateGraph] = {}


def get_agent_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Returns a cached compiled StateGraph for the given checkpointer.

    The graph is compiled on the first call and reused on all subsequent calls
    with the same checkpointer object, avoiding redundant compilation overhead.
    """
    cache_key = id(checkpointer)  # None maps to id(None), unique per process
    if cache_key not in _graph_cache:
        builder = build_agent_graph()
        _graph_cache[cache_key] = builder.compile(checkpointer=checkpointer)
        logger.debug("StateGraph compiled and cached (checkpointer id=%d)", cache_key)
    return _graph_cache[cache_key]


async def _build_human_message(
    message: str,
    images: list[str] | None = None,
    user_id: str = "root",
    thread_id: str | None = None,
) -> HumanMessage:
    """Constructs a text or multimodal HumanMessage supporting vision input for DeepSeek.

    Validates that any raw R2 keys prevent path traversal and are strictly authorized for
    the specified user_id or thread_id before minting presigned URLs for LLM multimodal ingestion.
    """
    if not images:
        return HumanMessage(content=message)

    content: list[dict[str, Any]] = [{"type": "text", "text": message}]
    for img in images:
        if img.startswith(("http://", "https://", "data:")):
            content.append({"type": "image_url", "image_url": {"url": img}})
            continue

        # Prevent path traversal and malformed keys
        if ".." in img or "\\" in img or img.startswith(("/", "\\")):
            raise ValueError(f"Invalid image key: path traversal sequences are forbidden in '{img}'")

        # Authorization check:
        # 1. Admin/root user can access storage keys
        # 2. Key is scoped to this user: users/{user_id}/...
        # 3. Key is scoped to this thread: threads/{thread_id}/...
        is_authorized = (
            user_id == "root"
            or img.startswith(f"users/{user_id}/")
            or (bool(thread_id) and img.startswith(f"threads/{thread_id}/"))
        )

        # 4. If not matched by prefix, check if key is an active registered asset in this thread
        if not is_authorized and thread_id:
            try:
                from app.core.database import AsyncSessionLocal
                from app.services.thread_service import get_thread_assets

                async with AsyncSessionLocal() as session:
                    assets = await get_thread_assets(session, thread_id)
                    if any(a.r2_key == img for a in assets):
                        is_authorized = True
            except Exception as check_exc:
                logger.warning("Failed checking thread assets for key '%s': %s", img, check_exc)

        if not is_authorized:
            raise ValueError(f"Access denied: storage key '{img}' is not authorized for user '{user_id}'")

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
    retrieval_mode: str = "hybrid",
    agent_id: str | None = None,
    agent_config: AgentConfig | dict[str, Any] | None = None,
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
        retrieval_mode: Retrieval strategy ('hybrid', 'lightrag', 'multi_query').
        agent_id: Optional UUID identifier of target agent.
        agent_config: Optional agent configuration schema or dict.
        checkpointer: Optional explicit checkpointer (defaults to global AsyncPostgresSaver).

    Returns:
        A tuple of (updated_agent_state, resolved_model_name).
    """
    if checkpointer is None:
        checkpointer = await get_checkpointer()

    norm_cfg = _normalize_agent_config(agent_config) if agent_config is not None else None
    if model:
        resolved_model = model
    elif norm_cfg and norm_cfg.model and norm_cfg.model.model_id:
        resolved_model = norm_cfg.model.model_id
    else:
        resolved_model = settings.LLM_MODEL or DEFAULT_MODEL

    graph = get_agent_graph(checkpointer=checkpointer)

    thread_config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "model": resolved_model,
            "use_rag": use_rag,
            "rag_preset": rag_preset or RAGPreset.BALANCED,
            "retrieval_mode": retrieval_mode,
            "agent_id": agent_id,
            "agent_config": norm_cfg,
        }
    }

    human_msg = await _build_human_message(
        message=message,
        images=images,
        user_id=user_id,
        thread_id=thread_id,
    )

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
    retrieval_mode: str = "hybrid",
    agent_id: str | None = None,
    agent_config: AgentConfig | dict[str, Any] | None = None,
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
        retrieval_mode: Retrieval strategy ('hybrid', 'lightrag', 'multi_query').
        agent_id: Optional UUID identifier of target agent.
        agent_config: Optional agent configuration schema or dict.
        checkpointer: Optional checkpointer override (defaults to global AsyncPostgresSaver).

    Yields:
        StreamEvent objects (token, reasoning, end, or error).
    """
    if checkpointer is None:
        checkpointer = await get_checkpointer()

    norm_cfg = _normalize_agent_config(agent_config) if agent_config is not None else None
    if model:
        resolved_model = model
    elif norm_cfg and norm_cfg.model and norm_cfg.model.model_id:
        resolved_model = norm_cfg.model.model_id
    else:
        resolved_model = settings.LLM_MODEL or DEFAULT_MODEL

    graph = get_agent_graph(checkpointer=checkpointer)

    thread_config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "model": resolved_model,
            "use_rag": use_rag,
            "rag_preset": rag_preset or RAGPreset.BALANCED,
            "retrieval_mode": retrieval_mode,
            "agent_id": agent_id,
            "agent_config": norm_cfg,
        }
    }

    human_msg = await _build_human_message(
        message=message,
        images=images,
        user_id=user_id,
        thread_id=thread_id,
    )

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
                    agent_id=agent_id,
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

