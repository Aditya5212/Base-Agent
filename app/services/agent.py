"""Single-node LangGraph conversational agent with persistent memory.

Implements the architecture specified in AGENTS.md:
- State schema: AgentState with messages, user_id, thread_id
- Graph topology: START -> chat_node -> END
- Persistence: AsyncPostgresSaver checkpointer
- Model Provider: DeepSeek via OpenAI-compatible endpoint
"""

from collections.abc import Sequence
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
from app.services.memory import get_checkpointer

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are Base Agent, an intelligent and helpful AI assistant powered by DeepSeek. "
    "You have persistent conversation memory and recall previous context within this thread."
)


class AgentState(TypedDict):
    """LangGraph agent state schema matching AGENTS.md."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str
    thread_id: str


async def chat_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Chat node: receives message history, invokes DeepSeek LLM, appends response."""
    messages = list(state.get("messages", []))
    configurable = config.get("configurable", {})
    model_name = configurable.get("model") or settings.LLM_MODEL or DEFAULT_MODEL

    llm = get_llm(model_name=model_name)

    if llm is None:
        last_human = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
            "Hello",
        )
        # Mock / development fallback when DEEPSEEK_API_KEY is not configured
        mock_response = (
            f"[Development Mode - No DEEPSEEK_API_KEY configured (Model: {model_name})]: "
            f"I received your message: '{last_human}'. Conversation history has been saved to PostgreSQL."
        )
        return {"messages": [AIMessage(content=mock_response)]}

    # Prepend system prompt if not present
    input_messages: list[BaseMessage] = []
    if not messages or not isinstance(messages[0], SystemMessage):
        input_messages.append(SystemMessage(content=DEFAULT_SYSTEM_PROMPT))
    input_messages.extend(messages)

    response = await llm.ainvoke(input_messages, config=config)
    return {"messages": [response]}


def build_agent_graph() -> StateGraph:
    """Constructs the single-node StateGraph topology (START -> chat_node -> END)."""
    builder = StateGraph(AgentState)
    builder.add_node("chat_node", chat_node)
    builder.add_edge(START, "chat_node")
    builder.add_edge("chat_node", END)
    return builder


def get_agent_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Compiles the StateGraph with the provided or default checkpointer."""
    builder = build_agent_graph()
    return builder.compile(checkpointer=checkpointer)


async def execute_agent(
    message: str,
    thread_id: str,
    user_id: str = "default_user",
    model: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> tuple[dict[str, Any], str]:
    """Executes a single conversational turn with persistent memory in PostgreSQL.

    Args:
        message: The user's input text.
        thread_id: Unique thread identifier for conversation isolation.
        user_id: Unique user identifier.
        model: Optional model override (e.g. DeepSeek-V4.1-Flash-1017 or DeepSeek-V4-Pro-0813).
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
        }
    }

    input_payload = {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id,
        "thread_id": thread_id,
    }

    result = await graph.ainvoke(input_payload, config=thread_config)
    return result, resolved_model
