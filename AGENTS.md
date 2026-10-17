# Base Agent — Architecture & Agent Design

This document describes the agent architecture for Base Agent, covering the current implementation and the future extensibility model for the Agent Builder platform.

---

## 1. Current Agent: Simple Conversational Bot

### Overview
The initial agent is a single-node LangGraph that handles conversational messages with persistent memory. It uses PostgreSQL as the checkpoint store to maintain conversation state across requests.

### State Schema

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str
    thread_id: str
    # Phase 2 additions:
    # rag_context: str
    # tool_calls: list
    # agent_config: dict
```

### Graph Topology

```
START → [chat_node] → END
```

- **chat_node**: Receives the current message history, invokes the LLM, and appends the response to the message list.

### Persistence
- **Checkpointer**: `AsyncPostgresSaver` (LangGraph native)
- **Storage**: Same PostgreSQL instance as the main application database
- **Key**: Composite key of `(user_id, thread_id)` ensures conversation isolation per user and per chat thread.

---

## 2. Future Agent Builder Architecture

### Concept
The Agent Builder will allow users to define custom agent workflows through a configuration API. Each "agent" becomes a configurable LangGraph that can be instantiated at runtime.

### Agent Configuration Schema (Draft)

```python
class AgentConfig(BaseModel):
    agent_id: str
    name: str
    description: str
    model: str                          # e.g., "gpt-4o", "claude-3"
    temperature: float = 0.0
    system_prompt: str
    tools: list[str] = []               # Registered tool names
    rag_enabled: bool = False
    rag_config: RAGConfig | None = None
    max_iterations: int = 10
    graph_type: str = "linear"          # "linear", "cyclic", "multi_agent"
```

### Graph Types

| Type        | Description                                      | Use Case              |
| ----------- | ------------------------------------------------ | --------------------- |
| `linear`    | Single node, input → LLM → output               | Simple chatbot        |
| `cyclic`    | LLM → Tool → LLM loop until done                | ReAct-style agents    |
| `multi_agent` | Multiple sub-graphs orchestrated by a supervisor | Complex workflows     |

### Extensibility Points

1. **Tools**: Agents can be equipped with callable tools (API calls, database queries, code execution).
2. **RAG Pipelines**: Agents can be attached to knowledge bases using BM25 or LightRAG retrieval.
3. **Sub-Agents**: Agents can spawn child agents for delegated tasks.
4. **Human-in-the-Loop**: Graphs can pause and wait for user approval before proceeding.

---

## 3. Node Registry (Future)

All available nodes will be registered in a central registry. The Agent Builder will reference these by name when constructing graphs.

```python
NODE_REGISTRY = {
    "chat": ChatNode,
    "rag_retriever": RAGRetrieverNode,
    "tool_executor": ToolExecutorNode,
    "human_approval": HumanApprovalNode,
    "summarizer": SummarizerNode,
}
```

---

## 4. Design Principles

1. **Stateless API, Stateful Graph**: FastAPI endpoints remain stateless; all conversation state lives in the LangGraph checkpointer.
2. **Configuration over Code**: New agent behaviors should be achievable through configuration, not code changes.
3. **Async First**: All nodes, tools, and database calls are fully async to maximize throughput.
4. **Observable**: Every agent execution logs to LangSmith for debugging and cost tracking.

