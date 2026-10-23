# Base Agent — Architecture & Agent Design

This document serves as the comprehensive architectural specification for **Base Agent**, detailing the current conversational agent implementation, persistence layer, LLM provider integration, repository structure, and the future Agent Builder platform.

---

## 1. Repository & Codebase Structure

The project follows production-grade FastAPI and LangGraph modular conventions:

```text
Base-Agent/
├── pyproject.toml              # Project dependencies, build system (uv), pytest configuration
├── uv.lock                     # Deterministic dependency lockfile
├── .python-version             # Python version pin (3.12)
├── .env.example                # Documented environment variables template
├── .env                        # Local secret configurations (git-ignored)
├── README.md                   # Project overview, getting started, roadmap
├── AGENTS.md                   # System architecture and agent design specifications
├── main.py                     # Root development runner (python main.py / uvicorn)
├── app/
│   ├── __init__.py             # Package init & WindowsSelectorEventLoopPolicy setup
│   ├── main.py                 # FastAPI application factory, lifespan, CORS & root route
│   ├── core/                   # Infrastructure & singleton configurations
│   │   ├── __init__.py         # Core exports
│   │   ├── config.py           # Pydantic-settings environment configuration
│   │   ├── database.py         # Async SQLAlchemy engine & session dependency
│   │   └── llm.py              # DeepSeek LLM configuration, model registry & client factory
│   ├── api/                    # HTTP presentation layer
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py       # API v1 aggregator router
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           ├── health.py   # Health & liveness probes (/health, /ping)
│   │           └── chat.py     # Conversation turns, thread history & memory deletion
│   ├── models/                 # Database persistence models
│   │   ├── __init__.py
│   │   └── base.py             # Declarative Base & UTC TimestampMixin
│   ├── schemas/                # Request & response Pydantic contracts
│   │   ├── __init__.py
│   │   ├── health.py           # Health response models
│   │   └── chat.py             # ChatRequest, ChatResponse, ThreadHistoryResponse
│   └── services/               # Business logic & agent workflows
│       ├── __init__.py
│       ├── agent.py            # LangGraph agent topology, state schema & execution
│       └── memory.py           # PostgreSQL checkpointer service (AsyncPostgresSaver)
├── scripts/
│   └── test_deepseek_live.py   # Interactive script testing live DeepSeek + PostgreSQL turns
└── tests/
    ├── __init__.py
    ├── test_api.py             # API route tests (health, ping, chat endpoints)
    ├── test_config.py          # Settings validation & URL normalization tests
    ├── test_agent.py           # Multi-turn memory & thread isolation tests
    └── test_llm.py             # DeepSeek model registry & alias resolution tests
```

---

## 2. Current Agent: Single-Node Conversational Agent

### Overview
The initial agent is a single-node **LangGraph** workflow that processes conversational turns with persistent short-term memory. State is automatically committed to PostgreSQL after each graph execution.

### State Schema

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str
    thread_id: str
    # Future Phase 2 additions:
    # rag_context: str
    # tool_calls: list
    # agent_config: dict
```

- **`messages`**: Chronological sequence of messages (`HumanMessage`, `AIMessage`, `SystemMessage`). The `add_messages` reducer handles appending new messages and processing removals.
- **`user_id`**: Identifier of the current authenticated user (prepares for Clerk auth).
- **`thread_id`**: Unique conversation session identifier ensuring memory isolation.

### Graph Topology

```
START ──► [chat_node] ──► END
```

- **`chat_node`**:
  1. Reads conversation history from the checkpointer.
  2. Ensures a system instruction prompt is prepended.
  3. Resolves the requested LLM model (`deepseek-flash` or `deepseek-v4-pro`).
  4. Invokes the model asynchronously.
  5. Appends the model response to the state message list.

---

## 3. Persistent Memory Architecture

### Single Connection Driver Strategy (`psycopg3`)
To avoid running duplicate connection pools and mismatched database drivers, both **SQLAlchemy** and **LangGraph** share the exact same `psycopg3` driver ecosystem:
- **Application ORM**: `postgresql+psycopg://` via async SQLAlchemy engine.
- **LangGraph Checkpointer**: `AsyncPostgresSaver` backed by a dedicated `psycopg_pool.AsyncConnectionPool`.

### PostgreSQL Checkpointer (`AsyncPostgresSaver`)
- **Tables**: `checkpoints`, `checkpoint_blobs`, `checkpoint_writes` (auto-created at startup via `setup_checkpointer()`).
- **Isolation**: Each conversation turn is isolated by `configurable = {"thread_id": "...", "user_id": "..."}`.
- **Operations Supported**:
  - `execute_agent()`: Resumes state, executes node, writes new checkpoint.
  - `get_thread_messages()`: Fetches full message history directly from storage.
  - `delete_thread_memory()`: Drops all checkpoints for a thread upon user request.

---

## 4. LLM Provider Design: DeepSeek

The application connects to **DeepSeek** via its OpenAI-compatible endpoint:

- **Base URL (OpenAI format)**: `https://api.deepseek.com`
- **Base URL (Anthropic format)**: `https://api.deepseek.com/anthropic`

### Supported Models

| Shorthand / API Identifier | Underlying Model Version | Context | Max Output | Vision | Concurrency | Primary Use Case |
| -------------------------- | ------------------------ | ------- | ---------- | ------ | ----------- | ---------------- |
| **`deepseek-flash`** (Default) | `DeepSeek-V4.1-Flash-1017` | 1M | 384K | Yes | 2500 | Fast, high-throughput conversational chatbot |
| **`deepseek-v4-pro`** | `DeepSeek-V4-Pro-0813` | 1M | 384K | No | 500 | Advanced reasoning & complex multi-step tasks |

### Model Resolution
The [resolve_model_name](file:///c:/Users/Aditya/Desktop/Base-Agent/app/core/llm.py) function transparently accepts:
- `deepseek-flash`, `DeepSeek-V4.1-Flash-1017`, `flash` &rarr; `deepseek-flash`
- `deepseek-v4-pro`, `DeepSeek-V4-Pro-0813`, `pro` &rarr; `deepseek-v4-pro`

---

## 5. API Endpoints

| Method | Path | Summary | Description |
| ------ | ---- | ------- | ----------- |
| `GET` | `/` | Root | Service info and API status |
| `GET` | `/api/v1/health` | Readiness Probe | Verifies PostgreSQL connectivity |
| `GET` | `/api/v1/ping` | Liveness Probe | Lightweight health check |
| `POST` | `/api/v1/chat` | Send Message | Executes a conversation turn with memory persistence |
| `GET` | `/api/v1/chat/{thread_id}/history` | Get History | Retrieves full conversation history from PostgreSQL |
| `DELETE` | `/api/v1/chat/{thread_id}` | Clear Memory | Deletes stored checkpoints for a thread |

---

## 6. Future Agent Builder Architecture (Phase 2)

### Concept
The Agent Builder will allow users to define custom agent workflows dynamically through a configuration API. Each user-defined agent becomes a configurable LangGraph instantiated at runtime.

### Agent Configuration Schema (Draft)

```python
class AgentConfig(BaseModel):
    agent_id: str
    name: str
    description: str
    model: str = "deepseek-flash"        # "deepseek-flash" or "deepseek-v4-pro"
    temperature: float = 0.0
    system_prompt: str
    tools: list[str] = []               # Registered tool names
    rag_enabled: bool = False
    rag_config: RAGConfig | None = None
    max_iterations: int = 10
    graph_type: str = "linear"          # "linear", "cyclic", "multi_agent"
```

### Graph Types

| Type | Description | Use Case |
| ---- | ----------- | -------- |
| `linear` | Input &rarr; LLM &rarr; Output | Conversational chatbot |
| `cyclic` | Input &rarr; LLM &rarr; Tool &rarr; LLM loop | ReAct-style agents with function execution |
| `multi_agent` | Supervisor orchestrating multiple sub-agents | Complex multi-stage business workflows |

### Planned Node Registry

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

## 7. Design Principles

1. **Stateless API, Stateful Graph**: FastAPI endpoints remain stateless; conversation state is persisted across turns via the checkpointer.
2. **Unified Driver Ecosystem**: Single `psycopg3` engine and pool for both ORM and LangGraph.
3. **Cross-Platform Compatibility**: Automatic selector loop policy for Windows (`WindowsSelectorEventLoopPolicy`) and standard selector on POSIX.
4. **Configuration over Hardcoding**: Dynamic model switching, base URLs, and parameters driven by environment and request payloads.
5. **Async First**: Fully async database queries, network calls, and agent invocations for maximum throughput.
