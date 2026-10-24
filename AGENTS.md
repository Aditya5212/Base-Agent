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
├── TODO.md                     # Roadmap enhancements (ParadeDB BM25 migration)
├── main.py                     # Root development runner (python main.py / uvicorn)
├── app/
│   ├── __init__.py             # Package init & WindowsSelectorEventLoopPolicy setup
│   ├── main.py                 # FastAPI application factory, lifespan, CORS & root route
│   ├── core/                   # Infrastructure & singleton configurations
│   │   ├── __init__.py         # Core exports
│   │   ├── config.py           # Pydantic-settings environment configuration
│   │   ├── database.py         # Async SQLAlchemy engine & session dependency
│   │   ├── llm.py              # DeepSeek LLM configuration, model registry & client factory
│   │   ├── embedding.py        # Decoupled OpenAI embedding factory (1536 dims)
│   │   └── rag_config.py       # Precision vs Recall presets & RRF configuration
│   ├── api/                    # HTTP presentation layer
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py       # API v1 aggregator router
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           ├── health.py   # Health & liveness probes (/health, /ping)
│   │           ├── chat.py     # Conversation turns, streaming (SSE), & memory deletion
│   │           └── documents.py# Document upload, search, and cascade delete
│   ├── models/                 # Database persistence models
│   │   ├── __init__.py
│   │   ├── base.py             # Declarative Base & UTC TimestampMixin
│   │   ├── user.py             # User model (root user auto-seeded)
│   │   ├── thread.py           # Thread & ThreadAsset models
│   │   └── document.py         # Document & DocumentChunk persistence models
│   ├── schemas/                # Request & response Pydantic contracts
│   │   ├── __init__.py
│   │   ├── health.py           # Health response models
│   │   ├── chat.py             # ChatRequest, ChatResponse, StreamEvent, ThreadAssetItem
│   │   ├── storage.py          # StorageObjectListResponse, PresignedUrlRequest
│   │   └── document.py         # DocumentUploadResponse, RAGSearchRequest/Response
│   └── services/               # Business logic & agent workflows
│       ├── __init__.py
│       ├── agent.py            # LangGraph agent topology (retrieve -> chat) & state schema
│       ├── memory.py           # PostgreSQL checkpointer service (AsyncPostgresSaver)
│       ├── thread_service.py   # Thread persistence, root user, and asset management
│       ├── tools.py            # LangChain agent tools (hybrid_document_search)
│       ├── document_loader.py  # Canonical Microsoft MarkItDown parser
│       ├── document_processor.py # Asynchronous RAG ingestion pipeline (BackgroundTasks / Taskiq)
│       ├── text_splitter.py    # Structure-aware token chunking (tiktoken cl100k_base)
│       ├── vector_store.py     # PGVector HNSW vector store service
│       ├── retriever.py        # Hybrid Search (Dense + FTS + RRF)
│       └── r2_storage.py       # Cloudflare R2 / S3 object storage service
├── scripts/
│   ├── test_rag_live.py             # Interactive script testing live RAG + PGVector + FTS + RRF
│   ├── test_deepseek_live.py        # Interactive script testing live DeepSeek + PostgreSQL turns
│   └── test_deepseek_stream_live.py # Interactive script testing live DeepSeek SSE streaming
└── tests/
    ├── __init__.py
    ├── conftest.py             # Pytest session setup & database table auto-initialization
    ├── test_api.py             # API route tests (health, ping, chat, multimodal images)
    ├── test_storage_api.py     # Object storage endpoint tests
    ├── test_config.py          # Settings validation & URL normalization tests
    ├── test_agent.py           # Multi-turn memory & thread isolation tests
    ├── test_llm.py             # DeepSeek model registry & alias resolution tests
    ├── test_document_loader.py # MarkItDown parser tests
    ├── test_text_splitter.py   # Token chunking & header tracking tests
    ├── test_embedding.py       # Embedding factory tests
    ├── test_retriever.py       # RRF fusion logic tests
    ├── test_documents_api.py   # Document upload, search, and delete endpoint tests
    └── test_rag_agent.py       # RAG-augmented agent execution and streaming tests
```

---

## 2. Conversational Agent & RAG Topology

### Overview
The agent is a stateful **LangGraph** workflow that combines conversation turn processing, persistent short-term PostgreSQL checkpointing, and dynamic knowledge retrieval from uploaded user documents.

### State Schema

```python
class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str
    thread_id: str
    rag_context: str | None
    rag_sources: list[dict[str, Any]]
    thread_assets: list[dict[str, Any]]
    thread_assets_context: str | None
```

- **`messages`**: Chronological sequence of messages (`HumanMessage`, `AIMessage`, `SystemMessage`), supporting multimodal text and image content blocks.
- **`user_id`**: Identifier of the current user (scopes document retrieval, thread ownership, and memory; defaults to `"root"`).
- **`thread_id`**: Unique conversation session identifier ensuring memory isolation.
- **`rag_context`**: Formatted passage text retrieved from PostgreSQL for the current turn.
- **`rag_sources`**: Structured metadata for retrieved document chunks (filenames, sections, citations).
- **`thread_assets`**: Structured records of uploaded media files (images, documents) associated with the thread, populated with fresh SigV4 presigned download URLs.
- **`thread_assets_context`**: Formatted prompt context listing all uploaded thread assets and presigned URLs for LLM awareness.

### Graph Topology

```
START ──► [retrieve_node] ──► [chat_node] ──► END
```

- **`retrieve_node`**:
  1. Resolves thread assets from PostgreSQL and generates fresh presigned download URLs for all uploaded images and documents.
  2. Formats `thread_assets_context` for system prompt injection.
  3. Checks `configurable.use_rag`. When false, passes through immediately with thread assets.
  4. Extracts the user's latest message text (handling both raw strings and multimodal content blocks).
  5. Executes hybrid retrieval (dense + sparse + RRF) scoped to the user's documents.
  6. Returns `rag_context`, `rag_sources`, `thread_assets`, and `thread_assets_context`.
- **`chat_node`**:
  1. Reads conversation history from the checkpointer.
  2. Injects system instructions combining knowledge grounding (if `rag_context` is present) and thread asset context with presigned URLs.
  3. Formats human input into OpenAI-compatible multimodal vision blocks (`image_url`) if image URLs or R2 keys are provided.
  4. Invokes the DeepSeek model asynchronously (or streams via SSE).
  5. Appends the model response to state.

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
  - `execute_agent()`: Resumes state, executes nodes, writes new checkpoint.
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

---

## 5. Hybrid Search & Knowledge Retrieval (RAG)

Base Agent implements a production-grade **Hybrid Retrieval** architecture combining dense semantic vector search and sparse keyword search:

### 1. Dense Semantic Search (PGVector + HNSW)
- **Model**: OpenAI `text-embedding-3-small` (1536-dimensional embeddings), decoupled strictly from conversational chat models.
- **Index**: PostgreSQL HNSW cosine distance index (`vector_cosine_ops`) providing sub-millisecond approximate nearest-neighbor retrieval.

### 2. Sparse Keyword Search (PostgreSQL Full-Text Search)
- **Index**: GIN index on `to_tsvector('english', document)`.
- **Query**: Executed via `plainto_tsquery` and ranked with `ts_rank_cd`.
- **Roadmap**: ParadeDB `pg_search` BM25 migration path documented in `TODO.md`.

### 3. Reciprocal Rank Fusion (RRF)
Combines dense and sparse ranked result lists into an optimal fused ranking:
$$\text{RRF}(d) = \frac{w_{\text{dense}}}{k + \text{rank}_{\text{dense}}} + \frac{w_{\text{sparse}}}{k + \text{rank}_{\text{sparse}}}$$

### 4. Precision vs Recall Presets

| Preset | Target Chunk Size | Overlap | Top-K Dense | Top-K Sparse | Use Case |
| ------ | ----------------- | ------- | ----------- | ------------ | -------- |
| `high_precision` | 200 tokens | 40 tokens | 5 | 5 | Exact fact lookup, specific definitions, Q&A |
| `balanced` (Default) | 500 tokens | 100 tokens | 10 | 10 | General question answering and thematic analysis |
| `high_recall` | 1000 tokens | 200 tokens | 20 | 20 | Broad summaries, cross-document synthesis |

---

## 6. Cloudflare R2 & Document Lifecycle Pipeline

### Ingestion Flow
1. **Upload**: User uploads document (`PDF`, `DOCX`, `TXT`, `MD`) via `POST /api/v1/documents/upload`.
2. **Object Storage**: Stored in Cloudflare R2 (S3-compatible) via async `aioboto3`.
3. **Immediate 202 Response**: Document record is initialized with status `PARSING`, and an HTTP 202 Accepted response is returned immediately to prevent client timeouts.
4. **Asynchronous Background Processing**: Dispatched to FastAPI `BackgroundTasks` executing `process_document_pipeline` in an isolated database session (roadmap for production scaling with Taskiq documented in `TODO.md`).
5. **Parsing**: Converted to structured markdown via Microsoft **MarkItDown** (`markitdown`).
6. **Chunking**: Structure-aware splitting using `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter.from_tiktoken_encoder` (`cl100k_base`), preserving section hierarchy.
7. **Embedding & Indexing**: Vectors indexed into `langchain_pg_embedding` in PostgreSQL with HNSW cosine index and PostgreSQL FTS GIN index. Status updated to `INDEXED` or `FAILED`.
8. **Cascade Delete**: When a document is deleted via `DELETE /api/v1/documents/{document_id}`, both R2 storage and all PostgreSQL chunks are purged atomically.

---

## 7. Event Streaming Architecture (SSE & LangChain v3 Protocol)

Real-time token and reasoning event streaming over HTTP via **Server-Sent Events (SSE)** using LangGraph's native `astream_events(..., version="v3")`.

### Wire Format
- `token`: Incremental response text chunk.
- `reasoning`: Incremental reasoning delta (DeepSeek thinking mode).
- `end`: Final turn snapshot containing `response`, `history_length`, and retrieved `sources`.
- `error`: Error payload.

---

## 8. API Endpoints

### Core & Conversation (`/api/v1/chat`)
| Method | Path | Summary | Description |
| ------ | ---- | ------- | ----------- |
| `GET` | `/` | Root | Service info and API status |
| `GET` | `/api/v1/health` | Readiness Probe | Verifies PostgreSQL connectivity |
| `GET` | `/api/v1/ping` | Liveness Probe | Lightweight health check |
| `POST` | `/api/v1/chat` | Send Message | Executes a conversation turn (optional RAG, multimodal images) |
| `POST` | `/api/v1/chat/stream` | Stream Message | Streams real-time tokens & reasoning via SSE with multimodal images |
| `POST` | `/api/v1/chat/{thread_id}/assets` | Upload Asset | Uploads an image or document attachment linked to thread |
| `GET` | `/api/v1/chat/{thread_id}/assets` | List Assets | Lists uploaded images & documents in thread with fresh signed URLs |
| `GET` | `/api/v1/chat/{thread_id}/details` | Thread Details | Retrieves thread metadata, title, and active assets |
| `GET` | `/api/v1/chat/{thread_id}/history` | Get History | Retrieves full conversation history |
| `DELETE` | `/api/v1/chat/{thread_id}` | Clear Memory | Deletes stored checkpoints for a thread |

### Knowledge Base Management (`/api/v1/documents`)
| Method | Path | Summary | Description |
| ------ | ---- | ------- | ----------- |
| `GET` | `/api/v1/documents/stats` | KB Statistics | Aggregated document, chunk, and storage metrics |
| `GET` | `/api/v1/documents` | List Documents | Lists user's documents with search filter and pagination |
| `POST` | `/api/v1/documents/upload` | Upload & Index | Uploads to R2, parses with MarkItDown, chunks & indexes |
| `GET` | `/api/v1/documents/{document_id}` | Document Details | Metadata, status, chunk count, and presigned download URL |
| `GET` | `/api/v1/documents/{document_id}/chunks` | Inspect Chunks | Granular inspection of document text chunks & headers |
| `POST` | `/api/v1/documents/{document_id}/reindex` | Re-index Document | Re-chunks/re-indexes from R2 source with new preset |
| `GET` | `/api/v1/documents/{document_id}/download` | Download Document | Streams raw binary document directly from R2 storage |
| `DELETE` | `/api/v1/documents/{document_id}` | Cascade Delete | Purges from R2, PGVector chunks, and database |
| `POST` | `/api/v1/documents/search` | Direct Hybrid Search | Hybrid retrieval (Dense + FTS + RRF) with preset tuning |

### Object Storage Management (`/api/v1/storage`)
| Method | Path | Summary | Description |
| ------ | ---- | ------- | ----------- |
| `GET` | `/api/v1/storage/stats` | Storage Stats | Bucket metrics, total files, and bytes consumed |
| `GET` | `/api/v1/storage/objects` | List Objects | Lists objects matching prefix with ETags and signed URLs |
| `POST` | `/api/v1/storage/upload` | Direct Upload | Uploads any asset directly to Cloudflare R2 |
| `GET` | `/api/v1/storage/objects/{key}` | Object Details | Metadata, size, content-type, and presigned URL |
| `GET` | `/api/v1/storage/download/{key}` | Download Object | Streams binary file directly from Cloudflare R2 |
| `DELETE` | `/api/v1/storage/objects/{key}` | Delete Object | Deletes object from Cloudflare R2 bucket |
| `POST` | `/api/v1/storage/presigned-url` | Generate Presigned URL | Generates temporary PUT (direct client upload) or GET URL |


---

## 9. Future Agent Builder Architecture (Phase 2)

Dynamic agent workflows defined through configuration API:
- Graph types: `linear`, `cyclic` (ReAct), `multi_agent`.
- Custom node registry: `chat`, `rag_retriever`, `tool_executor`, `human_approval`.

---

## 10. Design Principles

1. **Stateless API, Stateful Graph**: FastAPI endpoints remain stateless; conversation state is persisted across turns via the checkpointer.
2. **Unified Driver Ecosystem**: Single `psycopg3` engine and pool for ORM, LangGraph, and PGVector.
3. **Decoupled Embedding and LLM Providers**: Independent OpenAI embeddings and DeepSeek conversational reasoning.
4. **No In-Memory Production Traps**: Multi-worker persistence via PostgreSQL native FTS and PGVector.
5. **Cross-Platform Compatibility**: Automatic selector loop policy for Windows and standard selector on POSIX.
6. **Async First**: Fully async database queries, network calls, and agent invocations for maximum throughput.
