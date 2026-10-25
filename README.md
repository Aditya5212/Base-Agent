# Base Agent

A production-grade, extensible AI chatbot built with FastAPI and LangGraph. Starts as a simple conversational agent and is architected to evolve into a full **Agent Builder** platform.

## 🧠 Vision

**Phase 1 (Current):** A simple, persistent chatbot with conversation memory, authentication, and production-ready infrastructure.

**Phase 2 (Future):** An Agent Builder platform where users can define custom agent workflows, tools, RAG pipelines, and multi-step reasoning graphs.

## 🛠️ Tech Stack

| Layer          | Technology                          |
| -------------- | ----------------------------------- |
| Backend        | Python 3.12+, FastAPI               |
| Package Manager| `uv`                                |
| Agent Framework| LangGraph + LangChain               |
| Database / ORM | PostgreSQL + SQLAlchemy (async, psycopg3) |
| Vector Store   | `pgvector` (HNSW Cosine Index) via `langchain-postgres` |
| Sparse Search  | PostgreSQL Native Full-Text Search (GIN `to_tsvector`) |
| Graph RAG      | LightRAG (`PGTableGraphStorage` native tables on Neon) |
| Hybrid Ranking | Reciprocal Rank Fusion (RRF)        |
| Embeddings     | OpenAI `text-embedding-3-small` (1536-dim) |
| LLM Chat Model | DeepSeek (`deepseek-flash` & `deepseek-v4-pro`) |
| Parser         | Microsoft MarkItDown (`markitdown`) |
| Chunking       | Structure-aware token splitter (`tiktoken` `cl100k_base`) |
| Object Storage | Cloudflare R2 (S3-compatible via `aioboto3`) |
| Auth           | Clerk (Phase 2)                     |
| Frontend       | Next.js (Phase 3)                   |

## 📁 Project Structure

```
Base-Agent/
├── pyproject.toml              # Dependencies & pytest configuration
├── uv.lock                     # Deterministic dependency lockfile
├── .env.example                # Documented configuration template
├── README.md                   # Project overview and getting started
├── AGENTS.md                   # Agent and architecture specifications
├── TODO.md                     # Future enhancements (ParadeDB BM25)
├── main.py                     # Development runner
├── app/
│   ├── main.py                 # FastAPI factory & lifespan setup
│   ├── core/                   # Infrastructure configuration
│   │   ├── config.py           # Pydantic settings & postgres environment sync
│   │   ├── database.py         # Async SQLAlchemy engine & session dependency
│   │   ├── llm.py              # DeepSeek model registry & client factory
│   │   ├── embedding.py        # Decoupled OpenAI embedding factory
│   │   ├── rag_config.py       # Precision vs Recall presets & RRF config
│   │   └── rag/                # Pluggable KnowledgeProvider architecture
│   │       ├── protocols.py    # KnowledgeProvider protocol & data contracts
│   │       ├── workspace.py    # Multi-tenant workspace isolation (user, thread, global)
│   │       ├── registry.py     # Provider registry & decorator (@register_provider)
│   │       ├── factory.py      # Workspace-scoped cached provider factory with locks
│   │       └── providers/
│   │           ├── hybrid.py   # PGVector HNSW + PostgreSQL FTS + RRF provider
│   │           └── lightrag_provider.py # LightRAG + PGTableGraphStorage provider
│   ├── api/                    # HTTP presentation layer
│   │   └── v1/
│   │       ├── router.py       # API v1 aggregator
│   │       └── endpoints/
│   │           ├── health.py   # Health & liveness probes
│   │           ├── chat.py     # Persistent chat, streaming (SSE), & history
│   │           ├── documents.py# Knowledge Base management, chunks, reindex, search
│   │           └── storage.py  # Object storage management (upload, presigned URLs, stats)
│   ├── models/                 # SQLAlchemy ORM models
│   │   ├── base.py             # Declarative base & timestamps
│   │   ├── user.py             # User model (root user auto-seeded)
│   │   ├── thread.py           # Thread & ThreadAsset persistence models
│   │   └── document.py         # Document & DocumentChunk models
│   ├── schemas/                # Pydantic request & response contracts
│   │   ├── chat.py             # ChatRequest, ChatResponse, StreamEvent
│   │   ├── document.py         # Knowledge Base documents, chunks, reindex schemas
│   │   ├── health.py           # Health response schemas
│   │   └── storage.py          # Object storage items, presigned URLs, stats schemas
│   └── services/               # Core business logic
│       ├── agent.py            # LangGraph agent topology (retrieve -> chat)
│       ├── memory.py           # PostgreSQL checkpointer (AsyncPostgresSaver)
│       ├── thread_service.py   # Thread metadata, root user, and asset management
│       ├── tools.py            # LangChain agent tool factory (make_search_tool, user-scoped hybrid search)
│       ├── document_loader.py  # Canonical Microsoft MarkItDown parser
│       ├── document_processor.py # Asynchronous RAG ingestion pipeline (BackgroundTasks / Taskiq)
│       ├── text_splitter.py    # Structure-aware token chunking
│       ├── vector_store.py     # PGVector HNSW vector store service
│       ├── retriever.py        # Hybrid Search (Dense + FTS + RRF)
│       └── r2_storage.py       # Cloudflare R2 / S3 object storage service
├── scripts/
│   ├── test_rag_live.py             # Live RAG ingestion and hybrid search verification
│   ├── test_deepseek_live.py        # Live DeepSeek multi-turn chat test
│   ├── test_deepseek_stream_live.py # Live DeepSeek SSE streaming test
│   └── test_lightrag_live.py        # Live LightRAG + PGTableGraphStorage smoke test
└── tests/                           # Complete pytest suite (70 tests)
```

## 🔌 API Endpoints Summary

Interactive Swagger UI is available at `http://localhost:8000/docs`.

### Conversational Agent & Thread Management (`/api/v1/chat`)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service root and health check link |
| `GET` | `/api/v1/health` | Readiness probe (validates PostgreSQL connectivity) |
| `GET` | `/api/v1/ping` | Liveness probe |
| `POST` | `/api/v1/chat` | Multi-turn agent conversation turn (optional RAG, multimodal vision with key authorization) |
| `POST` | `/api/v1/chat/stream` | Server-Sent Events (SSE) streaming (tokens + reasoning) |
| `POST` | `/api/v1/chat/{thread_id}/assets` | Upload an image or document attachment to thread |
| `GET` | `/api/v1/chat/{thread_id}/assets` | List uploaded images & documents in thread with signed URLs |
| `GET` | `/api/v1/chat/{thread_id}/details` | Get thread metadata, ownership, and active assets |
| `GET` | `/api/v1/chat/{thread_id}/history` | Retrieve stored conversation history |
| `DELETE` | `/api/v1/chat/{thread_id}` | Clear PostgreSQL memory checkpoints for thread |

### Knowledge Base Management (`/api/v1/documents`)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/documents/stats` | Aggregated metrics: documents, chunks, bytes used, statuses |
| `GET` | `/api/v1/documents` | List knowledge base documents with search filter and pagination |
| `POST` | `/api/v1/documents/upload` | Upload raw document to R2, parse to Markdown, chunk & index in background |
| `GET` | `/api/v1/documents/{id}` | Document details with retry count, chunk count, status, and download URL |
| `GET` | `/api/v1/documents/{id}/chunks` | Inspect granular token chunks, section headers, and metadata |
| `POST` | `/api/v1/documents/{id}/reindex` | Re-index document from R2 with preset (purges chunks & graph entities) |
| `GET` | `/api/v1/documents/{id}/download` | Stream binary from R2 with `?inline=true` browser preview support |
| `DELETE` | `/api/v1/documents/{id}` | Cascade delete: R2 storage, PGVector embeddings, LightRAG graph & record |
| `POST` | `/api/v1/documents/search` | Direct Hybrid Search (Dense HNSW + FTS GIN + RRF fusion) |

### Object Storage Management (`/api/v1/storage` — Admin Only)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/storage/stats` | Bucket statistics: total files, total bytes, formatted size (Root only) |
| `GET` | `/api/v1/storage/objects` | List objects matching prefix with ETags and download URLs (Root only) |
| `POST` | `/api/v1/storage/upload` | Direct upload under deterministic user namespace (Root only) |
| `GET` | `/api/v1/storage/objects/{key}` | Object metadata, size, MIME type, and download URL (Root only) |
| `GET` | `/api/v1/storage/download/{key}` | Stream binary object directly from Cloudflare R2 (Root only) |
| `DELETE` | `/api/v1/storage/objects/{key}` | Delete object from Cloudflare R2 bucket (Root only) |
| `POST` | `/api/v1/storage/presigned-url` | Generate temporary PUT (upload) or GET (download) signed URL (Root only) |

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) package manager
- PostgreSQL with `vector` extension (e.g. [Neon](https://neon.tech))
- OpenAI API key (for embeddings)
- DeepSeek API key (for conversational agent)
- Cloudflare R2 bucket & API token (for object storage)

### Installation

```bash
# Clone and enter the repository
cd Base-Agent

# Install locked dependencies
uv sync

# Configure environment variables
cp .env.example .env
```

### Environment Variables

Configure `.env`:
```env
# Database
DATABASE_URL=postgresql+psycopg://user:password@host/base_agent

# DeepSeek Chat Models
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_MODEL=deepseek-flash

# OpenAI Embeddings (Decoupled from chat models)
OPENAI_API_KEY=your_openai_key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# Cloudflare R2 Object Storage (S3-compatible)
R2_ACCOUNT_ID=your_cloudflare_account_id
R2_ACCESS_KEY_ID=your_r2_access_key_id
R2_SECRET_ACCESS_KEY=your_r2_secret_access_key
R2_BUCKET_NAME=base-agent
R2_REGION=auto
# Optional CDN URL:
# R2_PUBLIC_BASE_URL=https://files.yourdomain.com
```

### Running the Server

```bash
uv run uvicorn app.main:app --reload
```

Interactive Swagger API documentation is available at:
`http://localhost:8000/docs`

### Running the Automated Tests

```bash
uv run pytest
```

### Running Live Interactive Scripts

```bash
# Test live RAG ingestion, PGVector HNSW, PostgreSQL FTS, and RRF
uv run python -m scripts.test_rag_live

# Test live multi-turn DeepSeek conversation with PostgreSQL memory
uv run python -m scripts.test_deepseek_live

# Test live DeepSeek SSE token and reasoning streaming
uv run python -m scripts.test_deepseek_stream_live

# Test live LightRAG entity/relationship extraction and graph retrieval on Neon
uv run python -m scripts.test_lightrag_live
```

## 🗺️ Roadmap

- [x] Project scaffolding with `uv` and FastAPI
- [x] LangGraph conversational agent with persistent PostgreSQL memory
- [x] Real-time Server-Sent Events (SSE) streaming (`POST /api/v1/chat/stream`)
- [x] DeepSeek integration (`deepseek-flash` & `deepseek-v4-pro`)
- [x] Cloudflare R2 object storage service (`aioboto3` with SigV4)
- [x] Cloudflare R2 object storage management APIs (`/api/v1/storage`: list, upload, presigned URLs, stats)
- [x] Canonical document ingestion with Microsoft MarkItDown (`markitdown`)
- [x] Structure-aware token chunking (`tiktoken` `cl100k_base`)
- [x] PGVector HNSW + PostgreSQL FTS Hybrid Search with Reciprocal Rank Fusion (RRF)
- [x] RAG Precision vs Recall presets (`high_precision`, `balanced`, `high_recall`)
- [x] Knowledge Base management APIs (`/api/v1/documents`: upload, list, chunk inspection, re-indexing, download, stats)
- [x] Asynchronous RAG document ingestion via FastAPI `BackgroundTasks` (with Taskiq migration roadmap in `TODO.md`)
- [x] Multimodal conversational agent with image uploads & vision handling (`images` / `thread_assets`)
- [x] LangChain hybrid search agent tool (`hybrid_document_search`)
- [x] Thread asset tracking with signed URLs and dynamic prompt context injection
- [x] Production DB tables (`users` with auto-seeded root user, `threads`, and `thread_assets`)
- [x] RAG-augmented LangGraph conversational agent with source citations (`use_rag=True`)
- [x] Pluggable RAG architecture (`KnowledgeProvider` protocol, registry, workspace isolation, factory)
- [x] LightRAG integration with `PGTableGraphStorage` (native graph tables on Neon, zero Apache AGE dependency)
- [x] Security hardening: user-scoped tool factory (`make_search_tool`), path-traversal guards, R2 key authorization, sanitized `Content-Disposition` headers, error messages stripped of internal details
- [x] Bug fixes: `user_id` default consistency across all endpoints (`root`), correct pagination `total` count via `COUNT(*)` subquery, atomic-safe cascade delete with R2 failure guard
- [x] Performance: PGVector and StateGraph singleton caches, parallel `asyncio.gather` dense+sparse retrieval, thread-asset 10 MB upload cap with chunked streaming read
- [ ] Clerk authentication middleware
- [ ] ParadeDB `pg_search` BM25 extension migration (see `TODO.md`)
- [ ] Agent Builder dynamic configuration API
- [ ] Next.js frontend
