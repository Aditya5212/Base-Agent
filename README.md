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
│   │   ├── config.py           # Pydantic settings
│   │   ├── database.py         # Async SQLAlchemy engine & session dependency
│   │   ├── llm.py              # DeepSeek model registry & client factory
│   │   ├── embedding.py        # Decoupled OpenAI embedding factory
│   │   └── rag_config.py       # Precision vs Recall presets & RRF config
│   ├── api/                    # HTTP presentation layer
│   │   └── v1/
│   │       ├── router.py       # API v1 aggregator
│   │       └── endpoints/
│   │           ├── health.py   # Health & liveness probes
│   │           ├── chat.py     # Persistent chat, streaming (SSE), & history
│   │           └── documents.py# Document upload, search, and cascade delete
│   ├── models/                 # SQLAlchemy ORM models
│   │   ├── base.py             # Declarative base & timestamps
│   │   └── document.py         # Document & DocumentChunk models
│   ├── schemas/                # Pydantic request & response contracts
│   │   ├── chat.py             # ChatRequest, ChatResponse, StreamEvent
│   │   ├── document.py         # Upload, list, and RAG search contracts
│   │   └── health.py           # Health response schemas
│   └── services/               # Core business logic
│       ├── agent.py            # LangGraph agent topology (retrieve -> chat)
│       ├── memory.py           # PostgreSQL checkpointer (AsyncPostgresSaver)
│       ├── document_loader.py  # Canonical Microsoft MarkItDown parser
│       ├── text_splitter.py    # Structure-aware token chunking
│       ├── vector_store.py     # PGVector HNSW vector store service
│       ├── retriever.py        # Hybrid Search (Dense + FTS + RRF)
│       └── r2_storage.py       # Cloudflare R2 / S3 object storage service
├── scripts/
│   ├── test_rag_live.py             # Live RAG ingestion and hybrid search verification
│   ├── test_deepseek_live.py        # Live DeepSeek multi-turn chat test
│   └── test_deepseek_stream_live.py # Live DeepSeek SSE streaming test
└── tests/                           # Complete pytest suite (39 tests)
```

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) package manager
- PostgreSQL with `vector` extension (e.g. [Neon](https://neon.tech))
- OpenAI API key (for embeddings)
- DeepSeek API key (for conversational agent)

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

# Cloudflare R2 Storage (Optional for local development)
CLOUDFLARE_R2_ACCESS_KEY=your_access_key
CLOUDFLARE_R2_SECRET_KEY=your_secret_key
CLOUDFLARE_R2_BUCKET_NAME=your_bucket
CLOUDFLARE_R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
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
```

## 🗺️ Roadmap

- [x] Project scaffolding with `uv` and FastAPI
- [x] LangGraph conversational agent with persistent PostgreSQL memory
- [x] Real-time Server-Sent Events (SSE) streaming (`POST /api/v1/chat/stream`)
- [x] DeepSeek integration (`deepseek-flash` & `deepseek-v4-pro`)
- [x] Cloudflare R2 object storage service (`aioboto3`)
- [x] Canonical document ingestion with Microsoft MarkItDown (`markitdown`)
- [x] Structure-aware token chunking (`tiktoken` `cl100k_base`)
- [x] PGVector HNSW + PostgreSQL FTS Hybrid Search with Reciprocal Rank Fusion (RRF)
- [x] RAG Precision vs Recall presets (`high_precision`, `balanced`, `high_recall`)
- [x] Document lifecycle endpoints (`POST /upload`, `GET /`, `DELETE /{id}`, `POST /search`)
- [x] RAG-augmented LangGraph conversational agent with source citations (`use_rag=True`)
- [ ] Clerk authentication middleware
- [ ] ParadeDB `pg_search` BM25 extension migration (see `TODO.md`)
- [ ] Agent Builder dynamic configuration API
- [ ] Next.js frontend

