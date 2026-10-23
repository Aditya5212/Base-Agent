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
| Database / ORM | PostgreSQL + SQLAlchemy (async)     |
| Auth           | Clerk                               |
| Object Storage | Cloudflare R2 (future)              |
| RAG / Search   | BM25 + LightRAG (future)            |
| Frontend       | Next.js (future)                    |

## 📁 Project Structure

```
base-agent/
├── pyproject.toml
├── .env
├── README.md
├── agents.md
└── app/
    ├── __init__.py
    ├── main.py              # FastAPI entry point
    ├── core/
    │   ├── __init__.py
    │   ├── config.py        # Environment settings
    │   ├── database.py      # Async SQLAlchemy engine
    │   └── auth.py          # Clerk JWT verification
    ├── api/
    │   ├── __init__.py
    │   └── routes.py        # API endpoints
    ├── models/              # SQLAlchemy ORM models
    │   └── __init__.py
    └── services/            # Business logic
        ├── __init__.py
        ├── agent.py         # LangGraph agent definition
        ├── memory.py        # Postgres checkpointer
        └── rag.py           # BM25 / LightRAG (future)
```

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) package manager
- PostgreSQL database (local or hosted)

### Installation

```bash
# Clone and enter the project
cd base-agent

# Install dependencies
uv sync

# Copy and configure environment variables
cp .env.example .env
```

### Environment Variables

```env
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/base_agent

# Auth (Clerk)
CLERK_SECRET_KEY=your_clerk_secret_key
CLERK_FRONTEND_API_URL=your-clerk-domain.clerk.accounts.dev

# LLM
OPENAI_API_KEY=your_openai_key

# Cloudflare R2 (Phase 2)
CLOUDFLARE_R2_ACCESS_KEY=
CLOUDFLARE_R2_SECRET_KEY=
CLOUDFLARE_R2_BUCKET_NAME=
```

### Running the Server

```bash
uv run uvicorn app.main:app --reload
```

API docs will be available at `http://localhost:8000/docs`.

## 🗺️ Roadmap

- [x] Project scaffolding with `uv` and FastAPI
- [x] LangGraph agent skeleton
- [x] Persistent conversation memory (Postgres checkpointer)
- [x] Conversation CRUD endpoints (chat turn, get history, delete thread)
- [ ] Clerk authentication middleware
- [ ] Cloudflare R2 file upload service
- [ ] BM25 + LightRAG retrieval pipeline
- [ ] Agent Builder configuration API
- [ ] Next.js frontend
