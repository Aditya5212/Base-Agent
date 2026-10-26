# Base Agent Console (Streamlit)

A test console for every Base Agent API endpoint: live chat with streaming, knowledge base management, admin storage, and system health.

## Run

```bash
# 1. Database (pgvector Postgres on host port 5433)
docker compose up -d db

# 2. API — on Windows keep --reload: it makes uvicorn use the selector event loop psycopg needs
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5433/base_agent \
  uv run python -m uvicorn app.main:app --port 8000 --reload --reload-dir app

# 3. Frontend (run from the repository root so .streamlit/config.toml is picked up)
uv run --group frontend python -m streamlit run frontend/streamlit_app.py
```

Open http://localhost:8501. Set `BASE_AGENT_API_URL` to point the console at another API.

Without `DEEPSEEK_API_KEY` the agent answers in mock mode; without `OPENAI_API_KEY` embeddings use a deterministic fake; without R2 credentials files live in per-process memory.

## Layout

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Entrypoint: sidebar (API URL, identity, live status) and navigation |
| `api_client.py` | `BaseAgentClient`, one method per API route, plus the `ENDPOINTS` map |
| `common.py` | Session state, error rendering, formatting helpers |
| `views/chat.py` | Threads, SSE streaming, RAG citations, attachments, history, memory |
| `views/knowledge.py` | KB stats, upload, documents table, chunk inspector, re-index, search |
| `views/storage.py` | Admin bucket browser, upload, download, delete, presigned URLs |
| `views/system.py` | Root, ping, health, and the endpoint → UI map |

## Identity

The API uses mock auth (`app/core/auth.py`). The sidebar's **User ID** and **Role** are sent as `X-User-ID` / `X-User-Role` headers on every request. Browser links (inline previews, citations) carry `?user_id=` instead, since browsers can't set headers.
