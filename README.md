# Base Agent

An extensible AI agent backend and **declarative workflow engine** built with FastAPI and LangGraph. Base Agent combines persistent conversations, agent configuration, tools, and declarative JSON graphs compiled into stateful LangGraph state machines.

Current execution support and outstanding release gates are tracked in [Audit remediation — 2026-11-01](AUDIT_REMEDIATION_2026-11-01.md). The workflow compiler supports eight node types; parallel/subgraph execution remains planned. Code execution is for trusted transformations and is terminated on timeout.

Tests run offline by default. To include database tests, set `TEST_DATABASE_URL` to a **disposable** PostgreSQL/pgvector database and run `pytest --run-integration`; this applies migrations to that test database. Apply the new Alembic revision before using the changed application models. Existing LightRAG workspaces with punctuation or long identities require reindexing under the new workspace names.

---

## 🧠 Platform Vision & Milestones

- **Phase 0.5 (Infrastructure baseline):** Async Alembic migrations, Taskiq PostgreSQL background task queue, development identity and ownership checks, rate limiting, and Docker packaging. Verified authentication and remaining deployment gates are deferred.
- **Phase 1 (Agent Configuration Core & Schemas — ✅ Complete):** Database-persisted custom agents with JSONB configs, auto-seeded default assistant, in-memory LRU config caching, and JSON Schema Draft 2020-12 registry.
- **Phase 2 (Workflow execution baseline):** Eight executable node types, conditional routing, retry/fallback wrappers, isolated persistent checkpoints, approval/resume, payload validation, and graph snapshots. Two additional schema node types (`parallel`, `subgraph`) are reserved and rejected until implemented.
- **Phase 3 (Tool Builder & Knowledge Integration — ✅ Complete):** Dynamic tool builder supporting HTTP/REST APIs, AST-sandboxed Python code execution, and read-only SQL queries with symmetric Fernet encryption at rest; runtime dynamic Pydantic schema compilation and LangChain `BaseTool` registry; per-node knowledge attachment; modular RAG research suite (Exploitation vs Exploration).
- **Phase 3.2 (Observability & Production Monitoring — 🚧 In Progress):** OpenTelemetry distributed tracing, Prometheus metrics exporter, and LLM observability instrumentation.

---

## 🛠️ Tech Stack

| Layer | Technology |
| --- | --- |
| **Backend Framework** | Python 3.12+, FastAPI, Pydantic v2 |
| **Package Manager** | `uv` (deterministic locking with `uv.lock`) |
| **Agent & Workflow Engine** | LangGraph, LangChain, Pregel (`StateGraph`, `interrupt()`, `Command`) |
| **Database & ORM** | PostgreSQL (Neon serverless), SQLAlchemy 2.0 (async), unified `psycopg3` |
| **Migrations** | Alembic (async psycopg3 runner with table inclusion filters) |
| **Task Queue** | Taskiq (0.12.6) + PostgreSQL Broker (`taskiq-pg` 0.2.0) |
| **Vector Database** | `pgvector` (HNSW Cosine Index) via `langchain-postgres` |
| **Sparse Keyword Search** | PostgreSQL Full-Text Search (GIN `to_tsvector('english')`) |
| **Knowledge Graph RAG** | LightRAG (`PGTableGraphStorage` native tables, zero Apache AGE dependency; user & thread workspaces) |
| **Hybrid Ranking** | Reciprocal Rank Fusion (RRF) combining dense & sparse scores |
| **Embeddings** | OpenAI `text-embedding-3-small` (1536 dimensions) |
| **Conversational LLM** | DeepSeek (`deepseek-flash` & `deepseek-v4-pro`) via OpenAI-compatible API |
| **Document Parser** | Microsoft MarkItDown (`markitdown` with `pdfminer.six` PDF support) |
| **Chunking** | Structure-aware token splitter (`tiktoken` `cl100k_base`) |
| **Object Storage** | Cloudflare R2 (S3-compatible via `aioboto3` with SigV4 presigned URLs) |
| **Rate Limiting** | `slowapi` tier-based limiter |
| **Schema Validation** | `jsonschema` (Draft 2020-12) & Pydantic Discriminated Unions |
| **Containerization** | Docker (multi-stage) + Docker Compose (API + Taskiq worker) |

---

## 📁 Repository Structure

```text
Base-Agent/
├── pyproject.toml              # Dependencies & pytest configuration
├── uv.lock                     # Deterministic dependency lockfile
├── .python-version             # Python version pin (3.12)
├── .env.example                # Documented configuration template
├── Dockerfile                  # Production multi-stage Docker build
├── docker-compose.yml          # Multi-container orchestration (api + taskiq worker)
├── docker-compose.override.yml # Local development overrides (hot-reloading & mounts)
├── .dockerignore               # Docker build exclusions
├── alembic.ini                 # Alembic migration configuration
├── README.md                   # Project overview and getting started
├── AGENTS.md                   # Comprehensive architecture and design specifications
├── TODO.md                     # Roadmap enhancements & observability items
├── Vision_and_Roadmap.md       # Long-term vision and phased milestone specification
├── main.py                     # Root development runner
├── alembic/                    # Alembic async migration environment
│   ├── env.py                  # Async migration runner with psycopg3 & model inclusion filters
│   ├── script.py.mako          # Migration template with pgvector support
│   └── versions/               # Versioned migration revision scripts
├── app/
│   ├── main.py                 # FastAPI application factory, lifespan, CORS, rate limiting & routes
│   ├── core/                   # Infrastructure & singleton configurations
│   │   ├── config.py           # Pydantic settings & postgres environment sync
│   │   ├── crypto.py           # Symmetric Fernet encryption & secret masking
│   │   ├── database.py         # Async SQLAlchemy engine, session dependency & auto-migrations
│   │   ├── auth.py             # Swappable UserContext, get_current_user & require_admin
│   │   ├── rate_limit.py       # slowapi Limiter configuration
│   │   ├── taskiq.py           # Taskiq PostgreSQL broker & lifecycle hooks
│   │   ├── llm.py              # DeepSeek model registry & client factory
│   │   ├── embedding.py        # Decoupled OpenAI embedding factory (1536 dims)
│   │   ├── rag_config.py       # Precision vs Recall presets & RRF configuration
│   │   └── rag/                # Pluggable KnowledgeProvider framework
│   │       ├── protocols.py    # KnowledgeProvider protocol & data contracts
│   │       ├── workspace.py    # Multi-tenant workspace isolation (user, thread, global)
│   │       ├── registry.py     # Provider registry & decorator (@register_provider)
│   │       ├── factory.py      # Workspace-scoped cached provider factory with locks
│   │       └── providers/
│   │           ├── hybrid.py   # PGVector HNSW + PostgreSQL FTS + RRF provider
│   │           └── lightrag_provider.py # LightRAG + PGTableGraphStorage provider
│   ├── api/                    # HTTP presentation layer
│   │   └── v1/
│   │       ├── router.py       # API v1 aggregator router
│   │       └── endpoints/
│   │           ├── health.py   # Health & liveness probes (/health, /ping)
│   │           ├── chat.py     # Persistent chat, streaming (SSE), assets, & history
│   │           ├── documents.py# Document upload, search, and cascade delete
│   │           ├── storage.py  # Admin-only raw Cloudflare R2 object storage operations
│   │           ├── agents.py   # Agent Builder CRUD, clone & promotion endpoints
│   │           ├── structured_schemas.py # JSON Schema registration & validation endpoints
│   │           ├── workflows.py# Dynamic Workflow Engine CRUD, compilation & execution endpoints
│   │           └── tools.py    # Dynamic Tool CRUD, test-execution & system tool listing
│   ├── models/                 # Database persistence models
│   │   ├── base.py             # Declarative Base & UTC TimestampMixin
│   │   ├── user.py             # User model (root user auto-seeded)
│   │   ├── thread.py           # Thread & ThreadAsset models
│   │   ├── document.py         # Document & DocumentChunk persistence models
│   │   ├── agent.py            # Agent persistence model with JSONB configuration
│   │   ├── structured_schema.py# Structured JSON schema definition model
│   │   ├── workflow.py         # Workflow & WorkflowRun state persistence models
│   │   └── tool.py             # Dynamic Tool model with JSONB config and input schema
│   ├── schemas/                # Request & response Pydantic contracts
│   │   ├── chat.py             # ChatRequest, ChatResponse, StreamEvent, ThreadAssetItem
│   │   ├── document.py         # Knowledge Base documents, chunks, reindex schemas
│   │   ├── health.py           # Health response schemas
│   │   ├── storage.py          # StorageObjectListResponse, PresignedUrlRequest
│   │   ├── agent.py            # AgentConfig, AgentCreate, AgentUpdate, AgentResponse
│   │   ├── structured_schema.py# StructuredSchemaCreate, StructuredSchemaResponse
│   │   ├── workflow.py         # Discriminated WorkflowNode, WorkflowEdge, execution contracts
│   │   └── tool.py             # Discriminated tool configurations, schemas, and execution payloads
│   ├── services/               # Core business logic
│   │   ├── agent.py            # LangGraph conversational agent (retrieve -> chat)
│   │   ├── agent_service.py    # Agent CRUD, cloning, default resolution & LRU config cache
│   │   ├── schema_service.py   # Structured JSON Schema validation and persistence
│   │   ├── workflow_service.py # Multi-tenant workflow persistence, run lifecycle & execution
│   │   ├── workflow/           # Declarative Workflow Engine & Graph Compiler
│   │   │   ├── state.py        # WorkflowState, template interpolation & safe AST evaluator
│   │   │   ├── validator.py    # Topological validation (START/END connectivity, cycles)
│   │   │   ├── compiler.py     # StateGraph compiler (JSON -> LangGraph StateGraph)
│   │   │   ├── middleware/     # Curated middleware pipeline (ModelRetry, ModelFallback)
│   │   │   └── nodes/          # Node builders (chat, retrieve, conditional, code, human, tool)
│   │   ├── tool_service.py     # Dynamic tool CRUD, secret masking & test execution
│   │   ├── tool_registry.py    # Dynamic tool compiler (BaseTool + runtime Pydantic schema) & cache
│   │   ├── tool_executors/     # Specialized tool execution engines (http, code, database)
│   │   ├── memory.py           # PostgreSQL checkpointer service (AsyncPostgresSaver)
│   │   ├── thread_service.py   # Thread persistence, root user, and asset management
│   │   ├── retrieval_service.py# Centralized RetrievalScope, scoped filtering, dual-layer LightRAG & RRF boosting
│   │   ├── tools.py            # LangChain agent tool factory & backwards-compatible re-exports
│   │   ├── rag_tools.py        # Modular RAG research tools (Exploitation & Exploration)
│   │   ├── multi_query.py      # LLM query expansion & parallel cross-query RRF fusion
│   │   ├── document_loader.py  # Canonical Microsoft MarkItDown parser
│   │   ├── document_processor.py # Asynchronous RAG ingestion pipeline (BackgroundTasks / Taskiq)
│   │   ├── text_splitter.py    # Structure-aware token chunking (tiktoken cl100k_base)
│   │   ├── vector_store.py     # PGVector HNSW vector store service
│   │   ├── retriever.py        # Hybrid Search (Dense + FTS + RRF)
│   │   └── r2_storage.py       # Cloudflare R2 / S3 object storage service
│   └── tasks/                  # Taskiq asynchronous distributed worker tasks
│       ├── __init__.py
│       └── document_tasks.py   # process_document_task worker task
├── scripts/
│   ├── test_rag_live.py             # Live RAG ingestion and hybrid search verification
│   ├── test_deepseek_live.py        # Live DeepSeek multi-turn chat test
│   ├── test_deepseek_stream_live.py # Live DeepSeek SSE streaming test
│   ├── test_lightrag_live.py        # Live LightRAG + PGTableGraphStorage smoke test
│   ├── test_taskiq_live.py          # Live Taskiq dispatch and PostgreSQL message storage
│   ├── test_workflow_live.py        # Live Workflow Engine + Graph Compiler verification
│   ├── test_multi_agent_structured_workflow.py # Live multi-agent advisory council workflow
│   └── test_rag_research_tools_live.py # Live Modular RAG Research Tool Suite verification
└── tests/                           # Offline and explicitly enabled database regression suites
```

---

## 🔌 API Endpoints Summary

Interactive Swagger UI is available at `http://localhost:8000/docs`.

### Conversational Agent & Thread Management (`/api/v1/chat`)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service root and health check link |
| `GET` | `/api/v1/health` | Readiness probe (validates PostgreSQL connectivity) |
| `GET` | `/api/v1/ping` | Liveness probe |
| `POST` | `/api/v1/chat` | Multi-turn agent turn (optional RAG, multimodal vision, dynamic agent resolution) |
| `POST` | `/api/v1/chat/stream` | Server-Sent Events (SSE) streaming (tokens + reasoning) |
| `POST` | `/api/v1/chat/{thread_id}/assets` | Upload an image or document attachment to thread |
| `GET` | `/api/v1/chat/{thread_id}/assets` | List uploaded images & documents in thread with signed URLs |
| `GET` | `/api/v1/chat/{thread_id}/details` | Get thread metadata, ownership, and active assets |
| `GET` | `/api/v1/chat/{thread_id}/history` | Retrieve stored conversation history |
| `DELETE` | `/api/v1/chat/{thread_id}` | Durable cascade clear thread: aborts active ingestion, purges memory/checkpoints, thread-scoped documents in R2/PGVector/LightRAG, and thread media assets (preserves records on external failure for safe retry) |

### Knowledge Base Management (`/api/v1/documents`)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/documents/stats` | Aggregated metrics: documents, chunks, bytes used, statuses |
| `GET` | `/api/v1/documents` | List knowledge base documents with search filter, pagination, and optional `thread_id` filter (`'global'` or thread UUID) |
| `POST` | `/api/v1/documents/upload` | Upload raw document to R2, parse to Markdown, chunk & index in background (supports optional `thread_id` for session-isolated docs) |
| `GET` | `/api/v1/documents/{id}` | Document details with retry count, chunk count, `thread_id`, status, and download URL |
| `GET` | `/api/v1/documents/{id}/chunks` | Inspect granular token chunks, section headers, and metadata |
| `POST` | `/api/v1/documents/{id}/reindex` | Re-index document from R2 with preset (purges chunks & graph entities) |
| `GET` | `/api/v1/documents/{id}/download` | Stream binary from R2 with `?inline=true` browser preview support |
| `DELETE` | `/api/v1/documents/{id}` | Cascade delete: R2 storage, PGVector embeddings, LightRAG graph & record |
| `POST` | `/api/v1/documents/search` | Direct Hybrid Search (Dense HNSW + FTS GIN + RRF fusion) supporting 4 explicit scopes (`global_only`, `thread_only`, `thread_with_global`, `all_user`) and active thread ranking boost |

### Agent Management (`/api/v1/agents`)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/agents/default` | Returns or auto-provisions user's default assistant |
| `GET` | `/api/v1/agents` | Lists user's agents with pagination and search filter |
| `POST` | `/api/v1/agents` | Creates a custom agent with validated `AgentConfig` |
| `GET` | `/api/v1/agents/{agent_id}` | Retrieves agent configuration and metadata by ID |
| `PUT` | `/api/v1/agents/{agent_id}` | Updates agent settings, auto-incrementing version |
| `POST` | `/api/v1/agents/{agent_id}/clone` | Clones configuration into a new non-default agent instance |
| `POST` | `/api/v1/agents/{agent_id}/set-default` | Promotes agent to user's default assistant |
| `DELETE` | `/api/v1/agents/{agent_id}` | Deletes agent (deletion of default agent rejected) |

### Structured Schema Management (`/api/v1/structured-schemas`)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/structured-schemas` | Lists user's registered JSON schemas with search |
| `POST` | `/api/v1/structured-schemas` | Validates against Draft 2020-12 and persists schema |
| `GET` | `/api/v1/structured-schemas/{schema_id}` | Retrieves schema definition and metadata by ID |
| `PUT` | `/api/v1/structured-schemas/{schema_id}` | Updates schema definition, auto-incrementing version |
| `DELETE` | `/api/v1/structured-schemas/{schema_id}` | Deletes registered schema definition |

### Dynamic Workflow Engine & Execution (`/api/v1/workflows`)
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/workflows` | Registers workflow with node & edge graph after topological validation |
| `GET` | `/api/v1/workflows` | Lists user workflows with pagination and search |
| `GET` | `/api/v1/workflows/{workflow_id}` | Retrieves graph topology, nodes, edges, and validation schemas |
| `PUT` | `/api/v1/workflows/{workflow_id}` | Updates workflow graph, re-validates topology, increments version |
| `DELETE` | `/api/v1/workflows/{workflow_id}` | Cascades deletion of workflow and all execution run logs |
| `POST` | `/api/v1/workflows/validate` | Dry-run topological validation of nodes and edges without saving |
| `POST` | `/api/v1/workflows/{workflow_id}/run` | Compiles and executes workflow state graph with persistent checkpointing |
| `GET` | `/api/v1/workflows/{workflow_id}/runs` | Lists historical runs and outcomes for a workflow |
| `GET` | `/api/v1/workflows/runs/{run_id}` | Inspects status, execution traces, node history, and error logs |
| `POST` | `/api/v1/workflows/runs/{run_id}/resume` | Resumes an INTERRUPTED workflow with human decision input |

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

---

## ⚡ Workflow Engine & Node Types Specification

The Workflow Engine (`app/services/workflow/`) translates declarative JSON graphs into LangGraph `StateGraph(WorkflowState)` state machines.

### Supported Node Types:
1. **`start`**: Ingests runtime inputs, validates against optional `input_schema`, and initializes default variables.
2. **`end`**: Terminal exit node. Validates outputs against optional `output_schema` and returns the designated `response_variable`.
3. **`chat`**: Invokes LLM with template-interpolated prompts (`{{variables.name}}`), conversation memory, and `ModelRetry`/`ModelFallback` middlewares.
4. **`retrieve`**: Multi-tenant hybrid knowledge search querying PostgreSQL PGVector HNSW and FTS with citations.
5. **`conditional`**: Dynamic branch routing evaluating AST expressions, variable comparisons, regex, or fallback paths.
6. **`code`**: Sandboxed Python script execution with AST forbidden call/import guards and execution timeouts.
7. **`human_approval`**: Human-in-the-loop review triggering LangGraph `interrupt()`, paused until resumed via API.
8. **`tool_call`**: Invokes built-in tools (`calculator`, `datetime`) or hybrid search with retry and error catching.
9. **`subgraph`**: Invokes another workflow as a child routine.
10. **`parallel`**: Concurrently triggers multiple nodes with `wait_all` merging.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) package manager
- PostgreSQL with `vector` extension (e.g. [Neon](https://neon.tech))
- OpenAI API key (for embeddings)
- DeepSeek API key (for conversational agent & workflow models)
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
```

### Database Migrations (Alembic)

Database schema migrations run automatically on application startup. To manage migrations manually:

```bash
# Run pending migrations
uv run alembic upgrade head

# Generate a new migration
uv run alembic revision --autogenerate -m "description_of_change"

# Check migration status
uv run alembic current
```

### Running the Server

```bash
uv run uvicorn app.main:app --reload
```

Interactive Swagger API documentation is available at `http://localhost:8000/docs`.

### Running the Distributed Background Worker (Taskiq)

```bash
uv run taskiq worker app.core.taskiq:broker app.tasks.document_tasks
```

*(Note: Dispatch failures fall back to in-process `BackgroundTasks`. A successful enqueue does not establish that a worker is running.)*

### Running with Docker Compose

```bash
# Build and run containers
docker compose up --build

# Run in background
docker compose up -d

# View logs
docker compose logs -f
```

---

## 🧪 Testing & Verification

### Running Automated Tests

```bash
uv run pytest
```

Offline result (service-backed tests are explicitly skipped):
```text
129 passed, 81 skipped
```

With `TEST_DATABASE_URL` set to a disposable PostgreSQL/pgvector database, `uv run pytest --run-integration` includes all service-backed tests. The PostgreSQL stabilization run on 2026-11-01 passed **204 tests** before subsequent type corrections and six additional regression tests. See the remediation report for verification scope and migration compatibility notes.

### Running Live Interactive Verification Scripts

```bash
# 1. Test live RAG ingestion, PGVector HNSW, PostgreSQL FTS, and RRF
uv run python scripts/test_rag_live.py

# 2. Test live multi-turn DeepSeek conversation with PostgreSQL memory
uv run python scripts/test_deepseek_live.py

# 3. Test live DeepSeek SSE token and reasoning streaming
uv run python scripts/test_deepseek_stream_live.py

# 4. Test live LightRAG entity/relationship extraction on Neon
uv run python scripts/test_lightrag_live.py

# 5. Test live Taskiq task dispatch and PostgreSQL message storage
uv run python scripts/test_taskiq_live.py

# 6. Test live Workflow Engine with branching routing & checkpointer
uv run python scripts/test_workflow_live.py

# 7. Test live Multi-Agent Advisory Council with Pydantic structured output
uv run python scripts/test_multi_agent_structured_workflow.py
```

---

## 🗺️ Roadmap & Completed Milestones

- [x] **Phase 0: Conversational Core & RAG**
  - [x] Project scaffolding with `uv` and FastAPI
  - [x] LangGraph conversational agent with persistent PostgreSQL memory (`AsyncPostgresSaver`)
  - [x] Real-time Server-Sent Events (SSE) streaming (`POST /api/v1/chat/stream`)
  - [x] DeepSeek integration (`deepseek-flash` & `deepseek-v4-pro`)
  - [x] Cloudflare R2 object storage service (`aioboto3` with SigV4 presigned URLs)
  - [x] Canonical document ingestion with Microsoft MarkItDown (`markitdown`)
  - [x] Structure-aware token chunking (`tiktoken` `cl100k_base`)
  - [x] PGVector HNSW + PostgreSQL FTS Hybrid Search with Reciprocal Rank Fusion (RRF)
  - [x] Multimodal vision agent handling (`images` / `thread_assets`)
  - [x] Pluggable RAG framework (`KnowledgeProvider` protocol & LightRAG integration)
- [x] **Phase 0.5: Production Hardening & Infrastructure**
  - [x] Database migrations via async Alembic with model filters & startup runner
  - [x] Distributed background task queue via Taskiq (`taskiq-pg` PostgreSQL broker)
  - [x] Swappable Mock Auth & multi-tenant isolation with thread & document ownership
  - [x] Rate limiting with `slowapi`
  - [x] Production multi-stage `Dockerfile` and `docker-compose.yml`
- [x] **Phase 1: Agent Configuration Core & Schemas**
  - [x] PostgreSQL `agents` table with JSONB `config` and auto-seeded user default assistants
  - [x] In-memory LRU configuration caching (`(agent_id, version)`)
  - [x] JSON Schema Draft 2020-12 registry (`structured_schemas` table and CRUD)
  - [x] Dynamic agent resolution in conversational chat endpoints
- [x] **Phase 2: Workflow Engine & Graph Compiler**
  - [x] PostgreSQL `workflows` and `workflow_runs` persistence models
  - [x] Polymorphic `WorkflowNode` discriminated union for 10 node types
  - [x] Topological graph validation (`validate_workflow_graph`)
  - [x] `WorkflowCompiler` translating declarative JSON into LangGraph `StateGraph`
  - [x] Safe template interpolation & AST condition evaluation
  - [x] Curated Agent Middleware Pipeline (`ModelRetryMiddleware`, `ModelFallbackMiddleware`)
  - [x] End-to-end multi-agent orchestration and structured JSON synthesis
- [x] **Phase 3: Tool Builder & Knowledge Integration**
  - [x] Custom HTTP/REST API tool builder with auth & response mapping
  - [x] AST-sandboxed Python code execution tools with timeout & whitelist
  - [x] Read-only SQL database querying tools with parameterized inputs
  - [x] Symmetric Fernet encryption for tool secrets at rest & masked payloads in logs/APIs
  - [x] Dynamic per-node knowledge base attachment & modular RAG research suite (hybrid, graph, multi-query)
  - [x] Thread-scoped document upload, dual-layer retrieval scoping, and atomic cascade deletion
  - [x] MarkItDown parser enhancement with full PDF dependency support (`pdfminer.six`)
- [ ] **Phase 4: Advanced Orchestration & Cross-Workflow Subgraphs**
- [ ] **Phase 5: Visual Drag-and-Drop Frontend (Next.js 15 & React Flow)**
