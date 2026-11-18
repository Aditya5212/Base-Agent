# Base Agent — Architectural Roadmap & Enhancements

This document tracks completed milestones, architectural capabilities, and the upcoming phased roadmap for Base Agent.

---

## 🚀 Execution Phases & Priority Order

| Order | Phase | Concrete Deliverables | Status |
|---|---|---|---|
| **Phase 1** | **Thread Knowledge Reliability & Locking Stabilization** | Row-locked thread admission (`FOR SHARE`) and deletion (`FOR UPDATE`), atomic R2 upload rollback on 409, busy lock detection, retry preservation, LightRAG `all_user` rejection, comprehensive integration & offline tests | ✅ **Complete & Verified** |
| **Phase 2** | **Execution Visibility & Control** | Append-only execution events with per-run sequence, automated redaction, state inspection (`GET /runs/{run_id}/state`), manual pause/resume, cancellation, active runtime budgets, worker lease heartbeats & recovery | Recovery corrections implemented; disposable PostgreSQL verification pending |
| **Phase 3 (Next)** | **Versioning Foundation** | Immutable agent, workflow, dynamic tool, and JSON schema revisions; pinned dependencies; version history, diffing, and rollback | ⏳ **Ready to Implement** |
| **Phase 4** | **Builder Frontend MVP** | Next.js agent editor, chat and thread-isolated document management, live dynamic tool testing, workflow run viewer, and React Flow visual graph editor for verified node types | 📅 Planned |
| **Phase 5** | **Advanced Orchestration** | Subgraphs, parallel execution with deterministic merge behavior, error routing branches, and multi-agent supervisor workflows | 📅 Planned |
| **Phase 6** | **Publishing & Deployment Readiness** | Verified production auth/RBAC (Clerk/OIDC), published REST endpoints, API keys, quota management, worker health monitoring, checkpoint retention, and disaster recovery | 📅 Planned |
| **Phase 7** | **Expansion & Optimization** | Webhooks, cron schedules, template marketplace, JSON/YAML import/export, ParadeDB BM25 search, cross-encoder reranking, multimodal OCR parsing | 📅 Planned |

---

## ✅ Completed Milestones

### 1. Thread Knowledge Reliability & Concurrency Serialization (Phase 1)
- **Shared vs Exclusive Row Locks**:
  - `verify_thread_admission(db, thread_id, user_id, for_share=True)` acquires `SELECT ... FOR SHARE` on `Thread`, serializing concurrent uploads and rejecting threads marked `DELETING` with HTTP 409 Conflict.
  - `delete_thread_cascade(db, thread_id, user_id)` acquires `SELECT ... FOR UPDATE` on `Thread`, atomically sets `status = "DELETING"` AND enumerates all documents and media assets within the same locked transaction before committing, closing concurrency windows with in-flight uploads.
  - Eliminated all catch-all exception handlers that retried reads without locks.
- **Atomic Two-Phase R2 Rollback**:
  - If thread admission fails or thread is `DELETING`, document upload (`/api/v1/documents/upload`) and thread asset upload (`/api/v1/chat/{thread_id}/assets`) immediately delete the uploaded R2 object and abort before database record creation.
- **Background Worker & Lock Coordination**:
  - Background document ingestion pipeline (`process_document_pipeline`) detects `DELETING` status at task start and chunk activation boundaries, aborting immediately to `FAILED`.
  - Advisory locks (`document_lock`) coordinate deletion with active workers. If locks are busy or external cleanups fail, thread database records are preserved for safe retry via the `DELETE` endpoint (`check_deleting=False`).
- **Scope Contract Integrity**:
  - `query_graph_rag_dual_layer` and workflow `retrieve` nodes explicitly reject `RetrievalScope.ALL_USER` with `ValueError`, enforcing LightRAG's per-workspace isolation model while supporting cross-thread search via PGVector/FTS hybrid search.
- **Regression Verification**:
  - 170 passed offline tests, 0 failures.
  - 3 disposable PostgreSQL integration tests verifying row-level lock serialization, ingestion termination on `DELETING`, and retry preservation.

### 2. Custom Dynamic Tool Builder Platform (Phase 3 Roadmap Core)
- **Database Persistence**: `tools` table supporting `HTTP`, `CODE`, and `DATABASE` tool types with Fernet-encrypted configuration and credentials.
- **Dynamic Compilation**: `ToolRegistry` caching compiled LangChain `BaseTool` instances with runtime-synthesized Pydantic input schemas.
- **Executors**:
  - `HTTPExecutor`: Async `httpx.AsyncClient` with variable interpolation (`{{variables.name}}`), auth headers, and timeout handling.
  - `CodeExecutor`: Python execution in isolated child processes with AST safety analysis, module whitelist, and process reaping on timeout.
  - `DatabaseExecutor`: Read-only parameterized SQL query execution.
- **Testing & Tool CRUD**: Live testing endpoints (`POST /api/v1/tools/{id}/test`), system tool registration, and agent/workflow tool attachment.

### 3. Execution Visibility & Transactional Lifecycle Control (Phase 2)
- **Increment 1 — History & State Inspection**:
  - `workflow_run_events` table with monotonic per-run sequence numbers (`1, 2, 3...`), unique `invocation_id` per node attempt/loop, and composite unique constraint on `(run_id, sequence)`.
  - Automated compile-time node wrapping (`wrap_node_with_tracing`) capturing inputs, outputs, errors, durations, LLM token counts, and tool invocations.
  - Multi-tier recursive redaction masking sensitive keys (`api_key`, `token`, `password`, `secret`), presigned URLs (`X-Amz-Signature`), and database connection URIs.
  - Cursor-paginated events endpoint (`GET /api/v1/workflows/runs/{run_id}/events?after_sequence=`) and state inspection endpoint (`GET /api/v1/workflows/runs/{run_id}/state`).
- **Increment 2 — Manual Pause/Resume & Cancellation**:
  - Cooperative node boundary checks evaluating `control_request` before node handler execution.
  - Manual pause (`POST /runs/{run_id}/pause`) suspending runs to `PAUSED` via LangGraph checkpointing interrupt.
  - Row-locked resume (`POST /runs/{run_id}/resume`) with concurrency serialization (`SELECT ... FOR UPDATE`), discriminating between manual pause resume and approval resume with deadline checks.
  - Active and inactive cancellation (`POST /runs/{run_id}/cancel`) transitioning running executions at boundaries and inactive runs immediately.
- **Increment 3 — Durable Execution, Timeouts, and Worker Recovery**:
  - Active execution budget tracking (`budget_seconds`, `consumed_seconds`, `execution_started_at`) strictly deducting running slices while ignoring paused/interrupted duration.
  - Boundary timeout enforcement transitioning over-budget runs to `TIMED_OUT` with `timed_out` trace event.
  - Worker lease, heartbeat tracking, and fencing token generation (`worker_id`, `lease_expires_at`, `generation`) preventing stale or zombie workers from mutating runs (`WorkerFencedError`).
  - Stale lease recovery sweeper (`reconcile_stale_workflow_runs`) detecting expired worker leases and transitioning abandoned executions to `FAILED`.
  - Transactional outbox dispatch table (`workflow_dispatches`) ensuring runs are created in `PENDING` and queued reliably with retry backoff (`retry_pending_dispatches`).
  - Asynchronous HTTP 202 Accepted response on `POST /workflows/{id}/run` returning immediately while background workers claim and execute the run.
  - Autonomous short-transaction event persistence with atomic monotonic sequence allocation (`WorkflowRun.next_event_sequence`) ensuring live visibility and zero sequence collisions.
  - Persisted tool invocation table (`workflow_tool_invocations`) recording identity, logical occurrence, and idempotency key (`{run_id}:{node_id}:{invocation_id}:{occurrence}`), allowing safe replay and reuse of succeeded results without duplicate external calls.
  - Decoupled execution metadata propagation injecting `X-Idempotency-Key` headers in `HttpToolExecutor` without polluting validated tool argument schemas.
  - Capabilities advertisement (`GET /api/v1/workflows/capabilities`) reporting supported visibility, control, budget, worker recovery, and dispatch features.
- **Regression & Integration Verification**:
  - 194 passed offline tests, 0 failures across the entire test suite.
  - 8 PostgreSQL integration tests in `tests/integration/test_workflow_execution_recovery.py` verifying duplicate claim mutual exclusion, cross-session pause boundary detection, checkpoint progress preservation on resume, live event visibility, atomic sequence uniqueness under concurrent writers, heartbeat fencing, tool replay idempotency, and outbox dispatch retry recovery.

---

## 🔍 Search & Retrieval Upgrades (Phase 7 Roadmap)

### 3. Upgrade Postgres FTS to ParadeDB (`pg_search`) Native BM25
- **Current State**: Using PostgreSQL native Full-Text Search (`tsvector` + GIN index + `plainto_tsquery` / `websearch_to_tsquery` with `ts_rank_cd`). This requires zero extra extensions, is 100% native to Neon, fully transactional, and performs well for 95% of RAG use cases.
- **Future Upgrade Trigger**: If users report that exact keyword lookups (e.g. specific alphanumeric error codes, part numbers, exact code snippets) require true BM25 scoring curves rather than standard TF-IDF `ts_rank`.
- **Implementation in Neon**:
  ```sql
  CREATE EXTENSION IF NOT EXISTS pg_search;

  CALL paradedb.create_bm25(
      index_name => 'document_chunks_search_idx',
      table_name => 'document_chunks',
      key_field => 'id',
      text_fields => paradedb.field('content')
  );
  ```

---

### 4. Optional Cross-Encoder Re-Ranker
- Add an optional post-retrieval neural re-ranking step after Reciprocal Rank Fusion (RRF):
  - Local cross-encoder (e.g. `flashrank` for zero-overhead, sub-50ms CPU re-ranking).
  - Hosted API re-ranker (e.g. Cohere Rerank or BGE Reranker).
- Configuration toggle in `app/core/rag_config.py` (`RERANKING_ENABLED=True/False`).

---

### 5. Advanced Multimodal Parsing
- Enable MarkItDown's optional LLM-assisted image description / OCR pipeline for scanned PDFs or images containing complex diagrams.

---

## 📊 Observability & Monitoring Details (Execution Visibility & Phase 6)

### 6. LangSmith Tracing & Evaluators
- Fine-grained run tracing for LangGraph nodes (`retrieve_node`, `chat_node`, workflow nodes).
- Token cost tracking, latency attribution, and automated LLM output evaluation datasets.

### 7. OpenTelemetry & Prometheus Metrics
- Standard OTel ASGI middleware exposing metrics (`/metrics`) for Prometheus scraping.
- Histograms for LLM generation latency, retrieval timing, and embedding batch durations.

### 8. Structured JSON Logging with Trace Correlation IDs
- Standardized logging format with request trace IDs linking FastAPI HTTP requests, Taskiq background jobs, and LangGraph checkpoints.

### 9. Sentry Error Tracking
- Production exception tracking with unhandled error alerting and stack trace capture.

---

## 🔐 Auth & Platform Features (Phase 6 Roadmap)

### 10. Production Clerk / OIDC Authentication
- Swap the header-based mock implementation in `app/core/auth.py` (`get_current_user`) for Clerk SDK JWT validation.
- All downstream route signatures (`UserContext = Depends(get_current_user)`) remain 100% stable with zero endpoint refactoring required.
