# Base Agent — Future Enhancements & TODO

This document tracks planned architectural enhancements, future infrastructure upgrades, and optimizations for Base Agent across upcoming development phases.

---

## 🔍 Search & Retrieval Upgrades

### 1. Upgrade Postgres FTS to ParadeDB (`pg_search`) Native BM25
- **Current State**: Using PostgreSQL native Full-Text Search (`tsvector` + GIN index + `plainto_tsquery` / `websearch_to_tsquery` with `ts_rank_cd`). This requires zero extra extensions, is 100% native to Neon, fully transactional, and performs well for 95% of RAG use cases.
- **Future Upgrade Trigger**: If users report that exact keyword lookups (e.g. specific alphanumeric error codes, part numbers, exact code snippets) require true BM25 scoring curves rather than standard TF-IDF `ts_rank`.
- **Implementation in Neon**:
  Neon natively supports the ParadeDB `pg_search` extension:
  ```sql
  -- 1. Enable extension in Neon
  CREATE EXTENSION IF NOT EXISTS pg_search;

  -- 2. Create BM25 index on document_chunks table
  CALL paradedb.create_bm25(
      index_name => 'document_chunks_search_idx',
      table_name => 'document_chunks',
      key_field => 'id',
      text_fields => paradedb.field('content')
  );
  ```
- **Query Pattern in Python / SQLAlchemy**:
  ```python
  stmt = (
      select(DocumentChunk)
      .where(text("document_chunks @@@ :query"))
      .order_by(text("paradedb.score(id) DESC"))
      .limit(10)
  )
  ```

---

### 2. Optional Cross-Encoder Re-Ranker
- Add an optional post-retrieval neural re-ranking step after Reciprocal Rank Fusion (RRF):
  - Local cross-encoder (e.g. `flashrank` for zero-overhead, sub-50ms CPU re-ranking).
  - Hosted API re-ranker (e.g. Cohere Rerank or BGE Reranker).
- Configuration toggle in `app/core/rag_config.py` (`RERANKING_ENABLED=True/False`).

---

### 3. Advanced Multimodal Parsing
- Enable MarkItDown's optional LLM-assisted image description / OCR pipeline for scanned PDFs or images containing complex diagrams.

---

## 📊 Observability & Monitoring (Phase 7 Roadmap)

### 4. LangSmith Tracing & Evaluators
- Fine-grained run tracing for LangGraph nodes (`retrieve_node`, `chat_node`).
- Token cost tracking, latency attribution, and automated LLM output evaluation datasets.

### 5. OpenTelemetry & Prometheus Metrics
- Standard OTel ASGI middleware exposing metrics (`/metrics`) for Prometheus scraping.
- Histograms for LLM generation latency, retrieval timing, and embedding batch durations.

### 6. Structured JSON Logging with Trace Correlation IDs
- Standardized logging format with request trace IDs linking FastAPI HTTP requests, Taskiq background jobs, and LangGraph checkpoints.

### 7. Sentry Error Tracking
- Production exception tracking with unhandled error alerting and stack trace capture.

---

## 🔐 Auth & Platform Features (Upcoming Phases)

### 8. Production Clerk / OIDC Authentication (Phase 1)
- Swap the header-based mock implementation in `app/core/auth.py` (`get_current_user`) for Clerk SDK JWT validation.
- All downstream route signatures (`UserContext = Depends(get_current_user)`) remain 100% stable with zero endpoint refactoring required.

### 9. Dynamic Agent Builder Platform (Phase 2)
- Dynamic graph topology compilation from user-submitted JSON / Pydantic specs.
- Pluggable node registry: `ChatNode`, `RAGRetrieverNode`, `ToolExecutorNode`, `HumanApprovalNode`.
