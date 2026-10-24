# Base Agent — Future Enhancements & TODO

This document tracks planned architectural enhancements, future infrastructure upgrades, and optimizations for Base Agent.

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
  # Swap FTS @@ query for ParadeDB's @@@ BM25 operator
  stmt = (
      select(DocumentChunk)
      .where(text("document_chunks @@@ :query"))
      .order_by(text("paradedb.score(id) DESC"))
      .limit(10)
  )
  ```
- **Tradeoffs**:
  - *Pros*: True BM25 term saturation and length normalization scoring curve natively in Postgres.
  - *Cons*: Additional extension dependency to manage instead of standard ANSI SQL `tsvector`.

---

### 2. Optional Cross-Encoder Re-Ranker
- Add an optional post-retrieval neural re-ranking step after Reciprocal Rank Fusion (RRF):
  - Local cross-encoder (e.g. `flashrank` for zero-overhead, sub-50ms CPU re-ranking).
  - Hosted API re-ranker (e.g. Cohere Rerank or BGE Reranker).
- Configuration toggle in `app/core/rag_config.py` (`RERANKING_ENABLED=True/False`).

---

## 📦 Storage & File Processing

### 3. Asynchronous Background Ingestion Pipeline & Distributed Scaling
- **Current State (FastAPI `BackgroundTasks`)**:
  - Implemented via [`app/services/document_processor.py`](file:///c:/Users/Aditya/Desktop/Base-Agent/app/services/document_processor.py) (`process_document_pipeline`).
  - Raw files are uploaded and committed to Cloudflare R2 immediately (<400ms HTTP response with `202 Accepted`).
  - Heavy parsing (`markitdown`), structure-aware token chunking (`tiktoken`), embedding generation (OpenAI `text-embedding-3-small`), and dual-indexing (`PGVector` + PostgreSQL FTS) run asynchronously in the background via FastAPI `BackgroundTasks`.
  - Zero new infrastructure required (no Redis or Celery cluster to manage), completely eliminating HTTP reverse proxy gateway timeouts and UI freezes.
  - Document status updates automatically in PostgreSQL from `PARSING` -> `INDEXED` (or `FAILED` with diagnostics).
- **Future Scaling Upgrade (Taskiq with PostgreSQL/Redis Backend)**:
  - When deploying to production and scaling out to multiple stateless API server containers and dedicated background worker processes:
  - Simply replace:
    ```python
    background_tasks.add_task(process_document_pipeline, ...)
    ```
    with:
    ```python
    await broker.kick(process_document_pipeline, ...)
    ```
    using **Taskiq** with a PostgreSQL or Redis broker backend.
  - The core logic in `process_document_pipeline` remains **exactly the same**! It is already completely modular, runs pure async/await, and manages its own isolated database sessions (`async_session_factory`).

### 4. Advanced Multimodal Parsing
- Enable MarkItDown's optional LLM-assisted image description / OCR pipeline for scanned PDFs or images containing complex diagrams.

---

## 🔐 Auth & Platform Features (Phase 2 Roadmap)

### 5. Clerk Authentication Middleware
- Integrate Clerk JWT verification on protected endpoints (`/api/v1/chat`, `/api/v1/documents`).
- Multi-tenant isolation at the database row level (`user_id` / `org_id`).

### 6. Dynamic Agent Builder Platform
- Dynamic graph topology compilation from user-submitted JSON / Pydantic specs.
- Pluggable node registry: `ChatNode`, `RAGRetrieverNode`, `ToolExecutorNode`, `HumanApprovalNode`.
