# Base Agent Builder — Vision & Production Roadmap

> **Document Type:** Product Vision & Engineering Roadmap
> **Project:** Base Agent → Base Agent Builder Platform
> **Status:** Living Document — Updated per phase completion
> **Last Updated:** 2026-11-01

> **Audit correction:** Historical completion labels below describe implementation milestones, not verified release readiness. [Current remediation status](AUDIT_REMEDIATION_2026-11-01.md) and `/api/v1/workflows/capabilities` take precedence for current support. Parallel/subgraph execution, thread/agent/global knowledge ingestion, automatic summarization, and weighted provider fusion remain unavailable and are explicitly rejected. Authentication remains a later deployment milestone.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Product Vision](#2-product-vision)
3. [Current State — What Is Already Built](#3-current-state)
4. [System Architecture Overview](#4-system-architecture-overview)
5. [Technology Stack & Rationale](#5-technology-stack--rationale)
6. [Phase-Wise Implementation Plan](#6-phase-wise-implementation-plan)
   - [Phase 0: Foundation (Complete)](#phase-0-foundation-complete)
   - [Phase 1: Agent Configuration Core](#phase-1-agent-configuration-core)
   - [Phase 2: Workflow Engine & Graph Compiler](#phase-2-workflow-engine--graph-compiler)
   - [Phase 3: Tool Builder & Knowledge Integration](#phase-3-tool-builder--knowledge-integration)
   - [Phase 4: Orchestration & Advanced Primitives](#phase-4-orchestration--advanced-primitives)
   - [Phase 5: Frontend & Visual Editor](#phase-5-frontend--visual-editor)
   - [Phase 6: Publishing, Versioning & API Management](#phase-6-publishing-versioning--api-management)
   - [Phase 7: Production Hardening & Scale](#phase-7-production-hardening--scale)
7. [Detailed Feature Matrix](#7-detailed-feature-matrix)
8. [Production Readiness Checklist](#8-production-readiness-checklist)
9. [Scaling Strategy](#9-scaling-strategy)
10. [Risk Assessment & Mitigations](#10-risk-assessment--mitigations)
11. [Appendix: Glossary](#11-appendix-glossary)

---

## 1. Executive Summary

**Base Agent Builder** is a production-grade, multi-tenant AI orchestration platform that
enables users to visually compose, configure, test, and publish intelligent agents and
multi-step workflows — comparable to Microsoft Copilot Studio, but built on open-source
primitives (LangGraph, LangChain, FastAPI, PostgreSQL) with full code-level extensibility.

The platform evolves from the existing **Base Agent** conversational backend (a
LangGraph-powered chatbot with persistent memory, hybrid RAG, knowledge graph retrieval,
multimodal vision, and Cloudflare R2 document storage) into a **declarative agent
composition engine** where every capability — LLM selection, retrieval strategy, tool
binding, structured output, guardrails, human-in-the-loop approval — is configurable
through a unified API and visual editor.

### Core Value Propositions

| Stakeholder | Value |
|---|---|
| **End Users** | Create custom AI assistants without writing code |
| **Developers** | Extend the platform with custom tools, providers, and node types via a plugin architecture |
| **Operators** | Full observability, cost tracking, version control, and rollback for every deployed agent |
| **Business** | Multi-tenant SaaS-ready architecture with per-user workspace isolation and quota management |

---

## 2. Product Vision

### 2.1 The "Copilot Studio" Parity Goal

Base Agent Builder targets feature parity with Microsoft Copilot Studio across these
capability areas, while adding open-source flexibility:

| Copilot Studio Concept | Base Agent Builder Equivalent |
|---|---|
| Topics | **Workflows** — visual state graphs with conditional branching |
| Entities | **State Variables** — typed global state flowing between nodes |
| Actions | **Tools** — HTTP APIs, custom code, database queries, built-in utilities |
| Knowledge | **Knowledge Providers** — pluggable RAG (PGVector, LightRAG, Hybrid) |
| Generative AI | **LLM Nodes** — configurable model, prompt, structured output |
| Analytics | **Observability** — LangSmith tracing, token cost dashboards, run history |
| Channels | **API Publishing** — REST endpoints, webhooks, scheduled triggers |
| Security Roles | **Multi-tenant RBAC** — Clerk auth, workspace isolation, per-user quotas |

### 2.2 Differentiators Beyond Copilot Studio

1. **True Graph Cycles:** LangGraph enables iterative loops (research → draft → critique
   → rewrite → loop until quality threshold met), not just linear decision trees.
2. **Knowledge Graph RAG:** Native LightRAG integration for multi-hop relational reasoning,
   not just flat vector similarity search.
3. **Code-First Fallback:** Any visual node can be "ejected" into raw Python for
   developers who hit UI limitations.
4. **Human-in-the-Loop as a Primitive:** True graph pausing (`interrupt_before`) with
   state inspection and manual state editing mid-execution.
5. **Open-Source Stack:** No vendor lock-in. Every component (LLM provider, vector DB,
   storage, auth) is swappable via the provider abstraction layer.

---

## 3. Current State

### 3.1 What Is Already Built and Operational

| Component | Status | Details |
|---|---|---|
| FastAPI Backend | ✅ Complete | Async-first, lifespan management, CORS, structured logging |
| LangGraph Agent | ✅ Complete | Linear topology (retrieve → chat), stateful execution |
| Persistent Memory | ✅ Complete | `AsyncPostgresSaver` with `psycopg3` unified driver |
| LLM Integration | ✅ Complete | DeepSeek via OpenAI-compatible API, vision support, SSE streaming |
| Hybrid RAG | ✅ Complete | PGVector HNSW + Postgres FTS + Reciprocal Rank Fusion |
| LightRAG Integration | ✅ Complete | `PGTableGraphStorage` on Neon, context-only mode, workspace isolation |
| Pluggable RAG Framework | ✅ Complete | `KnowledgeProvider` protocol, registry, factory, workspace scoping |
| Document Pipeline | ✅ Complete | MarkItDown parsing, token-aware chunking, background ingestion |
| Cloudflare R2 Storage | ✅ Complete | Async aioboto3, presigned URLs, atomic rollback, cascade delete |
| Stuck-Document Recovery | ✅ Complete | Advisory lock sweeper, 3-retry policy, FAILED status handling |
| Multimodal Vision | ✅ Complete | OpenAI-compatible image_url blocks, thread asset management |
| Thread Management | ✅ Complete | CRUD, asset linking, presigned URL generation, ownership enforcement |
| Mock Auth & Isolation | ✅ Complete | Swappable UserContext, headers & bearer auth, tenant isolation (Clerk in Phase 1) |
| Database Migrations | ✅ Complete | Async Alembic migrations (`alembic/`), auto-run on startup |
| Background Task Queue | ✅ Complete | Taskiq (0.12.6) + PostgreSQL broker (`taskiq-pg` 0.2.0) with dual dispatch |
| Rate Limiting | ✅ Complete | slowapi endpoint protection (`/chat`, `/chat/stream`, `/documents/upload`) |
| Containerization | ✅ Complete | Multi-stage Dockerfile + docker-compose orchestration (api + worker) |
| Agent Builder Core | ✅ Complete | Dynamic agent CRUD, JSONB config, clone & promote, LRU cache |
| JSON Schema Registry | ✅ Complete | Draft 2020-12 validation and persistence for structured agent outputs |
| Workflow Engine | ✅ Complete | Declarative JSON graph compiler to LangGraph StateGraph, 10 node types |
| Multi-Agent Council | ✅ Complete | Specialized evaluation workflows with Pydantic structured output extraction |
| Modular RAG Research Suite | ✅ Complete | Exploitation (Hybrid, LightRAG) & Exploration (MultiQuery) across 3 tiers |
| Tool Builder Platform | ✅ Complete | HTTP, sandboxed Python code, read-only SQL executors with Fernet encryption |
| Dynamic Tool Registry | ✅ Complete | Runtime Pydantic model compilation, LangChain BaseTool cache, knowledge attachment |
| Test Suite | ✅ Complete | 20 test modules covering API, RAG, agents, workflows, tools, storage (148 tests) |
| Interactive Test Scripts | ✅ Complete | Live RAG, DeepSeek, streaming, LightRAG validation scripts |

### 3.2 Technical Debt & Prerequisites for Next Phases

| Item | Priority | Status / Target Phase | Details |
|---|---|---|---|
| Alembic migration setup | High | ✅ Complete (Phase 0.5) | Fully operational async migrations tracking application schema |
| Taskiq + PostgreSQL broker | High | ✅ Complete (Phase 0.5) | Distributed queue with R2 payload decoupling and auto-fallback |
| Mock Auth & Tenant Isolation | High | ✅ Complete (Phase 0.5) | Swappable UserContext, dependency injection, cross-tenant isolation |
| Rate limiting middleware | Medium | ✅ Complete (Phase 0.5) | slowapi rate limiting on chat, streaming, and document endpoints |
| Docker setup | Medium | ✅ Complete (Phase 0.5) | Multi-stage build + docker-compose for api and background worker |
| Clerk auth provider integration | Medium | Phase 1 | Wire official Clerk JWT validation in place of mock provider |
| OpenTelemetry / LangSmith integration | Medium | Phase 7 | LLM tracing, spans, and distributed observability |
| GitHub Actions CI/CD pipeline | Medium | Phase 6 | Automated testing, linting, and automated cloud deployments |

---

## 4. System Architecture Overview

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                                 │
│  Next.js Frontend  │  REST API Consumers  │  Webhook Triggers       │
└──────────┬─────────┴──────────┬───────────┴──────────┬──────────────┘
           │                    │                      │
           ▼                    ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     API GATEWAY LAYER                                │
│  Clerk Auth  │  Rate Limiter  │  CORS  │  Request Validation        │
│  FastAPI Application (Stateless, Horizontally Scalable)             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────────────────┐
│  AGENT ENGINE    │ │  WORKFLOW    │ │  TOOL EXECUTION ENGINE       │
│  (LangGraph)     │ │  COMPILER    │ │  (LangChain Tools)           │
│                  │ │              │ │                              │
│ • Chat Nodes     │ │ • JSON →     │ │ • Built-in Tools             │
│ • Retrieve Nodes │ │   StateGraph │ │ • HTTP/API Tools             │
│ • Tool Nodes     │ │ • Conditional│ │ • Custom Code (Sandboxed)    │
│ • Human-in-Loop  │ │   Routing    │ │ • Database Query Tools       │
│ • Subgraphs      │ │ • State Mgmt │ │                              │
└────────┬─────────┘ └──────┬───────┘ └──────────────┬───────────────┘
         │                  │                        │
         ▼                  ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE & RETRIEVAL LAYER                      │
│                                                                     │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │ Hybrid RAG  │  │ LightRAG Graph  │  │ Future Providers        │ │
│  │ PGVector +  │  │ PGTableGraph    │  │ Pinecone, Weaviate,     │ │
│  │ Postgres FTS│  │ Storage         │  │ Qdrant, Milvus          │ │
│  │ + RRF       │  │                 │  │                         │ │
│  └─────────────┘  └─────────────────┘  └─────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA & STORAGE LAYER                             │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Neon Postgres│  │ Cloudflare R2│  │ Redis (Future)           │  │
│  │ • Checkpoints│  │ • Raw Files  │  │ • Task Queue Broker      │  │
│  │ • Vectors    │  │ • Documents  │  │ • Semantic Cache         │  │
│  │ • FTS Index  │  │ • Assets     │  │ • Rate Limiting          │  │
│  │ • Graph      │  │ • Media      │  │ • Session State          │  │
│  │ • App Data   │  │              │  │                          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow: Agent Execution

```
User Message
    │
    ▼
[Auth Middleware] ──► Verify Clerk JWT ──► Resolve user_id
    │
    ▼
[Agent Resolver] ──► Load AgentConfig from DB ──► Determine workspace
    │
    ▼
[Graph Compiler] ──► Build StateGraph from config ──► Compile with checkpointer
    │
    ▼
[Retrieve Node] ──► Fan-out to knowledge providers ──► RRF fusion ──► rag_context
    │
    ▼
[Chat Node] ──► Inject system prompt + context ──► Invoke LLM ──► Stream response
    │
    ▼
[Checkpointer] ──► Persist state to Postgres ──► Return SSE events to client
```

---

## 5. Technology Stack & Rationale

| Layer | Technology | Rationale |
|---|---|---|
| **Backend Framework** | FastAPI (Python 3.12) | Async-first, Pydantic-native validation, OpenAPI auto-generation |
| **Package Manager** | uv | 10-100x faster than pip, deterministic lockfiles |
| **Agent Framework** | LangGraph + LangChain | State machine orchestration, checkpointing, tool ecosystem |
| **LLM Provider** | DeepSeek (OpenAI-compatible) | Cost-effective, 1M context, vision support. Swappable via config |
| **Embeddings** | OpenAI text-embedding-3-small | 1536 dims, cost-effective, high quality. Decoupled from chat LLM |
| **Primary Database** | Neon Serverless Postgres | Serverless scaling, pgvector, FTS, branch-based testing |
| **Vector Search** | pgvector (HNSW) | Native Postgres, no extra infra, cosine similarity |
| **Full-Text Search** | Postgres FTS (tsvector + GIN) | Native, persistent, no in-memory BM25 traps |
| **Knowledge Graph** | LightRAG (PGTableGraphStorage) | Graph RAG on stock Postgres, no Apache AGE needed |
| **Object Storage** | Cloudflare R2 | S3-compatible, zero egress fees, presigned URLs |
| **Document Parsing** | MarkItDown (Microsoft) | Multi-format → Markdown, lightweight, no API cost |
| **Auth** | Clerk | JWT-based, multi-factor, session management, React SDK |
| **Background Tasks** | FastAPI BackgroundTasks → Taskiq | Phase 1: in-process. Phase 3+: distributed with Redis |
| **Task Queue (Future)** | Taskiq + Redis | Native async, Postgres broker option, no Celery sync issues |
| **Frontend (Future)** | Next.js 15 + Vercel AI SDK | Streaming SSE support, React Server Components |
| **Visual Editor (Future)** | React Flow | Node-based graph editor, customizable handles/edges |
| **Observability** | LangSmith + structlog | LLM tracing, structured logging, cost tracking |
| **Containerization** | Docker + Docker Compose | Reproducible environments, multi-service orchestration |
| **CI/CD** | GitHub Actions | Automated testing, linting, deployment |

---

## 6. Phase-Wise Implementation Plan

---

### Phase 0: Foundation ✅ COMPLETE

**Goal:** Production-grade conversational backend with persistent memory, hybrid RAG,
knowledge graph retrieval, document processing, and multimodal support.

**Deliverables (All Complete):**
- FastAPI application with async lifespan management
- LangGraph agent with `retrieve → chat` topology
- `AsyncPostgresSaver` checkpointer with unified `psycopg3` driver
- DeepSeek LLM integration with vision and SSE streaming
- Pluggable RAG framework (`KnowledgeProvider` protocol)
- Hybrid search provider (PGVector HNSW + Postgres FTS + RRF)
- LightRAG provider (`PGTableGraphStorage`, context-only mode)
- Multi-tenant workspace isolation (`user_workspace`, `thread_workspace`, `agent_workspace`)
- Cloudflare R2 storage with atomic rollback and cascade delete
- MarkItDown document parsing with token-aware chunking
- Background ingestion pipeline with stuck-document recovery
- Thread and asset management with presigned URLs
- Comprehensive test suite (15+ modules)
- Interactive validation scripts

**Production Decisions Made:**
- Unified `psycopg3` driver for SQLAlchemy + LangGraph (no asyncpg/psycopg split)
- `PGTableGraphStorage` over `PGGraphStorage` (Neon lacks Apache AGE)
- `only_need_context=True` for LightRAG (avoid double LLM calls)
- Advisory lock sweeper for stuck document recovery
- Presigned URLs for secure file access (no public bucket)

---

### Phase 1: Agent Configuration Core

**Goal:** Transform the hardcoded linear agent into a configurable, database-driven
agent system where every user has a customizable default agent.

**Duration:** 2 weeks
**Prerequisites:** Phase 0 complete, Alembic migration setup

#### 1.1 Implementation Tasks

| # | Task | Details | Priority |
|---|---|---|---|
| 1.1.1 | Set up Alembic | ✅ Complete (Phase 0.5). Async Alembic migrations operational (`alembic/`) and auto-run on startup. | P0 |
| 1.1.2 | Create `agents` table | ✅ Complete. PostgreSQL `agents` table with JSONB `config`, versioning, default flag, and foreign keys. | P0 |
| 1.1.3 | Create `structured_schemas` table | ✅ Complete. `structured_schemas` table with Draft 2020-12 validation and versioning. | P1 |
| 1.1.4 | Define `AgentConfig` Pydantic schema | ✅ Complete. Full contract with identity, model, knowledge providers, tools, guardrails, memory, and advanced settings. | P0 |
| 1.1.5 | Agent CRUD endpoints | ✅ Complete. `/api/v1/agents` endpoints for create, list, details, update, set-default, and delete. | P0 |
| 1.1.6 | Default agent seeding | ✅ Complete. Auto-provisions and protects default assistant per user (`get_or_create_default_agent`). | P0 |
| 1.1.7 | Agent resolution middleware | ✅ Complete. `/api/v1/chat` and `/api/v1/chat/stream` dynamically resolve agent & config with LRU caching. | P0 |
| 1.1.8 | Agent-to-workspace binding | ✅ Complete. Scoped knowledge isolation via `agent_workspace(user_id, agent_id)`. | P1 |
| 1.1.9 | Config validation | ✅ Complete. Validates JSON Schema and attached knowledge providers against registry on save/update. | P1 |
| 1.1.10 | Agent cloning | ✅ Complete. `POST /api/v1/agents/{id}/clone` duplicates agent configuration into a new version 1 instance. | P2 |

#### 1.2 AgentConfig Schema (Detailed)

```
AgentConfig:
├── identity
│   ├── system_prompt: str (required, min 10 chars)
│   ├── response_format: "text" | "markdown" | "json"
│   └── locale: str (default "English")
├── model
│   ├── provider: "deepseek" | "openai" | "anthropic" | "custom"
│   ├── model_id: str (e.g., "deepseek-flash")
│   ├── temperature: float [0.0 - 2.0]
│   ├── max_tokens: int [1 - 32000]
│   ├── enable_thinking: bool
│   └── base_url: str | None (for custom OpenAI-compatible endpoints)
├── knowledge_providers: list[ProviderSpec]
│   ├── name: "hybrid" | "lightrag"
│   ├── mode: str (provider-specific)
│   ├── weight: float [0.0 - 1.0]
│   ├── top_k: int [1 - 50]
│   └── enabled: bool
├── tools: list[ToolSpec]
│   ├── tool_id: str
│   ├── enabled: bool
│   └── config: dict (tool-specific params)
├── structured_output
│   ├── enabled: bool
│   ├── schema_id: str | None (references structured_schemas table)
│   └── enforce: bool (raise error vs. best-effort)
├── guardrails
│   ├── input_validation: str | None (regex or prompt)
│   ├── output_filtering: str | None (PII redaction rules)
│   └── blocked_topics: list[str]
├── memory
│   ├── context_window_messages: int [1 - 50]
│   ├── enable_long_term: bool
│   └── summarization_threshold: int (messages before summarizing)
└── advanced
    ├── enable_streaming: bool
    ├── max_retries: int
    └── timeout_seconds: int
```

#### 1.3 Production Decisions

| Decision | Rationale |
|---|---|
| Store config as JSON column, not normalized tables | Agent configs are read-heavy, write-light. JSON avoids schema migrations for every new config field. Pydantic validates on read/write. |
| One default agent per user, auto-seeded | Ensures backward compatibility. Existing `/api/v1/chat` continues working by resolving the default agent. |
| Agent config is immutable per version | Updates create a new version. Enables rollback and audit trail. |
| Workspace binding at agent level | Each agent gets its own knowledge workspace. Prevents cross-agent data leakage. |

#### 1.4 Scaling Considerations

- **Config caching:** Cache resolved `AgentConfig` in an in-memory LRU (keyed by agent_id + version). Invalidate on update. Avoids DB read on every chat turn.
- **Config size limit:** Cap JSON config at 64KB to prevent abuse.
- **Migration safety:** Alembic migrations must be backward-compatible. Use `server_default` for new columns.

#### 1.5 Testing Strategy

- Unit: AgentConfig validation (valid/invalid configs, boundary values)
- Unit: Agent CRUD with ownership enforcement
- Integration: Default agent seeding on user creation
- Integration: Chat endpoint resolves correct agent config
- Regression: Existing `/api/v1/chat` behavior unchanged after migration

---

### Phase 2: Workflow Engine & Graph Compiler

**Goal:** Build the declarative workflow engine that compiles JSON graph definitions
into executable LangGraph state machines.

**Duration:** 3 weeks
**Prerequisites:** Phase 1 complete (agents table, config schema)

#### 2.1 Implementation Tasks

| # | Task | Details | Priority |
|---|---|---|---|
| 2.1.1 | Create workflows table | ✅ Complete. PostgreSQL workflows table with nodes/edges JSONB, schemas, versioning, cascade delete. | P0 |
| 2.1.2 | Create workflow_runs table | ✅ Complete. PostgreSQL workflow_runs table with thread_id, status, input/output JSONB, execution log. | P0 |
| 2.1.3 | Define NodeConfig schemas | ✅ Complete. Strongly-typed Pydantic schemas and Discriminated Union for 10 node types. | P0 |
| 2.1.4 | Define EdgeConfig schema | ✅ Complete. WorkflowEdge schema with source, target, condition, and UI label. | P0 |
| 2.1.5 | Build WorkflowCompiler | ✅ Complete. Translates declarative JSON graph into executable LangGraph StateGraph(WorkflowState). | P0 |
| 2.1.6 | Implement Chat Node builder | ✅ Complete. Async handler with variable template interpolation, memory preservation, and middlewares. | P0 |
| 2.1.7 | Implement Retrieve Node builder | ✅ Complete. Knowledge retrieval querying hybrid search, formatted passages, and citations. | P0 |
| 2.1.8 | Implement Conditional Node builder | ✅ Complete. AST safe expression evaluation, variable comparison, regex matching, and default fallback. | P0 |
| 2.1.9 | Implement Tool Call Node builder | ✅ Complete. Tool invocation with input mappings, built-in utilities (calculator, datetime), search tool, retry & error isolation. | P1 |
| 2.1.10 | Workflow CRUD endpoints | ✅ Complete. Create, read, update, delete, list workflows with multi-tenant isolation. | P0 |
| 2.1.11 | Workflow execution endpoint | ✅ Complete. POST /workflows/{id}/run with checkpointer and recursion guards. | P0 |
| 2.1.12 | Run status & resume endpoints | ✅ Complete. GET /workflows/runs/{run_id} and POST /workflows/runs/{run_id}/resume. | P0 |
| 2.1.13 | Graph validation | ✅ Complete. alidate_workflow_graph checks start/end nodes, edge connectivity, conditional targets, reachability. | P1 |
| 2.1.14 | Workflow state schema | ✅ Complete. WorkflowState TypedDict with messages reducer, variables, thread_id, node_history. | P0 |
| 2.1.15 | Curated Agent Middleware Pipeline | ✅ Complete. Integrated ModelRetry, ModelFallback, and CallLimit wrappers with LangGraph. | P0 |

#### 2.2 Node Types Specification

| Node Type | Purpose | Config Fields | LangGraph Primitive |
|---|---|---|---|
| `start` | Entry point. Receives workflow input. | input_schema | `START` edge |
| `end` | Exit point. Returns workflow output. | output_schema | `END` edge |
| `chat` | LLM invocation with configurable prompt | system_prompt, model, enable_streaming | `add_node` with async function |
| `retrieve` | Knowledge retrieval from providers | providers, query_source, top_k | `add_node` with provider fan-out |
| `tool_call` | Execute a registered tool | tool_id, input_mapping, output_variable | `add_node` with tool execution |
| `conditional` | Branch based on state | condition_type, condition_expr, branches, default_branch | `add_conditional_edges` |
| `human_approval` | Pause for human input | message, approval_schema, timeout | `interrupt_before` |
| `code` | Execute custom Python | code, input_variables, output_variable | `add_node` with sandboxed exec |
| `subgraph` | Invoke another workflow/agent | subgraph_id, input_mapping | `add_node` with compiled subgraph |
| `parallel` | Fan-out to multiple nodes | target_nodes, merge_strategy | LangGraph `Send` API |

#### 2.3 WorkflowCompiler Design

```
Input:  { nodes: [...], edges: [...] }
            │
            ▼
    ┌───────────────────┐
    │  1. Parse & Validate │ ──► Reject invalid node types, missing edges
    └───────┬───────────┘
            ▼
    ┌───────────────────┐
    │  2. Build StateGraph │ ──► StateGraph(WorkflowState)
    └───────┬───────────┘
            ▼
    ┌───────────────────┐
    │  3. Add Nodes       │ ──► For each node, call node_builder[type](config)
    └───────┬───────────┘
            ▼
    ┌───────────────────┐
    │  4. Add Edges       │ ──► Static edges + conditional edges
    └───────┬───────────┘
            ▼
    ┌───────────────────┐
    │  5. Compile         │ ──► graph.compile(checkpointer=checkpointer)
    └───────┬───────────┘
            ▼
Output: CompiledStateGraph (executable, checkpointed)
```

#### 2.4 Production Decisions

| Decision | Rationale |
|---|---|
| Store graph as JSON in single column | Graphs are read-compiled, not queried relationally. JSON avoids complex join tables for nodes/edges. |
| Compile on every execution (no cached compiled graphs) | Graphs are mutable. Caching compiled graphs risks stale execution. Compilation is fast (<10ms). |
| Validate graph at save time AND compile time | Save-time validation gives immediate user feedback. Compile-time validation catches edge cases. |
| `recursion_limit` per workflow | Prevents infinite loops. Default 25, configurable per workflow. |
| Run history capped at 100 per workflow | Prevents unbounded table growth. Older runs archived to cold storage. |

#### 2.5 Scaling Considerations

- **Compilation is CPU-light:** JSON parsing + StateGraph construction takes <10ms. No bottleneck.
- **Execution is I/O-bound:** LLM calls and DB queries dominate. Async execution handles concurrency.
- **Checkpoint storage:** Each workflow run creates checkpoints. Implement TTL-based cleanup (e.g., delete checkpoints older than 30 days for non-published workflows).
- **Concurrent execution limit:** Cap concurrent runs per user (e.g., 5) to prevent resource exhaustion.

#### 2.6 Curated Agent Middleware Pipeline (`langchain.agents.middleware`)

To keep the Agent Builder robust and production-grade without unnecessary architectural bloat, we adopt a **curated, high-value subset** of provider-agnostic middlewares from LangChain (`langchain.agents.middleware`). These intercept model and tool executions via standard lifecycle hooks (`wrap_model_call`, `wrap_tool_call`, `before_agent`, `after_agent`).

##### Included Curated Middlewares

| Middleware | Hook Layer | Purpose in Agent Builder | Configuration Binding |
|---|---|---|---|
| **`ModelRetryMiddleware`** | `wrap_model_call` | Automatic exponential backoff on transient HTTP 429/5xx errors from LLM providers | `agent_config.advanced.max_retries` |
| **`ModelFallbackMiddleware`** | `wrap_model_call` | Failover to backup model (e.g. `deepseek-flash`) if primary reasoning model times out | `agent_config.model.fallback_model_id` |
| **`ModelCallLimitMiddleware`** | `before_model` | Guard against infinite recursion or runaway generation costs per session turn | `agent_config.advanced.max_model_calls` |
| **`SummarizationMiddleware`** | `before_agent` | Automatically condenses older conversation history when approaching token thresholds | `agent_config.memory.summarization_threshold` |
| **`ToolErrorMiddleware`** | `wrap_tool_call` | Catches unhandled tool execution exceptions and formats actionable error feedback for LLM self-correction (Phase 3) | `agent_config.tools[*].catch_errors` |
| **`ToolRetryMiddleware`** | `wrap_tool_call` | Automatic retry with backoff for transient HTTP tool network failures (Phase 3) | `agent_config.tools[*].retry_count` |
| **`HumanInTheLoopMiddleware`** | `wrap_tool_call` | Pauses execution using LangGraph `interrupt()` when sensitive actions require user confirmation | `agent_config.tools[*].requires_approval` |
| **`PIIMiddleware`** | `before_agent` / `after_model` | Redacts sensitive PII (emails, phone numbers, credit cards) before sending or saving | `agent_config.guardrails.output_filtering` |

> [!NOTE]
> **Explicitly Excluded Primitives:** Overly complex or niche middlewares (such as `DockerExecutionPolicy`, `CodexSandboxExecutionPolicy`, `LLMToolEmulator`, and `RubricGrading`) are deliberately omitted to avoid heavyweight dependencies, security attack surfaces, and runtime performance penalties.

#### 2.7 Testing Strategy

- Unit: WorkflowCompiler with valid/invalid graphs
- Unit: Each node builder in isolation
- Unit: Curated middleware pipeline (ModelRetry, Summarization, ToolError)
- Unit: Conditional routing logic (JSONPath, variable match)
- Integration: End-to-end workflow execution with 3+ nodes
- Integration: Workflow with conditional branching (verify correct path taken)
- Regression: Existing agent endpoints unaffected

---

### Phase 3: Tool Builder & Knowledge Integration

**Goal:** Enable users to create custom tools (HTTP APIs, code, database queries) and
attach knowledge bases to specific workflow nodes.

**Duration:** 3 weeks
**Prerequisites:** Phase 2 complete (workflow engine, tool_call node)

#### 3.1 Implementation Tasks

| # | Task | Details | Priority | Status |
|---|---|---|---|---|
| 3.1.1 | Create 	ools table | id, user_id, name, description, type, input_schema, output_schema, config | P0 | ✅ Complete |
| 3.1.2 | Define ToolType enum | uilt_in, pi, code, database | P0 | ✅ Complete |
| 3.1.3 | Build ToolRegistry | Dynamic LangChain tool creation from Tool models. Cache compiled tools. | P0 | ✅ Complete |
| 3.1.4 | HTTP/API tool executor | Async httpx client with auth, retries, timeout, response mapping | P0 | ✅ Complete |
| 3.1.5 | Code tool executor | Sandboxed Python execution (AST safety analysis & module whitelist) | P1 | ✅ Complete |
| 3.1.6 | Database tool executor | Read-only SQL execution against connected databases | P2 | ✅ Complete |
| 3.1.7 | Tool CRUD endpoints | Create, read, update, delete, list, test-execute | P0 | ✅ Complete |
| 3.1.8 | Built-in tool library | Calculator, modular RAG research suite (hybrid, LightRAG, multi-query) | P1 | ✅ Complete |
| 3.1.9 | Knowledge attachment API | POST /workflows/{id}/nodes/{node_id}/knowledge — attach provider config to retrieve node | P0 | ✅ Complete |
| 3.1.10 | Retrieval strategy config | Per-node provider selection, top_k, reranking toggle, metadata filters | P0 | ✅ Complete |
| 3.1.11 | Migrate BackgroundTasks to Taskiq | Replace FastAPI BackgroundTasks with Taskiq + Redis for distributed processing | P1 | Pending |
| 3.1.12 | Tool execution timeout | Per-tool timeout with graceful error handling | P1 | ✅ Complete |
| 3.1.13 | Tool result caching | Cache tool results for identical inputs (TTL-based) | P2 | Pending |
| 3.1.14 | Tool Execution Middleware | Hook ToolErrorMiddleware and ToolRetryMiddleware into dynamic tool node invocation | P0 | ✅ Complete |

#### 3.2 Tool Type Specifications

**HTTP/API Tool:**
```
Config:
├── url: str (supports variable interpolation: "https://api.example.com/{{order_id}}")
├── method: "GET" | "POST" | "PUT" | "DELETE"
├── headers: dict (supports auth tokens, API keys)
├── auth_type: "none" | "bearer" | "basic" | "api_key"
├── auth_config: dict (token, username/password, key_name/key_value)
├── body_template: dict | str (JSON body with variable interpolation)
├── timeout_seconds: int (default 30)
├── retry_policy: { max_retries: int, backoff_seconds: float }
└── response_mapping: { output_variable: str, json_path: str }
```

**Code Tool:**
```
Config:
├── code: str (Python function body)
├── input_schema: JSON Schema (parameter definitions)
├── output_schema: JSON Schema (return type definition)
├── allowed_imports: list[str] (whitelist: json, math, datetime, re)
├── max_execution_seconds: int (default 10)
└── sandbox_level: "restricted" | "subprocess"
```

**Database Tool:**
```
Config:
├── connection_id: str (references saved DB connections)
├── query: str (SQL with parameterized placeholders)
├── parameters: list[{name, type, source_variable}]
├── read_only: bool (always true for safety)
└── max_rows: int (default 100)
```

#### 3.3 Production Decisions

| Decision | Rationale |
|---|---|
| HTTP tools use `httpx` (not `requests`) | Async-native, connection pooling, HTTP/2 support |
| Code tools run in subprocess, not `exec()` | Security isolation. Prevents access to filesystem, network, env vars. |
| Database tools are read-only | Prevents accidental data mutation. Write operations require explicit admin approval flow. |
| Tool secrets stored encrypted | API keys and auth tokens encrypted at rest using Fernet symmetric encryption. Never logged. |
| Taskiq over Celery | Native async, no sync/async bridge overhead, Postgres broker option available |

#### 3.4 Scaling Considerations

- **Tool execution concurrency:** Limit concurrent tool executions per workflow run (e.g., 3 parallel tool calls).
- **HTTP connection pooling:** Reuse `httpx.AsyncClient` across tool executions. Pool size = 20.
- **Code sandbox resource limits:** CPU time limit (10s), memory limit (256MB), no network access.
- **Taskiq worker scaling:** Auto-scale workers based on queue depth. Minimum 2, maximum 10.

#### 3.5 Testing Strategy

- Unit: ToolRegistry creates valid LangChain tools from each ToolType
- Unit: HTTP tool executor with mock server (success, timeout, auth failure)
- Unit: Code tool sandboxing (verify no filesystem/network access)
- Integration: Workflow with tool_call node executing an HTTP tool
- Integration: Knowledge attachment and retrieval within workflow
- Security: Code tool cannot import `os`, `subprocess`, `socket`

---

### Phase 4: Orchestration & Advanced Primitives

**Goal:** Enable multi-agent coordination, human-in-the-loop, subgraphs, parallel
execution, and structured output enforcement.

**Duration:** 3 weeks
**Prerequisites:** Phase 3 complete (tool builder, knowledge integration)

#### 4.1 Implementation Tasks

| # | Task | Details | Priority |
|---|---|---|---|
| 4.1.1 | Human-in-the-Loop node | Implement `interrupt_before` with approval schema, timeout, resume API | P0 |
| 4.1.2 | Resume endpoint | `POST /workflows/runs/{run_id}/resume` with approval_data | P0 |
| 4.1.3 | Subgraph node | Invoke another workflow/agent as a node. Map parent state → child input. | P1 |
| 4.1.4 | Parallel execution node | Fan-out to N nodes using LangGraph `Send` API. Fan-in with merge strategy. | P1 |
| 4.1.5 | Structured output enforcement | Apply `.with_structured_output()` to chat nodes. Validate against schema. | P0 |
| 4.1.6 | Multi-agent coordination | Supervisor pattern: one agent routes tasks to specialized sub-agents | P2 |
| 4.1.7 | State inspection API | `GET /workflows/runs/{run_id}/state` — view current state variables | P1 |
| 4.1.8 | State editing API | `PUT /workflows/runs/{run_id}/state` — manually edit state (admin only) | P2 |
| 4.1.9 | Workflow replay | Re-run a workflow from a specific checkpoint | P2 |
| 4.1.10 | Error handling nodes | Try/catch pattern: route to error handler node on exception | P1 |

#### 4.2 Human-in-the-Loop Design

```
Workflow reaches HumanApproval node
    │
    ▼
Graph pauses (interrupt_before)
    │
    ▼
Run status → "paused", pending_approval_node set
    │
    ▼
Frontend polls run status → sees "paused"
    │
    ▼
Frontend renders approval UI (message + schema fields)
    │
    ▼
User submits approval/rejection
    │
    ▼
POST /workflows/runs/{run_id}/resume
    │
    ▼
Graph resumes from checkpoint with approval data injected
    │
    ▼
Workflow continues from next node
```

**Timeout handling:** If no response within `timeout_seconds`, auto-resume with
default action (approve/reject based on config) or escalate to a fallback node.

#### 4.3 Structured Output Design

| Mode | Behavior | LangChain Method |
|---|---|---|
| `strict` | LLM must return valid JSON matching schema. Error if invalid. | `.with_structured_output(schema, method="json_mode")` |
| `guided` | LLM attempts structured output. Falls back to text parsing if invalid. | `.with_structured_output(schema, method="function_calling")` |
| `none` | Free-form text output. No enforcement. | Standard `.ainvoke()` |

#### 4.4 Production Decisions

| Decision | Rationale |
|---|---|
| Human-in-the-loop uses LangGraph `interrupt_before` | Native checkpoint pausing. No custom polling logic needed. State persisted in Postgres. |
| Approval data stored in `workflow_runs.approval_data` | Audit trail. Enables replay with different approval decisions. |
| Subgraph execution uses separate thread_id | Prevents state collision between parent and child workflows. |
| Structured output defaults to `guided` mode | Strict mode can fail silently with some models. Guided provides graceful degradation. |
| Parallel execution limited to 5 concurrent branches | Prevents resource exhaustion from unbounded fan-out. |

#### 4.5 Scaling Considerations

- **Paused workflows:** Paused runs hold checkpoints in Postgres. Implement cleanup for runs paused > 7 days.
- **Subgraph depth limit:** Maximum nesting depth of 5 to prevent recursive subgraph loops.
- **Parallel branch isolation:** Each parallel branch gets its own state copy. Merge strategy resolves conflicts.

---

### Phase 5: Frontend & Visual Editor

**Goal:** Build the Next.js frontend with a visual workflow editor, chat interface,
document management, and knowledge graph explorer.

**Duration:** 4 weeks
**Prerequisites:** Phase 2 complete (workflow API), Phase 3 complete (tool API)

#### 5.1 Implementation Tasks

| # | Task | Details | Priority |
|---|---|---|---|
| 5.1.1 | Next.js 15 project setup | App Router, TypeScript, Tailwind CSS, shadcn/ui | P0 |
| 5.1.2 | Clerk auth integration | SignIn, SignUp, UserButton components. JWT passthrough to FastAPI. | P0 |
| 5.1.3 | Chat interface | Streaming SSE display, message history, thread management | P0 |
| 5.1.4 | Document upload UI | Drag-and-drop upload, progress tracking, status polling | P0 |
| 5.1.5 | Visual workflow editor | React Flow canvas with drag-and-drop nodes, edge drawing, zoom/pan | P0 |
| 5.1.6 | Node configuration panels | Side panel for editing node config (model, prompt, tools, providers) | P0 |
| 5.1.7 | Workflow execution viewer | Real-time state inspection, node highlighting during execution | P1 |
| 5.1.8 | Knowledge graph explorer | LightRAG graph visualization (entities, relationships) | P1 |
| 5.1.9 | Agent configuration UI | Form-based agent config editing with live preview | P1 |
| 5.1.10 | Tool builder UI | HTTP tool form, code editor with syntax highlighting | P1 |
| 5.1.11 | Run history dashboard | List of workflow runs with status, duration, cost | P2 |
| 5.1.12 | Mobile responsive layout | Chat and document management usable on mobile | P2 |

#### 5.2 Frontend Tech Stack

| Component | Technology | Rationale |
|---|---|---|
| Framework | Next.js 15 (App Router) | Server components, streaming, API routes |
| Language | TypeScript | Type safety for API contracts |
| Styling | Tailwind CSS + shadcn/ui | Rapid UI development, accessible components |
| Chat Streaming | Vercel AI SDK (`useChat`) | Native SSE support, maps to FastAPI StreamEvent schema |
| Visual Editor | React Flow | Industry-standard node graph editor, customizable |
| State Management | TanStack Query (React Query) | Server state caching, optimistic updates |
| Auth | Clerk React SDK | Pre-built UI components, session management |
| Code Editor | Monaco Editor (via @monaco-editor/react) | VS Code-like editing for code tools |
| Graph Visualization | D3.js or react-force-graph | Knowledge graph exploration |
| Forms | React Hook Form + Zod | Type-safe form validation matching Pydantic schemas |

#### 5.3 Production Decisions

| Decision | Rationale |
|---|---|
| Vercel AI SDK for chat streaming | Handles SSE parsing, reconnection, token buffering. Matches FastAPI StreamEvent format. |
| React Flow over custom canvas | Battle-tested, supports custom nodes/edges, undo/redo, minimap. Saves weeks of development. |
| Server Components for data-heavy pages | Reduces client bundle size. Workflow list, document list rendered server-side. |
| Client Components for interactive elements | Chat, visual editor, forms require client-side interactivity. |
| API proxy via Next.js API routes | Avoids CORS issues. Next.js proxies to FastAPI with auth headers. |

#### 5.4 Scaling Considerations

- **CDN for static assets:** Next.js static assets served via Vercel Edge Network.
- **Image optimization:** Next.js Image component with Cloudflare R2 as remote source.
- **SSE connection limits:** Browser limits concurrent SSE connections (~6 per domain). Use single multiplexed SSE stream per page.
- **React Flow performance:** Virtualize rendering for graphs with >100 nodes. Use `onlyRenderVisibleElements`.

---

### Phase 6: Publishing, Versioning & API Management

**Goal:** Enable publishing workflows/agents as versioned, accessible APIs with
rate limiting, API keys, and webhook triggers.

**Duration:** 2 weeks
**Prerequisites:** Phase 2 complete (workflow engine), Phase 5 started (frontend)

#### 6.1 Implementation Tasks

| # | Task | Details | Priority |
|---|---|---|---|
| 6.1.1 | Version management | Save workflow snapshots. List versions, diff between versions, rollback. | P0 |
| 6.1.2 | Publish/unpublish | Toggle workflow availability. Published workflows get stable API endpoint. | P0 |
| 6.1.3 | API key generation | Per-user API keys for programmatic access to published workflows. | P0 |
| 6.1.4 | Rate limiting | Per-key rate limits (RPM, TPM). Return 429 with Retry-After header. | P1 |
| 6.1.5 | Webhook triggers | `POST /webhooks/{workflow_id}` — trigger workflow execution via webhook | P1 |
| 6.1.6 | Scheduled triggers | Cron-based workflow execution (e.g., daily report generation) | P2 |
| 6.1.7 | Usage analytics | Track invocations, token usage, cost per workflow per day | P1 |
| 6.1.8 | Workflow templates | Pre-built workflow templates users can clone and customize | P2 |
| 6.1.9 | Export/import | Export workflow as JSON file. Import to another account. | P2 |

#### 6.2 Version Management Design

```
Workflow "Order Processor"
├── v1.0.0 (published) ──► Stable API endpoint
├── v1.1.0 (draft)     ──► In development
├── v1.2.0 (draft)     ──► In development
└── History:
    ├── v1.0.0 created 2026-10-01
    ├── v1.1.0 created 2026-10-15 (added refund node)
    └── v1.2.0 created 2026-10-25 (added parallel processing)
```

**Rollback:** `POST /workflows/{id}/rollback` with target version. Creates a new
version that is a copy of the target. Does not delete intermediate versions.

#### 6.3 Production Decisions

| Decision | Rationale |
|---|---|
| Versions are immutable copies, not pointers | Prevents accidental mutation of published versions. Each version is a full JSON snapshot. |
| API keys are hashed at rest | Store SHA-256 hash. Show full key only once at creation. |
| Rate limiting via Redis sliding window | Accurate, distributed rate limiting. Falls back to in-memory if Redis unavailable. |
| Webhook signatures (HMAC-SHA256) | Prevents unauthorized webhook triggers. Include timestamp to prevent replay attacks. |
| Usage analytics stored in separate table | High write volume. Separate from workflow config tables to avoid lock contention. |

---

### Phase 7: Production Hardening & Scale

**Goal:** Prepare the platform for production traffic with observability, security
hardening, deployment automation, and disaster recovery.

**Duration:** 3 weeks
**Prerequisites:** All previous phases functionally complete

#### 7.1 Implementation Tasks

| # | Task | Details | Priority |
|---|---|---|---|
| 7.1.1 | LangSmith integration | Trace every LLM call, tool execution, retrieval step. Link traces to workflow runs. | P0 |
| 7.1.2 | Structured logging (production) | JSON-formatted logs with correlation IDs. Ship to log aggregator. | P0 |
| 7.1.3 | Sentry error tracking | Capture unhandled exceptions with context (user, workflow, node). | P0 |
| 7.1.4 | Cost tracking dashboard | Per-user, per-agent, per-workflow token usage and cost estimation. | P1 |
| 7.1.5 | Docker multi-stage build | Optimize image size. Separate build and runtime stages. | P0 |
| 7.1.6 | Docker Compose (dev) | FastAPI + Postgres + Redis + LightRAG Server in one command. | P0 |
| 7.1.7 | GitHub Actions CI/CD | Lint → Test → Build → Deploy pipeline. | P0 |
| 7.1.8 | Database connection pooling | Neon pooler endpoint. Configure pool size per service. | P0 |
| 7.1.9 | Checkpoint pruning | Background job to delete checkpoints older than retention period. | P1 |
| 7.1.10 | Security audit | OWASP top 10 review. Input sanitization. SQL injection prevention. | P0 |
| 7.1.11 | CORS hardening | Restrict origins to frontend domain. Remove wildcard. | P0 |
| 7.1.12 | Secrets management | Move from .env to cloud secrets manager (e.g., Doppler, AWS Secrets Manager). | P1 |
| 7.1.13 | Load testing | k6 or Locust scripts simulating 100+ concurrent users. | P1 |
| 7.1.14 | Disaster recovery plan | Database backup strategy. R2 cross-region replication. Runbook. | P2 |

#### 7.2 Observability Stack

```
┌─────────────────────────────────────────────────────┐
│                  OBSERVABILITY                      │
│                                                     │
│  LangSmith ──► LLM traces, token counts, latency   │
│  structlog ──► JSON structured logs                 │
│  Sentry    ──► Error tracking with context          │
│  Prometheus──► Metrics (request rate, error rate)   │
│  Grafana   ──► Dashboards                           │
│                                                     │
│  Correlation ID: X-Request-ID header propagated     │
│  through all services and log entries               │
└─────────────────────────────────────────────────────┘
```

#### 7.3 Security Hardening Checklist

| Area | Action |
|---|---|
| Authentication | Clerk JWT verification on all endpoints. No unauthenticated access except /health. |
| Authorization | Row-level ownership checks. User A cannot access User B's agents/workflows. |
| Input Validation | Pydantic schema validation on all request bodies. Reject unknown fields. |
| SQL Injection | SQLAlchemy ORM parameterized queries only. No raw string interpolation. |
| File Upload | Type whitelist (PDF, DOCX, TXT, MD). Size limit (25MB). Virus scan (future). |
| Code Execution | Sandboxed subprocess. No network, filesystem, or env var access. |
| API Keys | Hashed at rest. Rate-limited. Revocable. |
| Secrets | No secrets in code or logs. Environment variables or secrets manager. |
| CORS | Explicit origin whitelist. No wildcards in production. |
| Rate Limiting | Per-user and per-IP rate limits on all endpoints. |
| Dependency Scanning | `uv pip audit` or `safety` in CI pipeline. |

#### 7.4 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT                     │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │ Vercel /     │    │ FastAPI     │    │ Taskiq Workers  │  │
│  │ Cloudflare   │───►│ (2+ pods)   │───►│ (2+ pods)       │  │
│  │ Pages        │    │             │    │                 │  │
│  │ (Frontend)   │    │ Stateless   │    │ Background      │  │
│  └─────────────┘    └──────┬──────┘    │ Processing      │  │
│                            │           └────────┬────────┘  │
│                            ▼                    ▼           │
│                     ┌─────────────┐    ┌─────────────────┐  │
│                     │ Neon Postgres│    │ Cloudflare R2   │  │
│                     │ (Serverless) │    │ (Object Store)  │  │
│                     │ + pgvector   │    │                 │  │
│                     │ + FTS        │    │                 │  │
│                     └─────────────┘    └─────────────────┘  │
│                                                             │
│                     ┌─────────────┐                         │
│                     │ Redis       │                         │
│                     │ (Task Queue) │                         │
│                     └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

#### 7.5 Production Decisions

| Decision | Rationale |
|---|---|
| Stateless FastAPI pods | Horizontal scaling. No in-memory state. All state in Postgres/R2. |
| Separate worker pods for Taskiq | Independent scaling. CPU-heavy document processing doesn't affect API latency. |
| Neon pooler endpoint | Connection multiplexing. Handles 1000s of connections on serverless Postgres. |
| Checkpoint retention: 30 days | Balance between debuggability and storage cost. Published workflows: 90 days. |
| Database backups: daily | Neon automatic backups. Point-in-time recovery enabled. |
| R2 lifecycle policy | Move objects older than 90 days to infrequent access tier. |

---

## 7. Detailed Feature Matrix

### 7.1 Agent Configuration Features

| Feature | Description | Config Options | Phase |
|---|---|---|---|
| System Prompt | Base instruction for the agent | Rich text, variable interpolation, locale | 1 |
| Model Selection | Choose LLM provider and model | Provider, model_id, temperature, max_tokens, thinking | 1 |
| Knowledge Providers | Attach retrieval backends | Provider name, mode, weight, top_k, enabled | 1 |
| Tool Binding | Attach tools to agent | Tool ID, enabled, config overrides | 1 |
| Structured Output | Force JSON/Pydantic responses | Schema reference, enforcement mode | 1 |
| Input Guardrails | Validate/filter user input | Regex, prompt-based check, blocked topics | 1 |
| Output Guardrails | Filter LLM responses | PII redaction, brand tone, hallucination check | 1 |
| Memory Config | Control conversation memory | Context window size, summarization threshold | 1 |
| Streaming Toggle | Enable/disable SSE streaming | Boolean | 1 |
| Agent Cloning | Duplicate agent for experimentation | Source agent ID | 1 |
| Agent Versioning | Track config changes over time | Auto-increment version on save | 6 |

### 7.2 Workflow Engine Features

| Feature | Description | Config Options | Phase |
|---|---|---|---|
| Visual Graph Editor | Drag-and-drop node/edge canvas | React Flow with custom node types | 5 |
| Node Types | 10 node types (see Phase 2 spec) | Per-type configuration panels | 2 |
| Conditional Routing | Branch based on state or LLM | JSONPath, variable match, LLM classification | 2 |
| Parallel Execution | Fan-out/fan-in pattern | Target nodes, merge strategy (concat, first, vote) | 4 |
| Subgraph Invocation | Nest workflows within workflows | Subgraph ID, input/output mapping | 4 |
| Human-in-the-Loop | Pause for human approval | Message, schema, timeout, default action | 4 |
| Error Handling | Try/catch with fallback nodes | Error node ID, error variable name | 4 |
| Loop Control | Iterative cycles with exit conditions | Max iterations, exit condition expression | 2 |
| State Variables | Typed variables flowing between nodes | Name, type, default value, scope | 2 |
| Workflow Validation | Check graph integrity before save | Connectivity, orphan detection, type checking | 2 |

### 7.3 Tool Builder Features

| Feature | Description | Config Options | Phase |
|---|---|---|---|
| HTTP/API Tools | Call external REST APIs | URL, method, headers, auth, body, retry | 3 |
| Code Tools | Custom Python functions | Code editor, input/output schema, sandbox | 3 |
| Database Tools | Read-only SQL queries | Connection, query, parameters, max_rows | 3 |
| Built-in Tools | Pre-built utilities | Web search, calculator, datetime, file reader | 3 |
| Tool Testing | Execute tool with sample input | Input form, output viewer | 3 |
| Tool Versioning | Track tool config changes | Auto-increment version | 6 |
| Tool Marketplace | Share tools across users (future) | Visibility, pricing (future) | Future |

### 7.4 Knowledge & Retrieval Features

| Feature | Description | Config Options | Phase |
|---|---|---|---|
| Hybrid Search | PGVector + FTS + RRF | Weights, top_k, preset (precision/balanced/recall) | 0 ✅ |
| LightRAG Graph | Knowledge graph retrieval | Mode (local/global/hybrid/mix), top_k | 0 ✅ |
| Provider Fan-out | Query multiple providers in parallel | Provider list, weights, RRF fusion | 1 |
| Per-Node Retrieval | Different retrieval config per node | Provider, top_k, filters per retrieve node | 3 |
| Query Rewriting | LLM rewrites query before search | Enable/disable, rewrite model | 3 |
| Metadata Filtering | Filter results by document metadata | Key-value filters, operators | 3 |
| Reranking | Re-rank retrieved chunks | Reranker model, enable/disable | Future |

### 7.5 Publishing & Operations Features

| Feature | Description | Config Options | Phase |
|---|---|---|---|
| Version Control | Snapshot workflow configs | Version list, diff, rollback | 6 |
| Publish/Unpublish | Toggle API availability | Published state, stable endpoint | 6 |
| API Keys | Programmatic access | Key generation, revocation, rate limits | 6 |
| Webhook Triggers | Event-driven execution | Webhook URL, HMAC signature, retry | 6 |
| Scheduled Triggers | Cron-based execution | Cron expression, timezone | 6 |
| Usage Analytics | Track invocations and cost | Dashboard, export, alerts | 6 |
| Workflow Templates | Pre-built starting points | Template gallery, clone | 6 |
| Export/Import | Portable workflow configs | JSON export, import validation | 6 |

---

## 8. Production Readiness Checklist

### 8.1 Pre-Launch (Must Have)

- [ ] All API endpoints authenticated (Clerk JWT)
- [ ] Row-level ownership enforcement on all resources
- [ ] Input validation on all request bodies (Pydantic)
- [ ] SQL injection prevention (parameterized queries only)
- [ ] File upload type and size validation
- [ ] Rate limiting on all public endpoints
- [ ] CORS restricted to frontend domain
- [ ] Secrets in environment variables (not code)
- [ ] Structured logging with correlation IDs
- [ ] Error tracking (Sentry) configured
- [ ] Database backups enabled (Neon automatic)
- [ ] Health check endpoints (/health, /ping)
- [ ] Graceful shutdown handling (lifespan)
- [ ] API documentation (OpenAPI/Swagger)

### 8.2 Launch (Should Have)

- [ ] LangSmith tracing enabled
- [ ] Cost tracking per user/workflow
- [ ] Docker Compose for local development
- [ ] CI/CD pipeline (lint → test → build → deploy)
- [ ] Load testing completed (100+ concurrent users)
- [ ] Checkpoint pruning job configured
- [ ] Stuck-task recovery (advisory lock sweeper)
- [ ] API key management UI

### 8.3 Post-Launch (Nice to Have)

- [ ] Multi-region deployment
- [ ] Database read replicas
- [ ] CDN for static assets
- [ ] Advanced caching (semantic cache for repeated queries)
- [ ] Workflow marketplace / template gallery
- [ ] Mobile app (React Native)
- [ ] SSO / SAML for enterprise

---

## 9. Scaling Strategy

### 9.1 Horizontal Scaling

| Component | Scaling Strategy | Trigger |
|---|---|---|
| FastAPI API | Add pods (stateless) | CPU > 70% or request latency > 500ms |
| Taskiq Workers | Add workers | Queue depth > 50 tasks |
| Neon Postgres | Serverless auto-scale (compute units) | Connection count > 80% of limit |
| Cloudflare R2 | No scaling needed (managed) | N/A |
| Redis | Managed service (Upstash/ElastiCache) | Memory > 80% |

### 9.2 Database Scaling

| Milestone | Strategy |
|---|---|
| < 10K users | Single Neon instance with pooler |
| 10K-100K users | Read replicas for search queries. Separate write primary. |
| 100K-1M users | Shard by user_id. Separate vector DB (Pinecone/Qdrant) for search. |
| > 1M users | Multi-region. Dedicated graph DB (Neo4j) for LightRAG. |

### 9.3 LLM Cost Optimization

| Strategy | Implementation |
|---|---|
| Semantic caching | Cache LLM responses for semantically similar queries (Redis + embedding similarity) |
| Model routing | Route simple queries to cheap models (deepseek-flash), complex to expensive (deepseek-v4-pro) |
| Context window management | Summarize conversation history beyond threshold instead of sending full history |
| Batch embeddings | Batch embedding requests (32 texts per call) to reduce API overhead |
| LightRAG LLM cache | Enable `enable_llm_cache=True` to avoid re-extraction on identical chunks |

---

## 10. Risk Assessment & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LLM provider outage | Medium | High | Fallback model chain. Circuit breaker pattern. Queue requests for retry. |
| Neon connection exhaustion | Medium | High | Use pooler endpoint. Limit pool size per service. Monitor connection count. |
| LightRAG ingestion timeout | High | Medium | Background task with retry. Advisory lock sweeper. Status tracking. |
| Checkpoint storage bloat | Medium | Medium | TTL-based pruning. Separate table for archived checkpoints. |
| Code tool security breach | Low | Critical | Subprocess sandboxing. No network/filesystem access. Resource limits. |
| Workflow infinite loop | Medium | Medium | `recursion_limit` per workflow. Timeout per execution. |
| Data leakage between tenants | Low | Critical | Row-level ownership checks. Workspace isolation. Integration tests. |
| Frontend SSE disconnection | Medium | Low | Auto-reconnect with exponential backoff. Resume from last checkpoint. |
| Dependency vulnerability | Medium | High | Automated dependency scanning in CI. Pin versions. Regular updates. |
| Cost overrun (LLM tokens) | Medium | High | Per-user token budgets. Cost tracking dashboard. Alert thresholds. |

---

## 11. Appendix: Glossary

| Term | Definition |
|---|---|
| **Agent** | A configured AI assistant with a system prompt, model, knowledge providers, and tools. Simplest unit of the platform. |
| **Workflow** | A multi-node state machine defining complex AI processes with branching, loops, and human interaction. |
| **Node** | A single processing step within a workflow (chat, retrieve, tool_call, conditional, etc.). |
| **Edge** | A connection between two nodes defining execution flow. Can be static or conditional. |
| **State** | The typed data (variables, messages) that flows between nodes during workflow execution. |
| **Checkpoint** | A persisted snapshot of workflow state at a specific point in execution. Enables pause/resume. |
| **Provider** | A knowledge retrieval backend (hybrid search, LightRAG, or future vector DBs). |
| **Workspace** | An isolation boundary for multi-tenancy. Scopes data access per user, thread, or agent. |
| **RRF** | Reciprocal Rank Fusion. Algorithm combining ranked results from multiple search engines. |
| **HNSW** | Hierarchical Navigable Small World. Vector index type for approximate nearest neighbor search. |
| **FTS** | Full-Text Search. PostgreSQL native text search using tsvector and GIN indexes. |
| **SSE** | Server-Sent Events. HTTP protocol for streaming real-time data from server to client. |
| **Structured Output** | LLM responses constrained to a specific JSON/Pydantic schema. |
| **Human-in-the-Loop** | Workflow pattern where execution pauses for human approval or data entry. |
| **Subgraph** | A workflow or agent invoked as a node within another workflow. |
| **Fan-out/Fan-in** | Parallel execution pattern. Fan-out distributes work to multiple nodes. Fan-in merges results. |
| **Checkpointer** | LangGraph component that persists state to a database. Enables pause/resume and memory. |
| **Presigned URL** | Time-limited URL for secure access to private cloud storage objects. |

---

*This document is a living artifact. Update it as phases are completed, decisions change,
or new requirements emerge. Each phase completion should trigger a review and update of
subsequent phase plans.*

*— Base Agent Builder Team*
