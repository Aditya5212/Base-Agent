# Base Agent / Agent Builder — Project Audit

**Review date:** 31 October 2026  
**Baseline commit:** `c9da61acdca211d2b2773b9b2a8c896385c7b5eb`  
**Scope:** Backend implementation, `AGENTS.md`, `Vision_and_Roadmap.md`, README, TODO, migrations, containers, and selected tests.  
**Development context:** Authentication is intentionally deferred while core functionality is built. It is treated as a future deployment requirement, not as evidence that the current development sequence is wrong.

## 1. Overall assessment

**This is a substantial backend prototype / early alpha with a useful architecture and several incomplete execution contracts.** The code goes well beyond scaffolding: agent CRUD, persisted conversation state, hybrid retrieval, document ingestion, reusable tools, graph compilation, branching, and run records have real implementations. The main weakness is the gap between accepting a configuration, persisting it, and actually enforcing it during execution.

The next milestone should be **reliable core execution**, before expanding the visual builder or advanced orchestration. In particular, fix approval lifecycle reporting, checkpoint identity, tool error propagation, and runtime schema enforcement. Adding more configuration fields before closing these gaps would increase the number of features that appear available but have no effect.

| Area | Assessment | What determines the assessment |
|---|---|---|
| Product direction | Coherent, but too broad for the next release | Agents + knowledge + tools + workflows fit together; a full Copilot Studio parity target spreads effort across many unrelated launch requirements |
| Module architecture | Good foundation | API/schema/model/service separation, node builders, provider protocol, reusable executors |
| Feature implementation | Broad, uneven integration | Many working primitives; tools, schemas, knowledge attachments, and approvals are not consistently connected end to end |
| Core correctness | Needs a focused stabilization milestone | Multiple defects reproduced with synthetic offline inputs |
| Maintainability | Reasonable for current size | Clear naming and small infrastructure modules; duplicated execution policies and silent fallbacks create drift |
| Verification | Useful happy-path coverage, insufficient contract coverage | 39 selected existing tests passed while targeted probes exposed important failures |
| Operational readiness | Partially implemented, not validated here | Docker, migrations, queue, recovery exist; live failure recovery and multi-process behavior still need integration tests |
| Authentication | Deliberately deferred | Appropriate to track separately; required before untrusted access |

This is a review of the repository, not a penetration test, performance benchmark, dependency vulnerability scan, or certification of deployment readiness.

## 2. Evidence and verification

### What was checked

- Read the architecture and roadmap, then traced the key API → service → compiler/provider/executor paths.
- Inspected database models, all four migration revisions, startup behavior, queue dispatch, document recovery, and container configuration.
- Parsed **118 Python files** under `app`, `tests`, and `alembic` successfully. The application contains **89 Python files / 12,922 lines**; tests contain **24 Python files / 3,511 lines**. These are physical line counts, not measures of quality.
- Counted **148 test functions** statically. This does not establish that 148 tests currently pass or account for parameterized case counts.
- Ran **39 selected existing tests: 39 passed, 2 warnings, 0.68 seconds**. Selection covered workflow compilation, node behavior, retrieval fusion, model factory behavior, RAG tools, crypto, HTTP mocks, code tools, SQLite parameter binding, and tool compilation.
- Ran targeted offline probes using `InMemorySaver`, mocked persistence/model calls, synthetic data, and a finite code timeout example.

### Reproducible artifacts

- [Offline audit runner](audit/run_core_audit.py)
- [Recorded probe results](audit/core-evidence.json)
- [Selected pytest results](audit/selected-tests.xml)

Run from the repository root:

```powershell
.venv/Scripts/python.exe audit/run_core_audit.py
.venv/Scripts/python.exe audit/run_core_audit.py --tests
```

The runner uses a dummy database URL, empty model API keys, and disabled tracing. For selected pytest tests, it replaces the autouse database initializer with a no-op. **This is an offline subset result, not a full-suite or live integration result.** Probe output intentionally includes a division-by-zero tool error and a timeout warning.

No live PostgreSQL/R2/DeepSeek/OpenAI ingestion, migration upgrade/downgrade, container build, load test, or end-to-end SSE test was run. Application source and existing tests were not changed. The audit added this report and the `audit/` artifacts.

## 3. What is already working well

1. **The decomposition is sensible.** FastAPI routes, Pydantic contracts, SQLAlchemy models, and business services have recognizable responsibilities. The compiler and node builder split gives a practical route to extending workflows.
2. **PostgreSQL is a reasonable consolidation choice.** ORM persistence, checkpoints, vectors, FTS, and a PostgreSQL task broker reduce the infrastructure needed for an early deployment. Budget connections across all pools rather than assuming they are one pool.
3. **Retrieval has a useful baseline.** Dense and sparse searches run concurrently, RRF is isolated in testable functions, and research tools bind user identity outside model-callable arguments.
4. **Document ingestion has real lifecycle thinking.** Upload rollback, background dispatch, inspection records, retries, reindexing, and cleanup paths exist. These are valuable foundations even though the consistency guarantees need improvement.
5. **There is a declarative workflow contract.** Typed node definitions, branch routing, template interpolation, recursion limits, run records, and an approval primitive form a viable builder backend.
6. **There is meaningful automated testing.** The selected tests passed quickly without infrastructure once initialization was isolated. Preserve that fast feedback loop while adding tests for actual execution contracts.
7. **The implementation avoids an early distributed-services rewrite.** Continue with the modular application and worker architecture. Most current issues can be fixed without introducing microservices or replacing LangGraph.

## 4. Feature audit: exposed configuration versus runtime behavior

“Implemented” below means a code path exists, not that the live service was validated.

| Feature | Current evidence | Assessment |
|---|---|---|
| Agent CRUD, cloning, defaults | Persistence and API service implementations | Implemented; concurrent default creation needs a database invariant |
| Persona, temperature, model ID, max tokens, message window | Read by the conversational `chat_node` | Connected to runtime |
| Agent tool attachments | Tool specifications and resolver exist; normal chat calls the model directly | Not integrated into normal chat tool execution |
| Agent structured output | Schema registry and config fields exist; chat does not apply them | Registry implemented, enforcement missing |
| Agent guardrails, summarization, long-term memory, advanced timeout/retries | Accepted fields; normal chat does not consume these controls | Mostly configuration surface / future work |
| Model provider selection | Schema accepts multiple providers; factory is DeepSeek-oriented | Broader provider selection is not implemented by the schema alone |
| Multiple knowledge providers, weights and top-k | Chat selects a retrieval path; no general weighted provider fusion | Partial; configuration promises exceed execution |
| Hybrid RAG | Dense + FTS + RRF code and fusion tests | Implemented baseline; live quality/latency unmeasured |
| LightRAG | Provider ingestion/query implementation and live script | Provider exists; standard document upload does not populate it |
| Multi-query search | Caller-supplied queries, concurrent hybrid calls, cross-query RRF | Implemented primitive; ordinary chat supplies only its current query |
| Knowledge scoped by thread/agent | Workspace helpers exist; hybrid paths generally filter only by user | Partial; identity naming alone does not enforce every scope |
| Workflow nodes | Ten schema node types; eight compiler handlers | `parallel` and `subgraph` are declared but unsupported |
| Human approval / resume | Interrupt node and resume service exist | Core lifecycle defect: paused runs can be marked completed |
| Workflow input/output validation | Schemas accepted and checked for schema syntax on save | Payload validation is not enforced during execution |
| Dynamic tools | HTTP/code/database compilation and workflow invocation | Implemented, with error and schema contract defects |
| Node knowledge attachment | Retrieve node consumes config; other nodes can store a `knowledge` property | Generic attachment is not active retrieval for chat/code/tool nodes |
| Versioning / publication | Mutable rows with version counters and flags | Revision history, pinned executions, rollback, and stable publication are future work |
| Queue and recovery | Taskiq PostgreSQL dispatch plus local fallback/sweeper | Implemented baseline; worker absence is not detected by successful enqueue |
| Visual builder and channel publishing | Roadmap proposals | Future work; not a missing backend implementation defect |

Evidence: [chat runtime](app/services/agent.py:263), [agent config](app/schemas/agent.py:1), [agent tool resolver](app/services/agent_service.py:375), [compiler](app/services/workflow/compiler.py:31), [knowledge attachment](app/services/workflow_service.py:181), [document pipeline](app/services/document_processor.py:24).

## 5. Prioritized core findings

Priorities are relative to the next **core functionality** milestone:

- **P0:** Fix before treating execution results as trustworthy.
- **P1:** Fix before broader internal adoption or a builder UI depends on the behavior.
- **P2:** Improve as integration and operational coverage expands.

### F01 — P0: Approval runs are reported as completed while still paused

**Confidence: reproduced.** `execute_workflow` and `resume_workflow_run` rely on catching `GraphInterrupt`, then otherwise mark the run `COMPLETED`. The installed LangGraph invocation returns interrupt information to the caller instead of letting that exception escape in the expected way.

The probe executed START → HUMAN_APPROVAL → END. The service returned `COMPLETED`, but the checkpoint still had pending node `h` and one interrupt. The resume API subsequently rejects a completed run. Approval timeout is only included in the interrupt payload; no expiry mechanism implements it.

**Fix:** Detect returned interrupts and/or inspect the state snapshot, persist pending approvals, and mark the run `INTERRUPTED`. Apply the same logic after resume, including a second approval. Require persistence for resumable graphs instead of silently compiling them without it. Implement expiry only when its behavior is defined.

**Acceptance:** First approval → INTERRUPTED → valid resume → COMPLETED; two approvals require two resumes; timeout handling follows an explicit policy.

Evidence: [execute service](app/services/workflow_service.py:248), [resume service](app/services/workflow_service.py:338), [approval node](app/services/workflow/nodes/human.py:1). LangGraph documents returned interrupt payloads for invocation and explicit resume behavior: [official interrupt documentation](https://docs.langchain.com/oss/python/langgraph/interrupts).

### F02 — P0: Workflow checkpoint keys collide across users and workflows

**Confidence: reproduced with in-memory checkpoints and mocked owned workflow lookup.** A caller-supplied `payload.thread_id` is passed directly as the checkpointer identity, without a tenant/workflow/run namespace. Separate users executing separate workflows with the same supplied ID can reuse message state.

The probe ran Alice's workflow and then Bob's workflow using the same thread ID. Bob's model input contained Alice's synthetic private message. This defect is independent of the deferred authentication implementation: the probe supplied different user identities and successful owned-workflow lookups.

**Fix:** Separate user-facing conversation IDs from internal checkpoint IDs. Derive checkpoint identity server-side from tenant, immutable workflow revision, and run identity. Only an explicitly authorized continuation should reuse a checkpoint. Keep chat and workflow checkpoint identities in separate namespaces.

**Acceptance:** Same visible thread ID across two users, two workflows, or a chat and workflow never shares state. Concurrent requests cannot unintentionally resume each other's runs.

Evidence: [thread selection](app/services/workflow_service.py:257), [memory access](app/services/memory.py:1). LangGraph describes `thread_id` as the pointer to persisted state: [official interrupt documentation](https://docs.langchain.com/oss/python/langgraph/interrupts).

### F03 — P0: Tool failures are converted into successful return values

**Confidence: reproduced.** `ToolRegistry` catches executor exceptions and returns an error string. `ToolService.execute_tool` sees a normal return and reports `success=True`, `error=None`. Workflow retry logic also cannot retry an exception that has already been converted to text. Unknown tools have an additional placeholder path that claims the tool executed.

The probe's division-by-zero tool returned `success: true` with an error message inside `output`.

**Fix:** Preserve typed executor failures through the registry. Convert them into model-readable feedback only at the orchestration boundary. Return a structured tool result with success/error/retryability and preserve native output values. Fail unknown tool resolution explicitly; make mock behavior an explicit test/development option.

**Acceptance:** Failed executor → failed API result; retryable failure triggers configured retries; unknown tool cannot produce a successful fake result.

Evidence: [registry wrapper](app/services/tool_registry.py:158), [execution API service](app/services/tool_service.py:245), [unknown-tool fallback](app/services/workflow/nodes/tool_call.py:115).

### F04 — P0: `requires_approval` is accepted but not enforced

**Confidence: reproduced.** `ToolCallNodeConfig.requires_approval=True` does not change execution. The probe ran a calculator node configured to require approval and obtained its result immediately, without an interrupt. This is a core execution-policy issue even in a trusted development environment.

**Fix:** Implement approval before tool side effects, with persisted decision state and a replay/idempotency policy, or reject the setting until supported. Reuse the repaired approval lifecycle from F01.

**Acceptance:** A configured approval prevents tool invocation until approval; rejection prevents invocation; repeated resume cannot duplicate a side effect.

Evidence: [configuration](app/schemas/workflow.py:228), [tool node](app/services/workflow/nodes/tool_call.py:1).

### F05 — P1: Agent configuration is broader than agent execution

**Confidence: source-confirmed; structured output and guardrail behavior reproduced.** Normal chat calls `llm.ainvoke` directly. It does not resolve/bind attached tools, execute returned tool calls, load the structured-output schema, or enforce guardrails. Several model, identity, memory, and advanced controls also have no corresponding runtime handling.

A synthetic model returned plain text despite `structured_output.enabled=True` and `enforce=True`; the configured blocked-topic example reached the model. The model stub needed only `ainvoke`, demonstrating that this path did not bind tools or structured output.

**Fix:** Introduce a capability-aware execution plan shared by streaming and non-streaming chat. Implement tool execution and schema enforcement first. Reject unsupported settings or expose them as planned capabilities. Resolve custom tools through one shared registry rather than the separate RAG-only synchronous resolver.

**Acceptance:** An attached calculator completes an actual model → tool → model loop; a registered schema validates the final result; unsupported config fails clearly; streaming follows the same policy.

Evidence: [direct model call](app/services/agent.py:352), [config contract](app/schemas/agent.py:1), [separate resolver](app/services/agent_service.py:375).

### F06 — P1: Workflow input and output schemas are not runtime validators

**Confidence: reproduced.** Schema syntax is checked at workflow creation, but execution does not validate `input_data` or final output against workflow-level schemas. Start/end node schemas are also unused during execution. A graph requiring missing input and output keys accepted empty variables.

**Fix:** Validate input before creating a running execution or invoking nodes; validate the final selected output before declaring success. Decide whether node schemas refine or replace workflow-level schemas. Return path-specific validation errors.

**Acceptance:** Invalid inputs run no nodes; invalid output produces a failed/validation outcome; valid payloads preserve their types.

Evidence: [execution](app/services/workflow_service.py:248), [start node](app/services/workflow/nodes/start.py:1), [end node](app/services/workflow/nodes/end.py:1).

### F07 — P1: Declared parallel/subgraph support does not exist in the compiler

**Confidence: reproduced.** Both node types raise `Unsupported node type`. Ordinary fan-out is also currently unsafe: two sibling nodes updating `variables` produced `InvalidUpdateError`, because `variables` and `node_history` do not have merge reducers.

**Fix:** Advertise eight executable types today. Either reject unsupported nodes on save or implement them with defined branch isolation and deterministic joins. Use per-node outputs/explicit merge semantics rather than blindly merging complete variable dictionaries. For subgraphs, add ownership, revision pinning, recursion depth, and state mapping.

**Acceptance:** Unsupported capabilities fail before saving a runnable workflow; supported fan-out has predictable conflict behavior and bounded concurrency.

Evidence: [compiler dispatch](app/services/workflow/compiler.py:91), [state channels](app/services/workflow/state.py:16), [schema enum](app/schemas/workflow.py:24).

### F08 — P1: Tool JSON Schema support is incomplete and silently weakens validation

**Confidence: reproduced.** The converter maps primitive `type` values but ignores constraints such as `enum`, `minimum`, nested item schemas, and composition. A value of `-2` passed a schema requiring a minimum of 10 and enum `[10,20]`. Valid nullable syntax `type: ["string", "null"]` caused `TypeError: unhashable type: 'list'`. Output schemas are stored but not validated in the execution wrapper.

**Fix:** Use the JSON Schema validator as the authoritative runtime validator for arguments and results; use generated Pydantic signatures only for tool interface compatibility. Explicitly support or reject schema subsets at registration. Revalidate type-specific config on update and compile before committing a tool as usable.

**Acceptance:** Nullable/nested schemas register successfully or receive clear capability errors; enum/bounds/additional-property rules are enforced; invalid updates do not persist.

Evidence: [converter](app/services/tool_registry.py:37), [tool update](app/services/tool_service.py:191).

### F09 — P1: Standard uploads do not populate LightRAG

**Confidence: source-confirmed integration gap.** The standard processing pipeline parses, chunks, indexes PGVector, and inserts inspection records. It does not call the LightRAG provider's ingestion method. That method is exercised by a separate live script. Selecting graph retrieval after a normal upload therefore does not itself make the uploaded document available to the graph.

**Fix:** Define requested ingestion targets and per-provider indexing status. Dispatch graph indexing explicitly, record readiness/errors independently, and expose graph availability to clients. Avoid mandatory graph extraction for every upload if its cost is unnecessary.

**Acceptance:** Upload → hybrid-ready and optionally graph-ready → graph query returns evidence from that upload; reindex/delete reconciles every configured target.

Evidence: [pipeline](app/services/document_processor.py:24), [provider ingestion](app/core/rag/providers/lightrag_provider.py:178), [standalone ingestion script](scripts/test_lightrag_live.py:56).

### F10 — P1: Knowledge scope and node attachments are only partially effective

**Confidence: source-confirmed.** Hybrid retrieval nodes calculate a workspace but query with only a `user_id` filter. Thread/agent scope is not applied to hybrid results. Workflow state does not define/populate an agent ID for agent-scoped retrieval. Normal chat's graph path uses the user workspace. Non-retrieve nodes can store `config.knowledge`, but the chat builder never reads it.

**Fix:** Define one knowledge binding contract: owner, workspace type, workspace ID, document selection, provider, mode, and result budget. Apply it at ingestion and retrieval. Until generic node attachments are supported, restrict attachment to retrieve nodes.

**Acceptance:** Two agents or threads under one user retrieve only their selected knowledge when configured for those scopes; generic attachment cannot silently do nothing.

Evidence: [retrieve handler](app/services/workflow/nodes/retrieve.py:1), [hybrid provider](app/core/rag/providers/hybrid.py:1), [attachment persistence](app/services/workflow_service.py:181), [chat node](app/services/workflow/nodes/chat.py:1).

### F11 — P1: Workspace sanitization creates identity collisions

**Confidence: reproduced.** `alice-bob` and `alice_bob` both become `u_alice_bob`; long IDs with different suffixes collapse to the same truncated name. This affects correctness of graph bucket selection independently of authentication.

**Fix:** Append a stable digest derived from the complete, structured identity and scope. Keep the readable prefix only for diagnostics. Plan migration/reindexing for existing workspace names rather than changing the function without addressing stored data.

**Acceptance:** Punctuation variants, long IDs, and ambiguous user/thread combinations yield different stable workspace identifiers.

Evidence: [workspace naming](app/core/rag/workspace.py:6).

### F12 — P1: Code timeouts stop waiting, not execution

**Confidence: reproduced.** Both code execution paths run in API-process threads. The finite probe timed out after 0.5 seconds, then completed its operation after the timeout. A stuck computation can continue occupying resources. In addition, tool callers can expand `allowed_modules`; the probe explicitly allowed and imported `os`, returning only its harmless platform name.

**Fix:** Use a separately managed execution process with termination and resource limits. For untrusted code, add operating-system/container isolation, restricted credentials/filesystem/network, and a server-controlled module policy; a subprocess alone is not a security boundary. Until then, label execution trusted-code-only and do not describe timeout as termination.

**Acceptance:** Timeout terminates the execution process and releases resources; request config cannot grant additional host capabilities. Python documents that already running futures cannot be cancelled: [official executor documentation](https://docs.python.org/3.12/library/concurrent.futures.html#concurrent.futures.Future.cancel).

Evidence: [code tool timeout](app/services/tool_executors/code_executor.py:228), [workflow code execution](app/services/workflow/nodes/code.py:1), [caller module setting](app/schemas/tool.py:110).

### F13 — P1: Ingestion/recovery lacks a strong idempotency and ownership contract

**Confidence: source-confirmed risk; duplicate delivery and concurrent delete/reindex were not exercised against live services.** Vector writes and inspection/status writes occur in separate transactions. An indexing success followed by metadata failure leaves inconsistent state. Initial retries do not use deterministic chunk identities. Recovery schedules in-process tasks after incrementing a retry count; there is no persisted execution lease or heartbeat to distinguish slow active work from abandoned work.

The sweeper also acquires a session-level PostgreSQL advisory lock through an ORM session, then commits before releasing it. Connection ownership across those boundaries needs a live test; a session advisory lock must be managed on the same physical connection.

**Fix:** Persist an ingestion attempt/generation, deterministic chunk IDs, lease owner/expiry, requested provider targets, and original preset. Claim work transactionally. Stage a new generation and activate it after successful persistence. Coordinate deletion with active jobs and make repeated delivery safe. Hold advisory locks on an explicitly retained connection, or use transaction-scoped coordination appropriate to the claim.

**Acceptance:** Duplicate delivery yields one active chunk generation; crash at each persistence boundary is recoverable; slow processing does not trigger concurrent destructive reindexing; deletion cannot be undone by a late worker.

Evidence: [pipeline write boundaries](app/services/document_processor.py:131), [recovery](app/services/document_processor.py:211).

### F14 — P1: Resume uses the current mutable workflow instead of the run's definition

**Confidence: source-confirmed.** Workflow updates overwrite nodes/edges and increment a counter. Run records contain no immutable workflow snapshot/revision reference. Resume reloads the current definition. Once F01 is repaired, editing a paused workflow could change the graph used to resume its saved state.

**Fix:** Store immutable workflow revisions or snapshot the execution definition on each run. Pin node/tool/schema revisions as required for reproducibility. Keep editing the draft separate from resuming historical runs.

**Acceptance:** Pause version 1, edit draft version 2, resume version 1 with its original graph and policies. Run history shows the executed revision.

Evidence: [workflow/run models](app/models/workflow.py:1), [resume recompilation](app/services/workflow_service.py:362).

### F15 — P2: Operational fallbacks can conceal broken dependencies

**Confidence: source-confirmed; live outages not injected.** Startup logs and continues after failed migrations/checkpoint/index setup. Readiness checks only database connectivity; Docker uses liveness as its health criterion. Retrieval errors become empty lists, so a backend failure can look like “no evidence found.” Successful queue dispatch cannot prove a worker is consuming tasks. Workflow execution may silently proceed without persistence.

**Fix:** Define required versus optional capabilities. Fail readiness for required initialization failures, surface degraded retrieval explicitly, and expose queue age/worker heartbeat plus provider status. Keep development mocks behind explicit configuration. Move migrations to one deployment step or coordinate startup migration execution. Execute CPU-heavy document parsing outside the event loop.

**Acceptance:** A missing checkpoint schema cannot appear resumable/healthy; a retrieval outage is distinguishable from an empty corpus; an absent worker creates an actionable queue-age signal.

Evidence: [startup](app/main.py:1), [readiness](app/api/v1/endpoints/health.py:1), [retrieval fallbacks](app/services/retriever.py:38), [queue dispatch](app/api/v1/endpoints/documents.py:171), [synchronous parsing](app/services/document_processor.py:105).

## 6. Deferred authentication and execution hardening

These items should be tracked separately from the core milestone. Authentication can remain deferred during local development; it does not repair the execution identity and policy bugs above.

| Future gate | Evidence / reason | Required behavior before untrusted use |
|---|---|---|
| Verified identity and role | `get_current_user` trusts headers/query/mock tokens and defaults to root; unrecognized bearer tokens can retain root | Explicit development auth mode; production verifies identity and fails closed |
| System tool administration | Normal create/update accepts `is_system`; system tools become accessible to other users | Administrator-only promotion, controlled names and sharing policy |
| HTTP/network tools | Arbitrary URL templates and automatic redirects | Egress policy, destination/redirect validation, bounded responses; retry side-effecting HTTP operations only with defined idempotency semantics |
| Database tools | String-based SQL guard; no read-only transaction or enforced credential restriction | Read-only database role, read-only transaction, database-side time limit and connection destination policy; SQLite needs read-only access/query controls |
| Secret handling | Only designated auth token/connection string encrypted; header secrets can remain raw; decryption failure can fall back to raw text | Explicit secret fields/references, masked responses/logs, fail-closed decryption, independent key and rotation plan |
| User media keys | Image builder signs supplied non-URL strings as object keys | Authorize keys through owned asset records before signing |
| Cost and concurrency budgets | Per-IP rate limiting is process-local; no complete run/model/tool budget | Per-user concurrency, maximum duration, token/tool-call budgets and distributed enforcement when required |

The SQL probe only checked validation and executed **no mutation**: `WITH moved AS (DELETE\nFROM audit_table RETURNING *) SELECT * FROM moved` passed the lexical guard. This demonstrates a faulty read-only classification; it does not claim a committed database write was observed. PostgreSQL's read-only transaction restrictions provide an additional database-level control: [official transaction documentation](https://www.postgresql.org/docs/current/sql-set-transaction.html).

Evidence: [auth](app/core/auth.py:1), [system tool creation](app/services/tool_service.py:82), [HTTP executor](app/services/tool_executors/http_executor.py:93), [SQL executor](app/services/tool_executors/database_executor.py:33), [crypto](app/core/crypto.py:1), [image key signing](app/services/agent.py:441).

## 7. Code quality and maintainability improvements

### Unify execution policy

The conversational agent, workflow chat node, tool registry, tool-call node, and RAG wrappers each handle errors/configuration differently. Introduce shared interfaces for model invocation, tool execution, and retrieval results. Keep errors typed until the API or model-feedback boundary. Make development mocks explicit and make unsupported features discoverable.

### Prefer strict contracts where users configure behavior

Many models ignore extra fields. That can make a misspelled or future setting appear saved while doing nothing. Use strict validation for execution-relevant config, add a capability/version field to stored definitions, and provide migration rules when schemas change. Preserve permissive metadata only in designated UI/custom fields.

### Enforce invariants in the database

Add a partial unique constraint/index for one default agent per user, plus transactional handling of concurrent creation/promotion. Use optimistic version checks for conflicting updates. Define a consistent transaction boundary: current services commit internally while the request dependency also commits.

### Bound caches and connections

The agent config cache is a bounded insertion-order dictionary, not a true access-updated LRU. It also does not eliminate the database lookup used to obtain the latest agent/version. The compiled tool cache and provider-instance/lock maps are unbounded. Bound them and close evicted resources. Keep revision keys; do not introduce shared caching until it solves a measured problem.

Document the actual pool topology: SQLAlchemy, checkpoint `AsyncConnectionPool`, Taskiq's asyncpg broker/backend, and LightRAG can own separate pools. One database and a common driver in some components do not imply a single shared pool.

### Add useful observability before a full analytics platform

Persist request/run/attempt correlation IDs, node start/end/error, model usage, provider status, and tool retry outcomes. Current `execution_log` largely stores final node history rather than granular traces. Add cancellation, run duration, and retention policies. Deleting workflow rows does not itself purge separately managed checkpoint storage.

## 8. Audit of AGENTS.md and Vision_and_Roadmap.md

### AGENTS.md

The document is helpful as an architecture inventory, but mixes implemented behavior, desired guarantees, and future plans. Keep it shorter and operational: repository map, commands, invariants, supported capabilities, and links to detailed design documents.

Specific corrections:

- State **eight executable node types**, with parallel/subgraph reserved/planned.
- Replace claims of arbitrary JSON Schema conversion with the actual supported subset until F08 is fixed.
- Replace “LRU” tool caching with “unbounded version-keyed dictionary” until bounded eviction is implemented.
- Clarify that sensitive custom headers are not currently encrypted automatically.
- Describe SQL checking as a lexical guard, not unconditional prevention of mutations.
- Describe thread timeouts as response deadlines, not execution termination.
- Document independent pools/drivers; Taskiq uses asyncpg and checkpoints use a dedicated psycopg pool.
- Remove absolute machine-specific `file:///c:/...` links.
- Replace absolute claims such as zero cross-tenant leakage or zero hallucination with scoped guarantees and supporting tests. Identity collisions and checkpoint collisions currently contradict the former.
- Separate registry availability, config persistence, and runtime enforcement for tools, structured output, guardrails, and knowledge attachments.

Evidence: [design principles](AGENTS.md:716), [tool platform description](AGENTS.md:663), [provider capability description](app/core/rag/providers/lightrag_provider.py:163).

### Vision_and_Roadmap.md

The vision captures a useful platform direction. The implementation plan needs acceptance criteria and dependency gates more than additional feature lists.

Specific corrections:

- The “complete” current-state table overstates end-to-end workflows, tools, and structured outputs; use `implemented`, `integrated`, `verified`, and `planned` as separate statuses.
- Phase 3 claims subprocess execution in production decisions while the code uses threads and `exec`.
- Taskiq is already PostgreSQL-backed, while parts of the technology table and migration tasks still describe future Redis adoption. Redis should be an optional scaling decision, not an automatic milestone.
- Immutable versions and rollback are future work; counters on overwritten rows do not provide revision history.
- Reranking and metadata-filter configuration are marked complete in a per-node task, but the actual retrieve schema does not expose those controls.
- “148 passing” should link to a reproducible CI run; this audit verified only the selected 39. The roadmap's “20 test modules” inventory is stale.
- Run retention, per-user concurrency, token budgeting, and detailed middleware controls should be labeled planned until implemented.
- Clean embedded control characters/malformed code names in the Phase 3 task table.
- Keep timing and performance statements as targets until benchmarked. Do not claim sub-millisecond retrieval from the architecture alone.

Evidence: [current state](Vision_and_Roadmap.md:94), [version promise](Vision_and_Roadmap.md:355), [middleware plan](Vision_and_Roadmap.md:466), [Phase 3 tasks](Vision_and_Roadmap.md:506), [subprocess decision](Vision_and_Roadmap.md:564).

## 9. Recommended implementation sequence

### Milestone A — Trustworthy execution outcomes

**Scope:** F01–F04, checkpoint namespace, run revision snapshot, explicit unsupported-node validation.

Deliver approval lifecycle handling, deterministic execution identity, typed tool failures, and enforced approval policy. Remove fake successful execution fallbacks. Pin run definitions before exposing stable resume behavior.

**Exit gate:** A single offline/integration scenario creates a workflow, pauses, resumes, fails a tool honestly, and records correct state transitions. Cross-user/workflow thread reuse tests pass.

### Milestone B — Configuration does what it says

**Scope:** F05–F08 and a supported-capabilities API/manifest.

Wire tool calling into agents, implement strict JSON output enforcement, validate workflow input/output, and enforce tool argument/result schemas. Use the same policy for SSE and non-streaming calls. Reject unsupported config explicitly.

**Exit gate:** Every user-visible execution setting has either a runtime behavior test or a clear unsupported capability response.

### Milestone C — Complete the knowledge lifecycle

**Scope:** F09–F11 and F13.

Unify scope bindings, add opt-in graph ingestion, eliminate workspace collisions, persist indexing generations/leases, and make deletion/reindexing idempotent. Add a small retrieval evaluation dataset before tuning reranking or BM25.

**Exit gate:** Upload → index → search → scoped agent/workflow answer → reindex → delete succeeds for each supported provider, including duplicate job delivery and worker restart.

### Milestone D — Thin builder UI and run inspector

Build the narrow journey first: create agent → select supported tools/knowledge → ask a question → inspect a workflow run → approve/resume. Add visual graph editing around proven backend capabilities. Display unsupported node types as unavailable. A state/trace inspector is more valuable initially than a graph explorer or many dashboard pages.

**Exit gate:** A new internal user can complete that journey without editing JSON or misinterpreting an execution status.

### Milestone E — Controlled deployment

Add verified authentication/RBAC, execution isolation, network/database policies, secret management, usage budgets, readiness, CI, and backup/restore checks before untrusted access. Sequence these requirements according to the deployment audience; they need not block local core development.

**Exit gate:** An isolated staging deployment passes authentication, authorization, recovery, cancellation, migration, and resource-limit scenarios.

## 10. Highest-value product enhancements

| Enhancement | Why it helps | Timing |
|---|---|---|
| Capability discovery | Prevents the UI/API advertising configuration the runtime ignores | Milestones A/B |
| Run debugger with state and typed errors | Makes workflow building understandable and cuts diagnosis time | A/D |
| Typed state references and missing-variable errors | Avoids silent empty-string substitution and preserves non-string tool arguments | B |
| Versioned agent/workflow templates | Gives users working starting points after contracts stabilize | B/D |
| Retrieval evaluation pack | Measures source recall, citation correctness, grounded answers, and empty-answer behavior | C |
| Provider-specific indexing status | Explains why hybrid works while graph is still processing or unavailable | C |
| Explicit run cancellation and deadlines | Essential for long workflows, stuck tools, and user confidence | A/E |
| Per-run usage and cost accounting | Makes agent experimentation predictable and supports later quotas | B/E |
| Export/import of versioned definitions | A practical developer feature before marketplace/channel expansion | After B |

Defer advanced supervisor patterns, dynamic map-reduce, many channels, graph visualization, and search-engine replacement until the basic build/run/debug loop is reliable. This is prioritization based on the current code gaps, not a claim that those features lack value.

## 11. Verification backlog and definition of done

Add regression tests for the reproduced failures first. The audit runner records current behavior and is **not** a substitute for tests that assert the repaired behavior.

| Layer | Required scenarios |
|---|---|
| Pure unit | Full schema constraints; nullable types; missing variables; strict config; workspace identity; typed retry decisions |
| Graph integration with in-memory checkpointing | Multiple approvals; rejection; fan-out/merge; cancellation; cross-run state isolation; pinned revision resume |
| Isolated PostgreSQL integration | Migration fresh/upgrade paths; checkpoint persistence; concurrent default creation; advisory lock ownership; duplicate ingestion claims |
| HTTP/API | Create/update validation parity; unknown tools; honest failure responses; normal and streaming policy parity |
| Storage/worker integration | Partial upload failure; duplicate delivery; worker absent/restart; slow ingestion; concurrent delete/reindex; cleanup reconciliation |
| RAG evaluation | Known answer/source pairs, unanswerable queries, scope boundaries, graph versus hybrid availability, multi-query duplicate handling |
| Staging operations | Required initialization failure; provider outage; worker queue age; timeout termination; backup/restore; quotas |

Move automatic database initialization out of the global autouse fixture for pure tests. Use explicit integration markers and a disposable database URL. The current fixture calls `init_db()` against configured settings and suppresses failures, which can both mutate a configured database and hide missing infrastructure. Add CI for offline tests first, then a PostgreSQL service job, lint/type checks, and a container smoke test.

For each roadmap feature, require: **a persisted contract, runtime implementation, explicit failure behavior, regression coverage, documentation matching reality, and one successful end-to-end usage example**. That standard would close the largest current gap without changing the project's basic architecture.
