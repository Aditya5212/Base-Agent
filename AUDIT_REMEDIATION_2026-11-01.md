# Audit remediation — 2026-11-01

This implements the core stabilization work identified by [the baseline audit](PROJECT_AUDIT_2026-10-31.md). Authentication remains intentionally deferred. The original audit and its recorded probes are historical evidence; use the regression suites below to check the current implementation.

## Findings and resulting behavior

| Audit finding | Current status | Implementation and practical limit |
|---|---|---|
| F01 Approval lifecycle | Core fix verified | Returned interrupts persist as INTERRUPTED with their payloads; repeated approvals work; completed runs cannot resume. Approval deadlines are enforced on resume and mark expired runs EXPIRED; no automatic expiry scheduler is claimed. |
| F02 Checkpoint collisions | Fixed and verified | Server-generated checkpoint keys include owner/workflow digest plus run UUID. Client thread IDs are descriptive, not continuation authority. Chat rejects the reserved workflow namespace. |
| F03 False tool success | Fixed and verified | Executor errors propagate; the test API reports failure; unknown tools do not return fabricated success; outputs retain native JSON types. |
| F04 Ignored tool approval | Fixed and verified | Approval interrupts occur before execution, outside retries/error-catching. Only `{"approved": true}` authorizes execution. PostgreSQL row locking serializes resume requests; subsequent requests reject completed/running runs. External side effects still require their own idempotency design. |
| F05 Ignored agent settings | Core contract repaired | Chat executes attached tools with model/tool call budgets, validates registered structured output, applies regex/topic filters, locale/style, model retries and thinking configuration. Unsupported memory/provider/fusion/tool-override settings reject explicitly. Protected SSE responses buffer the final checked answer. |
| F06 Workflow payload schemas | Fixed and verified | Workflow input, start input, end output, final output, tool arguments, and tool output are validated. Invalid final output fails the run. |
| F07 Unsupported nodes / fan-out | Guarded and verified | Eight executable node types are advertised. Parallel/subgraph, static fan-out, JSONPath routing, and unsupported retrieval scopes fail validation; those features were not implemented prematurely. |
| F08 Tool JSON Schema | Fixed and verified | Nullable/nested fields and schema constraints are enforced without implicit scalar coercion. Exact template references preserve JSON types. Invalid tool definitions/updates fail before commit. Local references only; object root for arguments. |
| F09 Graph upload integration | Implemented; provider invocation mocked in tests | `index_graph=true` opts standard upload into LightRAG. Document detail exposes independent provider readiness. Optional graph failure preserves hybrid readiness and reports FAILED for the graph. Real graph extraction/API calls were not run. |
| F10 Ineffective scope / attachment | Guarded and verified | Current ingestion contract supports user scope. Other scopes and attaching knowledge to non-retrieve nodes reject explicitly instead of pretending isolation works. Agent provider mode/top-k settings reach retrieval. |
| F11 Workspace collisions | Fixed; migration required for affected data | Ambiguous/long identities include a digest of the full structured identity. Existing graph data with affected names must be reindexed; no automatic graph migration was applied. |
| F12 Thread execution timeout | Lifecycle fixed; untrusted-code isolation remains open | Submitted code runs in a child process with a server-owned module policy, minimal environment, and termination/reaping on timeout or cancellation. This is trusted-code execution, not an OS sandbox or a guaranteed memory/CPU isolation boundary. |
| F13 Ingestion consistency | Primary index fixed; cross-provider reconciliation remains | Dedicated per-document PostgreSQL locks coordinate processing/delete/reindex/recovery. Pending generations have stable chunk IDs; both search paths exclude staged chunks. Generation activation/deletion and inspection rows commit in one database transaction; rollback preserves the previous visible index. Deleted/complete jobs skip, original preset is retained, and parsing runs off the event loop. LightRAG/storage reconciliation and durable outbox recovery across external failure boundaries remain open. |
| F14 Mutable workflow resume | Graph snapshot fixed and verified | Each run persists its execution graph, schemas, recursion limit, and version. Resume uses that snapshot even after editing the draft. Referenced tool/schema revision history and publishing/rollback are still future capabilities. |
| F15 Hidden operational failure | Partially resolved | Required initialization fails production startup; readiness includes initialization; Docker health uses readiness; missing model/embedding credentials cannot silently use mocks in production; retrieval outages raise. Fresh-DB setup initializes vector tables before indexes and uses autocommit checkpointer connections. Worker heartbeat/queue-age monitoring, coordinated multi-worker migrations, retention, and operational recovery instrumentation remain open. |

## Verification

The stabilization suite passed against a **new disposable PostgreSQL 15/pgvector Docker database** before the subsequent type-fix batches. The database used synthetic credentials on a random localhost port, with no host-mounted application data. The application database and cloud accounts were not used.

- PostgreSQL stabilization run: **204 passed, 0 skipped, 4 warnings** (12.67 seconds).
- Recorded offline stabilization run: **123 passed, 81 explicitly skipped, 2 warnings** (9.20 seconds).
- Final offline run after type fixes and six additional regression tests: **129 passed, 81 explicitly skipped, 2 warnings** (9.39 seconds). The PostgreSQL suite was not rerun after these follow-up changes.
- Pyright checked the entire `app/` directory after the type fixes: **0 errors, 0 warnings**. The temporary checker did not change project dependencies or the lockfile.

- All four existing Alembic revisions and the new revision applied to the fresh database.
- Downgrading the new revision and upgrading it again also succeeded on the temporary database.
- The full existing suite ran alongside new offline and real persistence regressions.
- A real PostgreSQL approval run persisted, the checkpoint pool was closed/reopened, the draft was edited, and resume executed the original snapshot.
- Two concurrent resume attempts yielded one completed run and one rejected request.
- Separate database connections contended on and released a document lock correctly.
- Repeated vector indexing upserted one stable chunk; hybrid retrieval returned it only for its owner.
- Staged chunks stayed hidden from both retrieval paths; rolling back activation preserved the old index, and committing activation exposed only the replacement generation.
- The actual ingestion pipeline was fault-injected during metadata activation: the old generation remained searchable and a retry published the replacement with one inspection row.
- Code timeout and cancellation terminated child processes.
- Offline SQL rendering verified the migration chain. Dependency deprecation warnings and the existing experimental v3 streaming warning remain.

Results: [full integration suite](audit/remediation-integration-tests.xml), [offline suite](audit/remediation-tests.xml). Test code: [offline regressions](tests/offline), [PostgreSQL regressions](tests/integration/test_persistence_regressions.py).

Normal testing is now offline by default:

```powershell
.venv/Scripts/python.exe -m pytest -q
```

Service-backed tests are explicitly skipped unless a disposable test database is provided:

```powershell
$env:TEST_DATABASE_URL = 'postgresql+psycopg://test_user:test_password@127.0.0.1:5432/disposable_test_db'
.venv/Scripts/python.exe -m pytest --run-integration -q
```

That command applies migrations to the named test database. Never point it at application data. Pure and service-backed tests share the suite; skipped tests are reported explicitly, and initialization errors are no longer swallowed.

## Migration and compatibility notes

- Apply Alembic revision `audit_execution_identity` before running the changed application models. It adds `workflow_runs.checkpoint_id`, `workflow_runs.definition_snapshot`, `documents.indexing_config`, and `documents.provider_status`. Your application database has not been migrated during this work.
- New runs use isolated checkpoint IDs. Pre-existing paused runs without snapshots must be restarted; they are rejected with HTTP 409 rather than resumed under a potentially different graph or colliding identity.
- Ordinary chat checkpoint IDs/history retain their existing format. The `workflow:` prefix is now reserved.
- LightRAG names for punctuation-bearing or long user/thread/agent IDs change. Reindex affected graph knowledge before relying on graph retrieval. Short unambiguous names, including `u_root`, are retained.
- Unsupported settings that previously appeared to save successfully now fail with explicit capability errors. The capabilities endpoint is `GET /api/v1/workflows/capabilities`.
- Exact references such as `{{variables.quantity}}` pass numeric/list/object/null values to tools without converting them to text. Mixed templates such as `Order {{quantity}}` remain strings.
- Tool results are native JSON values. Consumers that previously parsed JSON text in a successful result should use the native value.
- Protected/structured/tool-using SSE turns emit their checked answer after completion. Plain text turns retain token streaming.

## Remaining work before broader deployment

The code fixes do not close every release gate in the audit. Prioritize LightRAG/storage reconciliation and durable job/outbox recovery next. Add worker liveness/queue-age signals, cross-process startup migration coordination, checkpoint retention, and linked tool/schema revision pinning before relying on long-running published workflows. For untrusted users/code/tools, add verified authentication/RBAC, OS isolation, network egress policy, independent secret-key management/rotation, and distributed cost/concurrency controls.

No paid model calls, R2 operations, real LightRAG extraction, load test, or production deployment was performed. The temporary Docker database is used only for verification and is removed afterward.

Implementation follows the documented LangGraph interrupt/resume and model tool-call interfaces: [interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts), [model/tool calls](https://docs.langchain.com/oss/python/langchain/models). Thinking configuration follows [DeepSeek's API documentation](https://api-docs.deepseek.com/guides/thinking_mode/).
