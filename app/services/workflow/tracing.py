from __future__ import annotations

import asyncio
import contextvars
from datetime import datetime, timezone
import logging
import re
import time
from typing import Any, Callable
import uuid

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import AsyncSessionLocal
from app.models.workflow import WorkflowRunEvent
from app.services.workflow.state import WorkflowState

logger = logging.getLogger(__name__)

# Context variables for transaction/session awareness and test isolation
_CURRENT_DB_SESSION: contextvars.ContextVar[Any | None] = contextvars.ContextVar("_CURRENT_DB_SESSION", default=None)
_EVENT_HOOK: contextvars.ContextVar[Callable | None] = contextvars.ContextVar("_EVENT_HOOK", default=None)


def set_execution_session(session: Any):
    """Sets the active database session for tracing event writes."""
    return _CURRENT_DB_SESSION.set(session)


def reset_execution_session(token) -> None:
    """Resets the active database session token."""
    _CURRENT_DB_SESSION.reset(token)


def set_event_hook(hook: Callable):
    """Sets a custom event interception hook (e.g. for testing)."""
    return _EVENT_HOOK.set(hook)


def reset_event_hook(token) -> None:
    """Resets the event interception hook."""
    _EVENT_HOOK.reset(token)

# Patterns for sensitive keys and signed URLs
_SENSITIVE_KEYS_REGEX = re.compile(r"(?i)(api[_-]?key|token|password|secret|auth|credential|signature|private[_-]?key)")
_SIGNED_URL_REGEX = re.compile(r"https?://[^\s\"']+(?:X-Amz-Signature|signature|sig)=([^\s\"'&]+)", re.IGNORECASE)
_DB_URI_REGEX = re.compile(r"://([^:@]+):([^@]+)@", re.IGNORECASE)


def redact_value(value: Any, max_depth: int = 5, max_str_len: int = 500) -> Any:
    """Recursively redacts secrets, tokens, credentials, and truncates large payloads."""
    if max_depth <= 0:
        return "[MAX_DEPTH_REACHED]"

    if isinstance(value, dict):
        redacted = {}
        for k, v in value.items():
            if _SENSITIVE_KEYS_REGEX.search(str(k)):
                redacted[k] = "[REDACTED]"
            else:
                redacted[k] = redact_value(v, max_depth=max_depth - 1, max_str_len=max_str_len)
        return redacted

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if len(items) > 5:
            truncated = [redact_value(item, max_depth=max_depth - 1, max_str_len=max_str_len) for item in items[:5]]
            truncated.append(f"[+{len(items) - 5} items truncated]")
            return truncated
        return [redact_value(item, max_depth=max_depth - 1, max_str_len=max_str_len) for item in items]

    if isinstance(value, str):
        # Sanitize signed URLs
        sanitized = _SIGNED_URL_REGEX.sub("[REDACTED_URL]", value)
        # Sanitize connection strings
        sanitized = _DB_URI_REGEX.sub(r"://\1:[REDACTED]@", sanitized)
        if len(sanitized) > max_str_len:
            return sanitized[:max_str_len] + "...[truncated]"
        return sanitized

    return value


def redact_error_message(error: str | None) -> str | None:
    """Sanitizes sensitive connection strings and paths from error messages."""
    if not error:
        return None
    sanitized = _DB_URI_REGEX.sub(r"://\1:[REDACTED]@", error)
    sanitized = _SIGNED_URL_REGEX.sub("[REDACTED_URL]", sanitized)
    if len(sanitized) > 2000:
        return sanitized[:2000] + "...[truncated]"
    return sanitized


async def record_run_event(
    run_id: str | uuid.UUID,
    sequence: int | None = None,
    invocation_id: str | uuid.UUID | None = None,
    node_id: str = "system",
    node_type: str = "system",
    event_type: str = "node_started",
    attempt: int = 1,
    duration_ms: int | None = None,
    status: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    input_summary: dict[str, Any] | None = None,
    output_summary: dict[str, Any] | None = None,
    model_id: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    tool_name: str | None = None,
    tool_invocation_id: str | None = None,
    metadata_json: dict[str, Any] | None = None,
    event_session: AsyncSession | None = None,
) -> WorkflowRunEvent | None:
    """Persists a single immutable execution trace event."""
    try:
        run_uuid = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        inv_uuid = (
            uuid.UUID(str(invocation_id))
            if invocation_id is not None and not isinstance(invocation_id, uuid.UUID)
            else (invocation_id or uuid.uuid4())
        )

        event = WorkflowRunEvent(
            run_id=run_uuid,
            sequence=sequence or 1,
            invocation_id=inv_uuid,
            node_id=node_id,
            node_type=node_type,
            attempt=attempt,
            event_type=event_type,
            occurred_at=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            status=status,
            error_code=error_code,
            error_message=redact_error_message(error_message),
            input_summary=redact_value(input_summary, max_str_len=500) if input_summary is not None else None,
            output_summary=redact_value(output_summary, max_str_len=1000) if output_summary is not None else None,
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tool_name=tool_name,
            tool_invocation_id=tool_invocation_id,
            metadata_json=metadata_json or {},
        )

        # 1. Check custom event hook (e.g. testing or real-time event streaming)
        hook = _EVENT_HOOK.get()
        if hook is not None:
            if asyncio.iscoroutinefunction(hook):
                await hook(event)
            else:
                hook(event)
            return event

        # 2. Check ambient mock session (offline unit tests)
        active_session = _CURRENT_DB_SESSION.get()
        if active_session is not None and not isinstance(active_session, AsyncSession):
            if hasattr(active_session, "add"):
                if sequence is None:
                    next_s = getattr(active_session, "_seq_counter", 0) + 1
                    setattr(active_session, "_seq_counter", next_s)
                    event.sequence = next_s
                active_session.add(event)
            return event

        # Control transitions already hold the run lock; append in that transaction.
        if event_session is not None:
            from app.models.workflow import WorkflowRun
            from sqlalchemy import update

            allocated = await event_session.execute(
                update(WorkflowRun).where(WorkflowRun.id == run_uuid)
                .values(next_event_sequence=WorkflowRun.next_event_sequence + 1)
                .returning(WorkflowRun.next_event_sequence)
            )
            event.sequence = allocated.scalar_one() - 1
            event_session.add(event)
            await event_session.flush()
            return event

        # 3. Persist independently in an autonomous short transaction with atomic sequence allocation
        try:
            from app.models.workflow import WorkflowRun
            from sqlalchemy import select

            async with asyncio.timeout(3.0):
                async with AsyncSessionLocal() as session:
                    stmt = select(WorkflowRun).where(WorkflowRun.id == run_uuid).with_for_update()
                    res = await session.execute(stmt)
                    run_row = res.scalar_one_or_none()
                    if run_row is not None:
                        allocated_seq = run_row.next_event_sequence or 1
                        event.sequence = allocated_seq
                        run_row.next_event_sequence = allocated_seq + 1
                    elif sequence is not None:
                        event.sequence = sequence

                    session.add(event)
                    await session.commit()
                    return event
        except Exception as db_exc:
            raise ExecutionObservabilityDegradedError("Execution event could not be committed") from db_exc
    except Exception as exc:
        logger.warning(
            "Failed to record execution trace event [run_id=%s, node_id=%s, event=%s]: %s",
            run_id,
            node_id,
            event_type,
            exc,
        )
        raise ExecutionObservabilityDegradedError("Execution event persistence failed") from exc


class WorkflowCancelledError(Exception):
    """Raised when workflow execution is cooperatively cancelled at a node boundary."""
    pass


class WorkflowTimeoutError(Exception):
    """Raised when workflow execution exceeds its configured budget_seconds."""
    pass


class WorkerFencedError(Exception):
    """Raised when worker lease is lost or fenced by another worker."""
    pass


class ExecutionObservabilityDegradedError(Exception):
    """Raised when execution events cannot be persisted to storage."""
    pass


def wrap_node_with_tracing(node_id: str, node_type: str, handler: Callable) -> Callable:
    """Wraps a workflow node handler with automated lifecycle tracing and duration timing."""

    async def traced_handler(state: WorkflowState) -> dict[str, Any]:
        raw_run_id = state.get("run_id")
        if not raw_run_id:
            # If run_id is not in state (e.g. standalone test execution), execute directly
            return await handler(state)

        # -------------------------------------------------------------------
        # Cooperative boundary check: read fresh persisted control & fencing state
        # -------------------------------------------------------------------
        budget_sec = state.get("budget_seconds")
        consumed_sec = float(state.get("consumed_seconds", 0.0) or 0.0)
        started_at = state.get("execution_started_at")
        ctrl_req = state.get("control_request")
        current_worker_id = state.get("worker_id")
        current_generation = state.get("generation")

        active_sess = _CURRENT_DB_SESSION.get()
        if active_sess is not None and isinstance(active_sess, AsyncSession):
            # Real database execution: query fresh state bypassing identity map
            try:
                from app.models.workflow import WorkflowRun
                from sqlalchemy import select
                r_uuid = uuid.UUID(str(raw_run_id))

                async with AsyncSessionLocal() as read_sess:
                    stmt = select(
                        WorkflowRun.control_request,
                        WorkflowRun.status,
                        WorkflowRun.budget_seconds,
                        WorkflowRun.consumed_seconds,
                        WorkflowRun.execution_started_at,
                        WorkflowRun.worker_id,
                        WorkflowRun.generation,
                        WorkflowRun.lease_expires_at,
                    ).where(WorkflowRun.id == r_uuid)
                    row_res = await read_sess.execute(stmt)
                    row = row_res.first()
                    if row:
                        ctrl_req = row.control_request
                        run_status = row.status
                        if budget_sec is None and row.budget_seconds:
                            budget_sec = row.budget_seconds
                        if row.consumed_seconds is not None:
                            consumed_sec = float(row.consumed_seconds)
                        if row.execution_started_at is not None:
                            started_at = row.execution_started_at

                        # Fencing check: ensure another worker hasn't claimed or recovered the run
                        if current_worker_id and row.worker_id and row.worker_id != current_worker_id:
                            raise WorkerFencedError(
                                f"Worker fenced: run '{raw_run_id}' claimed by worker '{row.worker_id}', "
                                f"current worker is '{current_worker_id}'"
                            )
                        if current_generation is not None and row.generation is not None and row.generation != current_generation:
                            raise WorkerFencedError(
                                f"Worker fenced: run '{raw_run_id}' generation is {row.generation}, "
                                f"current worker generation is {current_generation}"
                            )
                        if current_worker_id and (
                            row.lease_expires_at is None
                            or row.lease_expires_at <= datetime.now(timezone.utc)
                        ):
                            raise WorkerFencedError("Worker lease expired before node execution")
                        if run_status in ("FAILED", "TIMED_OUT", "CANCELLED"):
                            raise WorkerFencedError(f"Worker fenced: run '{raw_run_id}' is in terminal status '{run_status}'")
            except WorkerFencedError:
                raise
            except Exception as exc:
                raise WorkerFencedError("Cannot verify current execution ownership") from exc
        elif active_sess is not None and hasattr(active_sess, "get"):
            # Offline test mode with mock session: read directly from mock session or state without network calls
            try:
                from app.models.workflow import WorkflowRun
                r_uuid = uuid.UUID(str(raw_run_id))
                run_row = await active_sess.get(WorkflowRun, r_uuid)
                if run_row and isinstance(run_row, WorkflowRun):
                    if budget_sec is None and run_row.budget_seconds:
                        budget_sec = run_row.budget_seconds
                    if run_row.consumed_seconds is not None:
                        consumed_sec = float(run_row.consumed_seconds)
                    if run_row.execution_started_at is not None:
                        started_at = run_row.execution_started_at
                    if run_row.control_request:
                        ctrl_req = run_row.control_request
            except Exception:
                pass

        if budget_sec and not isinstance(budget_sec, (int, float)):
            try:
                budget_sec = float(budget_sec)
            except (ValueError, TypeError):
                budget_sec = None

        if budget_sec is not None and budget_sec > 0:
            active_elapsed = 0.0
            if started_at:
                if isinstance(started_at, str):
                    try:
                        started_at = datetime.fromisoformat(started_at)
                    except Exception:
                        started_at = None
                if isinstance(started_at, datetime):
                    now = datetime.now(timezone.utc)
                    if started_at.tzinfo is None:
                        started_at = started_at.replace(tzinfo=timezone.utc)
                    active_elapsed = max(0.0, (now - started_at).total_seconds())
            total_active = consumed_sec + active_elapsed
            if total_active > float(budget_sec):
                inv_id = uuid.uuid4()
                current_seq = (state.get("event_sequence", 0) or 0) + 1
                await record_run_event(
                    run_id=raw_run_id,
                    sequence=current_seq,
                    invocation_id=inv_id,
                    node_id=node_id,
                    node_type=node_type,
                    event_type="timed_out",
                    status="timed_out",
                    error_message=f"Execution budget of {budget_sec}s exceeded (consumed {total_active:.2f}s)",
                )
                raise WorkflowTimeoutError(f"Workflow execution timed out: budget of {budget_sec}s exceeded")

        if ctrl_req == "cancel_requested":
            inv_id = uuid.uuid4()
            current_seq = (state.get("event_sequence", 0) or 0) + 1
            await record_run_event(
                run_id=raw_run_id,
                sequence=current_seq,
                invocation_id=inv_id,
                node_id=node_id,
                node_type=node_type,
                event_type="cancelled",
                status="cancelled",
                error_message="Workflow execution was cancelled by user request",
            )
            raise WorkflowCancelledError(f"Workflow execution cancelled before node '{node_id}'")

        if ctrl_req == "pause_requested":
            inv_id = uuid.uuid4()
            current_seq = (state.get("event_sequence", 0) or 0) + 1
            await record_run_event(
                run_id=raw_run_id,
                sequence=current_seq,
                invocation_id=inv_id,
                node_id=node_id,
                node_type=node_type,
                event_type="paused",
                status="interrupted",
            )
            from langgraph.types import interrupt
            interrupt({"type": "manual_pause", "node_id": node_id, "reason": "user_requested"})

        # Persist invocation identity deterministically across checkpoints & replays
        node_occurrences = dict(state.get("node_occurrences") or {})
        occurrence = node_occurrences.get(node_id, 0) + 1
        node_occurrences[node_id] = occurrence

        node_invocations = dict(state.get("node_invocations") or {})
        occ_key = f"{node_id}:{occurrence}"
        if occ_key not in node_invocations:
            node_invocations[occ_key] = str(uuid.uuid5(uuid.UUID(str(raw_run_id)), occ_key))
        invocation_id = uuid.UUID(node_invocations[occ_key])

        # Attach invocation_id to state for tool nodes or downstream introspection
        state["current_invocation_id"] = str(invocation_id)
        state["node_occurrences"] = node_occurrences
        state["node_invocations"] = node_invocations
        tool_invocation_id = f"{raw_run_id}:{node_id}:{invocation_id}:{occurrence}" if node_type == "tool_call" else None

        # 1. Event: node_started
        input_snapshot = dict(state.get("variables") or {})
        start_seq = (state.get("event_sequence", 0) or 0) + 1
        state["event_sequence"] = start_seq
        start_ev = await record_run_event(
            run_id=raw_run_id,
            sequence=start_seq,
            invocation_id=invocation_id,
            node_id=node_id,
            node_type=node_type,
            event_type="node_started",
            input_summary=redact_value(input_snapshot, max_str_len=500),
            tool_invocation_id=tool_invocation_id,
        )

        t0 = time.perf_counter()
        try:
            result = await handler(state)
            duration_ms = int((time.perf_counter() - t0) * 1000)

            # Determine outputs from result
            output_snapshot = None
            if isinstance(result, dict):
                output_snapshot = dict(result.get("variables") or {})

            # Extract LLM tokens or tool invocation info if present
            model_id = None
            prompt_tokens = None
            completion_tokens = None
            tool_name = None
            tool_invocation_id = None

            if node_type == "chat":
                # Check for model metadata
                new_msgs = result.get("messages") or []
                for msg in reversed(new_msgs):
                    usage = getattr(msg, "usage_metadata", None) or getattr(msg, "response_metadata", {}).get("token_usage")
                    if isinstance(usage, dict):
                        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
                        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
                        break

            elif node_type == "tool_call":
                tool_name = (output_snapshot or {}).get("tool") or node_id
                tool_invocation_id = f"{raw_run_id}:{node_id}:{invocation_id}:{occurrence}"

            # 2. Event: node_completed
            done_seq = (state.get("event_sequence", 0) or 0) + 1
            state["event_sequence"] = done_seq
            comp_ev = await record_run_event(
                run_id=raw_run_id,
                sequence=done_seq,
                invocation_id=invocation_id,
                node_id=node_id,
                node_type=node_type,
                event_type="node_completed",
                duration_ms=duration_ms,
                status="success",
                output_summary=redact_value(output_snapshot, max_str_len=1000) if output_snapshot is not None else None,
                model_id=model_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                tool_name=tool_name,
                tool_invocation_id=tool_invocation_id,
            )

            if isinstance(result, dict):
                result["node_occurrences"] = node_occurrences
                result["node_invocations"] = node_invocations
                result["event_sequence"] = comp_ev.sequence if (comp_ev is not None and isinstance(getattr(comp_ev, "sequence", None), int)) else done_seq
            return result

        except Exception as exc:
            duration_ms = int((time.perf_counter() - t0) * 1000)

            # Check if this is an expected interrupt (human-in-the-loop)
            is_interrupt = exc.__class__.__name__ == "GraphInterrupt"
            event_type = "approval_requested" if is_interrupt else "node_failed"
            status = "interrupted" if is_interrupt else "failed"

            fail_seq = (state.get("event_sequence", 0) or 0) + 1
            state["event_sequence"] = fail_seq
            await record_run_event(
                run_id=raw_run_id,
                sequence=fail_seq,
                invocation_id=invocation_id,
                node_id=node_id,
                node_type=node_type,
                event_type=event_type,
                duration_ms=duration_ms,
                status=status,
                error_code=exc.__class__.__name__,
                error_message=redact_error_message(str(exc)),
                tool_invocation_id=tool_invocation_id,
            )
            # Re-raise so LangGraph handles interrupt / error
            raise

    return traced_handler
