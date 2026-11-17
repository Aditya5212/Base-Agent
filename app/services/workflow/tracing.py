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
    sequence: int,
    invocation_id: str | uuid.UUID,
    node_id: str,
    node_type: str,
    event_type: str,
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
) -> WorkflowRunEvent | None:
    """Persists a single immutable execution trace event."""
    try:
        run_uuid = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        inv_uuid = uuid.UUID(str(invocation_id)) if not isinstance(invocation_id, uuid.UUID) else invocation_id

        event = WorkflowRunEvent(
            run_id=run_uuid,
            sequence=sequence,
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

        # 2. Check active ambient database session
        active_session = _CURRENT_DB_SESSION.get()
        if active_session is not None and hasattr(active_session, "add"):
            active_session.add(event)
            if hasattr(active_session, "flush") and asyncio.iscoroutinefunction(active_session.flush):
                try:
                    await active_session.flush()
                except Exception:
                    pass
            return event

        # 3. Fallback to independent connection session with timeout
        async with asyncio.timeout(2.0):
            async with AsyncSessionLocal() as session:
                session.add(event)
                await session.commit()
                return event
    except Exception as exc:
        logger.warning(
            "Failed to record execution trace event [run_id=%s, node_id=%s, event=%s]: %s",
            run_id,
            node_id,
            event_type,
            exc,
        )
        return None


def wrap_node_with_tracing(node_id: str, node_type: str, handler: Callable) -> Callable:
    """Wraps a workflow node handler with automated lifecycle tracing and duration timing."""

    async def traced_handler(state: WorkflowState) -> dict[str, Any]:
        raw_run_id = state.get("run_id")
        if not raw_run_id:
            # If run_id is not in state (e.g. standalone test execution), execute directly
            return await handler(state)

        invocation_id = uuid.uuid4()
        current_seq = state.get("event_sequence", 0)

        # 1. Event: node_started
        start_seq = current_seq + 1
        input_snapshot = dict(state.get("variables") or {})
        await record_run_event(
            run_id=raw_run_id,
            sequence=start_seq,
            invocation_id=invocation_id,
            node_id=node_id,
            node_type=node_type,
            event_type="node_started",
            input_summary=redact_value(input_snapshot, max_str_len=500),
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
                tool_invocation_id = f"{raw_run_id}:{node_id}:{invocation_id}:1"

            # 2. Event: node_completed
            complete_seq = start_seq + 1
            await record_run_event(
                run_id=raw_run_id,
                sequence=complete_seq,
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
                result["event_sequence"] = complete_seq
            return result

        except Exception as exc:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            failed_seq = start_seq + 1

            # Check if this is an expected interrupt (human-in-the-loop)
            is_interrupt = exc.__class__.__name__ == "GraphInterrupt"
            event_type = "approval_requested" if is_interrupt else "node_failed"
            status = "interrupted" if is_interrupt else "failed"

            await record_run_event(
                run_id=raw_run_id,
                sequence=failed_seq,
                invocation_id=invocation_id,
                node_id=node_id,
                node_type=node_type,
                event_type=event_type,
                duration_ms=duration_ms,
                status=status,
                error_code=exc.__class__.__name__,
                error_message=redact_error_message(str(exc)),
            )
            # Re-raise so LangGraph handles interrupt / error
            raise

    return traced_handler
