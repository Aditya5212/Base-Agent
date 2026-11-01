import re
import hashlib
import json

_VALID = re.compile(r"[^a-zA-Z0-9_]")


def sanitize(raw: str) -> str:
    """Sanitize workspace identifier.

    LightRAG workspaces only allow alphanumeric characters and underscores [a-zA-Z0-9_],
    with a maximum length of 60 characters to fit PostgreSQL identifier limits.
    """
    cleaned = _VALID.sub("_", raw).strip("_")
    # Collapse multiple consecutive underscores
    cleaned = re.sub(r"_+", "_", cleaned)
    return (cleaned[:60] or "default")


def user_workspace(user_id: str) -> str:
    """User-level knowledge graph bucket (global to all user documents)."""
    return _workspace("u", [user_id])


def thread_workspace(user_id: str, thread_id: str) -> str:
    """Thread-isolated knowledge graph bucket (per-session precision graph)."""
    return _workspace("t", [user_id, thread_id])


def agent_workspace(user_id: str, agent_id: str) -> str:
    """Isolated knowledge base per custom agent for the Agent Builder platform."""
    return _workspace("a", [user_id, agent_id])


def _workspace(kind: str, parts: list[str]) -> str:
    raw = f"u_{parts[0]}" if kind == "u" else f"u_{parts[0]}_{kind}_{parts[1]}"
    # Preserve short, unambiguous legacy names (including u_root).
    if all(re.fullmatch(r"[A-Za-z0-9]+", part) for part in parts) and len(raw) <= 60:
        return raw
    digest = hashlib.sha256(json.dumps([kind, *parts], ensure_ascii=False).encode()).hexdigest()[:24]
    return f"{sanitize(raw)[:34]}_{digest}"


def global_workspace() -> str:
    """Global system-wide knowledge base shared across users."""
    return "global_kb"
