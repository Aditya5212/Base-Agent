import re

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
    return sanitize(f"u_{user_id}")


def thread_workspace(user_id: str, thread_id: str) -> str:
    """Thread-isolated knowledge graph bucket (per-session precision graph)."""
    return sanitize(f"u_{user_id}_t_{thread_id}")


def agent_workspace(user_id: str, agent_id: str) -> str:
    """Isolated knowledge base per custom agent for the Agent Builder platform."""
    return sanitize(f"u_{user_id}_a_{agent_id}")


def global_workspace() -> str:
    """Global system-wide knowledge base shared across users."""
    return "global_kb"
