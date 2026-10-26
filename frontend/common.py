"""Shared Streamlit helpers: client construction, error rendering, formatting."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar

import httpx
import streamlit as st

from api_client import ApiError, BaseAgentClient, Identity

T = TypeVar("T")

DEFAULT_API_URL = os.getenv("BASE_AGENT_API_URL", "http://localhost:8000")
RAG_PRESETS = ["balanced", "high_precision", "high_recall"]
MODEL_OPTIONS = {
    "Server default": None,
    "deepseek-flash (DeepSeek-V4.1-Flash)": "deepseek-flash",
    "deepseek-v4-pro (DeepSeek-V4-Pro)": "deepseek-v4-pro",
}
STATUS_COLORS = {"INDEXED": "green", "PARSING": "orange", "PENDING": "gray", "FAILED": "red"}


def init_state() -> None:
    defaults = {
        "api_url": DEFAULT_API_URL,
        "user_id": "root",
        "role": "root",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_client() -> BaseAgentClient:
    """Returns a client bound to the current API URL and identity (one per session)."""
    url = st.session_state["api_url"].rstrip("/")
    client: BaseAgentClient | None = st.session_state.get("_client")
    if client is None or client.base_url != url:
        if client is not None:
            client.close()
        client = BaseAgentClient(url)
        st.session_state["_client"] = client
    client.identity = Identity(user_id=st.session_state["user_id"], role=st.session_state["role"])
    return client


def call(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T | None:
    """Runs an API call, rendering failures inline. Returns None on error."""
    try:
        return fn(*args, **kwargs)
    except ApiError as exc:
        hint = ""
        if exc.status_code == 403:
            hint = " — check the identity/role in the sidebar."
        elif exc.status_code == 429:
            hint = " — rate limit hit, wait a minute and retry."
        st.error(f"**{exc.status_code}** `{exc.method} {exc.path}`: {exc.detail}{hint}")
    except httpx.ConnectError:
        st.error(f"Cannot reach the API at `{st.session_state['api_url']}`. Is the server running?")
    except httpx.HTTPError as exc:
        st.error(f"Network error: {exc}")
    return None


def status_badge(status: str) -> str:
    color = STATUS_COLORS.get(status.upper(), "gray")
    return f":{color}-badge[{status}]"


def fmt_bytes(size: int | None) -> str:
    if size is None:
        return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def fmt_time(value: str | None) -> str:
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value
