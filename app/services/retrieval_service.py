"""Centralized Knowledge Retrieval Service and Scope Orchestration.

Defines the core retrieval contracts and multi-scope dispatch:
- Scope 1 (global_only): User global documents only (documents without a thread_id).
- Scope 2 (thread_only): Documents strictly belonging to the active conversation thread.
- Scope 3 (thread_with_global): Thread-specific documents prioritized + global documents fallback.
- Scope 4 (all_user): Explicit cross-thread search spanning all user documents.

Handles dual-layer Knowledge Graph (LightRAG) traversal and active-thread RRF ranking boosts.
"""

from __future__ import annotations

from enum import Enum
import logging
from typing import Any

from langchain_core.documents import Document

from app.core.rag.factory import get_provider
from app.core.rag.protocols import RetrievalQuery
from app.core.rag.workspace import thread_workspace, user_workspace

logger = logging.getLogger(__name__)


class RetrievalScope(str, Enum):
    """Explicit retrieval boundary scopes."""

    GLOBAL_ONLY = "global_only"
    THREAD_ONLY = "thread_only"
    THREAD_WITH_GLOBAL = "thread_with_global"
    ALL_USER = "all_user"


def resolve_retrieval_scope(
    scope: RetrievalScope | str | None = None,
    thread_id: str | None = None,
    thread_scoped_only: bool = False,
) -> RetrievalScope:
    """Resolves the active retrieval scope based on explicit parameter or thread presence.

    Rules:
    - If explicit scope is provided: normalize and validate.
    - If no thread_id is supplied: defaults to GLOBAL_ONLY to prevent cross-thread leaks.
    - If thread_id is supplied and thread_scoped_only is True: THREAD_ONLY.
    - If thread_id is supplied and thread_scoped_only is False: THREAD_WITH_GLOBAL.
    """
    if scope is not None:
        resolved = None
        if isinstance(scope, str):
            try:
                resolved = RetrievalScope(scope.strip().lower())
            except ValueError:
                valid_scopes = [s.value for s in RetrievalScope]
                raise ValueError(
                    f"Invalid retrieval scope '{scope}'. Must be one of: {valid_scopes}"
                )
        elif isinstance(scope, RetrievalScope):
            resolved = scope
        else:
            raise ValueError(f"Invalid retrieval scope type: {type(scope).__name__}")

        if resolved in (RetrievalScope.THREAD_ONLY, RetrievalScope.THREAD_WITH_GLOBAL) and not thread_id:
            raise ValueError(f"RetrievalScope '{resolved.value}' requires a valid thread_id")
        return resolved

    if thread_id:
        return RetrievalScope.THREAD_ONLY if thread_scoped_only else RetrievalScope.THREAD_WITH_GLOBAL

    return RetrievalScope.GLOBAL_ONLY


def build_scoped_filter(
    user_id: str,
    scope: RetrievalScope | str | None = None,
    thread_id: str | None = None,
    thread_scoped_only: bool = False,
) -> dict[str, Any]:
    """Constructs a structured filter compatible with PGVector and PostgreSQL FTS sparse search.

    Guarantees user boundary enforcement in every query.
    """
    active_scope = resolve_retrieval_scope(
        scope=scope,
        thread_id=thread_id,
        thread_scoped_only=thread_scoped_only,
    )

    user_clause = {"user_id": {"$eq": str(user_id)}}

    if active_scope == RetrievalScope.THREAD_ONLY:
        if not thread_id:
            raise ValueError("RetrievalScope.THREAD_ONLY requires a valid thread_id")
        return {
            "$and": [
                user_clause,
                {"thread_id": {"$eq": str(thread_id)}},
            ]
        }

    if active_scope == RetrievalScope.THREAD_WITH_GLOBAL:
        if not thread_id:
            # Fall back to global only if thread_id is absent
            return {
                "$and": [
                    user_clause,
                    {"thread_id": {"$exists": False}},
                ]
            }
        return {
            "$and": [
                user_clause,
                {
                    "$or": [
                        {"thread_id": {"$eq": str(thread_id)}},
                        {"thread_id": {"$exists": False}},
                    ]
                },
            ]
        }

    if active_scope == RetrievalScope.ALL_USER:
        return user_clause

    # Default: RetrievalScope.GLOBAL_ONLY
    return {
        "$and": [
            user_clause,
            {"thread_id": {"$exists": False}},
        ]
    }


async def query_graph_rag_dual_layer(
    user_id: str,
    query: str,
    mode: str = "mix",
    top_k: int = 5,
    thread_id: str | None = None,
    scope: RetrievalScope | str | None = None,
    provider_getter: Any | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Executes dual-layer LightRAG knowledge graph traversal across thread and user workspaces.

    - Queries thread_workspace first if thread_id is present and scope allows.
    - Supplements with user_workspace (global knowledge) if scope permits.
    - Labels contexts clearly as Thread-Specific vs Global Knowledge Graph Context.

    Returns:
        (formatted_context_string, list_of_source_metadata_dicts)
    """
    getter = provider_getter or get_provider
    active_scope = resolve_retrieval_scope(scope=scope, thread_id=thread_id)

    if active_scope == RetrievalScope.ALL_USER:
        raise ValueError(
            "Knowledge graph retrieval (LightRAG) does not support 'all_user' cross-thread scope. "
            "LightRAG workspaces are isolated per thread and user. Please specify 'global_only', "
            "'thread_only', or 'thread_with_global', or use hybrid search for cross-thread queries."
        )

    retrieval_query = RetrievalQuery(
        query=query,
        top_k=top_k,
        mode=mode if mode in {"mix", "local", "global", "hybrid", "naive"} else "mix",
    )

    thread_context = ""
    global_context = ""
    sources: list[dict[str, Any]] = []

    # 1. Thread workspace query (if applicable)
    if thread_id and active_scope in (
        RetrievalScope.THREAD_ONLY,
        RetrievalScope.THREAD_WITH_GLOBAL,
    ):
        try:
            ws_thread = thread_workspace(user_id, thread_id)
            thread_provider = await getter("lightrag", workspace=ws_thread)
            res_thread = await thread_provider.query(retrieval_query)
            if res_thread.context and res_thread.context.strip():
                thread_context = res_thread.context.strip()
            for chunk in res_thread.chunks:
                sources.append({
                    "source": "Knowledge Graph (Thread)",
                    "chunk_id": chunk.chunk_id,
                    "preview": chunk.content[:150] if chunk.content else "",
                    "content": chunk.content,
                    "scope": "thread",
                    "thread_id": thread_id,
                    "metadata": chunk.metadata or {},
                })
        except Exception as exc:
            logger.warning("LightRAG thread workspace query failed for thread %s: %s", thread_id, exc)
            if active_scope == RetrievalScope.THREAD_ONLY:
                raise RuntimeError(f"Thread knowledge graph query failed: {exc}") from exc

    # 2. Global user workspace query (if applicable)
    if active_scope in (RetrievalScope.GLOBAL_ONLY, RetrievalScope.THREAD_WITH_GLOBAL):
        try:
            ws_global = user_workspace(user_id)
            user_provider = await getter("lightrag", workspace=ws_global)
            res_global = await user_provider.query(retrieval_query)
            if res_global.context and res_global.context.strip():
                global_context = res_global.context.strip()
            for chunk in res_global.chunks:
                # Deduplicate against thread chunks
                if not any(s.get("chunk_id") == chunk.chunk_id for s in sources):
                    sources.append({
                        "source": "Knowledge Graph (Global)",
                        "chunk_id": chunk.chunk_id,
                        "preview": chunk.content[:150] if chunk.content else "",
                        "content": chunk.content,
                        "scope": "global",
                        "thread_id": None,
                        "metadata": chunk.metadata or {},
                    })
        except Exception as exc:
            logger.warning("LightRAG user workspace query failed for user %s: %s", user_id, exc)
            if not thread_context:
                raise RuntimeError(f"Global knowledge graph query failed: {exc}") from exc

    # Format combined context
    sections: list[str] = []
    if thread_context:
        sections.append(f"#### Thread-Specific Graph Context\n{thread_context}")
    if global_context:
        sections.append(f"#### Global Knowledge Graph Context\n{global_context}")

    if not sections:
        return "", []

    formatted_context = f"### Knowledge Graph Traversal [Mode: {mode.upper()}]\n\n" + "\n\n".join(sections)
    return formatted_context, sources
