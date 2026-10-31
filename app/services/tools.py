"""LangChain agent tools for hybrid document retrieval and knowledge search.

Re-exports the modular RAG research tool suite from app.services.rag_tools
while preserving complete backwards-compatibility for existing call-sites and test fixtures.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool, tool

from app.core.rag_config import get_rag_config
from app.services.rag_tools import (
    get_user_rag_tool_map,
    get_user_rag_tools,
    make_graph_rag_tool,
    make_hybrid_search_tool,
    make_multi_query_search_tool,
)
from app.services.retriever import hybrid_search

logger = logging.getLogger(__name__)


def make_search_tool(user_id: str) -> BaseTool:
    """Returns a hybrid_document_search tool pre-bound to *user_id*.

    Backwards-compatible implementation preserving patchability of app.services.tools.hybrid_search.
    """

    @tool
    async def hybrid_document_search(
        query: str,
        preset: str = "balanced",
    ) -> str:
        """Searches user uploaded documents and knowledge base using Hybrid Search (PGVector HNSW Dense + PostgreSQL Full-Text Sparse + Reciprocal Rank Fusion).

        Args:
            query: The natural language search question or keyword query.
            preset: Retrieval strategy ('high_precision', 'balanced', or 'high_recall').

        Returns:
            Formatted passages with source citations and section headers.
        """
        try:
            rag_cfg = get_rag_config(preset)
            fused_docs = await hybrid_search(
                query=query,
                config=rag_cfg,
                filter_dict={"user_id": user_id},
            )
            if not fused_docs:
                return f"No relevant documents found in knowledge base for query: '{query}'."

            results = []
            for idx, doc in enumerate(fused_docs, 1):
                filename = doc.metadata.get("filename") or doc.metadata.get("source") or "Document"
                headers = [v for k, v in doc.metadata.items() if k.startswith("header_")]
                section_str = f" | Section: {' > '.join(headers)}" if headers else ""
                results.append(f"[Source {idx}: {filename}{section_str}]\n{doc.page_content.strip()}")

            return "\n\n---\n\n".join(results)
        except Exception as exc:
            logger.error("hybrid_document_search tool failed: %s", exc, exc_info=True)
            return f"Error executing hybrid search: {exc}"

    return hybrid_document_search


# Default tool instances (user_id="root") kept for backwards-compatibility with
# existing code that imports AGENT_TOOLS directly.
AGENT_TOOLS: list[BaseTool] = [
    make_search_tool("root"),
    make_graph_rag_tool("root"),
    make_multi_query_search_tool("root"),
]

__all__ = [
    "AGENT_TOOLS",
    "hybrid_search",
    "get_rag_config",
    "make_search_tool",
    "make_hybrid_search_tool",
    "make_graph_rag_tool",
    "make_multi_query_search_tool",
    "get_user_rag_tools",
    "get_user_rag_tool_map",
]
