"""LangChain agent tools for hybrid document retrieval and knowledge search."""

import logging
from langchain_core.tools import tool

from app.core.rag_config import get_rag_config
from app.services.retriever import hybrid_search

logger = logging.getLogger(__name__)


@tool
async def hybrid_document_search(
    query: str,
    preset: str = "balanced",
    user_id: str = "root",
) -> str:
    """Searches user uploaded documents and knowledge base using Hybrid Search (PGVector HNSW Dense + PostgreSQL Full-Text Sparse + Reciprocal Rank Fusion).

    Use this tool whenever you need to find factual information, technical specifications, documentation,
    or citations from uploaded files to answer a user's question accurately.

    Args:
        query: The natural language search question or keyword query.
        preset: Retrieval strategy ('high_precision', 'balanced', or 'high_recall').
        user_id: User identifier scoping the document search (defaults to 'root').

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
            results.append(f"[Source {idx}: {filename}{section_str}]\n{doc.page_content}")

        return "\n\n---\n\n".join(results)
    except Exception as exc:
        logger.error("hybrid_document_search tool failed: %s", exc, exc_info=True)
        return f"Error executing hybrid search: {exc}"


AGENT_TOOLS = [hybrid_document_search]
