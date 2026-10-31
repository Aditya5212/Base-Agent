"""Modular RAG research tools covering Exploitation (precision facts & knowledge graphs)
and Exploration (agent-driven multi-query parallel hybrid search).

Designed for 3-tier reusability:
1. Chatbot Level: Directly callable in multi-turn chat sessions (/api/v1/chat)
2. Agent Node Level: Declaratively bound via AgentConfig.tools in Agent Builder
3. Workflow Engine Level: Executable inside StateGraph tool_call and retrieve nodes

Tenant Security:
Every factory accepts user_id at construction and closes over it.
The LLM-callable tool parameters never expose user_id or workspace paths.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.tools import BaseTool, tool

from app.core.rag.factory import get_provider
from app.core.rag.protocols import RetrievalQuery
from app.core.rag.workspace import user_workspace
from app.core.rag_config import get_rag_config
from app.services.multi_query import multi_query_hybrid_search
from app.services.retriever import hybrid_search

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exploitation Tool 1: Hybrid Document Search (PGVector + PostgreSQL FTS + RRF)
# ---------------------------------------------------------------------------

def make_hybrid_search_tool(user_id: str) -> BaseTool:
    """Returns a hybrid_document_search tool pre-bound to *user_id*.

    Targeted precision retrieval combining Dense vector search with Sparse keyword FTS via RRF.
    Best for finding specific factual answers, code references, numbers, or exact quotes for a single query.
    """

    @tool
    async def hybrid_document_search(
        query: str,
        preset: Literal["high_precision", "balanced", "high_recall"] = "balanced",
        top_k: int = 5,
    ) -> str:
        """Searches user uploaded documents and knowledge base using Hybrid Search (PGVector HNSW Dense + PostgreSQL Full-Text Sparse + Reciprocal Rank Fusion).

        Use this tool whenever you need targeted, factual information, technical specifications, documentation,
        or exact citations for a specific question or topic.

        Args:
            query: The natural language search question or keyword query.
            preset: Retrieval strategy ('high_precision', 'balanced', or 'high_recall').
            top_k: Maximum number of relevant passages to return.

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

            selected_docs = fused_docs[:top_k]
            results: list[str] = []
            for idx, doc in enumerate(selected_docs, 1):
                filename = doc.metadata.get("filename") or doc.metadata.get("source") or "Document"
                headers = [v for k, v in doc.metadata.items() if k.startswith("header_")]
                section_str = f" | Section: {' > '.join(headers)}" if headers else ""
                results.append(f"[Source {idx}: {filename}{section_str}]\n{doc.page_content.strip()}")

            return "\n\n---\n\n".join(results)
        except Exception as exc:
            logger.error("hybrid_document_search failed: %s", exc, exc_info=True)
            return f"Error executing hybrid search: {exc}"

    return hybrid_document_search


# ---------------------------------------------------------------------------
# Exploitation Tool 2: Knowledge Graph Traversal (LightRAG)
# ---------------------------------------------------------------------------

def make_graph_rag_tool(user_id: str) -> BaseTool:
    """Returns a graph_rag_search tool pre-bound to *user_id*.

    Navigates the LightRAG knowledge graph (entities, relations, and communities) stored in PostgreSQL.
    Best for conceptual questions, entity relationship mapping, and multi-hop reasoning.
    """

    @tool
    async def graph_rag_search(
        query: str,
        mode: Literal["mix", "local", "global", "hybrid", "naive"] = "mix",
        top_k: int = 5,
    ) -> str:
        """Explores the user's knowledge graph (entities, relationships, communities, and thematic graphs) extracted from documents using LightRAG.

        Use this tool when answering conceptual questions, tracing relationships between entities, exploring
        system architectures, or generating high-level thematic summaries that span across multiple documents.

        Args:
            query: The conceptual question, entity name, or relational query.
            mode: LightRAG traversal mode ('mix' for integrated local+global, 'local' for neighborhood entities,
                  'global' for thematic relations, 'hybrid', or 'naive').
            top_k: Number of graph entities/chunks to retrieve.

        Returns:
            Formatted graph context including entity descriptions, relationships, and community insights.
        """
        try:
            workspace = user_workspace(user_id)
            provider = await get_provider("lightrag", workspace=workspace)

            retrieval_query = RetrievalQuery(
                query=query,
                top_k=top_k,
                mode=mode,
            )
            result = await provider.query(retrieval_query)

            if not result.context or not result.context.strip():
                return f"No graph entities or relationships found in knowledge base for query: '{query}'."

            return (
                f"### Knowledge Graph Traversal [Mode: {mode.upper()}]\n\n"
                f"{result.context.strip()}"
            )
        except Exception as exc:
            logger.warning("graph_rag_search tool encountered error: %s", exc, exc_info=True)
            return f"Notice: Knowledge graph traversal unavailable for query '{query}': {exc}"

    return graph_rag_search


# ---------------------------------------------------------------------------
# Exploration Tool: Multi-Query Search (Agent-Provided Queries Array)
# ---------------------------------------------------------------------------

def make_multi_query_search_tool(user_id: str) -> BaseTool:
    """Returns a multi_query_search tool pre-bound to *user_id*.

    Exploration tool that accepts a list of distinct queries from the agent, executes parallel
    hybrid retrievals across all queries, and fuses the results using cross-query RRF.
    """

    @tool
    async def multi_query_search(
        queries: list[str],
        preset: Literal["high_precision", "balanced", "high_recall"] = "balanced",
        top_k: int = 8,
    ) -> str:
        """Explores the knowledge base with multiple complementary search queries simultaneously.

        Provide an array of diverse queries (e.g. synonyms, architectural angles, edge cases, or sub-topics)
        to search in parallel. The results from each query are executed via hybrid search (vector + keyword)
        and fused together using Reciprocal Rank Fusion, with bonuses for documents matching multiple queries.

        Args:
            queries: Array of distinct search queries/perspectives to explore in parallel.
            preset: Retrieval strategy preset ('high_precision', 'balanced', or 'high_recall').
            top_k: Maximum number of fused, deduplicated passages to return.

        Returns:
            Ranked, deduplicated passages annotated with the specific queries that matched each document.
        """
        try:
            cleaned = [q.strip() for q in queries if q and q.strip()]
            if not cleaned:
                return "Error: At least one non-empty query string must be provided in 'queries'."

            fused_docs = await multi_query_hybrid_search(
                queries=cleaned,
                user_id=user_id,
                preset=preset,
                top_k_final=top_k,
            )

            if not fused_docs:
                return (
                    f"No relevant documents found across {len(cleaned)} queries explored:\n"
                    + "\n".join(f"- {q}" for q in cleaned)
                )

            header_section = [
                f"### Multi-Query Exploration Results ({len(cleaned)} queries searched in parallel):",
            ]
            for q in cleaned:
                header_section.append(f"- {q}")

            header_section.append("\n**Ranked & Fused Passages:**\n")

            passages: list[str] = []
            for idx, doc in enumerate(fused_docs, 1):
                filename = doc.metadata.get("filename") or doc.metadata.get("source") or "Document"
                headers = [v for k, v in doc.metadata.items() if k.startswith("header_")]
                section_str = f" | Section: {' > '.join(headers)}" if headers else ""
                matched_qs = doc.metadata.get("matched_queries") or []
                matched_str = f" | Matched Queries: {', '.join(matched_qs)}" if matched_qs else ""
                passages.append(
                    f"[Source {idx}: {filename}{section_str}{matched_str}]\n{doc.page_content.strip()}"
                )

            return "\n".join(header_section) + "\n\n---\n\n".join(passages)
        except Exception as exc:
            logger.error("multi_query_search failed: %s", exc, exc_info=True)
            return f"Error executing multi-query search: {exc}"

    return multi_query_search


# ---------------------------------------------------------------------------
# Factory Aggregators & Lookup Helpers
# ---------------------------------------------------------------------------

def get_user_rag_tools(user_id: str) -> list[BaseTool]:
    """Returns the modular RAG research tools bound to the specified *user_id*.

    Includes:
    - Exploitation: hybrid_document_search, graph_rag_search
    - Exploration: multi_query_search
    """
    return [
        make_hybrid_search_tool(user_id=user_id),
        make_graph_rag_tool(user_id=user_id),
        make_multi_query_search_tool(user_id=user_id),
    ]


def get_user_rag_tool_map(user_id: str) -> dict[str, BaseTool]:
    """Returns a dictionary mapping tool names and standard aliases to pre-bound tool instances."""
    tools = get_user_rag_tools(user_id=user_id)
    tool_map: dict[str, BaseTool] = {}

    for t in tools:
        tool_map[t.name] = t

    # Helpful aliases for workflow and agent resolution
    hybrid_tool = tool_map.get("hybrid_document_search")
    if hybrid_tool:
        tool_map["hybrid_search"] = hybrid_tool
        tool_map["document_search"] = hybrid_tool
        tool_map["search"] = hybrid_tool

    graph_tool = tool_map.get("graph_rag_search")
    if graph_tool:
        tool_map["lightrag_search"] = graph_tool
        tool_map["graph_search"] = graph_tool

    multi_query_tool = tool_map.get("multi_query_search")
    if multi_query_tool:
        tool_map["multi_query"] = multi_query_tool
        tool_map["multiquery"] = multi_query_tool
        tool_map["parallel_search"] = multi_query_tool

    return tool_map
