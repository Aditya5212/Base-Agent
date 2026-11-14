from __future__ import annotations

import logging
from typing import Any

from app.core.rag.factory import get_provider
from app.core.rag.protocols import RetrievalQuery
from app.core.rag.workspace import (
    agent_workspace,
    global_workspace,
    thread_workspace,
    user_workspace,
)
from app.core.rag_config import get_rag_config
from app.schemas.workflow import RetrieveNodeConfig
from app.services.multi_query import multi_query_hybrid_search
from app.services.retrieval_service import (
    RetrievalScope,
    query_graph_rag_dual_layer,
    resolve_retrieval_scope,
)
from app.services.retriever import hybrid_search
from app.services.workflow.state import WorkflowState, render_template

logger = logging.getLogger(__name__)


def _resolve_workspace(workspace_type: str, user_id: str, state: WorkflowState) -> str:
    """Resolves the tenant workspace identifier for graph/knowledge isolation."""
    if workspace_type == "thread":
        thread_id = str(state.get("thread_id") or "default")
        return thread_workspace(user_id, thread_id)
    if workspace_type == "agent":
        agent_id = str(state.get("agent_id") or "default")
        return agent_workspace(user_id, agent_id)
    if workspace_type == "global":
        return global_workspace()
    return user_workspace(user_id)


def build_retrieve_node(node_id: str, config_dict: dict[str, Any]) -> Any:
    """Builds a knowledge retrieval node handler querying RAG knowledge bases.

    Supports providers:
    - 'hybrid' (default): Dense PGVector + Sparse FTS + RRF
    - 'lightrag': Knowledge graph traversal via LightRAG (PGTableGraphStorage)
    - 'multi_query': LLM query expansion + parallel hybrid retrieval + RRF fusion
    - 'deep_research': Dual-engine vector exploration + graph traversal
    """
    config = RetrieveNodeConfig.model_validate(config_dict or {})
    if config.provider_name not in ("hybrid", "lightrag", "multi_query"):
        raise ValueError(f"Unsupported retrieval provider '{config.provider_name}'")
    if config.workspace_type not in ("user", "thread", "global"):
        raise ValueError(f"Unsupported knowledge scope '{config.workspace_type}'")

    async def retrieve_handler(state: WorkflowState) -> dict[str, Any]:
        logger.debug(
            "Executing RETRIEVE node '%s' [provider=%s, top_k=%d]",
            node_id,
            config.provider_name,
            config.top_k,
        )

        rendered_query = render_template(config.query_template, state).strip()
        user_id = state.get("user_id", "root")
        raw_thread_id = state.get("thread_id")
        effective_thread_id = raw_thread_id if raw_thread_id and not raw_thread_id.startswith("wf_run_") else None
        if config.workspace_type == "thread" and raw_thread_id:
            effective_thread_id = raw_thread_id

        if config.scope:
            resolved_scope = resolve_retrieval_scope(config.scope, thread_id=effective_thread_id)
        elif config.workspace_type == "thread":
            resolved_scope = RetrievalScope.THREAD_WITH_GLOBAL if effective_thread_id else RetrievalScope.GLOBAL_ONLY
        elif config.workspace_type == "global":
            resolved_scope = RetrievalScope.GLOBAL_ONLY
        else:
            resolved_scope = RetrievalScope.THREAD_WITH_GLOBAL if effective_thread_id else RetrievalScope.GLOBAL_ONLY

        workspace = _resolve_workspace(config.workspace_type, user_id, state)

        variables = dict(state.get("variables") or {})
        history = list(state.get("node_history") or [])
        history.append(node_id)

        if not rendered_query:
            logger.warning("Retrieve node '%s' received an empty query template result", node_id)
            variables[config.output_variable] = ""
            variables[config.sources_variable] = []
            return {"variables": variables, "node_history": history}

        try:
            provider = (config.provider_name or "hybrid").lower().strip()

            # ---------------------------------------------------------------
            # 1. LightRAG Knowledge Graph Provider
            # ---------------------------------------------------------------
            if provider in ("lightrag", "graph", "knowledge_graph"):
                if effective_thread_id and resolved_scope == RetrievalScope.THREAD_WITH_GLOBAL:
                    formatted_context, sources = await query_graph_rag_dual_layer(
                        query=rendered_query,
                        user_id=user_id,
                        thread_id=effective_thread_id,
                        top_k=config.top_k,
                        mode="mix",
                        provider_getter=get_provider,
                    )
                    variables[config.output_variable] = formatted_context
                    variables[config.sources_variable] = sources
                else:
                    rag_provider = await get_provider("lightrag", workspace=workspace)
                    retrieval_query = RetrievalQuery(
                        query=rendered_query,
                        top_k=config.top_k,
                        mode="mix",
                    )
                    res = await rag_provider.query(retrieval_query)
                    formatted_context = res.context.strip() if res.context else ""
                    sources = [
                        {
                            "index": idx,
                            "source": chunk.source,
                            "chunk_id": chunk.chunk_id,
                            "metadata": chunk.metadata,
                        }
                        for idx, chunk in enumerate(res.chunks, 1)
                    ]
                    variables[config.output_variable] = formatted_context
                    variables[config.sources_variable] = sources

            # ---------------------------------------------------------------
            # 2. Multi-Query Exploration Provider (Agent-Provided Queries)
            # ---------------------------------------------------------------
            elif provider in ("multi_query", "multiquery", "multi_query_search"):
                # Split multiline or comma-separated queries if template renders multiple queries
                if "\n" in rendered_query:
                    queries_list = [line.strip() for line in rendered_query.splitlines() if line.strip()]
                elif "," in rendered_query:
                    queries_list = [item.strip() for item in rendered_query.split(",") if item.strip()]
                else:
                    queries_list = [rendered_query]

                fused_docs = await multi_query_hybrid_search(
                    queries=queries_list,
                    user_id=user_id,
                    thread_id=effective_thread_id if resolved_scope in (RetrievalScope.THREAD_ONLY, RetrievalScope.THREAD_WITH_GLOBAL) else None,
                    scope=resolved_scope,
                    top_k_final=config.top_k,
                )
                context_chunks = []
                sources = []
                for idx, doc in enumerate(fused_docs, 1):
                    src_name = doc.metadata.get("source") or doc.metadata.get("filename") or f"doc_{idx}"
                    section = doc.metadata.get("header") or doc.metadata.get("section") or ""
                    header_info = f" ({section})" if section else ""
                    matched_qs = doc.metadata.get("matched_queries") or []
                    matched_str = f" [Matched: {', '.join(matched_qs)}]" if matched_qs else ""
                    is_th = doc.metadata.get("is_thread_document", False)
                    th_tag = " [Thread Doc]" if is_th else ""
                    context_chunks.append(
                        f"[{idx}] Source: {src_name}{header_info}{th_tag}{matched_str}\n{doc.page_content.strip()}"
                    )
                    sources.append({
                        "index": idx,
                        "source": src_name,
                        "section": section,
                        "matched_queries": matched_qs,
                        "is_thread_document": is_th,
                        "metadata": doc.metadata,
                    })

                variables[config.output_variable] = "\n\n---\n\n".join(context_chunks)
                variables[config.sources_variable] = sources

            # ---------------------------------------------------------------
            # 3. Standard Hybrid Search (Dense + Sparse + RRF) Default
            # ---------------------------------------------------------------
            else:
                documents = await hybrid_search(
                    query=rendered_query,
                    filter_dict={"user_id": user_id},
                    thread_id=effective_thread_id if resolved_scope in (RetrievalScope.THREAD_ONLY, RetrievalScope.THREAD_WITH_GLOBAL) else None,
                    scope=resolved_scope,
                    config=get_rag_config(top_k_final=config.top_k, top_k_dense=max(20, config.top_k), top_k_sparse=max(20, config.top_k)),
                )

                selected_docs = documents[:config.top_k]
                context_chunks = []
                sources = []

                for idx, doc in enumerate(selected_docs, 1):
                    src_name = doc.metadata.get("source") or doc.metadata.get("filename") or f"doc_{idx}"
                    section = doc.metadata.get("header") or doc.metadata.get("section") or ""
                    header_info = f" ({section})" if section else ""
                    is_th = doc.metadata.get("is_thread_document", False)
                    th_tag = " [Thread Doc]" if is_th else ""
                    context_chunks.append(f"[{idx}] Source: {src_name}{header_info}{th_tag}\n{doc.page_content.strip()}")
                    sources.append({
                        "index": idx,
                        "source": src_name,
                        "section": section,
                        "is_thread_document": is_th,
                        "metadata": doc.metadata,
                    })

                variables[config.output_variable] = "\n\n---\n\n".join(context_chunks)
                variables[config.sources_variable] = sources

        except Exception as exc:
            logger.error("Retrieve node '%s' error during search: %s", node_id, exc, exc_info=True)
            raise RuntimeError(f"Knowledge retrieval failed at node '{node_id}'") from exc

        return {
            "variables": variables,
            "node_history": history,
        }

    return retrieve_handler
