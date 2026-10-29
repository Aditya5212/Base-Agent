from __future__ import annotations

import logging
from typing import Any

from app.schemas.workflow import RetrieveNodeConfig
from app.services.retriever import hybrid_search
from app.services.workflow.state import WorkflowState, render_template

logger = logging.getLogger(__name__)


def build_retrieve_node(node_id: str, config_dict: dict[str, Any]):
    """Builds a knowledge retrieval node handler querying RAG knowledge bases."""
    config = RetrieveNodeConfig.model_validate(config_dict or {})

    async def retrieve_handler(state: WorkflowState) -> dict[str, Any]:
        logger.debug("Executing RETRIEVE node '%s' [provider=%s]", node_id, config.provider_name)

        rendered_query = render_template(config.query_template, state).strip()
        user_id = state.get("user_id", "root")

        variables = dict(state.get("variables") or {})
        history = list(state.get("node_history") or [])
        history.append(node_id)

        if not rendered_query:
            logger.warning("Retrieve node '%s' received an empty query template result", node_id)
            variables[config.output_variable] = ""
            variables[config.sources_variable] = []
            return {"variables": variables, "node_history": history}

        try:
            filter_dict = {"user_id": user_id}
            documents = await hybrid_search(
                query=rendered_query,
                filter_dict=filter_dict,
            )

            # Limit to top_k
            selected_docs = documents[:config.top_k]

            # Format retrieved context
            context_chunks = []
            sources = []

            for idx, doc in enumerate(selected_docs, 1):
                src_name = doc.metadata.get("source") or doc.metadata.get("filename") or f"doc_{idx}"
                section = doc.metadata.get("header") or doc.metadata.get("section") or ""
                header_info = f" ({section})" if section else ""
                context_chunks.append(f"[{idx}] Source: {src_name}{header_info}\n{doc.page_content.strip()}")
                sources.append({
                    "index": idx,
                    "source": src_name,
                    "section": section,
                    "metadata": doc.metadata,
                })

            formatted_context = "\n\n---\n\n".join(context_chunks)
            variables[config.output_variable] = formatted_context
            variables[config.sources_variable] = sources
        except Exception as exc:
            logger.error("Retrieve node '%s' error during search: %s", node_id, exc, exc_info=True)
            variables[config.output_variable] = f"Error retrieving knowledge context: {str(exc)}"
            variables[config.sources_variable] = []

        return {
            "variables": variables,
            "node_history": history,
        }

    return retrieve_handler

