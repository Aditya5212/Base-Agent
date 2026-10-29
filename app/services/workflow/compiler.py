from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.schemas.workflow import NodeType
from app.services.workflow.nodes import (
    build_chat_node,
    build_code_node,
    build_conditional_node,
    build_conditional_router,
    build_end_node,
    build_human_approval_node,
    build_retrieve_node,
    build_start_node,
    build_tool_call_node,
)
from app.services.workflow.state import WorkflowState
from app.services.workflow.validator import GraphValidationError, validate_workflow_graph

logger = logging.getLogger(__name__)


class WorkflowCompiler:
    """Compiles declarative workflow JSON specifications into executable LangGraph StateGraphs."""

    @classmethod
    def compile(
        cls,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        checkpointer: Any | None = None,
        validate: bool = True,
    ) -> CompiledStateGraph:
        """Translates nodes and edges into an executable LangGraph StateGraph.

        Args:
            nodes: List of node definition dictionaries.
            edges: List of directed edge definition dictionaries.
            checkpointer: Optional LangGraph checkpointer for persistence/memory.
            validate: Whether to perform graph topology validation prior to compilation.

        Returns:
            CompiledStateGraph ready for execution via .ainvoke().

        Raises:
            GraphValidationError: If graph topology or configuration is invalid.
        """
        if validate:
            is_valid, errors, warnings = validate_workflow_graph(nodes, edges)
            if not is_valid:
                raise GraphValidationError(errors)
            for w in warnings:
                logger.warning("Workflow graph warning: %s", w)

        builder = StateGraph(WorkflowState)
        node_map: dict[str, dict[str, Any]] = {n["id"]: n for n in nodes}
        start_node_id: str | None = None
        end_node_ids: list[str] = []
        conditional_nodes: set[str] = set()

        # 1. Register all nodes
        for node in nodes:
            node_id = str(node["id"])
            node_type = str(node.get("type", "")).strip().lower()
            config = node.get("config", {})

            if node_type == NodeType.START.value:
                start_node_id = node_id
                handler = build_start_node(node_id, config)
            elif node_type == NodeType.END.value:
                end_node_ids.append(node_id)
                handler = build_end_node(node_id, config)
            elif node_type == NodeType.CHAT.value:
                handler = build_chat_node(node_id, config)
            elif node_type == NodeType.RETRIEVE.value:
                handler = build_retrieve_node(node_id, config)
            elif node_type == NodeType.CONDITIONAL.value:
                conditional_nodes.add(node_id)
                handler = build_conditional_node(node_id, config)
            elif node_type == NodeType.CODE.value:
                handler = build_code_node(node_id, config)
            elif node_type == NodeType.HUMAN_APPROVAL.value:
                handler = build_human_approval_node(node_id, config)
            elif node_type == NodeType.TOOL_CALL.value:
                handler = build_tool_call_node(node_id, config)
            else:
                raise GraphValidationError([f"Unsupported node type '{node_type}' for node '{node_id}'."])

            builder.add_node(node_id, handler)

        # 2. Connect Entry Point
        if start_node_id:
            builder.add_edge(START, start_node_id)

        # 3. Connect Exit Points
        for end_id in end_node_ids:
            builder.add_edge(end_id, END)

        # 4. Connect Conditional Edges
        for cond_node_id in conditional_nodes:
            cond_node = node_map[cond_node_id]
            cond_cfg = cond_node.get("config", {})
            router_fn = build_conditional_router(cond_node_id, cond_cfg)

            # Collect all possible branch targets
            target_ids: set[str] = set()
            d_target = cond_cfg.get("default_target_node_id")
            if d_target:
                target_ids.add(str(d_target))
            for branch in cond_cfg.get("branches", []):
                b_target = branch.get("target_node_id")
                if b_target:
                    target_ids.add(str(b_target))

            target_mapping = {t: t for t in target_ids}
            builder.add_conditional_edges(cond_node_id, router_fn, target_mapping)

        # 5. Connect Static Edges (excluding edges originating from conditional nodes)
        for edge in edges:
            src = str(edge["source"])
            tgt = str(edge["target"])

            if src in conditional_nodes:
                # Already handled by add_conditional_edges
                continue

            builder.add_edge(src, tgt)

        # 6. Compile graph with checkpointer
        if checkpointer:
            return builder.compile(checkpointer=checkpointer)
        return builder.compile()

