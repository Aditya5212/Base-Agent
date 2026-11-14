from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from app.schemas.workflow import NodeType

SUPPORTED_NODE_TYPES = frozenset({"start", "end", "chat", "retrieve", "tool_call", "conditional", "human_approval", "code"})


class GraphValidationError(Exception):
    """Raised when a workflow graph fails topological validation."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def validate_workflow_graph(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[bool, list[str], list[str]]:
    """Validate graph topology, node configurations, and connectivity.

    Returns:
        tuple[is_valid, errors, warnings]
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not nodes:
        errors.append("Workflow must contain at least one node.")
        return False, errors, warnings

    # 1. Unique Node IDs & Node Type Distribution
    node_ids: set[str] = set()
    start_nodes: list[str] = []
    end_nodes: list[str] = []
    node_map: dict[str, dict[str, Any]] = {}

    for node in nodes:
        node_id = str(node.get("id", "")).strip()
        node_type = str(node.get("type", "")).strip().lower()

        if node_type not in SUPPORTED_NODE_TYPES:
            errors.append(f"Unsupported node type '{node_type}' for node '{node_id}'.")
        config = node.get("config") or {}
        if "knowledge" in config:
            errors.append(f"Node '{node_id}' must use an explicit retrieve node for knowledge attachment.")
        if node_type == "retrieve" and (config.get("workspace_type", "user") not in {"user", "thread", "global"} or config.get("provider_name", "hybrid") not in {"hybrid", "lightrag", "multi_query"}):
            errors.append(f"Node '{node_id}' uses an unsupported provider or knowledge scope.")
        if node_type == "conditional" and any(b.get("condition_type") == "jsonpath" for b in config.get("branches", [])):
            errors.append(f"Node '{node_id}' uses unimplemented JSONPath routing.")

        if not node_id:
            errors.append("Every node must have a non-empty 'id'.")
            continue

        if node_id in node_ids:
            errors.append(f"Duplicate node ID detected: '{node_id}'.")
        node_ids.add(node_id)
        node_map[node_id] = node

        if node_type == NodeType.START.value:
            start_nodes.append(node_id)
        elif node_type == NodeType.END.value:
            end_nodes.append(node_id)

    if len(start_nodes) == 0:
        errors.append("Workflow must have exactly one 'start' node (found 0).")
    elif len(start_nodes) > 1:
        errors.append(f"Workflow must have exactly one 'start' node (found {len(start_nodes)}: {start_nodes}).")

    if len(end_nodes) == 0:
        errors.append("Workflow must have at least one 'end' node (found 0).")

    # 2. Edge Endpoints Validation
    adjacency: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, list[str]] = defaultdict(list)

    for idx, edge in enumerate(edges):
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()

        if not source or not target:
            errors.append(f"Edge at index {idx} must specify both 'source' and 'target'.")
            continue

        if source not in node_ids:
            errors.append(f"Edge source '{source}' does not exist in nodes.")
        if target not in node_ids:
            errors.append(f"Edge target '{target}' does not exist in nodes.")

        if source in node_ids and target in node_ids:
            # Start node cannot have incoming edges
            source_type = node_map[source].get("type", "").lower()
            target_type = node_map[target].get("type", "").lower()

            if target_type == NodeType.START.value:
                errors.append(f"Start node '{target}' cannot have incoming edges (from '{source}').")
            if source_type == NodeType.END.value:
                errors.append(f"End node '{source}' cannot have outgoing edges (to '{target}').")

            adjacency[source].append(target)
            incoming[target].append(source)

    for source, targets in adjacency.items():
        if len(targets) > 1 and node_map[source].get("type") != "conditional":
            errors.append(f"Node '{source}' has multiple outgoing edges; parallel execution is not supported yet.")

    # 3. Conditional Node Configuration Validation
    for node_id, node in node_map.items():
        node_type = str(node.get("type", "")).strip().lower()
        if node_type == NodeType.CONDITIONAL.value:
            config = node.get("config", {})
            default_target = str(config.get("default_target_node_id", "")).strip()
            if not default_target:
                errors.append(f"Conditional node '{node_id}' is missing required 'default_target_node_id'.")
            elif default_target not in node_ids:
                errors.append(f"Conditional node '{node_id}' default_target_node_id '{default_target}' does not exist.")

            branches = config.get("branches", [])
            for b_idx, branch in enumerate(branches):
                target = str(branch.get("target_node_id", "")).strip()
                if not target:
                    errors.append(f"Conditional node '{node_id}' branch {b_idx} is missing 'target_node_id'.")
                elif target not in node_ids:
                    errors.append(f"Conditional node '{node_id}' branch {b_idx} target '{target}' does not exist.")

    # 4. Reachability from START to at least one END node
    if start_nodes and end_nodes and not errors:
        start_id = start_nodes[0]
        visited: set[str] = set()
        queue: deque[str] = deque([start_id])

        while queue:
            curr = queue.popleft()
            if curr in visited:
                continue
            visited.add(curr)

            # Traverse standard edge adjacency
            for neighbor in adjacency.get(curr, []):
                if neighbor not in visited:
                    queue.append(neighbor)

            # Also traverse conditional branches if curr is conditional
            curr_node = node_map.get(curr, {})
            if curr_node.get("type", "").lower() == NodeType.CONDITIONAL.value:
                cfg = curr_node.get("config", {})
                d_target = cfg.get("default_target_node_id")
                if d_target and d_target in node_ids and d_target not in visited:
                    queue.append(d_target)
                for br in cfg.get("branches", []):
                    b_target = br.get("target_node_id")
                    if b_target and b_target in node_ids and b_target not in visited:
                        queue.append(b_target)

        reachable_ends = [end_id for end_id in end_nodes if end_id in visited]
        if not reachable_ends:
            errors.append("No 'end' node is reachable from the 'start' node.")

        # Identify orphan / unreachable nodes
        unreachable = node_ids - visited
        if unreachable:
            warnings.append(f"The following nodes are unreachable from start: {sorted(list(unreachable))}")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings

