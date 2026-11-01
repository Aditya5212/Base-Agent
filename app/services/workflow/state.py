from __future__ import annotations

import ast
import operator
import re
from copy import deepcopy
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Workflow State Schema
# ---------------------------------------------------------------------------

class WorkflowState(TypedDict, total=False):
    """Execution state passed through the LangGraph workflow pipeline."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    variables: dict[str, Any]
    user_id: str
    workflow_id: str
    thread_id: str
    node_history: list[str]
    error: str | None


# ---------------------------------------------------------------------------
# Template Interpolation
# ---------------------------------------------------------------------------

_TEMPLATE_REGEX = re.compile(r"\{\{\s*([\w\.\-]+)\s*\}\}")


def resolve_path_value(path: str, state: WorkflowState) -> Any:
    """Resolve a dot-separated variable path against state."""
    parts = path.strip().split(".")
    
    # Direct top-level state keys
    first = parts[0]
    if first in ("variables", "state"):
        current = state.get("variables", {})
        parts = parts[1:]
    elif first in state:
        current = state.get(first)
        parts = parts[1:]
    else:
        # Default lookup in state["variables"]
        current = state.get("variables", {})

    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    return current


def render_template(template: str, state: WorkflowState) -> str:
    """Safely interpolate {{variables.key}} placeholders into a template string."""
    if not template or "{{" not in template:
        return template

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        val = resolve_path_value(key, state)
        if val is None:
            # Fallback to direct variable lookup
            val = state.get("variables", {}).get(key, "")
        if isinstance(val, (dict, list)):
            import json
            return json.dumps(val)
        return str(val) if val is not None else ""

    return _TEMPLATE_REGEX.sub(_replace, template)


def render_template_value(value: Any, state: WorkflowState) -> Any:
    """An exact variable reference preserves JSON types; mixed templates stay text."""
    if isinstance(value, dict):
        return {key: render_template_value(item, state) for key, item in value.items()}
    if isinstance(value, list):
        return [render_template_value(item, state) for item in value]
    if not isinstance(value, str):
        return value
    match = _TEMPLATE_REGEX.fullmatch(value.strip())
    if match is None:
        return render_template(value, state)
    parts = match.group(1).split(".")
    if parts[0] in {"variables", "state"}:
        current = state.get("variables", {})
        parts = parts[1:]
    elif parts[0] in state:
        current = state
    else:
        current = state.get("variables", {})
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ValueError(f"Missing workflow variable '{match.group(1)}'")
    return deepcopy(current)


# ---------------------------------------------------------------------------
# Safe Condition Evaluation
# ---------------------------------------------------------------------------

_OPERATORS = {
    "eq": operator.eq,
    "==": operator.eq,
    "neq": operator.ne,
    "!=": operator.ne,
    "gt": operator.gt,
    ">": operator.gt,
    "gte": operator.ge,
    ">=": operator.ge,
    "lt": operator.lt,
    "<": operator.lt,
    "lte": operator.le,
    "<=": operator.le,
    "contains": lambda a, b: b in a if a is not None else False,
    "in": lambda a, b: a in b if b is not None else False,
}

_AST_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


def _safe_eval_node(node: ast.AST, context: dict[str, Any]) -> Any:
    """Recursively evaluate simple literal/comparison expressions safely via AST."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return context.get(node.id)
    if isinstance(node, ast.Attribute):
        val = _safe_eval_node(node.value, context)
        if isinstance(val, dict):
            return val.get(node.attr)
        return getattr(val, node.attr, None)
    if isinstance(node, ast.Compare):
        left = _safe_eval_node(node.left, context)
        for op_type, comparator in zip(node.ops, node.comparators, strict=False):
            op_fn = _AST_OPERATORS.get(type(op_type))
            if not op_fn:
                return False
            right = _safe_eval_node(comparator, context)
            if not op_fn(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_safe_eval_node(val, context) for val in node.values)
        if isinstance(node.op, ast.Or):
            return any(_safe_eval_node(val, context) for val in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _safe_eval_node(node.operand, context)
    return False


def evaluate_condition_expression(expression: str, state: WorkflowState) -> bool:
    """Safely evaluates a python-like expression using AST parsing without eval()."""
    if not expression or not expression.strip():
        return False
    try:
        parsed = ast.parse(expression.strip(), mode="eval")
        context = {
            "variables": state.get("variables", {}),
            "state": state,
            **state.get("variables", {}),
        }
        return bool(_safe_eval_node(parsed.body, context))
    except Exception:
        return False


def evaluate_variable_match(
    variable_name: str,
    operator_name: str,
    expected_value: Any,
    state: WorkflowState,
) -> bool:
    """Evaluate variable_match condition against workflow state."""
    actual_value = resolve_path_value(variable_name, state)
    op_fn = _OPERATORS.get(operator_name.lower())
    if not op_fn:
        return False

    try:
        # Type coercion if comparing numbers with string
        if isinstance(expected_value, (int, float)) and isinstance(actual_value, str):
            actual_value = float(actual_value) if "." in actual_value else int(actual_value)
        return bool(op_fn(actual_value, expected_value))
    except Exception:
        return False

