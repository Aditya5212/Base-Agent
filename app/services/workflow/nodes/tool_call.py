from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import Any

from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.schemas.workflow import ToolCallNodeConfig
from app.services.tools import make_search_tool
from app.services.workflow.state import WorkflowState, render_template

logger = logging.getLogger(__name__)


def _execute_builtin_tool(tool_name: str, args: dict[str, Any], user_id: str) -> Any:
    """Execute simple built-in utilities if registered."""
    if tool_name == "calculator":
        expr = str(args.get("expression") or args.get("expr") or "").strip()
        # Safe arithmetic evaluation
        import ast
        import operator
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }
        def eval_expr(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp):
                return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
            if isinstance(node, ast.UnaryOp):
                return ops[type(node.op)](eval_expr(node.operand))
            raise ValueError(f"Unsupported mathematical expression: {ast.dump(node)}")

        parsed = ast.parse(expr, mode="eval")
        return eval_expr(parsed.body)

    if tool_name == "datetime":
        return datetime.now(timezone.utc).isoformat()

    return None


def build_tool_call_node(node_id: str, config_dict: dict[str, Any]):
    """Builds a tool execution node handler supporting retries and error isolation."""
    config = ToolCallNodeConfig.model_validate(config_dict or {})

    async def tool_call_handler(state: WorkflowState) -> dict[str, Any]:
        logger.debug("Executing TOOL_CALL node '%s' [tool=%s]", node_id, config.tool_name)
        user_id = state.get("user_id", "root")

        # 1. Resolve inputs from mappings
        resolved_args: dict[str, Any] = {}
        for arg_key, arg_val in config.input_mapping.items():
            if isinstance(arg_val, str):
                rendered = render_template(arg_val, state)
                # Try JSON deserialization if string looks like JSON
                if rendered.startswith(("{", "[")) and rendered.endswith(("}", "]")):
                    try:
                        resolved_args[arg_key] = json.loads(rendered)
                        continue
                    except Exception:
                        pass
                resolved_args[arg_key] = rendered
            else:
                resolved_args[arg_key] = arg_val

        variables = dict(state.get("variables") or {})
        history = list(state.get("node_history") or [])
        history.append(node_id)

        async def _execute():
            # Check built-in tools
            builtin_res = _execute_builtin_tool(config.tool_name, resolved_args, user_id)
            if builtin_res is not None:
                return builtin_res

            # Check hybrid search tool
            if config.tool_name in ("hybrid_document_search", "document_search", "search"):
                tool_instance = make_search_tool(user_id=user_id)
                query = resolved_args.get("query") or resolved_args.get("q") or ""
                preset = resolved_args.get("preset", "balanced")
                return await tool_instance.ainvoke({"query": query, "preset": preset})

            # Placeholder for future dynamic tool registry (Phase 3)
            return f"[Tool '{config.tool_name}' executed with args: {resolved_args}]"

        # 2. Execute with ToolRetryMiddleware and ToolErrorMiddleware
        try:
            if config.retry_count > 0:
                retryer = AsyncRetrying(
                    stop=stop_after_attempt(config.retry_count + 1),
                    wait=wait_exponential(multiplier=1.0, min=1.0, max=5.0),
                    retry=retry_if_exception_type(Exception),
                    reraise=True,
                )
                async for attempt in retryer:
                    with attempt:
                        result = await _execute()
            else:
                result = await _execute()

            variables[config.output_variable] = result
        except Exception as exc:
            logger.error("Tool execution node '%s' failed: %s", node_id, exc, exc_info=True)
            if config.catch_errors:
                variables[config.output_variable] = {
                    "error": True,
                    "message": str(exc),
                    "tool": config.tool_name,
                }
            else:
                raise exc

        return {
            "variables": variables,
            "node_history": history,
        }

    return tool_call_handler

