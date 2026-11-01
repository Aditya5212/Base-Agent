from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import Any

from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from langgraph.types import interrupt

from app.schemas.workflow import ToolCallNodeConfig
from app.services.rag_tools import get_user_rag_tool_map
from app.services.workflow.state import WorkflowState, render_template_value

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
    """Builds a tool execution node handler supporting retries, error isolation,
    and modular RAG research tool resolution.
    """
    config = ToolCallNodeConfig.model_validate(config_dict or {})

    async def tool_call_handler(state: WorkflowState) -> dict[str, Any]:
        logger.debug("Executing TOOL_CALL node '%s' [tool=%s]", node_id, config.tool_name)
        user_id = state.get("user_id", "root")

        # 1. Resolve inputs from mappings
        resolved_args: dict[str, Any] = {}
        input_mapping = config.input_mapping or (config_dict or {}).get("arguments_template") or (config_dict or {}).get("arguments") or {}
        for arg_key, arg_val in input_mapping.items():
            rendered = render_template_value(arg_val, state)
            if isinstance(rendered, str):
                # Try JSON deserialization if string looks like JSON
                if rendered.startswith(("{", "[")) and rendered.endswith(("}", "]")):
                    try:
                        resolved_args[arg_key] = json.loads(rendered)
                        continue
                    except Exception:
                        pass
                resolved_args[arg_key] = rendered
            else:
                resolved_args[arg_key] = rendered

        # Outside the error/retry wrapper: interrupts must reach the graph runtime.
        if config.requires_approval:
            decision = interrupt({"type": "tool_approval", "node_id": node_id,
                                  "tool": config.tool_name, "arguments": resolved_args,
                                  "timeout_seconds": 86400})
            if not isinstance(decision, dict) or decision.get("approved") is not True:
                raise PermissionError(f"Approval denied for tool '{config.tool_name}'")

        variables = dict(state.get("variables") or {})
        history = list(state.get("node_history") or [])
        history.append(node_id)

        async def _execute():
            # Check built-in tools
            builtin_res = _execute_builtin_tool(config.tool_name, resolved_args, user_id)
            if builtin_res is not None:
                return builtin_res

            # Check modular RAG research tools (hybrid, graph, multi_query)
            rag_tools = get_user_rag_tool_map(user_id=user_id)
            tool_key = config.tool_name.lower().strip()
            if tool_key in rag_tools:
                tool_instance = rag_tools[tool_key]
                call_args = dict(resolved_args)
                # Normalize common parameter aliases
                if "q" in call_args and "query" not in call_args:
                    call_args["query"] = call_args.pop("q")
                if "query" in call_args and "queries" not in call_args and tool_key in ("multi_query_search", "multi_query", "multiquery", "parallel_search"):
                    raw_q = call_args.pop("query")
                    call_args["queries"] = [raw_q] if isinstance(raw_q, str) else list(raw_q)
                return await tool_instance.ainvoke(call_args)

            # Check dynamic custom tool registry (Phase 3.1)
            from app.core.database import AsyncSessionLocal
            from app.services.tool_registry import ToolRegistry

            async with AsyncSessionLocal() as session:
                custom_tools = await ToolRegistry.resolve_tools(
                    user_id=user_id,
                    tool_names=[config.tool_name],
                    db=session,
                )
                if custom_tools:
                    return await custom_tools[0].ainvoke(resolved_args)

            raise ValueError(f"Unknown or inaccessible tool '{config.tool_name}'")

        async def _execute_with_retries() -> Any:
            if config.retry_count > 0:
                retryer = AsyncRetrying(
                    stop=stop_after_attempt(config.retry_count + 1),
                    wait=wait_exponential(multiplier=1.0, min=1.0, max=5.0),
                    retry=retry_if_exception_type(Exception),
                    reraise=True,
                )
                async for attempt in retryer:
                    with attempt:
                        return await _execute()
            else:
                return await _execute()
            raise RuntimeError("Tool retry loop completed without a result")

        # 2. Execute with ToolRetryMiddleware and ToolErrorMiddleware
        try:
            variables[config.output_variable] = await _execute_with_retries()
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
