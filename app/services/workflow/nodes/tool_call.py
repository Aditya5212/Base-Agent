from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import Any
import uuid

from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from langgraph.types import interrupt
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.workflow import ToolCallNodeConfig
from app.services.rag_tools import get_user_rag_tool_map
from app.services.workflow.state import WorkflowState, render_template_value
from app.services.workflow.tracing import _CURRENT_DB_SESSION, redact_value

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

        # Resolve deterministic invocation identity and idempotency key
        run_id = state.get("run_id")
        occurrence = 1
        tool_idempotency_key = None
        inv_id_uuid = None

        if run_id:
            node_occurrences = state.setdefault("node_occurrences", {})
            occurrence = node_occurrences.get(node_id, 1)
            node_invocations = state.setdefault("node_invocations", {})
            occ_key = f"{node_id}:{occurrence}"
            if occ_key not in node_invocations:
                node_invocations[occ_key] = str(uuid.uuid5(uuid.UUID(str(run_id)), occ_key))
            inv_id_str = node_invocations[occ_key]
            try:
                inv_id_uuid = uuid.UUID(inv_id_str)
            except Exception:
                inv_id_uuid = uuid.uuid4()
            tool_idempotency_key = f"{run_id}:{node_id}:{inv_id_str}:{occurrence}"

        # Check persisted tool invocation record for safe replay / result reuse
        variables = dict(state.get("variables") or {})
        history = list(state.get("node_history") or [])
        history.append(node_id)

        active_sess = _CURRENT_DB_SESSION.get()
        use_db_invocations = active_sess is None or isinstance(active_sess, AsyncSession)

        if run_id and tool_idempotency_key and use_db_invocations:
            try:
                from app.core.database import AsyncSessionLocal
                from app.models.workflow import WorkflowToolInvocation
                from sqlalchemy import select

                async with AsyncSessionLocal() as inv_sess:
                    # Serialize preparation with a unique logical occurrence, even after replay.
                    from sqlalchemy.dialects.postgresql import insert
                    await inv_sess.execute(
                        insert(WorkflowToolInvocation).values(
                            id=inv_id_uuid, run_id=uuid.UUID(str(run_id)), node_id=node_id,
                            logical_occurrence=occurrence, idempotency_key=tool_idempotency_key,
                            status="prepared", tool_name=config.tool_name,
                            arguments_json=redact_value(resolved_args, max_str_len=500),
                        ).on_conflict_do_nothing(constraint="uq_tool_invocation_occurrence")
                    )
                    stmt = select(WorkflowToolInvocation).where(
                        WorkflowToolInvocation.run_id == uuid.UUID(str(run_id)),
                        WorkflowToolInvocation.node_id == node_id,
                        WorkflowToolInvocation.logical_occurrence == occurrence,
                    ).with_for_update()
                    res = await inv_sess.execute(stmt)
                    cached_inv = res.scalar_one_or_none()
                    if cached_inv is None:
                        raise RuntimeError("Tool invocation preparation failed")
                    tool_idempotency_key = cached_inv.idempotency_key
                    inv_id_uuid = cached_inv.id
                    if cached_inv.status in ("running", "uncertain", "failed"):
                        raise RuntimeError("Previous tool outcome requires reconciliation before replay")

                    if cached_inv and cached_inv.status == "succeeded" and cached_inv.result_data is not None:
                        logger.info("Reusing cached tool result for idempotency key '%s'", tool_idempotency_key)
                        res_val = (
                            cached_inv.result_data.get("result")
                            if isinstance(cached_inv.result_data, dict) and "result" in cached_inv.result_data and len(cached_inv.result_data) == 1
                            else cached_inv.result_data
                        )
                        variables[config.output_variable] = res_val
                        return {"variables": variables, "node_history": history}

                    await inv_sess.commit()
            except Exception as inv_err:
                raise RuntimeError("Cannot safely prepare or replay tool invocation") from inv_err

        # Outside the error/retry wrapper: interrupts must reach the graph runtime.
        if config.requires_approval:
            decision = interrupt({"type": "tool_approval", "node_id": node_id,
                                  "tool": config.tool_name, "arguments": resolved_args,
                                  "timeout_seconds": 86400})
            if not isinstance(decision, dict) or decision.get("approved") is not True:
                raise PermissionError(f"Approval denied for tool '{config.tool_name}'")

        if run_id and tool_idempotency_key and use_db_invocations:
            try:
                from app.core.database import AsyncSessionLocal
                from app.models.workflow import WorkflowToolInvocation
                from sqlalchemy import update

                async with AsyncSessionLocal() as inv_sess:
                    started = await inv_sess.execute(
                        update(WorkflowToolInvocation)
                        .where(
                            WorkflowToolInvocation.idempotency_key == tool_idempotency_key,
                            WorkflowToolInvocation.status == "prepared",
                        )
                        .values(status="running", updated_at=datetime.now(timezone.utc))
                        .returning(WorkflowToolInvocation.id)
                    )
                    if started.scalar_one_or_none() is None:
                        raise RuntimeError("Tool invocation already started; reconciliation required")
                    await inv_sess.commit()
            except Exception:
                raise RuntimeError("Cannot persist tool execution start")

        execution_metadata = {
            "idempotency_key": tool_idempotency_key,
            "run_id": str(run_id) if run_id else None,
            "node_id": node_id,
            "invocation_id": str(inv_id_uuid) if inv_id_uuid else None,
        }

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
            from app.services.tool_registry import ToolRegistry, set_tool_execution_metadata, reset_tool_execution_metadata

            async with AsyncSessionLocal() as session:
                custom_tools = await ToolRegistry.resolve_tools(
                    user_id=user_id,
                    tool_names=[config.tool_name],
                    db=session,
                )
                if custom_tools:
                    token = set_tool_execution_metadata(execution_metadata)
                    try:
                        return await custom_tools[0].ainvoke(resolved_args)
                    finally:
                        reset_tool_execution_metadata(token)

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
            exec_result = await _execute_with_retries()
            variables[config.output_variable] = exec_result

            if run_id and tool_idempotency_key and use_db_invocations:
                try:
                    from app.core.database import AsyncSessionLocal
                    from app.models.workflow import WorkflowToolInvocation
                    from sqlalchemy import update

                    save_res = {"result": exec_result}
                    async with AsyncSessionLocal() as inv_sess:
                        await inv_sess.execute(
                            update(WorkflowToolInvocation)
                            .where(WorkflowToolInvocation.idempotency_key == tool_idempotency_key)
                            .values(
                                status="succeeded",
                                result_data=save_res,
                                updated_at=datetime.now(timezone.utc),
                            )
                        )
                        await inv_sess.commit()
                except Exception as exc:
                    raise RuntimeError("Tool completed but its result could not be persisted") from exc

        except Exception as exc:
            logger.error("Tool execution node '%s' failed: %s", node_id, exc, exc_info=True)

            if run_id and tool_idempotency_key and use_db_invocations:
                try:
                    from app.core.database import AsyncSessionLocal
                    from app.models.workflow import WorkflowToolInvocation
                    from sqlalchemy import update

                    is_timeout = isinstance(exc, (TimeoutError, asyncio.TimeoutError))
                    fail_status = "uncertain" if is_timeout else "failed"
                    async with AsyncSessionLocal() as inv_sess:
                        await inv_sess.execute(
                            update(WorkflowToolInvocation)
                            .where(WorkflowToolInvocation.idempotency_key == tool_idempotency_key)
                            .values(
                                status=fail_status,
                                error_message=str(exc),
                                updated_at=datetime.now(timezone.utc),
                            )
                        )
                        await inv_sess.commit()
                except Exception:
                    pass

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
