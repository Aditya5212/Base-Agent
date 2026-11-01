from __future__ import annotations
from typing import Any
from app.schemas.workflow import CodeNodeConfig
from app.services.tool_executors.code_executor import validate_code_ast, CodeSecurityError
from app.services.tool_executors.code_process import run_code_process
from app.services.workflow.state import WorkflowState

_ALLOWED_MODULES = ["math", "json", "datetime", "re", "collections"]


def build_code_node(node_id: str, config_dict: dict[str, Any]):
    """Build a trusted-code transformation with process termination on timeout."""
    config = CodeNodeConfig.model_validate(config_dict or {})
    try:
        validate_code_ast(config.code, _ALLOWED_MODULES)
    except CodeSecurityError as exc:
        raise PermissionError(f"Code is not permitted: {exc}") from exc

    async def code_handler(state: WorkflowState) -> dict[str, Any]:
        variables = dict(state.get("variables") or {})
        local_scope = {key: variables.get(key) for key in config.input_variables} if config.input_variables else dict(variables)
        result = await run_code_process({
            "mode": "script", "code": config.code, "arguments": local_scope,
            "allowed_modules": _ALLOWED_MODULES, "output_variable": config.output_variable,
        }, float(config.timeout_seconds))
        variables[config.output_variable] = result
        return {"variables": variables, "node_history": [*(state.get("node_history") or []), node_id]}

    return code_handler
