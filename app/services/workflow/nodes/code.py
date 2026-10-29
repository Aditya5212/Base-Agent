from __future__ import annotations

import ast
import asyncio
import collections
import datetime
import json
import logging
import math
import re
from typing import Any

from app.schemas.workflow import CodeNodeConfig
from app.services.workflow.state import WorkflowState

logger = logging.getLogger(__name__)

# Safe whitelist of modules and builtins
_ALLOWED_MODULES = {
    "math": math,
    "json": json,
    "datetime": datetime,
    "re": re,
    "collections": collections,
}

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root_mod = name.split(".")[0]
    if root_mod in _ALLOWED_MODULES:
        return __import__(name, globals, locals, fromlist, level)
    raise PermissionError(f"Import of module '{name}' is not permitted in sandboxed code.")


_SAFE_BUILTINS = {
    "__import__": _safe_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "None": None,
    "True": True,
    "False": False,
}

_FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "exit",
    "quit",
}


def _verify_safe_ast(code: str) -> None:
    """Inspects AST to ensure no dangerous operations or forbidden imports exist."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name.split(".")[0] for alias in node.names] if isinstance(node, ast.Import) else [node.module.split(".")[0] if node.module else ""]
            for mod_name in names:
                if mod_name not in _ALLOWED_MODULES:
                    raise PermissionError(f"Import of module '{mod_name}' is not permitted in sandboxed code.")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                raise PermissionError(f"Invocation of '{node.func.id}()' is prohibited.")
            elif isinstance(node.func, ast.Attribute) and node.func.attr.startswith("__"):
                raise PermissionError("Access to dunder attributes is prohibited.")


def build_code_node(node_id: str, config_dict: dict[str, Any]):
    """Builds a sandboxed Python execution node handler."""
    config = CodeNodeConfig.model_validate(config_dict or {})

    # Pre-validate AST at construction time
    _verify_safe_ast(config.code)

    def _sync_execute(local_env: dict[str, Any]) -> Any:
        exec_globals = {
            "__builtins__": _SAFE_BUILTINS,
            **_ALLOWED_MODULES,
        }
        exec(config.code, exec_globals, local_env)
        if config.output_variable in local_env:
            return local_env[config.output_variable]
        if "result" in local_env:
            return local_env["result"]
        return None

    async def code_handler(state: WorkflowState) -> dict[str, Any]:
        logger.debug("Executing CODE node '%s'", node_id)
        variables = dict(state.get("variables") or {})
        history = list(state.get("node_history") or [])
        history.append(node_id)

        # Build local scope from selected input variables
        local_scope: dict[str, Any] = {}
        if config.input_variables:
            for var_key in config.input_variables:
                local_scope[var_key] = variables.get(var_key)
        else:
            # If no input_variables specified, pass all variables
            local_scope.update(variables)

        try:
            # Run execution in thread with timeout
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _sync_execute, local_scope),
                timeout=float(config.timeout_seconds),
            )
            variables[config.output_variable] = result
        except asyncio.TimeoutError:
            logger.error("Code node '%s' exceeded timeout of %ds", node_id, config.timeout_seconds)
            variables[config.output_variable] = None
            variables["_code_error"] = f"Execution timed out after {config.timeout_seconds} seconds."
        except Exception as exc:
            logger.error("Code node '%s' execution error: %s", node_id, exc, exc_info=True)
            variables[config.output_variable] = None
            variables["_code_error"] = str(exc)

        return {
            "variables": variables,
            "node_history": history,
        }

    return code_handler

