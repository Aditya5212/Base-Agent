"""Sandboxed Python code tool executor.

Executes user-defined Python functions with AST-level security verification,
restricted built-ins, module import whitelisting, and execution timeouts.
"""

from __future__ import annotations

import ast
import asyncio
from datetime import datetime, timezone
import json
import logging
import math
import random
import re
from typing import Any
import urllib.parse

from app.schemas.tool import CodeToolConfig

logger = logging.getLogger(__name__)


class CodeSecurityError(Exception):
    """Raised when submitted code violates sandbox security policies."""


class CodeExecutionError(Exception):
    """Raised when code execution fails or times out."""


# Attributes that must never be accessed in sandboxed code
FORBIDDEN_ATTRIBUTES: frozenset[str] = frozenset({
    "__subclasses__",
    "__globals__",
    "__code__",
    "__class__",
    "__bases__",
    "__mro__",
    "__builtins__",
    "__import__",
    "__dict__",
    "gi_frame",
    "f_locals",
    "f_globals",
    "cr_frame",
})

# Dangerous builtins that cannot be called
FORBIDDEN_CALLS: frozenset[str] = frozenset({
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "breakpoint",
    "exit",
    "quit",
    "input",
    "help",
})

# Whitelisted standard modules permitted for code tools
DEFAULT_MODULE_WHITELIST: dict[str, Any] = {
    "math": math,
    "json": json,
    "datetime": datetime,
    "re": re,
    "random": random,
    "urllib.parse": urllib.parse,
}


def validate_code_ast(code: str, allowed_modules: list[str]) -> None:
    """Parses code and asserts that no forbidden AST nodes, modules, or attributes are used."""
    try:
        tree = ast.parse(code)
    except SyntaxError as err:
        raise CodeSecurityError(f"Syntax error in code: {err}") from err

    allowed_set = set(allowed_modules or [])

    for node in ast.walk(tree):
        # 1. Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if alias.name not in allowed_set and root_pkg not in allowed_set:
                    raise CodeSecurityError(
                        f"Import of module '{alias.name}' is prohibited. Allowed: {sorted(allowed_set)}"
                    )

        if isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            root_pkg = mod_name.split(".")[0]
            if mod_name not in allowed_set and root_pkg not in allowed_set:
                raise CodeSecurityError(
                    f"Import from module '{mod_name}' is prohibited. Allowed: {sorted(allowed_set)}"
                )

        # 2. Check forbidden attribute access
        if isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRIBUTES:
                raise CodeSecurityError(
                    f"Access to private or internal attribute '{node.attr}' is prohibited."
                )

        # 3. Check forbidden function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise CodeSecurityError(
                    f"Call to built-in function '{node.func.id}' is prohibited."
                )


def _make_safe_import(allowed_modules: list[str]) -> Any:
    """Returns a restricted __import__ implementation enforcing the module whitelist."""
    allowed_set = set(allowed_modules or [])

    def _safe_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        root_pkg = name.split(".")[0]
        if name not in allowed_set and root_pkg not in allowed_set:
            raise ImportError(f"Import of module '{name}' is not permitted.")
        return __import__(name, globals, locals, fromlist, level)

    return _safe_import


def _get_safe_builtins(allowed_modules: list[str]) -> dict[str, Any]:
    """Returns a restricted, safe subset of standard Python builtins with controlled import."""
    return {
        "__import__": _make_safe_import(allowed_modules),
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "chr": chr,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "hex": hex,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "True": True,
        "False": False,
        "None": None,
    }


class CodeToolExecutor:
    """Executes sandboxed Python code tools in isolated thread pools with timeouts."""

    def __init__(self, config: CodeToolConfig):
        self.config = config
        validate_code_ast(self.config.code, self.config.allowed_modules)

    def _sync_execute(self, arguments: dict[str, Any]) -> Any:
        """Executes the function inside an isolated dictionary environment."""
        builtins_dict = _get_safe_builtins(self.config.allowed_modules)
        exec_scope: dict[str, Any] = {
            "__builtins__": builtins_dict,
        }

        # Inject pre-imported modules
        for mod_name in self.config.allowed_modules:
            if mod_name in DEFAULT_MODULE_WHITELIST:
                exec_scope[mod_name] = DEFAULT_MODULE_WHITELIST[mod_name]
            elif mod_name == "timezone":
                exec_scope["timezone"] = timezone

        # Compile and execute definition into exec_scope
        compiled = compile(self.config.code, "<tool_script>", "exec")
        exec(compiled, exec_scope)

        entrypoint = self.config.entrypoint
        if entrypoint in exec_scope:
            target_func = exec_scope[entrypoint]
        else:
            # Fallback: check if a user-defined function exists in scope
            user_funcs = [
                v for k, v in exec_scope.items()
                if callable(v) and k not in builtins_dict and k not in DEFAULT_MODULE_WHITELIST
            ]
            if len(user_funcs) == 1:
                target_func = user_funcs[0]
            else:
                raise CodeExecutionError(
                    f"Entrypoint function '{entrypoint}' not found in code script."
                )

        if not callable(target_func):
            raise CodeExecutionError(f"Entrypoint '{entrypoint}' is not callable.")

        try:
            return target_func(**arguments)
        except TypeError:
            # Fallback to single dict argument if function signature expects a dictionary
            try:
                return target_func(arguments)
            except Exception as inner_exc:
                raise CodeExecutionError(f"Error executing entrypoint: {inner_exc}") from inner_exc
        except Exception as exc:
            raise CodeExecutionError(f"Error executing entrypoint: {exc}") from exc

    async def execute(self, arguments: dict[str, Any]) -> Any:
        """Executes the sandboxed code asynchronously with timeout enforcement."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._sync_execute, arguments or {}),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError as err:
            logger.warning("Code tool execution exceeded %s seconds", self.config.timeout_seconds)
            raise CodeExecutionError(
                f"Execution timed out after {self.config.timeout_seconds} seconds."
            ) from err
        except (CodeSecurityError, CodeExecutionError):
            raise
        except Exception as exc:
            logger.error("Unhandled error during code tool execution: %s", exc)
            raise CodeExecutionError(f"Execution failed: {exc}") from exc

