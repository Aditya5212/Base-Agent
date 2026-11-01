"""Trusted-code execution with AST guards and a killable child process.

Untrusted code still requires an operating-system sandbox at deployment.
"""
from __future__ import annotations
import ast
import asyncio
from typing import Any
from app.schemas.tool import CodeToolConfig
from app.services.tool_executors.code_process import run_code_process
from app.services.tool_executors.code_worker import MODULES


class CodeSecurityError(Exception):
    """Submitted code exceeds server policy."""


class CodeExecutionError(Exception):
    """Code failed or its child process was terminated."""


FORBIDDEN_CALLS = frozenset({"eval", "exec", "compile", "open", "__import__", "breakpoint", "exit", "quit", "input", "help", "globals", "locals", "vars", "getattr", "setattr", "delattr"})
FORBIDDEN_ATTRIBUTES = frozenset({"gi_frame", "f_locals", "f_globals", "cr_frame"})


def validate_code_ast(code: str, allowed_modules: list[str]) -> None:
    allowed = set(allowed_modules)
    if not allowed <= MODULES:
        raise CodeSecurityError("Module whitelist exceeds the server's permitted modules")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CodeSecurityError(f"Syntax error in code: {exc}") from exc
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise CodeSecurityError("Relative imports are prohibited")
            names = [node.module or ""]
            if any(alias.name.startswith("_") for alias in node.names):
                raise CodeSecurityError("Private imports are prohibited")
        for name in names:
            if name not in allowed and name.split(".")[0] not in allowed:
                raise CodeSecurityError(f"Import of module '{name}' is prohibited")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES):
            raise CodeSecurityError(f"Access to private or internal attribute '{node.attr}' is prohibited")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            raise CodeSecurityError(f"Call to built-in function '{node.func.id}' is prohibited")


class CodeToolExecutor:
    def __init__(self, config: CodeToolConfig):
        self.config = config
        validate_code_ast(config.code, config.allowed_modules)

    async def execute(self, arguments: dict[str, Any]) -> Any:
        try:
            return await run_code_process({
                "mode": "tool", "code": self.config.code, "entrypoint": self.config.entrypoint,
                "allowed_modules": self.config.allowed_modules, "arguments": arguments or {},
            }, self.config.timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise CodeExecutionError(f"Execution timed out after {self.config.timeout_seconds} seconds; process terminated") from exc
        except Exception as exc:
            raise CodeExecutionError(f"Execution failed: {exc}") from exc
