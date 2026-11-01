"""Standalone stdlib-only worker. Receives JSON, never application credentials."""
import builtins
import importlib
import inspect
import json
import sys
from typing import Any

MODULES = {"math", "json", "datetime", "re", "random", "urllib.parse", "collections", "itertools", "typing"}
BUILTINS = ("abs", "all", "any", "bin", "bool", "chr", "dict", "divmod", "enumerate", "filter", "float", "format", "hex", "int", "isinstance", "issubclass", "len", "list", "map", "max", "min", "oct", "ord", "pow", "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip")


def execute(payload):
    allowed = set(payload["allowed_modules"])
    if not allowed <= MODULES:
        raise ValueError("Requested module is not allowed by server policy")

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or (name not in allowed and name.split(".")[0] not in allowed):
            raise ImportError(f"Import of '{name}' is not permitted")
        return builtins.__import__(name, globals, locals, fromlist, level)

    safe_builtins: dict[str, Any] = {name: getattr(builtins, name) for name in BUILTINS}
    safe_builtins["__import__"] = safe_import
    scope: dict[str, Any] = {"__builtins__": safe_builtins}
    for name in allowed:
        # A dotted import binds the root module, matching Python import syntax.
        importlib.import_module(name)
        scope[name.split(".")[0]] = importlib.import_module(name.split(".")[0])
    if payload["mode"] == "script":
        scope.update(payload["arguments"])
        exec(compile(payload["code"], "<workflow_code>", "exec"), scope)
        return scope.get(payload["output_variable"], scope.get("result"))
    exec(compile(payload["code"], "<tool_code>", "exec"), scope)
    target = scope.get(payload["entrypoint"])
    if not callable(target):
        raise ValueError(f"Entrypoint '{payload['entrypoint']}' is missing or not callable")
    args = payload["arguments"]
    # Bind before invocation: a TypeError inside the function must not execute it twice.
    try:
        inspect.signature(target).bind(**args)
    except TypeError:
        inspect.signature(target).bind(args)
        return target(args)
    return target(**args)


if __name__ == "__main__":
    try:
        answer = {"result": execute(json.load(sys.stdin))}
        encoded = json.dumps(answer, allow_nan=False)
        if len(encoded.encode()) > 2_000_000:
            raise ValueError("Code output exceeds 2 MB")
    except BaseException as exc:
        encoded = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    sys.stdout.write(encoded)
