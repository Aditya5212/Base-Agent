"""JSON Schema validation shared by contracts, workflows, and tools."""
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


def validate_schema(schema: dict[str, Any]) -> None:
    Draft202012Validator.check_schema(schema)
    def local_refs(node):
        if isinstance(node, dict):
            for key, item in node.items():
                if key in {"$ref", "$dynamicRef"} and isinstance(item, str) and not item.startswith("#"):
                    raise ValueError("Only local JSON Schema references are supported")
                local_refs(item)
        elif isinstance(node, list):
            for item in node:
                local_refs(item)
    local_refs(schema)


def validate_payload(value: Any, schema: dict[str, Any] | None, label: str) -> None:
    if schema is None:
        return
    validate_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    error = next(validator.iter_errors(value), None)
    if error is not None:
        path = ".".join(str(part) for part in error.absolute_path) or "$"
        raise ValueError(f"{label} at {path}: {error.message}")
