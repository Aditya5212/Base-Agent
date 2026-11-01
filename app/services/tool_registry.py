"""Dynamic Tool Registry for compiling, caching, and resolving custom and built-in tools.

Compiles persisted Tool database entities (HTTP APIs, sandboxed Python code, read-only SQL)
into LangChain BaseTool instances with dynamic Pydantic argument schemas and tenant isolation.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from copy import deepcopy
from typing import Any
import uuid

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tool import Tool
from app.core.json_validation import validate_payload, validate_schema
from app.schemas.tool import (
    CodeToolConfig,
    DatabaseToolConfig,
    HttpToolConfig,
    ToolType,
)
from app.services.rag_tools import get_user_rag_tool_map, get_user_rag_tools
from app.services.tool_executors.code_executor import CodeToolExecutor
from app.services.tool_executors.database_executor import DatabaseToolExecutor
from app.services.tool_executors.http_executor import HttpToolExecutor

logger = logging.getLogger(__name__)

# In-memory compiled tool cache: (user_id, tool_id_str, version) -> BaseTool
_COMPILED_TOOL_CACHE: OrderedDict[tuple[str, str, int], BaseTool] = OrderedDict()
_MAX_COMPILED_TOOLS = 256


def json_schema_to_pydantic_model(schema: dict[str, Any], model_name: str) -> type[BaseModel]:
    """Dynamically converts a JSON Schema properties dictionary into a Pydantic model class."""
    validate_schema(schema)
    if schema.get("type", "object") != "object":
        raise ValueError("Tool arguments must use an object JSON Schema")
    properties = schema.get("properties") or {}
    required_fields = set(schema.get("required") or [])

    class SchemaArguments(BaseModel):
        model_config = ConfigDict(extra="allow")

        @model_validator(mode="before")
        @classmethod
        def validate_schema(cls, value: Any) -> Any:
            validate_payload(value, schema, "Tool arguments")
            return value

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return deepcopy(schema)

    # create_model mixes named control keywords with dynamic field keywords.
    fields: dict[str, Any] = {}

    for prop_name, prop_spec in properties.items():
        if not isinstance(prop_spec, dict):
            fields[prop_name] = (Any, Field(default=None))
            continue

        # JSON Schema is authoritative; Any avoids coercing or losing nested values.
        py_type = Any
        description = prop_spec.get("description", "")
        default_val = prop_spec.get("default", None)

        if prop_name in required_fields:
            fields[prop_name] = (py_type, Field(..., description=description))
        else:
            fields[prop_name] = (py_type | None, Field(default=default_val, description=description))

    if not fields:
        # Fallback empty model
        return create_model(model_name, __base__=SchemaArguments)

    return create_model(model_name, __base__=SchemaArguments, **fields)


def _make_calculator_tool() -> BaseTool:
    """Built-in safe mathematical calculator tool."""
    from langchain_core.tools import tool

    @tool
    def calculator(expression: str) -> str:
        """Evaluates basic arithmetic expressions safely (e.g. '128 * 4 + 15')."""
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

        def _eval_node(node: Any) -> float:
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.BinOp):
                left = _eval_node(node.left)
                right = _eval_node(node.right)
                op_type = type(node.op)
                if op_type in ops:
                    return ops[op_type](left, right)
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            if isinstance(node, ast.UnaryOp):
                operand = _eval_node(node.operand)
                op_type = type(node.op)
                if op_type in ops:
                    return ops[op_type](operand)
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")

        try:
            tree = ast.parse(expression.strip(), mode="eval")
            res = _eval_node(tree.body)
            return str(round(res, 6) if isinstance(res, float) and res.is_integer() is False else int(res))
        except Exception as exc:
            return f"Error evaluating expression '{expression}': {exc}"

    return calculator


class ToolRegistry:
    """Manages compilation, tenant-scoping, and dynamic execution of tools."""

    @classmethod
    def compile_tool(cls, tool_orm: Tool, *, cache: bool = True) -> BaseTool:
        """Compiles a persisted Tool ORM entity into an executable LangChain BaseTool."""
        cache_key = (tool_orm.user_id, str(tool_orm.id), tool_orm.version)
        if cache and cache_key in _COMPILED_TOOL_CACHE:
            _COMPILED_TOOL_CACHE.move_to_end(cache_key)
            return _COMPILED_TOOL_CACHE[cache_key]

        tool_type = ToolType(tool_orm.type)
        tool_config = tool_orm.config or {}
        if tool_orm.output_schema is not None:
            validate_schema(tool_orm.output_schema)

        # 1. Instantiate the appropriate executor
        executor: Any
        if tool_type == ToolType.API:
            http_cfg = HttpToolConfig.model_validate(tool_config)
            executor = HttpToolExecutor(http_cfg)
        elif tool_type == ToolType.CODE:
            code_cfg = CodeToolConfig.model_validate(tool_config)
            executor = CodeToolExecutor(code_cfg)
        elif tool_type == ToolType.DATABASE:
            db_cfg = DatabaseToolConfig.model_validate(tool_config)
            executor = DatabaseToolExecutor(db_cfg)
        elif tool_type == ToolType.BUILT_IN:
            if tool_orm.name == "calculator":
                return _make_calculator_tool()
            raise ValueError(f"Unknown built-in tool handler: {tool_orm.name}")
        else:
            raise ValueError(f"Unsupported tool type: {tool_type}")

        # 2. Build dynamic Pydantic schema for parameters
        clean_model_name = "".join(c for c in tool_orm.name.title() if c.isalnum()) + "Args"
        args_schema = json_schema_to_pydantic_model(tool_orm.input_schema or {}, clean_model_name)

        # 3. Create async execution handler
        async def _execute_coroutine(**kwargs: Any) -> Any:
            validate_payload(kwargs, tool_orm.input_schema, "Tool arguments")
            res = await executor.execute(kwargs)
            validate_payload(res, tool_orm.output_schema, "Tool output")
            return res

        compiled = StructuredTool.from_function(
            coroutine=_execute_coroutine,
            name=tool_orm.name,
            description=tool_orm.description,
            args_schema=args_schema,
        )

        if cache:
            _COMPILED_TOOL_CACHE[cache_key] = compiled
            while len(_COMPILED_TOOL_CACHE) > _MAX_COMPILED_TOOLS:
                _COMPILED_TOOL_CACHE.popitem(last=False)
        return compiled

    @classmethod
    def invalidate_tool(cls, user_id: str, tool_id: str) -> None:
        """Removes all cached compiled versions for the given tool."""
        keys_to_remove = [k for k in _COMPILED_TOOL_CACHE if k[0] == user_id and k[1] == str(tool_id)]
        for k in keys_to_remove:
            _COMPILED_TOOL_CACHE.pop(k, None)

    @classmethod
    def invalidate_user_tools(cls, user_id: str) -> None:
        """Clears all cached tools for a specific tenant user."""
        keys_to_remove = [k for k in _COMPILED_TOOL_CACHE if k[0] == user_id]
        for k in keys_to_remove:
            _COMPILED_TOOL_CACHE.pop(k, None)

    @classmethod
    def clear_cache(cls) -> None:
        """Clears the entire compilation cache."""
        _COMPILED_TOOL_CACHE.clear()

    @classmethod
    async def resolve_tools(
        cls,
        user_id: str,
        tool_names: list[str],
        db: AsyncSession | None = None,
    ) -> list[BaseTool]:
        """Resolves tool names/IDs into executable LangChain BaseTool instances.

        Supports:
        - RAG research tools: 'hybrid_document_search', 'graph_rag_search', 'multi_query_search', 'all_rag_tools'
        - Standard built-in tools: 'calculator'
        - Custom user-created tools stored in database
        """
        if not tool_names:
            return []

        resolved: list[BaseTool] = []
        rag_tools_map = get_user_rag_tool_map(user_id=user_id)
        db_tool_names_or_ids: list[str] = []
        found_identifiers: set[str] = set()

        for name in tool_names:
            normalized = name.lower().strip()
            if normalized == "all_rag_tools":
                resolved.extend(get_user_rag_tools(user_id=user_id))
            elif normalized in rag_tools_map:
                resolved.append(rag_tools_map[normalized])
            elif normalized in ("calculator", "calc", "math"):
                resolved.append(_make_calculator_tool())
            else:
                db_tool_names_or_ids.append(name.strip())

        # If custom database tools were referenced and a db session is provided
        if db_tool_names_or_ids and db is not None:
            # Query custom tools owned by this user or global system tools
            stmt = select(Tool).where(
                or_(Tool.user_id == user_id, Tool.is_system.is_(True))
            )

            # Check if any identifiers look like UUIDs
            uuid_filters = []
            name_filters = []
            for item in db_tool_names_or_ids:
                try:
                    uuid_filters.append(uuid.UUID(item))
                except ValueError:
                    name_filters.append(item)

            conditions = []
            if uuid_filters:
                conditions.append(Tool.id.in_(uuid_filters))
            if name_filters:
                conditions.append(Tool.name.in_(name_filters))

            if conditions:
                stmt = stmt.where(or_(*conditions))
                result = await db.execute(stmt)
                db_tools = result.scalars().all()

                for tool_record in db_tools:
                    found_identifiers.update((tool_record.name, str(tool_record.id)))
                    try:
                        compiled_tool = cls.compile_tool(tool_record)
                        resolved.append(compiled_tool)
                    except Exception as exc:
                        raise ValueError(f"Cannot compile tool '{tool_record.name}': {exc}") from exc

        missing = set(db_tool_names_or_ids) - found_identifiers
        if missing:
            raise ValueError(f"Unknown or inaccessible tools: {', '.join(sorted(missing))}")
        return list({tool.name: tool for tool in resolved}.values())

