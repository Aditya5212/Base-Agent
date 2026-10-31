"""Dynamic Tool Registry for compiling, caching, and resolving custom and built-in tools.

Compiles persisted Tool database entities (HTTP APIs, sandboxed Python code, read-only SQL)
into LangChain BaseTool instances with dynamic Pydantic argument schemas and tenant isolation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Coroutine
import uuid

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tool import Tool
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
_COMPILED_TOOL_CACHE: dict[tuple[str, str, int], BaseTool] = {}


def json_schema_to_pydantic_model(schema: dict[str, Any], model_name: str) -> type[BaseModel]:
    """Dynamically converts a JSON Schema properties dictionary into a Pydantic model class."""
    properties = schema.get("properties") or {}
    required_fields = set(schema.get("required") or [])

    type_mapping: dict[str, Any] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    fields: dict[str, tuple[Any, Any]] = {}

    for prop_name, prop_spec in properties.items():
        if not isinstance(prop_spec, dict):
            fields[prop_name] = (Any, Field(default=None))
            continue

        raw_type = prop_spec.get("type", "string")
        py_type = type_mapping.get(raw_type, Any)
        description = prop_spec.get("description", "")
        default_val = prop_spec.get("default", None)

        if prop_name in required_fields:
            fields[prop_name] = (py_type, Field(..., description=description))
        else:
            fields[prop_name] = (py_type | None, Field(default=default_val, description=description))

    if not fields:
        # Fallback empty model
        return create_model(model_name)

    return create_model(model_name, **fields)


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
    def compile_tool(cls, tool_orm: Tool) -> BaseTool:
        """Compiles a persisted Tool ORM entity into an executable LangChain BaseTool."""
        cache_key = (tool_orm.user_id, str(tool_orm.id), tool_orm.version)
        if cache_key in _COMPILED_TOOL_CACHE:
            return _COMPILED_TOOL_CACHE[cache_key]

        tool_type = ToolType(tool_orm.type)
        tool_config = tool_orm.config or {}

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
        async def _execute_coroutine(**kwargs: Any) -> str:
            try:
                res = await executor.execute(kwargs)
                if isinstance(res, (dict, list)):
                    return json.dumps(res, default=str)
                return str(res)
            except Exception as exc:
                logger.error("Error executing tool '%s': %s", tool_orm.name, exc, exc_info=True)
                return f"Error executing tool '{tool_orm.name}': {exc}"

        compiled = StructuredTool.from_function(
            coroutine=_execute_coroutine,
            name=tool_orm.name,
            description=tool_orm.description,
            args_schema=args_schema,
        )

        _COMPILED_TOOL_CACHE[cache_key] = compiled
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
                    try:
                        compiled_tool = cls.compile_tool(tool_record)
                        resolved.append(compiled_tool)
                    except Exception as exc:
                        logger.error("Failed to compile tool '%s': %s", tool_record.name, exc)

        return resolved

