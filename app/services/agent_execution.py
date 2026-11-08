"""Bounded model/tool execution and final response validation."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.database import AsyncSessionLocal
from app.schemas.agent import AgentConfig
from app.core.json_validation import validate_payload
from app.services.schema_service import get_structured_schema_by_id
from app.services.tool_registry import ToolRegistry


def validate_runtime_config(cfg: AgentConfig) -> None:
    """Reject reserved settings instead of silently ignoring them."""
    if cfg.model.provider != "deepseek":
        raise ValueError("Only the deepseek model provider is supported; use base_url for a compatible endpoint")
    if cfg.memory.enable_long_term or cfg.memory.summarization_threshold != 40:
        raise ValueError("Long-term memory and automatic summarization are not implemented")
    if any(t.config for t in cfg.tools if t.enabled):
        raise ValueError("Per-agent tool config overrides are not implemented; update the registered tool")
    active = [p for p in cfg.knowledge_providers if p.enabled]
    if len(active) > 1 or any(p.weight != 1.0 for p in active):
        raise ValueError("Weighted multi-provider fusion is not implemented; select one provider with weight 1")
    for provider in active:
        allowed = {"mix", "local", "global", "hybrid", "naive"} if provider.name == "lightrag" else {"balanced", "high_precision", "high_recall", "multi_query"}
        if provider.mode not in allowed:
            raise ValueError(f"Unsupported mode '{provider.mode}' for provider '{provider.name}'")
    if cfg.structured_output.enabled and not cfg.structured_output.schema_id:
        raise ValueError("Structured output requires a registered schema_id")
    if cfg.structured_output.enforce and not cfg.structured_output.enabled:
        raise ValueError("Structured output enforcement requires enabled=True")
    for pattern in (cfg.guardrails.input_validation, cfg.guardrails.output_filtering):
        if pattern:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"Invalid guardrail pattern: {exc}") from exc


def check_agent_input(messages: list[Any], cfg: AgentConfig) -> None:
    text = next((m.text for m in reversed(messages) if isinstance(m, HumanMessage)), "")
    if any(topic.casefold() in text.casefold() for topic in cfg.guardrails.blocked_topics if topic):
        raise ValueError("Input contains a configured blocked topic")
    if cfg.guardrails.input_validation and not re.fullmatch(cfg.guardrails.input_validation, text):
        raise ValueError("Input does not match the configured validation pattern")


async def invoke_agent(llm: Any, messages: list[Any], cfg: AgentConfig, user_id: str, runnable_config: Any) -> list[Any]:
    validate_runtime_config(cfg)
    check_agent_input(messages, cfg)
    schema = None
    tools = []
    enabled = [t.tool_id for t in cfg.tools if t.enabled]
    configurable = runnable_config.get("configurable", {}) if isinstance(runnable_config, dict) else getattr(runnable_config, "configurable", {})
    thread_id = configurable.get("thread_id") if isinstance(configurable, dict) else None
    if enabled or cfg.structured_output.enabled:
        async with AsyncSessionLocal() as db:
            if enabled:
                tools = await ToolRegistry.resolve_tools(user_id, enabled, db=db, thread_id=thread_id)
            if cfg.structured_output.enabled:
                schema_id = cfg.structured_output.schema_id
                if not schema_id:
                    raise ValueError("Structured output requires a registered schema_id")
                record = await get_structured_schema_by_id(db, schema_id, user_id)
                schema = record.schema_definition
    # Resolver supports aliases; detect missing names there rather than guessing here.
    model = llm.bind_tools(tools) if tools else llm
    pending = list(messages)
    if schema is not None:
        pending.insert(0, SystemMessage(content="Return a JSON value matching this JSON Schema: " + json.dumps(schema)))
    emitted = []
    tool_map = {tool.name: tool for tool in tools}
    async with asyncio.timeout(cfg.advanced.timeout_seconds):
        for _ in range(cfg.advanced.max_model_calls):
            reply = await model.ainvoke(pending, config=runnable_config)
            calls = getattr(reply, "tool_calls", [])
            if not calls:
                text = reply.text
                if cfg.guardrails.output_filtering:
                    text = re.sub(cfg.guardrails.output_filtering, "[REDACTED]", text)
                if schema is not None or cfg.identity.response_format == "json":
                    try:
                        parsed = json.loads(text)
                        validate_payload(parsed, schema, "Agent output")
                    except (ValueError, TypeError):
                        if schema is None or cfg.structured_output.enforce:
                            raise
                    else:
                        text = json.dumps(parsed, ensure_ascii=False)
                reply = reply.model_copy(update={"content": text})
                emitted.append(reply)
                return emitted
            if len(calls) > cfg.advanced.max_tool_calls - sum(isinstance(m, ToolMessage) for m in emitted):
                raise ValueError("Agent tool-call limit exceeded")
            emitted.append(reply)
            pending.append(reply)
            for call in calls:
                tool = tool_map.get(call["name"])
                if tool is None:
                    raise ValueError(f"Model requested an unattached tool '{call['name']}'")
                try:
                    output = await tool.ainvoke(call["args"], config=runnable_config)
                    message = ToolMessage(content=json.dumps(output, default=str), tool_call_id=call["id"])
                except Exception as exc:
                    message = ToolMessage(content=json.dumps({"error": str(exc)}), tool_call_id=call["id"], status="error")
                pending.append(message)
                emitted.append(message)
        raise ValueError("Agent model-call limit exceeded")
