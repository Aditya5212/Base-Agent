import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.rag.workspace import user_workspace, thread_workspace, agent_workspace
from app.schemas.agent import AgentConfig
from app.schemas.tool import CodeToolConfig
from app.services.agent_execution import invoke_agent, validate_runtime_config
from app.services.tool_executors.code_executor import CodeToolExecutor, CodeExecutionError, CodeSecurityError
from app.services.tool_registry import _make_calculator_tool, ToolRegistry


@asynccontextmanager
async def fake_db():
    yield object()


@pytest.mark.asyncio
async def test_agent_completes_tool_loop_and_preserves_tool_messages():
    cfg = AgentConfig.model_validate({"tools": [{"tool_id": "calculator"}]})
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=[
        AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "2+2"}, "id": "call1"}]),
        AIMessage(content="4"),
    ]))
    model.bind_tools = lambda tools: model
    with patch("app.services.agent_execution.AsyncSessionLocal", fake_db), patch.object(ToolRegistry, "resolve_tools", AsyncMock(return_value=[_make_calculator_tool()])):
        messages = await invoke_agent(model, [HumanMessage(content="2+2")], cfg, "alice", {})
    assert len(messages) == 3
    assert isinstance(messages[1], ToolMessage)
    assert messages[-1].content == "4"
    assert model.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_agent_model_call_limit_is_enforced():
    cfg = AgentConfig.model_validate({"tools": [{"tool_id": "calculator"}], "advanced": {"max_model_calls": 1}})
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "2+2"}, "id": "call1"}])))
    model.bind_tools = lambda tools: model
    with patch("app.services.agent_execution.AsyncSessionLocal", fake_db), patch.object(ToolRegistry, "resolve_tools", AsyncMock(return_value=[_make_calculator_tool()])):
        with pytest.raises(ValueError, match="model-call limit"):
            await invoke_agent(model, [HumanMessage(content="calculate")], cfg, "alice", {})


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ["plain text", '{"n": -1}'])
async def test_structured_agent_output_is_checked_against_owned_schema(response):
    cfg = AgentConfig.model_validate({"structured_output": {"enabled": True, "schema_id": "schema-id", "enforce": True}})
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content=response)))
    get_schema = AsyncMock(return_value=SimpleNamespace(schema_definition={"type": "object", "properties": {"n": {"minimum": 0}}}))
    with patch("app.services.agent_execution.AsyncSessionLocal", fake_db), patch("app.services.agent_execution.get_structured_schema_by_id", get_schema):
        with pytest.raises(ValueError):
            await invoke_agent(model, [HumanMessage(content="data")], cfg, "alice", {})
    assert get_schema.call_args.args[2] == "alice"


@pytest.mark.asyncio
async def test_input_guardrail_blocks_model_and_output_filter_redacts():
    cfg = AgentConfig.model_validate({"guardrails": {"blocked_topics": ["secret-topic"], "output_filtering": "1234"}})
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content="code 1234")))
    with pytest.raises(ValueError, match="blocked topic"):
        await invoke_agent(model, [HumanMessage(content="SECRET-TOPIC")], cfg, "alice", {})
    model.ainvoke.assert_not_called()
    answer = await invoke_agent(model, [HumanMessage(content="allowed")], cfg, "alice", {})
    assert answer[-1].content == "code [REDACTED]"


@pytest.mark.asyncio
async def test_filtered_sse_only_emits_validated_final_answer():
    from app.services.agent import stream_agent_events
    cfg = AgentConfig.model_validate({"guardrails": {"output_filtering": "private"}})
    final = {"messages": [AIMessage(content="[REDACTED]")]}
    with patch("app.services.agent.execute_agent", AsyncMock(return_value=(final, "test-model"))) as execute:
        events = [event async for event in stream_agent_events("hi", "thread", agent_config=cfg)]
    execute.assert_awaited_once()
    assert [e.event for e in events] == ["token", "end"]
    assert "private" not in str([e.data for e in events])


@pytest.mark.parametrize("cfg", [
    {"model": {"provider": "anthropic"}},
    {"memory": {"enable_long_term": True}},
    {"tools": [{"tool_id": "calculator", "config": {"ignored": True}}]},
    {"knowledge_providers": [{"name": "hybrid", "weight": .5}]},
    {"structured_output": {"enabled": True}},
])
def test_unimplemented_agent_settings_fail_explicitly(cfg):
    with pytest.raises(ValueError):
        validate_runtime_config(AgentConfig.model_validate(cfg))


def test_workspace_identity_is_stable_and_collision_resistant():
    assert user_workspace("root") == "u_root"
    assert user_workspace("alice-bob") != user_workspace("alice_bob")
    assert user_workspace("x" * 70 + "a") != user_workspace("x" * 70 + "b")
    assert thread_workspace("a_t_b", "c") != thread_workspace("a", "b_t_c")
    assert agent_workspace("alice", "uuid-with-dashes") == agent_workspace("alice", "uuid-with-dashes")
    assert len(agent_workspace("x" * 100, "y" * 100)) <= 60


@pytest.mark.asyncio
async def test_code_runs_in_child_and_return_value_remains_json_typed():
    executor = CodeToolExecutor(CodeToolConfig(code="def execute(n):\n    return {'n': n * 2}"))
    assert await executor.execute({"n": 4}) == {"n": 8}


@pytest.mark.asyncio
async def test_code_timeout_kills_and_reaps_process():
    import subprocess
    original = subprocess.Popen
    children = []
    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        children.append(process)
        return process
    executor = CodeToolExecutor(CodeToolConfig(code="def execute():\n    while True:\n        pass", timeout_seconds=0.5))
    with patch("app.services.tool_executors.code_process.subprocess.Popen", spawn):
        with pytest.raises(CodeExecutionError, match="process terminated"):
            await executor.execute({})
    assert len(children) == 1 and children[0].poll() is not None


@pytest.mark.asyncio
async def test_code_cancellation_kills_process():
    executor = CodeToolExecutor(CodeToolConfig(code="def execute():\n    while True:\n        pass", timeout_seconds=10))
    task = asyncio.create_task(executor.execute({}))
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_caller_cannot_expand_module_policy():
    with pytest.raises(CodeSecurityError, match="server"):
        CodeToolExecutor(CodeToolConfig(code="import os\ndef execute():\n    return os.name", allowed_modules=["os"]))


@pytest.mark.asyncio
async def test_unknown_tool_is_a_failure():
    with pytest.raises(ValueError, match="Unknown or inaccessible"):
        await ToolRegistry.resolve_tools("alice", ["missing-tool"])


@pytest.mark.asyncio
async def test_development_response_still_applies_output_filter():
    from app.services.agent import chat_node
    cfg = AgentConfig.model_validate({"guardrails": {"output_filtering": "1234"}})
    with patch("app.services.agent.get_llm", return_value=None):
        answer = await chat_node({"messages": [HumanMessage(content="code 1234")]}, {"configurable": {"agent_config": cfg}})
    assert "1234" not in answer["messages"][-1].text
    assert "[REDACTED]" in answer["messages"][-1].text


@pytest.mark.asyncio
async def test_message_window_keeps_complete_tool_call_and_result():
    from app.services.agent import chat_node
    cfg = AgentConfig.model_validate({"memory": {"context_window_messages": 3}})
    history = [HumanMessage(content="first"),
               AIMessage(content="", tool_calls=[{"name": "calculator", "args": {}, "id": "old"}]),
               ToolMessage(content="4", tool_call_id="old"), AIMessage(content="4"), HumanMessage(content="next")]
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content="answer")))
    with patch("app.services.agent.get_llm", return_value=model):
        await chat_node({"messages": history}, {"configurable": {"agent_config": cfg}})
    sent = model.ainvoke.call_args.args[0]
    assert any(isinstance(m, AIMessage) and m.tool_calls for m in sent)
    assert any(isinstance(m, ToolMessage) for m in sent)
