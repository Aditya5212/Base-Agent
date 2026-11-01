from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import uuid

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import ValidationError

from app.models.tool import Tool
from app.schemas.workflow import WorkflowResumeRequest, WorkflowRunRequest
from app.services import workflow_service as svc
from app.services.tool_registry import ToolRegistry, json_schema_to_pydantic_model
from app.services.tool_service import ToolService
from app.services.workflow.compiler import WorkflowCompiler
from app.services.workflow.validator import validate_workflow_graph


def workflow(*middle, input_schema=None, output_schema=None):
    nodes = [{"id": "s", "type": "start", "config": {}}, *middle, {"id": "e", "type": "end", "config": {}}]
    edges = [{"source": a["id"], "target": b["id"]} for a, b in zip(nodes, nodes[1:])]
    return SimpleNamespace(nodes=nodes, edges=edges, input_schema=input_schema,
                           output_schema=output_schema, recursion_limit=25, version=1)


def session():
    return SimpleNamespace(add=lambda item: None, commit=AsyncMock(), refresh=AsyncMock(), rollback=AsyncMock())


@pytest.mark.asyncio
async def test_two_approvals_and_resume_are_pinned_to_original_definition():
    wf = workflow(*[{"id": n, "type": "human_approval", "config": {}} for n in ("h1", "h2")])
    db, cp = session(), InMemorySaver()
    with patch.object(svc, "get_workflow_by_id", AsyncMock(return_value=wf)), patch.object(svc, "get_checkpointer", AsyncMock(return_value=cp)):
        run = await svc.execute_workflow(db, uuid.uuid4(), "alice", WorkflowRunRequest())
        assert run.status == "INTERRUPTED"
        assert run.completed_at is None
        assert run.execution_log[-1]["interrupts"][0]["value"]["node_id"] == "h1"
        # Changes to the draft must not affect an existing run.
        wf.nodes[1]["type"] = "unsupported"
        with patch.object(svc, "get_workflow_run", AsyncMock(return_value=run)):
            await svc.resume_workflow_run(db, run.id, "alice", WorkflowResumeRequest(resume_data={"approved": True}))
            assert run.status == "INTERRUPTED"
            assert run.execution_log[-1]["interrupts"][0]["value"]["node_id"] == "h2"
            await svc.resume_workflow_run(db, run.id, "alice", WorkflowResumeRequest(resume_data={"approved": True}))
            assert run.status == "COMPLETED"
            assert run.completed_at is not None
            with pytest.raises(HTTPException) as exc:
                await svc.resume_workflow_run(db, run.id, "alice", WorkflowResumeRequest())
            assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_same_client_thread_is_isolated_between_runs_and_users():
    wf = workflow({"id": "c", "type": "chat", "config": {"system_prompt": "Test", "user_prompt_template": "{{input}}"}})
    db, cp = session(), InMemorySaver()
    model = AsyncMock(return_value=AIMessage(content="answer"))
    with patch.object(svc, "get_workflow_by_id", AsyncMock(return_value=wf)), patch.object(svc, "get_checkpointer", AsyncMock(return_value=cp)), patch("app.services.workflow.nodes.chat.invoke_chat_model_with_middlewares", model):
        runs = []
        for user, text in [("alice", "private"), ("bob", "public"), ("bob", "again")]:
            runs.append(await svc.execute_workflow(db, uuid.uuid4(), user, WorkflowRunRequest(thread_id="same", input_data={"input": text})))
        assert len({run.checkpoint_id for run in runs}) == 3
        assert {run.thread_id for run in runs} == {"same"}
        assert all(m.content != "private" for m in model.call_args_list[1].kwargs["messages"])
        assert all(m.content != "public" for m in model.call_args_list[2].kwargs["messages"])


@pytest.mark.asyncio
async def test_invalid_workflow_input_fails_before_run_or_model():
    wf = workflow(input_schema={"type": "object", "required": ["name"]})
    db = session()
    db.add = AsyncMock()
    with patch.object(svc, "get_workflow_by_id", AsyncMock(return_value=wf)):
        with pytest.raises(HTTPException) as exc:
            await svc.execute_workflow(db, uuid.uuid4(), "alice", WorkflowRunRequest())
    assert exc.value.status_code == 422
    db.add.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_workflow_output_is_failed():
    wf = workflow(output_schema={"type": "object", "required": ["answer"]})
    with patch.object(svc, "get_workflow_by_id", AsyncMock(return_value=wf)), patch.object(svc, "get_checkpointer", AsyncMock(return_value=InMemorySaver())):
        run = await svc.execute_workflow(session(), uuid.uuid4(), "alice", WorkflowRunRequest())
    assert run.status == "FAILED"
    assert "Workflow output" in run.error_message


@pytest.mark.asyncio
async def test_workflow_does_not_run_without_persistence():
    with patch.object(svc, "get_workflow_by_id", AsyncMock(return_value=workflow())), patch.object(svc, "get_checkpointer", AsyncMock(side_effect=RuntimeError("offline"))):
        with pytest.raises(HTTPException) as exc:
            await svc.execute_workflow(session(), uuid.uuid4(), "alice", WorkflowRunRequest())
    assert exc.value.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [True, False])
async def test_tool_approval_prevents_execution_until_explicit_approval(approved):
    wf = workflow({"id": "t", "type": "tool_call", "config": {"tool_name": "calculator", "requires_approval": True, "input_mapping": {"expression": "2+2"}}})
    graph = WorkflowCompiler.compile(wf.nodes, wf.edges, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    with patch("app.services.workflow.nodes.tool_call._execute_builtin_tool", return_value=4) as execute:
        first = await graph.ainvoke({"variables": {}}, config=config)
        assert first["__interrupt__"]
        execute.assert_not_called()
        if approved:
            result = await graph.ainvoke(Command(resume={"approved": True}), config=config)
            assert result["variables"]["tool_result"] == 4
            execute.assert_called_once()
        else:
            with pytest.raises(PermissionError):
                await graph.ainvoke(Command(resume={"approved": False}), config=config)
            execute.assert_not_called()


def test_declared_but_unimplemented_nodes_and_fanout_are_rejected():
    for kind in ("parallel", "subgraph", "unknown"):
        wf = workflow({"id": "n", "type": kind, "config": {}})
        valid, errors, _ = validate_workflow_graph(wf.nodes, wf.edges)
        assert not valid
        assert any("Unsupported" in error for error in errors)
    wf = workflow({"id": "n", "type": "code", "config": {"code": "result=1"}})
    wf.edges.append({"source": "s", "target": "e"})
    assert not validate_workflow_graph(wf.nodes, wf.edges)[0]


def tool(**overrides):
    values = dict(id=uuid.uuid4(), user_id="alice", version=1, name="test_tool", description="Test execution",
                  type="code", config={"code": "def execute():\n    return 1 / 0"}, input_schema={"type": "object"})
    values.update(overrides)
    return Tool(**values)


@pytest.mark.asyncio
async def test_tool_failure_is_reported_as_failure():
    item = tool()
    with patch.object(ToolService, "get_tool", AsyncMock(return_value=item)):
        result = await ToolService.execute_tool(session(), "alice", item.id, {})
    assert result.success is False
    assert result.output is None
    assert "division by zero" in result.error


@pytest.mark.asyncio
async def test_tool_output_schema_is_enforced_and_values_remain_typed():
    item = tool(config={"code": "def execute():\n    return {'n': 2}"}, output_schema={"type": "object", "properties": {"n": {"minimum": 10}}})
    with pytest.raises(ValueError, match="Tool output"):
        await ToolRegistry.compile_tool(item).ainvoke({})
    item = tool(config={"code": "def execute():\n    return {'n': 2}"})
    assert await ToolRegistry.compile_tool(item).ainvoke({}) == {"n": 2}


@pytest.mark.parametrize("invalid", [{"n": -2}, {"n": 15}, {"n": "10"}, {"n": 10, "extra": 1}])
def test_tool_argument_constraints_are_authoritative(invalid):
    model = json_schema_to_pydantic_model({"type": "object", "properties": {"n": {"type": "integer", "minimum": 10, "enum": [10, 20]}}, "required": ["n"], "additionalProperties": False}, "Args")
    with pytest.raises(ValidationError):
        model.model_validate(invalid)


def test_nullable_and_nested_tool_schemas():
    model = json_schema_to_pydantic_model({"type": "object", "properties": {"n": {"type": ["string", "null"]}, "items": {"type": "array", "items": {"type": "integer"}}}}, "Args")
    assert model.model_validate({"n": None, "items": [1]}).items == [1]
    with pytest.raises(ValidationError):
        model.model_validate({"items": ["wrong"]})


@pytest.mark.asyncio
async def test_node_input_and_output_schema_validation():
    from app.services.workflow.nodes.start import build_start_node
    from app.services.workflow.nodes.end import build_end_node
    schema = {"required": ["name"]}
    with pytest.raises(ValueError, match="Start input"):
        await build_start_node("s", {"input_schema": schema})({"variables": {}})
    with pytest.raises(ValueError, match="End output"):
        await build_end_node("e", {"output_schema": schema})({"variables": {}})


def test_internal_workflow_checkpoint_namespace_cannot_be_used_by_chat():
    from app.schemas.chat import ChatRequest
    from app.services.memory import validate_chat_thread_id
    with pytest.raises(ValidationError):
        ChatRequest(message="test", thread_id="workflow:private-checkpoint")
    with pytest.raises(ValueError, match="reserved"):
        validate_chat_thread_id("workflow:private-checkpoint")


@pytest.mark.asyncio
async def test_expired_and_legacy_runs_cannot_resume():
    from datetime import datetime, timedelta, timezone
    run = SimpleNamespace(status="INTERRUPTED", definition_snapshot={"nodes": []}, checkpoint_id="id",
                          execution_log=[{"interrupts": [{"expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}]}])
    with patch.object(svc, "get_workflow_run", AsyncMock(return_value=run)):
        result = await svc.resume_workflow_run(session(), uuid.uuid4(), "alice", WorkflowResumeRequest())
    assert result.status == "EXPIRED"
    run.status = "INTERRUPTED"
    run.definition_snapshot = None
    with patch.object(svc, "get_workflow_run", AsyncMock(return_value=run)):
        with pytest.raises(HTTPException) as exc:
            await svc.resume_workflow_run(session(), uuid.uuid4(), "alice", WorkflowResumeRequest())
    assert exc.value.status_code == 409
