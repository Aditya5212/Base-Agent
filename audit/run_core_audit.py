"""Offline audit evidence. Uses synthetic inputs and in-memory/mock infrastructure.

Run from repository root: .venv/Scripts/python.exe audit/run_core_audit.py
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update({
    "PYTHON_DOTENV_DISABLED": "1",
    "DATABASE_URL": "postgresql+psycopg://audit:audit@127.0.0.1:1/audit",
    "DEEPSEEK_API_KEY": "", "OPENAI_API_KEY": "",
    "SECRET_ENCRYPTION_KEY": "offline-audit-synthetic-key",
    "DEBUG": "false", "ENVIRONMENT": "test",
    "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
})


async def run_probes():
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from app.core.rag.workspace import user_workspace
    from app.schemas.agent import AgentConfig
    from app.schemas.tool import CodeToolConfig
    from app.schemas.workflow import WorkflowRunRequest
    from app.services.agent import chat_node
    from app.services.tool_executors.code_executor import CodeToolExecutor, CodeExecutionError
    from app.services.tool_executors.database_executor import validate_readonly_sql
    from app.services.tool_registry import ToolRegistry, json_schema_to_pydantic_model
    from app.services.tool_service import ToolService
    from app.services.workflow.compiler import WorkflowCompiler
    from app.services.workflow.nodes.tool_call import build_tool_call_node
    from app.services import workflow_service as svc

    evidence = {}
    evidence["workspace_collision"] = {
        "alice-bob": user_workspace("alice-bob"),
        "alice_bob": user_workspace("alice_bob"),
        "long_ids_collide": user_workspace("x" * 70 + "a") == user_workspace("x" * 70 + "b"),
    }

    async def noop(*args, **kwargs):
        pass

    session = SimpleNamespace(add=lambda obj: None, commit=noop, refresh=noop)
    workflow_id = uuid.uuid4()
    human_nodes = [
        {"id": "s", "type": "start", "config": {}},
        {"id": "h", "type": "human_approval", "config": {}},
        {"id": "e", "type": "end", "config": {}},
    ]
    human_edges = [{"source": "s", "target": "h"}, {"source": "h", "target": "e"}]
    wf = SimpleNamespace(nodes=human_nodes, edges=human_edges, recursion_limit=25)
    cp = InMemorySaver()
    with patch.object(svc, "get_workflow_by_id", AsyncMock(return_value=wf)), patch.object(svc, "get_checkpointer", AsyncMock(return_value=cp)):
        run = await svc.execute_workflow(session, workflow_id, "alice", WorkflowRunRequest(input_data={}))
        snapshot = await WorkflowCompiler.compile(human_nodes, human_edges, checkpointer=cp).aget_state({"configurable": {"thread_id": run.thread_id}})
        evidence["human_approval_service_status"] = {"reported_status": run.status, "pending_nodes": list(snapshot.next), "interrupt_count": sum(len(t.interrupts) for t in snapshot.tasks)}

    schema_nodes = [
        {"id": "s", "type": "start", "config": {"input_schema": {"type": "object", "required": ["required_key"]}}},
        {"id": "e", "type": "end", "config": {"output_schema": {"type": "object", "required": ["required_output"]}}},
    ]
    result = await WorkflowCompiler.compile(schema_nodes, [{"source": "s", "target": "e"}]).ainvoke({"variables": {}})
    evidence["workflow_schemas_not_enforced"] = {"invalid_empty_input_and_output_accepted": result["variables"] == {}}

    approval_tool = build_tool_call_node("approved_calc", {"tool_name": "calculator", "input_mapping": {"expression": "2 + 2"}, "requires_approval": True})
    approval_result = await approval_tool({"variables": {}, "user_id": "alice"})
    evidence["tool_requires_approval_ignored"] = {"executed_without_interrupt": approval_result["variables"]["tool_result"] == 4}

    for node_type in ("parallel", "subgraph"):
        nodes = [{"id": "s", "type": "start"}, {"id": "n", "type": node_type, "config": {}}, {"id": "e", "type": "end"}]
        try:
            WorkflowCompiler.compile(nodes, [{"source": "s", "target": "n"}, {"source": "n", "target": "e"}])
        except Exception as exc:
            evidence[node_type + "_compile"] = str(exc)

    fork_nodes = [
        {"id": "s", "type": "start"},
        {"id": "a", "type": "code", "config": {"code": "result = 1"}},
        {"id": "b", "type": "code", "config": {"code": "result = 2"}},
        {"id": "e", "type": "end"},
    ]
    fork_edges = [{"source": "s", "target": "a"}, {"source": "s", "target": "b"}, {"source": "a", "target": "e"}, {"source": "b", "target": "e"}]
    try:
        await WorkflowCompiler.compile(fork_nodes, fork_edges).ainvoke({"variables": {}})
    except Exception as exc:
        evidence["ordinary_fanout"] = {"error_type": type(exc).__name__, "message": str(exc).split("\n")[0]}

    mock_llm = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content="plain unstructured answer")))
    cfg = AgentConfig.model_validate({"tools": [{"tool_id": "calculator"}], "structured_output": {"enabled": True, "schema_id": str(uuid.uuid4()), "enforce": True}, "guardrails": {"blocked_topics": ["forbidden-example"]}})
    with patch("app.services.agent.get_llm", return_value=mock_llm):
        reply = await chat_node({"messages": [HumanMessage(content="forbidden-example: calculate 2 + 2")]}, {"configurable": {"agent_config": cfg}})
    evidence["chat_config_runtime"] = {"structured_enforce_returned_plain_text": reply["messages"][0].content, "llm_invoked_with_blocked_topic": mock_llm.ainvoke.await_count == 1, "tool_binding_available_on_stub": False, "call_received_plain_messages": isinstance(mock_llm.ainvoke.call_args.args[0], list)}

    chat_nodes = [
        {"id": "s", "type": "start"},
        {"id": "c", "type": "chat", "config": {"system_prompt": "Audit test", "user_prompt_template": "{{input}}"}},
        {"id": "e", "type": "end"},
    ]
    chat_wf = SimpleNamespace(nodes=chat_nodes, edges=[{"source": "s", "target": "c"}, {"source": "c", "target": "e"}], recursion_limit=25)
    cp2 = InMemorySaver()
    fake_invoke = AsyncMock(return_value=AIMessage(content="synthetic response"))
    with patch.object(svc, "get_workflow_by_id", AsyncMock(return_value=chat_wf)), patch.object(svc, "get_checkpointer", AsyncMock(return_value=cp2)), patch("app.services.workflow.nodes.chat.invoke_chat_model_with_middlewares", fake_invoke):
        await svc.execute_workflow(session, uuid.uuid4(), "alice", WorkflowRunRequest(thread_id="same-client-thread", input_data={"input": "ALICE_SYNTHETIC_PRIVATE_TEXT"}))
        await svc.execute_workflow(session, uuid.uuid4(), "bob", WorkflowRunRequest(thread_id="same-client-thread", input_data={"input": "Bob request"}))
    evidence["workflow_checkpoint_collision"] = {"bob_model_context_contains_alice_text": any(m.content == "ALICE_SYNTHETIC_PRIVATE_TEXT" for m in fake_invoke.call_args.kwargs["messages"])}

    bad_tool = SimpleNamespace(id=uuid.uuid4(), user_id="alice", version=1, name="audit_failure", description="Synthetic failure", type="code", config={"code": "def execute():\n    return 1 / 0"}, input_schema={"type": "object"})
    with patch.object(ToolService, "get_tool", AsyncMock(return_value=bad_tool)):
        tool_result = await ToolService.execute_tool(session, "alice", bad_tool.id, {})
    evidence["failed_tool_reported_success"] = {"success": tool_result.success, "error": tool_result.error, "output": tool_result.output}

    model = json_schema_to_pydantic_model({"type": "object", "properties": {"n": {"type": "integer", "minimum": 10, "enum": [10, 20]}}}, "AuditSchema")
    evidence["tool_schema_constraints_ignored"] = model.model_validate({"n": -2}).model_dump()
    try:
        json_schema_to_pydantic_model({"type": "object", "properties": {"n": {"type": ["string", "null"]}}}, "AuditNullableSchema")
    except Exception as exc:
        evidence["valid_nullable_schema_compile_failure"] = type(exc).__name__ + ": " + str(exc)

    code = CodeToolExecutor(CodeToolConfig(code="import os\ndef execute():\n    return os.name", allowed_modules=["os"]))
    evidence["caller_controlled_module_allowlist"] = {"os_import_accepted_and_executed": await code.execute({})}
    done = threading.Event()
    timed_code = CodeToolExecutor(CodeToolConfig(code="import time\ndef execute(done):\n    time.sleep(0.8)\n    done.set()", allowed_modules=["time"], timeout_seconds=0.5))
    try:
        await timed_code.execute({"done": done})
    except CodeExecutionError:
        evidence["code_timeout"] = {"finished_at_timeout": done.is_set()}
    await asyncio.sleep(0.4)
    evidence["code_timeout"]["finished_after_timeout"] = done.is_set()

    sql_examples = ["WITH moved AS (DELETE\nFROM audit_table RETURNING *) SELECT * FROM moved", "SELECT pg_sleep(30)"]
    evidence["readonly_sql_lexical_checks"] = []
    for sql in sql_examples:
        validate_readonly_sql(sql)
        evidence["readonly_sql_lexical_checks"].append({"query": sql, "accepted_by_validator_only_not_executed": True})

    ToolRegistry.clear_cache()
    return evidence


def run_existing_tests():
    import pytest
    from app.core import database
    async def offline_init():
        pass
    database.init_db = offline_init
    # Tests selected after source inspection; no service-backed API tests.
    modules = ["tests/test_workflow_compiler.py", "tests/test_retriever.py", "tests/test_llm.py", "tests/test_rag_tools.py"]
    tool_names = ["secret_encryption_and_decryption", "secret_masking", "corrupted_token_decryption_error", "http_tool_executor_get_with_params", "http_tool_executor_post_with_bearer_auth", "code_tool_executor_valid_execution", "code_tool_executor_blocks_forbidden_import", "code_tool_executor_blocks_eval_and_exec", "database_tool_executor_blocks_forbidden_sql", "database_tool_executor_parameterized_query", "json_schema_to_pydantic_model", "tool_registry_compiles_code_tool_with_cache", "tool_registry_resolves_built_ins_and_rag"]
    nodes = ["tests/test_tools_api.py::test_" + n for n in tool_names]
    nodes += ["tests/test_workflow_nodes.py::test_" + n for n in ["code_node_execution", "code_node_sandbox_security", "chat_node_execution", "retrieve_node_execution", "tool_call_calculator"]]
    return pytest.main([*modules, *nodes, "-q", "-p", "no:cacheprovider", "--junitxml=audit/selected-tests.xml"])


if __name__ == "__main__":
    evidence = asyncio.run(run_probes())
    (ROOT / "audit" / "core-evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    if "--tests" in sys.argv:
        raise SystemExit(run_existing_tests())
