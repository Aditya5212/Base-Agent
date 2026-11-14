from unittest.mock import AsyncMock, patch
import pytest
from langchain_core.documents import Document
from app.core.rag_config import get_rag_config
from app.services.retrieval_service import RetrievalScope

from app.services.workflow.nodes.chat import build_chat_node
from app.services.workflow.nodes.code import build_code_node
from app.services.workflow.nodes.retrieve import build_retrieve_node
from app.services.workflow.nodes.tool_call import build_tool_call_node


@pytest.mark.asyncio
async def test_code_node_execution():
    code_fn = build_code_node(
        node_id="math_1",
        config_dict={
            "code": "result = a + b * 2",
            "input_variables": ["a", "b"],
            "output_variable": "calc_out",
        },
    )

    state = {
        "variables": {"a": 10, "b": 5},
        "node_history": [],
    }
    res = await code_fn(state)
    assert res["variables"]["calc_out"] == 20
    assert "math_1" in res["node_history"]


def test_code_node_sandbox_security():
    # Attempting to import forbidden module should fail AST inspection
    with pytest.raises(PermissionError, match="not permitted"):
        build_code_node(
            node_id="bad_import",
            config_dict={"code": "import os\nos.system('dir')"},
        )

    # Attempting forbidden call
    with pytest.raises(PermissionError, match="prohibited"):
        build_code_node(
            node_id="bad_call",
            config_dict={"code": "eval('1+1')"},
        )


@pytest.mark.asyncio
async def test_chat_node_execution():
    chat_fn = build_chat_node(
        node_id="chat_1",
        config_dict={
            "system_prompt": "You are assisting {{name}}.",
            "user_prompt_template": "Query: {{query}}",
            "output_variable": "answer",
            "enable_memory": True,
        },
    )

    state = {
        "variables": {"name": "Bob", "query": "What is AI?"},
        "messages": [],
        "node_history": [],
    }

    res = await chat_fn(state)
    assert "answer" in res["variables"]
    assert len(res["variables"]["answer"]) > 0
    assert len(res["messages"]) == 2  # HumanMessage + AIMessage


@pytest.mark.asyncio
async def test_retrieve_node_execution():
    retrieve_fn = build_retrieve_node(
        node_id="ret_1",
        config_dict={
            "query_template": "{{user_query}}",
            "top_k": 2,
            "output_variable": "rag_context",
            "sources_variable": "rag_sources",
        },
    )

    mock_docs = [
        Document(
            page_content="Antigravity architecture guide.",
            metadata={"source": "guide.md", "header": "Overview"},
        ),
        Document(
            page_content="Vector indexing with PGVector.",
            metadata={"source": "vector.md", "header": "HNSW"},
        ),
    ]

    with patch("app.services.workflow.nodes.retrieve.hybrid_search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_docs

        state = {
            "variables": {"user_query": "Explain architecture"},
            "user_id": "test_user",
            "node_history": [],
        }

        res = await retrieve_fn(state)
        mock_search.assert_awaited_once_with(
            query="Explain architecture",
            filter_dict={"user_id": "test_user"},
            thread_id=None,
            scope=RetrievalScope.GLOBAL_ONLY,
            config=get_rag_config(top_k_final=2, top_k_dense=20, top_k_sparse=20),
        )
        assert "Antigravity architecture guide" in res["variables"]["rag_context"]
        assert len(res["variables"]["rag_sources"]) == 2
        assert res["variables"]["rag_sources"][0]["source"] == "guide.md"


@pytest.mark.asyncio
async def test_retrieve_node_thread_scope():
    retrieve_fn = build_retrieve_node(
        node_id="ret_thread",
        config_dict={
            "query_template": "{{user_query}}",
            "top_k": 3,
            "workspace_type": "thread",
            "output_variable": "rag_context",
            "sources_variable": "rag_sources",
        },
    )

    mock_docs = [
        Document(
            page_content="Thread-specific project requirements.",
            metadata={"source": "spec.md", "is_thread_document": True},
        ),
    ]

    with patch("app.services.workflow.nodes.retrieve.hybrid_search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_docs

        state = {
            "variables": {"user_query": "What are the specs?"},
            "user_id": "user_alpha",
            "thread_id": "conv-thread-99",
            "node_history": [],
        }

        res = await retrieve_fn(state)
        mock_search.assert_awaited_once_with(
            query="What are the specs?",
            filter_dict={"user_id": "user_alpha"},
            thread_id="conv-thread-99",
            scope=RetrievalScope.THREAD_WITH_GLOBAL,
            config=get_rag_config(top_k_final=3, top_k_dense=20, top_k_sparse=20),
        )
        assert "Thread-specific project requirements" in res["variables"]["rag_context"]
        assert "[Thread Doc]" in res["variables"]["rag_context"]
        assert res["variables"]["rag_sources"][0]["is_thread_document"] is True


@pytest.mark.asyncio
async def test_retrieve_node_explicit_scope():
    retrieve_fn = build_retrieve_node(
        node_id="ret_explicit",
        config_dict={
            "query_template": "{{user_query}}",
            "top_k": 5,
            "scope": "thread_only",
            "output_variable": "rag_context",
            "sources_variable": "rag_sources",
        },
    )

    with patch("app.services.workflow.nodes.retrieve.hybrid_search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []

        state = {
            "variables": {"user_query": "strictly thread"},
            "user_id": "user_beta",
            "thread_id": "thread-abc",
            "node_history": [],
        }

        await retrieve_fn(state)
        mock_search.assert_awaited_once_with(
            query="strictly thread",
            filter_dict={"user_id": "user_beta"},
            thread_id="thread-abc",
            scope=RetrievalScope.THREAD_ONLY,
            config=get_rag_config(top_k_final=5, top_k_dense=20, top_k_sparse=20),
        )


@pytest.mark.asyncio
async def test_tool_call_calculator():
    tool_fn = build_tool_call_node(
        node_id="calc_tool",
        config_dict={
            "tool_name": "calculator",
            "input_mapping": {"expression": "{{var_a}} * 10 + 5"},
            "output_variable": "calc_result",
        },
    )

    state = {
        "variables": {"var_a": 3},
        "node_history": [],
    }
    res = await tool_fn(state)
    assert res["variables"]["calc_result"] == 35

