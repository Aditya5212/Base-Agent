"""Unit and integration tests for the modular RAG research tool suite.

Validates:
- Exploitation tools: hybrid_document_search, graph_rag_search
- Exploration tool: multi_query_search (agent-provided queries list)
- Cross-query RRF fusion and multi-query coverage tracking
- 3-tier integration: Chatbot level, Agent node level, and Workflow engine level
- Tenant scoping and security isolation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from langchain_core.documents import Document

from app.core.rag.protocols import RetrievalQuery, RetrievalResult, RetrievedChunk
from app.schemas.agent import AgentConfig, ToolSpec
from app.services.agent_service import resolve_agent_tools
from app.services.multi_query import (
    cross_query_reciprocal_rank_fusion,
    multi_query_hybrid_search,
)
from app.services.rag_tools import (
    get_user_rag_tool_map,
    get_user_rag_tools,
    make_graph_rag_tool,
    make_hybrid_search_tool,
    make_multi_query_search_tool,
)
from app.services.workflow.nodes.retrieve import build_retrieve_node
from app.services.workflow.nodes.tool_call import build_tool_call_node


# ---------------------------------------------------------------------------
# Test 1: Cross-Query Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def test_cross_query_reciprocal_rank_fusion():
    """Verifies cross-query RRF scores documents and awards multi-hit bonuses."""
    doc_a = Document(
        page_content="PostgreSQL connection pooling with psycopg3",
        metadata={"id": "chunk_1", "filename": "db.py"},
    )
    doc_b = Document(
        page_content="Asyncpg vs Psycopg3 performance benchmark",
        metadata={"id": "chunk_2", "filename": "benchmarks.md"},
    )
    doc_c = Document(
        page_content="Docker compose networking and DNS setup",
        metadata={"id": "chunk_3", "filename": "docker.md"},
    )

    # doc_a matches both query 1 and query 2
    # doc_b matches only query 1
    # doc_c matches only query 2
    query_results = [
        ("query 1", [doc_a, doc_b]),
        ("query 2", [doc_a, doc_c]),
    ]

    fused = cross_query_reciprocal_rank_fusion(query_results, top_k=5)
    assert len(fused) == 3

    # doc_a matched in both queries, so it should rank #1 due to double RRF + multi-hit bonus
    top_doc = fused[0]
    assert top_doc.metadata["id"] == "chunk_1"
    assert top_doc.metadata["matched_query_count"] == 2
    assert "query 1" in top_doc.metadata["matched_queries"]
    assert "query 2" in top_doc.metadata["matched_queries"]
    assert top_doc.metadata["cross_rrf_score"] > 0


# ---------------------------------------------------------------------------
# Test 2: Multi-Query Hybrid Search Helper
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multi_query_hybrid_search_execution():
    """Verifies multi_query_hybrid_search executes hybrid_search for all agent-provided queries in parallel."""
    doc1 = Document(page_content="Async engine pool size", metadata={"id": "c1", "source": "db.py"})
    doc2 = Document(page_content="FastAPI lifespan teardown", metadata={"id": "c2", "source": "main.py"})

    with patch("app.services.multi_query.hybrid_search", new_callable=AsyncMock) as mock_hs:
        mock_hs.side_effect = [[doc1], [doc2]]

        results = await multi_query_hybrid_search(
            queries=["connection pool", "lifespan shutdown"],
            user_id="alice",
            preset="balanced",
            top_k_final=5,
        )

        assert mock_hs.await_count == 2
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Test 3: Exploitation Tool — Hybrid Document Search
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_make_hybrid_search_tool():
    """Verifies hybrid_document_search tool metadata, tenant isolation, and formatted output."""
    tool = make_hybrid_search_tool(user_id="alice")
    assert tool.name == "hybrid_document_search"

    mock_doc = Document(
        page_content="The maximum connection limit is 100.",
        metadata={"filename": "config.md", "header_1": "Database", "header_2": "Limits"},
    )

    with patch("app.services.rag_tools.hybrid_search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [mock_doc]

        result = await tool.ainvoke({"query": "connection limit", "preset": "high_precision", "top_k": 3})

        # Ensure user_id was passed down into filter_dict
        mock_search.assert_awaited_once()
        _, kwargs = mock_search.call_args
        assert kwargs["filter_dict"] == {"user_id": "alice"}
        assert kwargs["query"] == "connection limit"

        # Check output formatting
        assert "Source 1: config.md | Section: Database > Limits" in result
        assert "The maximum connection limit is 100." in result


@pytest.mark.asyncio
async def test_make_hybrid_search_tool_empty():
    """Verifies hybrid_document_search tool returns a friendly notice when no documents match."""
    tool = make_hybrid_search_tool(user_id="bob")
    with patch("app.services.rag_tools.hybrid_search", new_callable=AsyncMock, return_value=[]):
        result = await tool.ainvoke({"query": "non-existent topic"})
        assert "No relevant documents found in knowledge base" in result


# ---------------------------------------------------------------------------
# Test 4: Exploitation Tool — Knowledge Graph Search (LightRAG)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_make_graph_rag_tool():
    """Verifies graph_rag_search tool queries LightRAG provider and formats entity context."""
    tool = make_graph_rag_tool(user_id="alice")
    assert tool.name == "graph_rag_search"

    mock_provider = MagicMock()
    mock_provider.query = AsyncMock(
        return_value=RetrievalResult(
            provider="lightrag",
            query="Workflow Engine",
            context="Entity: WorkflowEngine -> DependsOn -> AsyncPostgresSaver",
            chunks=[
                RetrievedChunk(
                    chunk_id="chunk_g1",
                    content="Entity: WorkflowEngine -> DependsOn -> AsyncPostgresSaver",
                    source="lightrag",
                )
            ],
        )
    )

    with patch("app.services.rag_tools.get_provider", return_value=mock_provider) as mock_get_provider:
        result = await tool.ainvoke({"query": "Workflow Engine", "mode": "mix", "top_k": 5})

        mock_get_provider.assert_awaited_once_with("lightrag", workspace="u_alice")
        mock_provider.query.assert_awaited_once()
        retrieval_query: RetrievalQuery = mock_provider.query.call_args[0][0]
        assert retrieval_query.query == "Workflow Engine"
        assert retrieval_query.mode == "mix"

        assert "### Knowledge Graph Traversal [Mode: MIX]" in result
        assert "WorkflowEngine -> DependsOn -> AsyncPostgresSaver" in result


# ---------------------------------------------------------------------------
# Test 5: Exploration Tool — Multi-Query Search (Agent-Provided Queries)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_make_multi_query_search_tool():
    """Verifies multi_query_search tool accepts an array of queries directly from agent and fuses results."""
    tool = make_multi_query_search_tool(user_id="charlie")
    assert tool.name == "multi_query_search"

    mock_doc = Document(
        page_content="LightRAG PGTableGraphStorage runs on stock PostgreSQL tables.",
        metadata={"filename": "architecture.md", "matched_queries": ["LightRAG storage", "PostgreSQL Neon"]},
    )

    with patch(
        "app.services.rag_tools.multi_query_hybrid_search",
        new_callable=AsyncMock,
    ) as mock_mq_search:
        mock_mq_search.return_value = [mock_doc]

        # Agent provides a list of queries to explore different perspectives
        agent_queries = [
            "LightRAG storage backend architecture",
            "PostgreSQL Neon graph tables",
        ]
        result = await tool.ainvoke({"queries": agent_queries})

        mock_mq_search.assert_awaited_once_with(
            queries=agent_queries,
            user_id="charlie",
            preset="balanced",
            top_k_final=8,
            thread_id=None,
        )
        assert "Multi-Query Exploration Results (2 queries searched in parallel):" in result
        assert "LightRAG storage backend architecture" in result
        assert "PostgreSQL Neon graph tables" in result
        assert "Source 1: architecture.md" in result
        assert "Matched Queries: LightRAG storage, PostgreSQL Neon" in result


# ---------------------------------------------------------------------------
# Test 6: Agent Node Level — resolve_agent_tools
# ---------------------------------------------------------------------------

def test_resolve_agent_tools():
    """Verifies resolve_agent_tools resolves individual tools and macro 'all_rag_tools'."""
    # Test specific tool selection
    cfg = AgentConfig(
        tools=[
            ToolSpec(tool_id="hybrid_document_search", enabled=True),
            ToolSpec(tool_id="multi_query_search", enabled=True),
            ToolSpec(tool_id="graph_rag_search", enabled=False),  # disabled
        ]
    )
    tools = resolve_agent_tools(cfg, user_id="agent_owner")
    names = [t.name for t in tools]
    assert "hybrid_document_search" in names
    assert "multi_query_search" in names
    assert "graph_rag_search" not in names

    # Test macro 'all_rag_tools'
    cfg_macro = AgentConfig(
        tools=[ToolSpec(tool_id="all_rag_tools", enabled=True)]
    )
    macro_tools = resolve_agent_tools(cfg_macro, user_id="agent_owner")
    macro_names = [t.name for t in macro_tools]
    assert len(macro_tools) == 3
    assert set(macro_names) == {
        "hybrid_document_search",
        "graph_rag_search",
        "multi_query_search",
    }


# ---------------------------------------------------------------------------
# Test 7: Workflow Engine Level — ToolCallNode with Multi-Query Search
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_workflow_tool_call_node_with_multi_query_search():
    """Verifies that Workflow ToolCallNode executes multi_query_search with agent queries."""
    config = {
        "tool_name": "multi_query_search",
        "input_mapping": {"queries": '["connection pool", "database timeout"]'},
        "output_variable": "mq_result",
    }
    handler = build_tool_call_node("node_tool_mq", config)

    state = {
        "user_id": "workflow_user",
        "variables": {},
        "node_history": [],
    }

    mock_doc = Document(page_content="Pool timeout configuration", metadata={"filename": "pool.py"})

    with patch("app.services.rag_tools.multi_query_hybrid_search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [mock_doc]
        new_state = await handler(state)

        assert "mq_result" in new_state["variables"]
        assert "Pool timeout configuration" in new_state["variables"]["mq_result"]
        assert "node_tool_mq" in new_state["node_history"]


# ---------------------------------------------------------------------------
# Test 8: Workflow Engine Level — RetrieveNode Providers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_workflow_retrieve_node_multi_query_provider():
    """Verifies RetrieveNode executes 'multi_query' provider and populates sources."""
    config = {
        "provider_name": "multi_query",
        "query_template": "{{variables.search_queries}}",
        "top_k": 3,
        "output_variable": "rag_ctx",
        "sources_variable": "rag_srcs",
    }
    handler = build_retrieve_node("node_retrieve_mq", config)

    state = {
        "user_id": "test_user",
        "variables": {"search_queries": "2PC protocol\ndistributed consensus"},
        "node_history": [],
    }

    mock_doc = Document(
        page_content="2PC protocol details.",
        metadata={"filename": "transactions.md", "matched_queries": ["2PC protocol"]},
    )

    with patch("app.services.workflow.nodes.retrieve.multi_query_hybrid_search", new_callable=AsyncMock) as mock_mq:
        mock_mq.return_value = [mock_doc]

        new_state = await handler(state)

        assert "2PC protocol details." in new_state["variables"]["rag_ctx"]
        assert len(new_state["variables"]["rag_srcs"]) == 1
        assert new_state["variables"]["rag_srcs"][0]["source"] == "transactions.md"
