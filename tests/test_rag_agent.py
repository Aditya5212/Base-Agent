"""Unit tests for RAG-augmented conversational agent workflow."""

from unittest.mock import AsyncMock, patch
import pytest
from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver

from app.core.rag_config import RAGPreset
from app.services.agent import execute_agent, stream_agent_events


@pytest.mark.asyncio
async def test_agent_with_rag_enabled_injects_sources():
    saver = InMemorySaver()
    mock_retrieved_docs = [
        Document(
            page_content="Nike was incorporated in 1967 in Oregon.",
            metadata={"source": "nike_10k.md", "header_1": "Business", "user_id": "test_user"},
        ),
        Document(
            page_content="Nike has eight significant distribution centers in the US.",
            metadata={"source": "nike_10k.md", "header_1": "Properties", "user_id": "test_user"},
        ),
    ]

    with patch("app.services.agent.hybrid_search", new_callable=AsyncMock) as mock_search, \
         patch("app.services.agent.get_llm", return_value=None):
        mock_search.return_value = mock_retrieved_docs

        res, model = await execute_agent(
            message="When was Nike incorporated?",
            thread_id="test-rag-thread-1",
            user_id="test_user",
            use_rag=True,
            rag_preset=RAGPreset.HIGH_PRECISION,
            checkpointer=saver,
        )

        # Verify hybrid_search was called with query and user filter
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["query"] == "When was Nike incorporated?"
        assert call_kwargs["filter_dict"] == {"user_id": "test_user"}

        # Verify RAG context and sources in state
        assert res.get("rag_context") is not None
        assert "Nike was incorporated in 1967" in res["rag_context"]
        sources = res.get("rag_sources", [])
        assert len(sources) == 2
        assert sources[0]["filename"] == "nike_10k.md"
        assert sources[0]["section"] == "Business"


@pytest.mark.asyncio
async def test_agent_streaming_with_rag_emits_sources():
    saver = InMemorySaver()
    mock_retrieved_docs = [
        Document(
            page_content="LangGraph supports persistent checkpoints with AsyncPostgresSaver.",
            metadata={"source": "langgraph.md", "header_1": "Persistence", "user_id": "u1"},
        ),
    ]

    with patch("app.services.agent.hybrid_search", new_callable=AsyncMock) as mock_search, \
         patch("app.services.agent.get_llm", return_value=None):
        mock_search.return_value = mock_retrieved_docs

        events = []
        async for event in stream_agent_events(
            message="How does persistence work in LangGraph?",
            thread_id="test-stream-rag-1",
            user_id="u1",
            use_rag=True,
            checkpointer=saver,
        ):
            events.append(event)

        end_events = [e for e in events if e.event == "end"]
        assert len(end_events) == 1
        end_event = end_events[0]
        assert "sources" in end_event.data
        sources = end_event.data["sources"]
        assert len(sources) == 1
        assert sources[0]["filename"] == "langgraph.md"
