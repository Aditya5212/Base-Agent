from unittest.mock import patch
import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.core.llm import MODEL_DEEPSEEK_FLASH, MODEL_DEEPSEEK_PRO
from app.services.agent import execute_agent, get_agent_graph


@pytest.mark.asyncio
async def test_agent_graph_compilation():
    saver = InMemorySaver()
    graph = get_agent_graph(checkpointer=saver)
    assert graph is not None


@pytest.mark.asyncio
async def test_agent_short_term_memory_persistence():
    saver = InMemorySaver()
    thread_id = "test-mem-thread-1"

    with patch("app.services.agent.get_llm", return_value=None):
        # Turn 1
        res1, model1 = await execute_agent(
            message="Hi! My name is Bob.",
            thread_id=thread_id,
            user_id="user_test",
            checkpointer=saver,
        )
        assert model1 == MODEL_DEEPSEEK_FLASH
        messages1 = res1["messages"]
        assert len(messages1) == 2  # Human + AI
        assert messages1[0].content == "Hi! My name is Bob."

        # Turn 2: Same thread should remember Turn 1
        res2, model2 = await execute_agent(
            message="What is my name?",
            thread_id=thread_id,
            user_id="user_test",
            checkpointer=saver,
        )
        assert model2 == MODEL_DEEPSEEK_FLASH
        messages2 = res2["messages"]
        assert len(messages2) == 4  # Human + AI + Human + AI
        assert messages2[0].content == "Hi! My name is Bob."
        assert messages2[2].content == "What is my name?"


@pytest.mark.asyncio
async def test_agent_model_override():
    saver = InMemorySaver()
    with patch("app.services.agent.get_llm", return_value=None):
        res, model = await execute_agent(
            message="Test pro model",
            thread_id="test-pro-thread",
            model=MODEL_DEEPSEEK_PRO,
            checkpointer=saver,
        )
        assert model == MODEL_DEEPSEEK_PRO
        assert len(res["messages"]) == 2
        assert "deepseek-v4-pro" in res["messages"][1].content


@pytest.mark.asyncio
async def test_agent_thread_isolation():
    saver = InMemorySaver()

    with patch("app.services.agent.get_llm", return_value=None):
        # Thread A
        res_a, _ = await execute_agent(
            message="Message for thread A",
            thread_id="thread-a",
            checkpointer=saver,
        )
        assert len(res_a["messages"]) == 2

        # Thread B should start fresh
        res_b, _ = await execute_agent(
            message="Message for thread B",
            thread_id="thread-b",
            checkpointer=saver,
        )
        assert len(res_b["messages"]) == 2
        assert res_b["messages"][0].content == "Message for thread B"
