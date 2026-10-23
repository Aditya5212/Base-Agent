from unittest.mock import patch
import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.core.llm import MODEL_DEEPSEEK_FLASH, MODEL_DEEPSEEK_PRO
from app.services.agent import execute_agent, get_agent_graph, stream_agent_events



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


@pytest.mark.asyncio
async def test_stream_agent_events_single_turn():
    saver = InMemorySaver()
    thread_id = "stream-thread-1"

    with patch("app.services.agent.get_llm", return_value=None):
        events = []
        async for event in stream_agent_events(
            message="Hello streaming",
            thread_id=thread_id,
            user_id="stream_user",
            checkpointer=saver,
        ):
            events.append(event)

        assert len(events) >= 2
        token_events = [e for e in events if e.event == "token"]
        end_events = [e for e in events if e.event == "end"]

        assert len(token_events) >= 1
        assert len(end_events) == 1
        assert end_events[0].data["thread_id"] == thread_id
        assert end_events[0].data["history_length"] == 2
        assert "Hello streaming" in end_events[0].data["response"]


@pytest.mark.asyncio
async def test_stream_agent_events_multi_turn_memory():
    saver = InMemorySaver()
    thread_id = "stream-mem-thread"

    with patch("app.services.agent.get_llm", return_value=None):
        # Turn 1
        turn1_events = []
        async for event in stream_agent_events(
            message="I am Alice",
            thread_id=thread_id,
            user_id="user_alice",
            checkpointer=saver,
        ):
            turn1_events.append(event)

        turn1_end = [e for e in turn1_events if e.event == "end"][0]
        assert turn1_end.data["history_length"] == 2

        # Turn 2: same thread_id
        turn2_events = []
        async for event in stream_agent_events(
            message="Who am I?",
            thread_id=thread_id,
            user_id="user_alice",
            checkpointer=saver,
        ):
            turn2_events.append(event)

        turn2_end = [e for e in turn2_events if e.event == "end"][0]
        assert turn2_end.data["history_length"] == 4


@pytest.mark.asyncio
async def test_stream_agent_events_with_fake_llm():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    fake_model = FakeListChatModel(responses=["Streaming works great!"])
    saver = InMemorySaver()

    with patch("app.services.agent.get_llm", return_value=fake_model):
        tokens = []
        end_event = None
        async for event in stream_agent_events(
            message="Say hi",
            thread_id="fake-llm-thread",
            checkpointer=saver,
        ):
            if event.event == "token":
                tokens.append(event.data["delta"])
            elif event.event == "end":
                end_event = event

        assert len(tokens) > 0
        full_text = "".join(tokens)
        assert full_text == "Streaming works great!"
        assert end_event is not None
        assert end_event.data["history_length"] == 2

