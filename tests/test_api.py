from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Base Agent"
    assert data["status"] == "online"
    assert data["docs_url"] == "/docs"


def test_ping_endpoint():
    response = client.get("/api/v1/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}


def test_health_check_healthy():
    with patch("app.api.v1.endpoints.health.check_db_connection", new_callable=AsyncMock) as mock_db:
        mock_db.return_value = (True, None)
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"]["connected"] is True


def test_health_check_degraded():
    with patch("app.api.v1.endpoints.health.check_db_connection", new_callable=AsyncMock) as mock_db:
        mock_db.return_value = (False, "Connection refused")
        response = client.get("/api/v1/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["database"]["connected"] is False
        assert data["database"]["details"] == "Connection refused"


def test_chat_endpoint_success():
    mock_messages = [
        HumanMessage(content="Hello agent"),
        AIMessage(content="Hello human! How can I assist you?"),
    ]
    with patch("app.api.v1.endpoints.chat.execute_agent", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = (
            {
                "messages": mock_messages,
                "thread_id": "test-t-1",
                "user_id": "user-1",
            },
            "deepseek-flash",
        )

        payload = {"message": "Hello agent", "thread_id": "test-t-1", "user_id": "user-1"}
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"] == "test-t-1"
        assert data["response"] == "Hello human! How can I assist you?"
        assert data["model"] == "deepseek-flash"
        assert data["history_length"] == 2


def test_chat_history_endpoint():
    mock_messages = [
        HumanMessage(content="Message 1"),
        AIMessage(content="Response 1"),
    ]
    with patch("app.api.v1.endpoints.chat.get_thread_messages", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_messages
        response = client.get("/api/v1/chat/test-t-1/history")
        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"] == "test-t-1"
        assert data["total_messages"] == 2
        assert data["messages"][0]["role"] == "human"
        assert data["messages"][1]["role"] == "ai"


def test_clear_memory_endpoint():
    with patch("app.api.v1.endpoints.chat.delete_thread_memory", new_callable=AsyncMock) as mock_del:
        mock_del.return_value = True
        response = client.delete("/api/v1/chat/test-t-1")
        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True
        assert data["thread_id"] == "test-t-1"


def test_chat_stream_endpoint_success():
    from app.schemas.chat import StreamEvent

    async def mock_stream_events(**kwargs):
        yield StreamEvent.token(delta="Hello")
        yield StreamEvent.token(delta=" world!")
        yield StreamEvent.end(
            thread_id="test-stream-t1",
            user_id="user-1",
            response="Hello world!",
            model="deepseek-flash",
            history_length=2,
        )

    with patch("app.api.v1.endpoints.chat.stream_agent_events", side_effect=mock_stream_events):
        payload = {"message": "Hello", "thread_id": "test-stream-t1", "user_id": "user-1"}
        response = client.post("/api/v1/chat/stream", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = response.text
        assert "event: token" in body
        assert "Hello" in body
        assert " world!" in body
        assert "event: end" in body
        assert "test-stream-t1" in body


def test_chat_stream_endpoint_error_handling():
    async def mock_stream_error(**kwargs):
        raise RuntimeError("LLM connection timeout")
        yield  # Make it an async generator

    with patch("app.api.v1.endpoints.chat.stream_agent_events", side_effect=mock_stream_error):
        payload = {"message": "Hello", "thread_id": "test-stream-err", "user_id": "user-1"}
        response = client.post("/api/v1/chat/stream", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = response.text
        assert "event: error" in body
        assert "LLM connection timeout" in body

