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


def test_chat_endpoint_with_images():
    mock_messages = [
        HumanMessage(content="Describe this image"),
        AIMessage(content="This is a diagram of the LangGraph architecture."),
    ]
    with patch("app.api.v1.endpoints.chat.execute_agent", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = (
            {
                "messages": mock_messages,
                "thread_id": "test-img-thread",
                "user_id": "root",
                "thread_assets": [{"file_name": "diag.png", "presigned_url": "https://r2/diag.png"}],
            },
            "deepseek-flash",
        )

        payload = {
            "message": "Describe this image",
            "thread_id": "test-img-thread",
            "user_id": "root",
            "images": ["https://r2/diag.png"],
        }
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"] == "test-img-thread"
        assert "diagram of the LangGraph" in data["response"]
        assert len(data["thread_assets"]) == 1
        assert data["thread_assets"][0]["file_name"] == "diag.png"


def test_upload_and_list_thread_assets():
    with patch("app.api.v1.endpoints.chat.r2_storage_service.upload_file", new_callable=AsyncMock) as mock_upload, \
         patch("app.api.v1.endpoints.chat.r2_storage_service.generate_presigned_url", new_callable=AsyncMock) as mock_url:
        mock_upload.return_value = "threads/test-asset-t1/mock_image.png"
        mock_url.return_value = "https://r2.storage/signed/mock_image.png"

        files = {"file": ("mock_image.png", b"\x89PNG\r\n\x1a\nfakeimage", "image/png")}
        response = client.post(
            "/api/v1/chat/test-asset-t1/assets",
            files=files,
            data={"user_id": "root"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["thread_id"] == "test-asset-t1"
        assert data["file_name"] == "mock_image.png"
        assert data["asset_type"] == "image"
        assert data["presigned_url"] == "https://r2.storage/signed/mock_image.png"

        # List assets
        list_res = client.get("/api/v1/chat/test-asset-t1/assets")
        assert list_res.status_code == 200
        assets = list_res.json()
        assert len(assets) >= 1
        assert assets[0]["file_name"] == "mock_image.png"


import pytest
from app.services.tools import hybrid_document_search


@pytest.mark.asyncio
async def test_hybrid_document_search_tool():
    with patch("app.services.tools.hybrid_search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []
        result = await hybrid_document_search.ainvoke({"query": "What is LangGraph?"})
        assert "No relevant documents found" in result


def test_chat_endpoint_value_error_returns_400():
    """ValueError from invalid image keys/traversal returns HTTP 400 Bad Request."""
    with patch("app.api.v1.endpoints.chat.execute_agent", new_callable=AsyncMock) as mock_exec:
        mock_exec.side_effect = ValueError("Invalid image key: path traversal sequences are forbidden in '../../etc/passwd'")

        payload = {
            "message": "Analyze this file",
            "thread_id": "test-sec-thread",
            "user_id": "user123",
            "images": ["../../etc/passwd"],
        }
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 400
        assert "path traversal" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_build_human_message_path_traversal_guard():
    """_build_human_message raises ValueError on path traversal sequences."""
    from app.services.agent import _build_human_message

    with pytest.raises(ValueError, match="path traversal"):
        await _build_human_message(
            message="Test traversal",
            images=["../../etc/shadow"],
            user_id="user123",
            thread_id="t1",
        )

    with pytest.raises(ValueError, match="path traversal"):
        await _build_human_message(
            message="Test leading slash",
            images=["/root/private.key"],
            user_id="user123",
            thread_id="t1",
        )


@pytest.mark.asyncio
async def test_build_human_message_unauthorized_key_guard():
    """_build_human_message raises ValueError when accessing another user's key."""
    from app.services.agent import _build_human_message

    with pytest.raises(ValueError, match="Access denied"):
        await _build_human_message(
            message="Steal user image",
            images=["users/victim_user/doc.png"],
            user_id="attacker_user",
            thread_id="t1",
        )


@pytest.mark.asyncio
async def test_build_human_message_authorized_keys():
    """_build_human_message accepts keys owned by user or thread."""
    from app.services.agent import _build_human_message

    with patch("app.services.agent.r2_storage_service.generate_presigned_url", new_callable=AsyncMock) as mock_url:
        mock_url.return_value = "https://r2/signed/pic.png"

        # 1. User-scoped key
        msg = await _build_human_message(
            message="Describe image",
            images=["users/user123/pic.png"],
            user_id="user123",
            thread_id="t1",
        )
        assert isinstance(msg.content, list)
        assert msg.content[0]["text"] == "Describe image"
        assert msg.content[1]["image_url"]["url"] == "https://r2/signed/pic.png"

        # 2. Thread-scoped key
        msg_thread = await _build_human_message(
            message="Describe thread image",
            images=["threads/t1/pic.png"],
            user_id="user123",
            thread_id="t1",
        )
        assert isinstance(msg_thread.content, list)

        # 3. Root admin user can access any valid key
        msg_root = await _build_human_message(
            message="Admin inspect",
            images=["users/someone_else/pic.png"],
            user_id="root",
            thread_id="t1",
        )
        assert isinstance(msg_root.content, list)



