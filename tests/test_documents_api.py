"""Integration tests for Documents & RAG API endpoints."""

from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app

client = TestClient(app)


def test_search_documents_endpoint():
    mock_results = [
        Document(
            page_content="LangGraph supports cyclical agent graphs.",
            metadata={
                "source": "langgraph.md",
                "chunk_index": 0,
                "rrf_score": 0.016393,
                "dense_rank": 1,
                "sparse_rank": 1,
            },
        )
    ]

    with patch("app.api.v1.endpoints.documents.hybrid_search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_results

        payload = {"query": "What is LangGraph?", "preset": "balanced", "user_id": "test_user"}
        response = client.post("/api/v1/documents/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is LangGraph?"
        assert data["total_results"] == 1
        assert "cyclical agent graphs" in data["results"][0]["content"]
        assert data["results"][0]["source"] == "langgraph.md"
        assert data["results"][0]["rrf_score"] == 0.016393


def test_upload_empty_file_rejected():
    files = {"file": ("empty.txt", b"", "text/plain")}
    response = client.post("/api/v1/documents/upload", files=files, data={"user_id": "u1"})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_get_knowledge_base_stats():
    response = client.get("/api/v1/documents/stats?user_id=test_stats_user")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "total_chunks" in data
    assert "formatted_storage" in data
    assert data["active_preset_default"] == "balanced"


def test_get_document_invalid_uuid():
    response = client.get("/api/v1/documents/not-a-valid-uuid")
    assert response.status_code == 400
    assert "invalid uuid" in response.json()["detail"].lower()


def test_get_document_not_found():
    response = client.get("/api/v1/documents/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_document_chunks_not_found():
    response = client.get("/api/v1/documents/00000000-0000-0000-0000-000000000000/chunks")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_reindex_document_not_found():
    response = client.post(
        "/api/v1/documents/00000000-0000-0000-0000-000000000000/reindex",
        json={"preset": "high_precision"},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

