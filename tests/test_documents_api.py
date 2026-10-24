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
