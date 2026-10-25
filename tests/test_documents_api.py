import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app
from app.services.document_processor import process_document_pipeline

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


def test_upload_document_background_dispatch():
    with patch("app.api.v1.endpoints.documents.r2_storage_service.upload_file", new_callable=AsyncMock) as mock_r2, \
         patch("app.api.v1.endpoints.documents.process_document_pipeline", new_callable=AsyncMock) as mock_pipeline:
        mock_r2.return_value = "documents/test_user/test_guide.md"

        files = {"file": ("test_guide.md", b"# LangGraph & RAG Guide\nSample documentation.", "text/markdown")}
        response = client.post(
            "/api/v1/documents/upload",
            files=files,
            data={"user_id": "test_user_bg", "preset": "balanced"},
        )

        assert response.status_code == 202
        data = response.json()
        assert data["file_name"] == "test_guide.md"
        assert data["status"] == "PARSING"
        assert data["chunks_count"] == 0
        assert "background" in data["message"].lower()

        # Verify background task was executed
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args.kwargs
        assert call_kwargs["user_id"] == "test_user_bg"
        assert call_kwargs["file_name"] == "test_guide.md"
        assert call_kwargs["preset"] == "balanced"
        assert call_kwargs["is_reindex"] is False


@pytest.mark.asyncio
async def test_process_document_pipeline_invalid_uuid():
    with pytest.raises(ValueError):
        await process_document_pipeline(
            document_id="not-a-valid-uuid",
            r2_key="dummy_key",
            user_id="u1",
            file_name="test.txt",
        )


@pytest.mark.asyncio
async def test_download_document_inline_and_attachment():
    """Verifies that download_document sets Content-Disposition inline or attachment correctly."""
    import uuid
    from app.models.document import Document, DocumentStatus
    from app.core.database import async_session_factory

    doc_id = uuid.uuid4()
    r2_key = f"documents/test_dl_user/{doc_id}_report.pdf"

    async with async_session_factory() as session:
        doc = Document(
            id=doc_id,
            user_id="test_dl_user",
            file_name="report.pdf",
            r2_key=r2_key,
            mime_type="application/pdf",
            file_size=1024,
            status=DocumentStatus.INDEXED,
        )
        session.add(doc)
        await session.commit()

    with patch("app.api.v1.endpoints.documents.r2_storage_service.download_file", new_callable=AsyncMock) as mock_dl:
        mock_dl.return_value = b"%PDF-1.4 dummy binary content"

        # 1. Default download: attachment
        res_attachment = client.get(f"/api/v1/documents/{doc_id}/download?user_id=test_dl_user")
        assert res_attachment.status_code == 200
        assert 'attachment; filename="report.pdf"' in res_attachment.headers["content-disposition"]

        # 2. Inline preview
        res_inline = client.get(f"/api/v1/documents/{doc_id}/download?user_id=test_dl_user&inline=true")
        assert res_inline.status_code == 200
        assert 'inline; filename="report.pdf"' in res_inline.headers["content-disposition"]

        # 3. Unauthorized access by another non-root user
        res_unauthorized = client.get(f"/api/v1/documents/{doc_id}/download?user_id=other_user")
        assert res_unauthorized.status_code == 403
        assert "Access denied" in res_unauthorized.json()["detail"]

        # 4. Root admin access allowed
        res_root = client.get(f"/api/v1/documents/{doc_id}/download?user_id=root&inline=true")
        assert res_root.status_code == 200


@pytest.mark.asyncio
async def test_delete_document_unauthorized():
    """Deleting a document owned by another user without root privileges returns 403."""
    import uuid
    from app.models.document import Document, DocumentStatus
    from app.core.database import async_session_factory

    doc_id = uuid.uuid4()
    r2_key = f"documents/owner_user/{doc_id}_file.txt"

    async with async_session_factory() as session:
        doc = Document(
            id=doc_id,
            user_id="owner_user",
            file_name="file.txt",
            r2_key=r2_key,
            mime_type="text/plain",
            file_size=100,
            status=DocumentStatus.INDEXED,
        )
        session.add(doc)
        await session.commit()

    res = client.delete(f"/api/v1/documents/{doc_id}?user_id=different_user")
    assert res.status_code == 403
    assert "Access denied" in res.json()["detail"]


@pytest.mark.asyncio
async def test_recover_stuck_documents():
    """Verifies that recover_stuck_documents recovers documents in PARSING older than threshold."""
    import uuid
    from datetime import datetime, timedelta, timezone
    from app.services.document_processor import recover_stuck_documents
    from app.models.document import Document, DocumentStatus
    from app.core.database import async_session_factory

    doc_id = uuid.uuid4()
    stuck_time = datetime.now(timezone.utc) - timedelta(seconds=400)

    async with async_session_factory() as session:
        doc = Document(
            id=doc_id,
            user_id="test_rec_user",
            file_name="stuck_file.txt",
            r2_key=f"documents/test_rec_user/{doc_id}_stuck_file.txt",
            mime_type="text/plain",
            file_size=120,
            status=DocumentStatus.PARSING,
            retry_count=0,
            created_at=stuck_time,
            updated_at=stuck_time,
        )
        session.add(doc)
        await session.commit()

    with patch("app.services.document_processor.r2_storage_service.file_exists", new_callable=AsyncMock) as mock_exists, \
         patch("app.services.document_processor.process_document_pipeline", new_callable=AsyncMock) as mock_pipe:
        mock_exists.return_value = True

        recovered = await recover_stuck_documents(stuck_threshold_seconds=300)
        assert str(doc_id) in recovered

        async with async_session_factory() as session:
            refreshed = await session.get(Document, doc_id)
            assert refreshed is not None
            assert refreshed.retry_count == 1



