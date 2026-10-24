"""Integration tests for Object Storage Management API endpoints."""

from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_storage_stats_endpoint():
    with patch("app.api.v1.endpoints.storage.r2_storage_service.get_storage_stats", new_callable=AsyncMock) as mock_stats:
        mock_stats.return_value = {
            "configured": True,
            "bucket_name": "test-bucket",
            "region": "auto",
            "endpoint_url": "https://test.r2.cloudflarestorage.com",
            "total_files": 12,
            "total_bytes": 1048576 * 5,  # 5 MB
        }

        response = client.get("/api/v1/storage/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["configured"] is True
        assert data["bucket_name"] == "test-bucket"
        assert data["total_files"] == 12
        assert data["total_bytes"] == 5242880
        assert "MB" in data["formatted_storage"]


def test_list_storage_objects_endpoint():
    with patch("app.api.v1.endpoints.storage.r2_storage_service.list_objects", new_callable=AsyncMock) as mock_list:
        mock_list.return_value = {
            "objects": [
                {
                    "key": "uploads/image.png",
                    "size": 2048,
                    "last_modified": None,
                    "etag": "abcd1234",
                    "presigned_url": "https://r2/download/image.png",
                    "public_url": None,
                }
            ],
            "total_count": 1,
            "is_truncated": False,
            "next_continuation_token": None,
        }

        response = client.get("/api/v1/storage/objects?prefix=uploads/")
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert data["objects"][0]["key"] == "uploads/image.png"
        assert data["objects"][0]["formatted_size"] == "2.0 KB"


def test_upload_storage_object_endpoint():
    with patch("app.api.v1.endpoints.storage.r2_storage_service.upload_file", new_callable=AsyncMock) as mock_upload, \
         patch("app.api.v1.endpoints.storage.r2_storage_service.generate_presigned_url", new_callable=AsyncMock) as mock_url:
        mock_upload.return_value = "uploads/doc.pdf"
        mock_url.return_value = "https://r2/download/doc.pdf"

        files = {"file": ("doc.pdf", b"%PDF-1.4 dummy", "application/pdf")}
        response = client.post("/api/v1/storage/upload", files=files, data={"prefix": "uploads"})
        assert response.status_code == 201
        data = response.json()
        assert data["key"] == "uploads/doc.pdf"
        assert data["file_name"] == "doc.pdf"
        assert data["presigned_url"] == "https://r2/download/doc.pdf"


def test_create_presigned_url_endpoint():
    with patch("app.api.v1.endpoints.storage.r2_storage_service.generate_presigned_upload_url", new_callable=AsyncMock) as mock_put_url:
        mock_put_url.return_value = "https://r2/upload/signed-put"

        payload = {
            "file_key": "raw/dataset.csv",
            "operation": "put_object",
            "content_type": "text/csv",
            "expires_in": 1800,
        }
        response = client.post("/api/v1/storage/presigned-url", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["url"] == "https://r2/upload/signed-put"
        assert data["operation"] == "put_object"
        assert data["expires_in"] == 1800


def test_delete_storage_object_endpoint():
    with patch("app.api.v1.endpoints.storage.r2_storage_service.delete_file", new_callable=AsyncMock) as mock_del:
        mock_del.return_value = True

        response = client.delete("/api/v1/storage/objects/uploads/old_file.txt")
        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True
        assert data["key"] == "uploads/old_file.txt"
