from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

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
