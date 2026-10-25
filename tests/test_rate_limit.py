"""Tests for API Rate Limiting via slowapi."""

from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_rate_limiter_allows_requests_under_threshold():
    """Requests under rate limit succeed with 200 OK."""
    with patch("app.api.v1.endpoints.documents.hybrid_search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []
        payload = {"query": "test query", "preset": "balanced", "top_k": 3}
        response = client.post("/api/v1/documents/search", json=payload)
        assert response.status_code == 200


def test_rate_limit_exceeded_returns_429():
    """Exceeding rate limit triggers HTTP 429 Too Many Requests."""
    # We can test an endpoint configured with a limit by firing repeated requests
    # or registering a temporary test route on app with 2/minute limit
    from slowapi.errors import RateLimitExceeded
    from fastapi import Request
    from app.core.rate_limit import limiter

    @app.get("/api/v1/test-rate-limit")
    @limiter.limit("2/minute")
    def rate_limited_endpoint(request: Request):
        return {"ok": True}

    # Request 1: OK
    r1 = client.get("/api/v1/test-rate-limit")
    assert r1.status_code == 200

    # Request 2: OK
    r2 = client.get("/api/v1/test-rate-limit")
    assert r2.status_code == 200

    # Request 3: Rate limit exceeded -> 429
    r3 = client.get("/api/v1/test-rate-limit")
    assert r3.status_code == 429
    assert "rate limit exceeded" in r3.text.lower() or r3.status_code == 429
