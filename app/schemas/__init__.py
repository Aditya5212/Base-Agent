"""Pydantic schemas for request and response validation."""

from app.schemas.health import DatabaseHealth, HealthResponse

__all__ = ["HealthResponse", "DatabaseHealth"]
