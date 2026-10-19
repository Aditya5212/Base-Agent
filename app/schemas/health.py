from pydantic import BaseModel, Field


class DatabaseHealth(BaseModel):
    """Database connectivity health details."""

    connected: bool = Field(..., description="Whether database connection is active")
    details: str | None = Field(default=None, description="Optional diagnostic message if unhealthy")


class HealthResponse(BaseModel):
    """Overall system health check response."""

    status: str = Field(default="healthy", description="Overall service status")
    environment: str = Field(..., description="Current deployment environment")
    version: str = Field(..., description="Application version")
    database: DatabaseHealth = Field(..., description="Database connectivity status")
