from fastapi import APIRouter, Response, status

from app.core.config import settings
from app.core.database import check_db_connection
from app.schemas.health import DatabaseHealth, HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Readiness Check",
    description="Checks the health of the application and its connection to PostgreSQL.",
)
async def health_check(response: Response) -> HealthResponse:
    """Returns application health and database connection status."""
    db_connected, error_message = await check_db_connection()

    if not db_connected:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    return HealthResponse(
        status=overall_status,
        environment=settings.ENVIRONMENT,
        version=settings.VERSION,
        database=DatabaseHealth(
            connected=db_connected,
            details=error_message,
        ),
    )


@router.get(
    "/ping",
    summary="Process Liveness Check",
    description="Lightweight ping to verify the FastAPI process is alive without touching external services.",
)
async def ping() -> dict[str, str]:
    """Lightweight process liveness endpoint."""
    return {"status": "pong"}
