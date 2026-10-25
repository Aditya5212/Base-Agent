import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.database import check_db_connection, close_db_connection
from app.services.memory import close_checkpointer_pool, setup_checkpointer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan event handler for startup and shutdown procedures."""
    logger.info(
        "Starting %s v%s [env=%s, debug=%s]",
        settings.PROJECT_NAME,
        settings.VERSION,
        settings.ENVIRONMENT,
        settings.DEBUG,
    )

    sweeper_task = None

    # Check database connectivity during startup
    db_connected, err = await check_db_connection()
    if db_connected:
        logger.info("Database connection successfully established.")
        try:
            from app.core.database import init_db
            await init_db()
            logger.info("Application database tables verified/created successfully.")
        except Exception as init_err:
            logger.warning("Failed to initialize database tables: %s", init_err)

        try:
            await setup_checkpointer()
            logger.info("PostgreSQL checkpointer initialized successfully.")
        except Exception as checkpointer_err:
            logger.warning("Failed to initialize checkpointer tables: %s", checkpointer_err)

        try:
            from app.services.vector_store import setup_vector_store
            await setup_vector_store()
            logger.info("PostgreSQL vector store and FTS indexes initialized successfully.")
        except Exception as vs_err:
            logger.warning("Failed to initialize vector store indexes: %s", vs_err)

        try:
            from app.services.document_processor import recover_stuck_documents, start_document_sweeper
            recovered = await recover_stuck_documents()
            if recovered:
                logger.info("Startup recovery processed %d stuck document(s): %s", len(recovered), recovered)
            sweeper_task = asyncio.create_task(start_document_sweeper(interval_seconds=300))
            logger.info("Stuck-document background sweeper started.")
        except Exception as rec_err:
            logger.warning("Failed to initialize document recovery/sweeper: %s", rec_err)
    else:
        logger.warning("Database connection failed during startup: %s", err)

    yield

    # Graceful shutdown
    logger.info("Shutting down %s...", settings.PROJECT_NAME)
    if sweeper_task:
        sweeper_task.cancel()
        try:
            await sweeper_task
        except asyncio.CancelledError:
            pass
        except Exception as sw_err:
            logger.debug("Error while cancelling sweeper: %s", sw_err)

    try:
        from app.core.rag import shutdown_all_providers
        await shutdown_all_providers()
    except Exception as rag_err:
        logger.warning("Error shutting down knowledge providers: %s", rag_err)

    await close_checkpointer_pool()
    await close_db_connection()
    logger.info("Shutdown completed.")


def create_application() -> FastAPI:
    """Factory creating and configuring the FastAPI application."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Configure CORS
    if settings.CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Register API Routers
    app.include_router(api_router, prefix=settings.API_V1_STR)

    @app.get("/", tags=["Root"], summary="Root service status")
    async def root() -> JSONResponse:
        return JSONResponse(
            content={
                "name": settings.PROJECT_NAME,
                "version": settings.VERSION,
                "status": "online",
                "docs_url": "/docs",
                "api_v1": f"{settings.API_V1_STR}/health",
            }
        )

    return app


app = create_application()
