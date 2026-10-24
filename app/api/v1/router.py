from fastapi import APIRouter

from app.api.v1.endpoints import chat, documents, health

api_router = APIRouter()

# Register endpoint routers
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(documents.router)

