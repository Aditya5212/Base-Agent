from fastapi import APIRouter

from app.api.v1.endpoints import (
    agents,
    chat,
    documents,
    health,
    storage,
    structured_schemas,
    tools,
    users,
    workflows,
)

api_router = APIRouter()

# Register endpoint routers
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(users.router)
api_router.include_router(agents.router)
api_router.include_router(workflows.router)
api_router.include_router(structured_schemas.router)
api_router.include_router(tools.router)
api_router.include_router(documents.router)
api_router.include_router(storage.router)




