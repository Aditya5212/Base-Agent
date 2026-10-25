"""Core modules: configuration, database, LLM provider, and authentication."""

from app.core.auth import UserContext, get_current_user, require_admin
from app.core.config import get_settings, settings
from app.core.database import AsyncSessionLocal, engine, get_db
from app.core.llm import (
    DEEPSEEK_FLASH,
    DEEPSEEK_PRO,
    MODEL_DEEPSEEK_FLASH,
    MODEL_DEEPSEEK_PRO,
    get_llm,
    resolve_model_name,
)

__all__ = [
    "settings",
    "get_settings",
    "engine",
    "AsyncSessionLocal",
    "get_db",
    "get_llm",
    "resolve_model_name",
    "DEEPSEEK_FLASH",
    "DEEPSEEK_PRO",
    "MODEL_DEEPSEEK_FLASH",
    "MODEL_DEEPSEEK_PRO",
    "UserContext",
    "get_current_user",
    "require_admin",
]
