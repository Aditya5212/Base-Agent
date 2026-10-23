import json
import urllib.parse
from functools import lru_cache
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings validated by pydantic-settings."""

    # Project metadata
    PROJECT_NAME: str = "Base Agent"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = (
        "Production-grade, extensible AI chatbot and Agent Builder platform built with FastAPI and LangGraph."
    )
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"  # development, staging, production
    DEBUG: bool = False

    # Security & CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        if isinstance(v, (list, tuple)):
            return list(v)
        return []

    # Database Settings
    DATABASE_URL: str = "postgresql+psycopg://postgres:postgres@localhost:5432/base_agent"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_POOL_RECYCLE: int = 1800
    DATABASE_ECHO: bool = False

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def enforce_psycopg_scheme(cls, v: Any) -> str:
        if not isinstance(v, str):
            return v

        # Convert standard postgresql://, postgres://, or postgresql+asyncpg:// to postgresql+psycopg://
        if v.startswith("postgresql+asyncpg://"):
            return v.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+psycopg://", 1)
        if v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql+psycopg://", 1)

        return v

    @property
    def raw_database_url(self) -> str:
        """Returns standard postgresql:// DSN without SQLAlchemy driver prefix.

        Useful for LangGraph's AsyncPostgresSaver and direct psycopg AsyncConnectionPool.
        """
        return self.DATABASE_URL.replace("postgresql+psycopg://", "postgresql://", 1)

    # Auth (Clerk)
    CLERK_SECRET_KEY: str | None = None
    CLERK_FRONTEND_API_URL: str | None = None

    # LLM / AI Providers (DeepSeek OpenAI-compatible format)
    DEEPSEEK_API_KEY: str | None = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-flash"

    # Generic fallbacks
    LLM_BASE_URL: str = "https://api.deepseek.com"
    LLM_MODEL: str = "deepseek-flash"
    DEFAULT_TEMPERATURE: float = 0.0
    OPENAI_API_KEY: str | None = None

    @property
    def effective_llm_api_key(self) -> str | None:
        """Returns DEEPSEEK_API_KEY if present, falling back to OPENAI_API_KEY."""
        return self.DEEPSEEK_API_KEY or self.OPENAI_API_KEY

    # Object Storage (Cloudflare R2 - Phase 2)
    CLOUDFLARE_R2_ACCESS_KEY: str | None = None
    CLOUDFLARE_R2_SECRET_KEY: str | None = None
    CLOUDFLARE_R2_BUCKET_NAME: str | None = None
    CLOUDFLARE_R2_ENDPOINT_URL: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Returns cached Settings instance."""
    return Settings()


settings = get_settings()
