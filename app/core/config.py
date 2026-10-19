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
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/base_agent"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_POOL_RECYCLE: int = 1800
    DATABASE_ECHO: bool = False

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def enforce_asyncpg_scheme(cls, v: Any) -> str:
        if not isinstance(v, str):
            return v

        # Convert standard postgresql:// or postgres:// to postgresql+asyncpg://
        if v.startswith("postgresql://"):
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif v.startswith("postgres://"):
            v = v.replace("postgres://", "postgresql+asyncpg://", 1)

        # Normalize parameters for asyncpg if using asyncpg driver
        parsed = urllib.parse.urlsplit(v)
        if "asyncpg" in parsed.scheme and parsed.query:
            query_params = urllib.parse.parse_qs(parsed.query)
            clean_params: dict[str, str] = {}
            for k, val_list in query_params.items():
                if not val_list:
                    continue
                # asyncpg expects 'ssl' parameter rather than libpq's 'sslmode'
                if k == "sslmode":
                    clean_params["ssl"] = val_list[0]
                # channel_binding is a libpq parameter unsupported by asyncpg
                elif k == "channel_binding":
                    continue
                else:
                    clean_params[k] = val_list[0]
            new_query = urllib.parse.urlencode(clean_params)
            v = urllib.parse.urlunsplit(
                (parsed.scheme, parsed.netloc, parsed.path, new_query, parsed.fragment)
            )

        return v

    # Auth (Clerk)
    CLERK_SECRET_KEY: str | None = None
    CLERK_FRONTEND_API_URL: str | None = None

    # LLM Settings
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o"
    DEFAULT_TEMPERATURE: float = 0.0

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
