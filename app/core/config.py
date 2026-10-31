import json
import os
import urllib.parse
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# MUST run before Settings() and before any LightRAG import/init
load_dotenv(override=False)


def _sync_postgres_env_vars() -> None:
    """Ensure POSTGRES_* environment variables required by LightRAG are set.

    If discrete variables are already in os.environ (e.g. from .env), they take precedence.
    Otherwise, parse DATABASE_URL and populate POSTGRES_HOST, PORT, USER, PASSWORD, DATABASE.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return

    cleaned_url = (
        db_url.replace("postgresql+psycopg://", "postgresql://")
        .replace("postgresql+asyncpg://", "postgresql://")
    )
    parsed = urllib.parse.urlparse(cleaned_url)

    if "POSTGRES_HOST" not in os.environ and parsed.hostname:
        os.environ["POSTGRES_HOST"] = parsed.hostname
    if "POSTGRES_PORT" not in os.environ:
        os.environ["POSTGRES_PORT"] = str(parsed.port) if parsed.port else "5432"
    if "POSTGRES_USER" not in os.environ and parsed.username:
        os.environ["POSTGRES_USER"] = urllib.parse.unquote(parsed.username)
    if "POSTGRES_PASSWORD" not in os.environ and parsed.password:
        os.environ["POSTGRES_PASSWORD"] = urllib.parse.unquote(parsed.password)
    if "POSTGRES_DATABASE" not in os.environ and parsed.path:
        os.environ["POSTGRES_DATABASE"] = parsed.path.lstrip("/")
    if "POSTGRES_SSL_MODE" not in os.environ:
        if "sslmode=require" in (parsed.query or "").lower() or (
            parsed.hostname and "neon.tech" in parsed.hostname
        ):
            os.environ["POSTGRES_SSL_MODE"] = "require"


_sync_postgres_env_vars()


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

    # Object Storage (Cloudflare R2 - S3-compatible)
    R2_ACCOUNT_ID: str | None = None
    R2_ACCESS_KEY_ID: str | None = None
    R2_SECRET_ACCESS_KEY: str | None = None
    R2_BUCKET_NAME: str | None = None
    R2_REGION: str = "auto"
    R2_ENDPOINT: str | None = None
    R2_PUBLIC_BASE_URL: str | None = None

    # Backward compatibility aliases
    CLOUDFLARE_R2_ACCESS_KEY: str | None = None
    CLOUDFLARE_R2_SECRET_KEY: str | None = None
    CLOUDFLARE_R2_BUCKET_NAME: str | None = None
    CLOUDFLARE_R2_ENDPOINT_URL: str | None = None

    @property
    def r2_effective_access_key(self) -> str | None:
        """Returns R2_ACCESS_KEY_ID or fallback CLOUDFLARE_R2_ACCESS_KEY."""
        return self.R2_ACCESS_KEY_ID or self.CLOUDFLARE_R2_ACCESS_KEY

    @property
    def r2_effective_secret_key(self) -> str | None:
        """Returns R2_SECRET_ACCESS_KEY or fallback CLOUDFLARE_R2_SECRET_KEY."""
        return self.R2_SECRET_ACCESS_KEY or self.CLOUDFLARE_R2_SECRET_KEY

    @property
    def r2_effective_bucket_name(self) -> str | None:
        """Returns R2_BUCKET_NAME or fallback CLOUDFLARE_R2_BUCKET_NAME."""
        return self.R2_BUCKET_NAME or self.CLOUDFLARE_R2_BUCKET_NAME

    @property
    def r2_effective_endpoint_url(self) -> str | None:
        """Returns custom endpoint, or derives from R2_ACCOUNT_ID."""
        if self.R2_ENDPOINT:
            return self.R2_ENDPOINT
        if self.CLOUDFLARE_R2_ENDPOINT_URL:
            return self.CLOUDFLARE_R2_ENDPOINT_URL
        if self.R2_ACCOUNT_ID:
            return f"https://{self.R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
        return None

    # Security & Secret Encryption
    SECRET_ENCRYPTION_KEY: str | None = None

    # Embeddings & Vector Search (OpenAI text-embedding-3-small)
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536
    RAG_COLLECTION_NAME: str = "base_agent_rag"
    EMBEDDING_TOKEN_LIMIT: int = 8192
    EMBEDDING_BASE_URL: str | None = None

    # LightRAG Configuration
    LIGHTRAG_WORKING_DIR: str = "./data/lightrag"
    LIGHTRAG_LLM_MODEL: str = "gpt-4o-mini"
    LIGHTRAG_MAX_ASYNC_LLM: int = 4
    LIGHTRAG_MAX_PARALLEL_INSERT: int = 3
    LIGHTRAG_MAX_CONTEXT_TOKENS: int = 4096
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP_SIZE: int = 100
    SUMMARY_LANGUAGE: str = "English"

    @property
    def embedding_dim(self) -> int:
        return self.EMBEDDING_DIMENSIONS

    @property
    def embedding_model(self) -> str:
        return self.EMBEDDING_MODEL

    @property
    def embedding_token_limit(self) -> int:
        return self.EMBEDDING_TOKEN_LIMIT

    @property
    def embedding_base_url(self) -> str | None:
        return self.EMBEDDING_BASE_URL

    @property
    def lightrag_working_dir(self) -> str:
        return self.LIGHTRAG_WORKING_DIR

    @property
    def lightrag_llm_model(self) -> str:
        return self.LIGHTRAG_LLM_MODEL

    @property
    def lightrag_max_async_llm(self) -> int:
        return self.LIGHTRAG_MAX_ASYNC_LLM

    @property
    def lightrag_max_parallel_insert(self) -> int:
        return self.LIGHTRAG_MAX_PARALLEL_INSERT

    @property
    def lightrag_max_context_tokens(self) -> int:
        return self.LIGHTRAG_MAX_CONTEXT_TOKENS

    @property
    def chunk_size(self) -> int:
        return self.CHUNK_SIZE

    @property
    def chunk_overlap_size(self) -> int:
        return self.CHUNK_OVERLAP_SIZE

    @property
    def summary_language(self) -> str:
        return self.SUMMARY_LANGUAGE

    @property
    def llm_api_key(self) -> str | None:
        return self.effective_llm_api_key

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
