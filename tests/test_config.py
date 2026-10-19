from app.core.config import Settings


def test_default_settings():
    settings = Settings()
    assert settings.PROJECT_NAME == "Base Agent"
    assert settings.VERSION == "0.1.0"
    assert settings.API_V1_STR == "/api/v1"
    assert isinstance(settings.CORS_ORIGINS, list)
    assert len(settings.CORS_ORIGINS) > 0


def test_cors_origins_parsing():
    settings = Settings(CORS_ORIGINS="http://example.com, https://agent.app")
    assert settings.CORS_ORIGINS == ["http://example.com", "https://agent.app"]


def test_database_url_asyncpg_enforcement():
    settings = Settings(DATABASE_URL="postgresql://user:pass@localhost:5432/testdb")
    assert settings.DATABASE_URL == "postgresql+asyncpg://user:pass@localhost:5432/testdb"


def test_database_url_neon_sslmode_conversion():
    neon_url = "postgresql://user:pass@ep-test.neon.tech/neondb?sslmode=require&channel_binding=require"
    settings = Settings(DATABASE_URL=neon_url)
    assert settings.DATABASE_URL == "postgresql+asyncpg://user:pass@ep-test.neon.tech/neondb?ssl=require"
