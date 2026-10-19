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


def test_database_url_psycopg_enforcement():
    settings = Settings(DATABASE_URL="postgresql://user:pass@localhost:5432/testdb")
    assert settings.DATABASE_URL == "postgresql+psycopg://user:pass@localhost:5432/testdb"
    assert settings.raw_database_url == "postgresql://user:pass@localhost:5432/testdb"


def test_database_url_preserves_query_params_for_psycopg():
    neon_url = "postgresql://user:pass@ep-test.neon.tech/neondb?sslmode=require&channel_binding=require"
    settings = Settings(DATABASE_URL=neon_url)
    assert settings.DATABASE_URL == "postgresql+psycopg://user:pass@ep-test.neon.tech/neondb?sslmode=require&channel_binding=require"
    assert settings.raw_database_url == neon_url
