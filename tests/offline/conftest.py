"""Explicitly offline tests: run with --confcutdir=tests/offline."""
import os

os.environ.update({
    "PYTHON_DOTENV_DISABLED": "1", "DEBUG": "false", "ENVIRONMENT": "test",
    "DEEPSEEK_API_KEY": "", "OPENAI_API_KEY": "",
    "SECRET_ENCRYPTION_KEY": "offline-regression-key",
    "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
})
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://test:test@127.0.0.1:1/offline_tests")
