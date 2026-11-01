"""Offline by default; database tests require an explicit disposable database."""
import asyncio
import os
from pathlib import Path
import sys

import pytest

_integration = "--run-integration" in sys.argv
_test_database = os.environ.get("TEST_DATABASE_URL") if _integration else None
os.environ.update({
    "PYTHON_DOTENV_DISABLED": "1", "DEBUG": "false", "ENVIRONMENT": "test",
    "DATABASE_URL": _test_database or "postgresql+psycopg://test:test@127.0.0.1:1/offline_tests",
    "DEEPSEEK_API_KEY": "", "OPENAI_API_KEY": "",
    "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
    "SECRET_ENCRYPTION_KEY": "offline-regression-key",
})

OFFLINE_MODULES = {
    "test_config.py", "test_document_loader.py", "test_embedding.py", "test_text_splitter.py",
    "test_retriever.py", "test_llm.py", "test_rag_tools.py", "test_taskiq.py",
    "test_workflow_compiler.py", "test_workflow_nodes.py", "test_rate_limit.py",
}
OFFLINE_TOOL_TESTS = {
    "test_secret_encryption_and_decryption", "test_secret_masking", "test_corrupted_token_decryption_error",
    "test_http_tool_executor_get_with_params", "test_http_tool_executor_post_with_bearer_auth",
    "test_code_tool_executor_valid_execution", "test_code_tool_executor_blocks_forbidden_import",
    "test_code_tool_executor_blocks_eval_and_exec", "test_code_tool_executor_timeout_guard",
    "test_database_tool_executor_blocks_forbidden_sql", "test_database_tool_executor_parameterized_query",
    "test_json_schema_to_pydantic_model", "test_tool_registry_compiles_code_tool_with_cache",
    "test_tool_registry_resolves_built_ins_and_rag",
}


def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", help="Run service-backed tests using TEST_DATABASE_URL (migrations will run)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        if not os.environ.get("TEST_DATABASE_URL"):
            raise pytest.UsageError("--run-integration requires TEST_DATABASE_URL pointing to a disposable PostgreSQL database")
        return
    for item in items:
        file = Path(str(item.path))
        name = item.originalname or item.name
        offline = "offline" in file.parts or file.name in OFFLINE_MODULES
        offline |= file.name == "test_tools_api.py" and name in OFFLINE_TOOL_TESTS
        offline |= file.name == "test_agents.py" and name in {"test_agent_config_defaults", "test_agent_config_custom_values", "test_agent_config_validation_errors", "test_agent_config_lru_cache"}
        offline |= file.name == "test_auth.py" and name in {"test_get_current_user_default_is_root", "test_get_current_user_from_headers", "test_get_current_user_from_mock_bearer", "test_require_admin_guard"}
        offline |= file.name == "test_rag_providers.py" and name != "test_hybrid_provider_capabilities_and_health"
        if not offline:
            item.add_marker(pytest.mark.skip(reason="Service-backed test: use --run-integration and TEST_DATABASE_URL"))


@pytest.fixture(scope="session", autouse=True)
def initialize_disposable_database(request):
    if request.config.getoption("--run-integration"):
        from app.core.database import init_db
        async def initialize():
            from app.core.database import engine
            from app.services.memory import setup_checkpointer, close_checkpointer_pool
            from app.services.vector_store import setup_vector_store
            await init_db()
            await setup_checkpointer()
            await setup_vector_store()
            await close_checkpointer_pool()
            await engine.dispose()
        asyncio.run(initialize())  # Fail visibly; never suppress missing infrastructure.
