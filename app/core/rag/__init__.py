from app.core.rag.factory import get_provider, shutdown_all_providers
from app.core.rag.protocols import (
    IngestDocument,
    IngestResult,
    KnowledgeProvider,
    ProviderCapabilities,
    ProviderHealth,
    RetrievalQuery,
    RetrievalResult,
    RetrievedChunk,
)
import app.core.rag.providers  # noqa: F401 - ensures @register_provider fires
from app.core.rag.registry import available_providers, register_provider
from app.core.rag.workspace import (
    agent_workspace,
    global_workspace,
    sanitize,
    thread_workspace,
    user_workspace,
)

__all__ = [
    "KnowledgeProvider",
    "IngestDocument",
    "RetrievedChunk",
    "RetrievalQuery",
    "RetrievalResult",
    "IngestResult",
    "ProviderCapabilities",
    "ProviderHealth",
    "get_provider",
    "shutdown_all_providers",
    "available_providers",
    "register_provider",
    "sanitize",
    "user_workspace",
    "thread_workspace",
    "agent_workspace",
    "global_workspace",
]
