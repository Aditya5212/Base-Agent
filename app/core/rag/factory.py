from __future__ import annotations

import asyncio
import structlog
from typing import TYPE_CHECKING

from app.core.rag.registry import get_provider_class

if TYPE_CHECKING:
    from app.core.rag.protocols import KnowledgeProvider

logger = structlog.get_logger(__name__)

_instances: dict[tuple[str, str], KnowledgeProvider] = {}
_locks: dict[tuple[str, str], asyncio.Lock] = {}
_global_lock = asyncio.Lock()


async def get_provider(provider_name: str, workspace: str = "default") -> KnowledgeProvider:
    """Returns a cached, initialized knowledge provider bound to a specific workspace.

    Thread/async safe: Uses a per-workspace lock to prevent thundering herd initialization.
    """
    key = (provider_name, workspace)

    async with _global_lock:
        if key not in _locks:
            _locks[key] = asyncio.Lock()
        lock = _locks[key]

    async with lock:
        if key not in _instances:
            cls = get_provider_class(provider_name)
            provider: KnowledgeProvider = cls(workspace=workspace)
            logger.info(
                "initializing_knowledge_provider",
                provider=provider_name,
                workspace=workspace,
            )
            await provider.initialize()
            _instances[key] = provider

        return _instances[key]


async def shutdown_all_providers() -> None:
    """Gracefully shuts down all active knowledge provider instances and flushes pools."""
    async with _global_lock:
        providers = list(_instances.values())
        _instances.clear()
        _locks.clear()

    for provider in providers:
        try:
            logger.info("shutting_down_knowledge_provider", provider=provider.name)
            await provider.shutdown()
        except Exception as exc:
            logger.warning(
                "error_shutting_down_provider",
                provider=getattr(provider, "name", "unknown"),
                error=str(exc),
            )
