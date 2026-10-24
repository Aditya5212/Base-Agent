from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.core.rag.protocols import KnowledgeProvider

PROVIDER_REGISTRY: dict[str, Any] = {}


def register_provider(cls: Any) -> Any:
    """Decorator to register a KnowledgeProvider class into the global provider registry."""
    provider_name = getattr(cls, "name", None)
    if not provider_name:
        raise ValueError(f"Provider class {cls.__name__} must define a 'name' attribute.")
    PROVIDER_REGISTRY[provider_name] = cls
    return cls


def available_providers() -> list[str]:
    """Returns a sorted list of registered knowledge provider identifiers."""
    return sorted(PROVIDER_REGISTRY.keys())


def get_provider_class(name: str) -> Any:
    """Retrieves provider class by name, raising KeyError if not registered."""
    if name not in PROVIDER_REGISTRY:
        raise KeyError(
            f"Knowledge provider '{name}' is not registered. Available: {available_providers()}"
        )
    return PROVIDER_REGISTRY[name]
