from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable
from pydantic import BaseModel, Field


class IngestDocument(BaseModel):
    """Normalized document payload submitted for ingestion into a knowledge provider."""

    doc_id: str
    content: str
    file_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """Standardized granular chunk or entity relation retrieved from a provider."""

    chunk_id: str
    content: str
    score: float = 1.0
    source: str  # provider name
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalQuery(BaseModel):
    """Query parameters submitted to a knowledge provider."""

    query: str
    top_k: int = 10
    mode: str = "default"  # provider-specific (e.g. LightRAG: local, global, hybrid, mix, naive)
    filters: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Unified retrieval result returned by all knowledge providers."""

    provider: str
    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    context: str = ""  # assembled context string, ready for LLM prompt injection
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResult(BaseModel):
    """Batch ingestion outcome summary."""

    provider: str
    accepted: int = 0
    failed: int = 0
    errors: list[str] = Field(default_factory=list)


class ProviderCapabilities(BaseModel):
    """Static metadata consumed by the Agent Builder to render configuration options."""

    display_name: str
    description: str
    supports_graph: bool = False
    supports_multimodal: bool = False
    supported_modes: list[str] = Field(default_factory=lambda: ["default"])
    incremental_delete: bool = True
    cost_tier: Literal["free", "low", "medium", "high"] = "low"
    config_schema: dict[str, Any] = Field(default_factory=dict)


class ProviderHealth(BaseModel):
    """Runtime health probe result for a provider instance."""

    healthy: bool
    latency_ms: float = 0.0
    detail: str = ""


@runtime_checkable
class KnowledgeProvider(Protocol):
    """The unified contract that all knowledge and graph retrieval providers must implement."""

    name: str

    async def initialize(self) -> None:
        """Initialize connections, pools, and storage schemas."""
        ...

    async def shutdown(self) -> None:
        """Gracefully close background pools and clients."""
        ...

    async def health_check(self) -> ProviderHealth:
        """Validate live database and backend connectivity."""
        ...

    def capabilities(self) -> ProviderCapabilities:
        """Return provider static capabilities and schema."""
        ...

    async def ingest(self, documents: list[IngestDocument]) -> IngestResult:
        """Ingest and index a batch of documents."""
        ...

    async def query(self, q: RetrievalQuery) -> RetrievalResult:
        """Retrieve contextual information relevant to the query."""
        ...

    async def delete_document(self, doc_id: str) -> None:
        """Purge all document chunks/entities for the given document identifier."""
        ...
