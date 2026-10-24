from __future__ import annotations

import time
from typing import Any
from sqlalchemy import text

from app.core.database import engine
from app.core.rag_config import get_rag_preset
from app.core.rag.protocols import (
    IngestDocument,
    IngestResult,
    ProviderCapabilities,
    ProviderHealth,
    RetrievalQuery,
    RetrievalResult,
    RetrievedChunk,
)
from app.core.rag.registry import register_provider
from app.services.retriever import hybrid_search
from app.services.vector_store import get_vector_store


@register_provider
class HybridProvider:
    """Knowledge provider wrapping dense PGVector HNSW and PostgreSQL Full-Text Search with RRF."""

    name = "hybrid"

    def __init__(self, workspace: str = "default"):
        self.workspace = workspace

    async def initialize(self) -> None:
        """Shares the existing SQLAlchemy and PGVector connection pool."""
        pass

    async def shutdown(self) -> None:
        """No isolated connection pool to tear down; managed by core database engine."""
        pass

    async def health_check(self) -> ProviderHealth:
        t0 = time.perf_counter()
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return ProviderHealth(healthy=True, latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception as exc:
            return ProviderHealth(healthy=False, detail=str(exc))

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            display_name="Hybrid Search (pgvector + FTS + RRF)",
            description=(
                "Dense embeddings fused with native PostgreSQL Full-Text Search via Reciprocal Rank Fusion. "
                "Sub-millisecond retrieval, low compute overhead, optimal for exact facts and direct lookup."
            ),
            supports_graph=False,
            supports_multimodal=False,
            supported_modes=["default", "high_precision", "balanced", "high_recall"],
            incremental_delete=True,
            cost_tier="low",
        )

    async def ingest(self, documents: list[IngestDocument]) -> IngestResult:
        result = IngestResult(provider=self.name)
        vector_store = get_vector_store()
        from langchain_core.documents import Document

        lc_docs = []
        for doc in documents:
            try:
                meta = dict(doc.metadata)
                meta["document_id"] = doc.doc_id
                if doc.file_path:
                    meta["file_path"] = doc.file_path
                lc_docs.append(Document(page_content=doc.content, metadata=meta))
                result.accepted += 1
            except Exception as exc:
                result.failed += 1
                result.errors.append(f"{doc.doc_id}: {exc}")

        if lc_docs:
            await vector_store.aadd_documents(lc_docs)

        return result

    async def query(self, q: RetrievalQuery) -> RetrievalResult:
        t0 = time.perf_counter()
        preset_name = q.mode if q.mode in ("high_precision", "balanced", "high_recall") else "balanced"
        preset = get_rag_preset(preset_name)

        # Apply user/workspace filters if provided
        filters = dict(q.filters) if q.filters else None

        docs = await hybrid_search(
            query=q.query,
            config=preset,
            filter_dict=filters,
        )

        chunks: list[RetrievedChunk] = []
        context_parts: list[str] = []

        for i, doc in enumerate(docs[: q.top_k]):
            chunk_id = str(doc.metadata.get("id") or doc.metadata.get("chunk_id") or f"c_{i}")
            score = float(doc.metadata.get("rrf_score", 1.0))
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    content=doc.page_content,
                    score=score,
                    source=self.name,
                    metadata=doc.metadata,
                )
            )
            context_parts.append(f"[Document Chunk {i+1}]:\n{doc.page_content.strip()}")

        context = "\n\n".join(context_parts)
        latency_ms = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            provider=self.name,
            query=q.query,
            chunks=chunks,
            context=context,
            latency_ms=latency_ms,
            metadata={"mode": preset_name, "workspace": self.workspace},
        )

    async def delete_document(self, doc_id: str) -> None:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "DELETE FROM langchain_pg_embedding "
                    "WHERE cmetadata->>'document_id' = :doc_id"
                ),
                {"doc_id": doc_id},
            )
