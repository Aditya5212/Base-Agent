from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import setup_logger, wrap_embedding_func_with_attrs

from app.core.config import settings
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

logger = structlog.get_logger(__name__)
setup_logger("lightrag", level="INFO")


def _build_embedding_func():
    """Builds wrapped embedding function using OpenAI text-embedding-3-small (1536 dims)."""
    @wrap_embedding_func_with_attrs(
        embedding_dim=settings.embedding_dim,
        max_token_size=settings.embedding_token_limit,
        model_name=settings.embedding_model,
    )
    async def embedding_func(texts: list[str]) -> np.ndarray:
        return await openai_embed.func(
            texts,
            model=settings.embedding_model,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.embedding_base_url or None,
        )

    return embedding_func


async def _llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> str:
    """Invokes OpenAI-compatible completion for entity and relationship extraction."""
    # Determine provider credentials based on available keys
    if settings.OPENAI_API_KEY:
        api_key = settings.OPENAI_API_KEY
        # If user explicitly configured an OpenAI-compatible proxy, use it; otherwise None defaults to official OpenAI
        base_url = (
            settings.LLM_BASE_URL
            if settings.LLM_BASE_URL and "deepseek.com" not in settings.LLM_BASE_URL
            else None
        )
        model = settings.lightrag_llm_model
    else:
        # Fall back to DeepSeek
        api_key = settings.effective_llm_api_key
        base_url = settings.DEEPSEEK_BASE_URL
        model = settings.DEEPSEEK_MODEL

    return await openai_complete_if_cache(
        model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=api_key,
        base_url=base_url,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )


@register_provider
class LightRAGProvider:
    """Knowledge Graph RAG provider powered by LightRAG and PostgreSQL.

    Configured with PGTableGraphStorage for native stock PostgreSQL graph tables
    without Apache AGE extension dependency, running natively on managed services like Neon.
    """

    name = "lightrag"

    def __init__(self, workspace: str = "default"):
        self.workspace = workspace
        self._rag: LightRAG | None = None
        self._ready = asyncio.Event()

    async def initialize(self) -> None:
        """Initializes LightRAG storages and verifies PostgreSQL tables."""
        if self._rag is not None:
            return

        Path(settings.lightrag_working_dir).mkdir(parents=True, exist_ok=True)

        self._rag = LightRAG(
            working_dir=settings.lightrag_working_dir,
            workspace=self.workspace,
            # Native PostgreSQL storages in Neon
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            graph_storage="PGTableGraphStorage",  # Stock PostgreSQL tables - NO Apache AGE!
            doc_status_storage="PGDocStatusStorage",
            llm_model_func=_llm_model_func,
            llm_model_name=settings.lightrag_llm_model,
            llm_model_max_async=settings.lightrag_max_async_llm,
            embedding_func=_build_embedding_func(),
            embedding_batch_num=16,
            embedding_func_max_async=8,
            chunk_token_size=settings.chunk_size,
            chunk_overlap_token_size=settings.chunk_overlap_size,
            max_parallel_insert=settings.lightrag_max_parallel_insert,
            enable_llm_cache=True,
            enable_llm_cache_for_entity_extract=True,
            addon_params={"language": settings.summary_language},
        )

        await self._rag.initialize_storages()
        self._ready.set()
        logger.info("lightrag_provider_initialized", workspace=self.workspace)

    async def shutdown(self) -> None:
        """Finalizes LightRAG storages and releases connections."""
        if self._rag is not None:
            await self._rag.finalize_storages()
            self._rag = None
            self._ready.clear()
            logger.info("lightrag_provider_shutdown", workspace=self.workspace)

    async def _get(self) -> LightRAG:
        if self._rag is None:
            await self.initialize()
        if self._rag is None:
            raise RuntimeError(f"LightRAGProvider[{self.workspace}] failed to initialize")
        return self._rag

    async def health_check(self) -> ProviderHealth:
        t0 = time.perf_counter()
        try:
            rag = await self._get()
            await rag.get_processing_status()
            return ProviderHealth(
                healthy=True,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            return ProviderHealth(
                healthy=False,
                latency_ms=(time.perf_counter() - t0) * 1000,
                detail=str(exc),
            )

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            display_name="LightRAG (Knowledge Graph)",
            description=(
                "Extracts structured entities and relationships into an interconnected knowledge graph. "
                "Delivers zero-hallucination multi-hop reasoning, thematic summaries, and relationship traversal."
            ),
            supports_graph=True,
            supports_multimodal=False,
            supported_modes=["mix", "local", "global", "hybrid", "naive"],
            incremental_delete=True,
            cost_tier="high",
        )

    async def ingest(self, documents: list[IngestDocument]) -> IngestResult:
        rag = await self._get()
        result = IngestResult(provider=self.name)

        for doc in documents:
            try:
                await rag.ainsert(
                    doc.content,
                    ids=[doc.doc_id],
                    file_paths=[doc.file_path or doc.doc_id],
                )
                result.accepted += 1
                logger.info(
                    "lightrag_document_ingested",
                    doc_id=doc.doc_id,
                    workspace=self.workspace,
                )
            except Exception as exc:
                result.failed += 1
                result.errors.append(f"{doc.doc_id}: {exc}")
                logger.error(
                    "lightrag_ingest_failed",
                    doc_id=doc.doc_id,
                    workspace=self.workspace,
                    error=str(exc),
                )

        return result

    async def query(self, q: RetrievalQuery) -> RetrievalResult:
        rag = await self._get()
        t0 = time.perf_counter()

        mode = q.mode if q.mode in ("mix", "local", "global", "hybrid", "naive") else "mix"

        # only_need_context=True guarantees LightRAG returns retrieved entity/relation/chunk context
        # without invoking a duplicate LLM call to generate a redundant answer.
        param = QueryParam(
            mode=mode,  # type: ignore
            only_need_context=True,
            top_k=max(q.top_k * 4, 20),
            chunk_top_k=q.top_k,
            max_total_tokens=settings.lightrag_max_context_tokens,
            enable_rerank=False,
        )

        context = await rag.aquery(q.query, param=param)
        context_str = str(context) if context else ""
        latency_ms = (time.perf_counter() - t0) * 1000

        chunks: list[RetrievedChunk] = []
        if context_str:
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"graph_{self.workspace}_{int(time.time())}",
                    content=context_str,
                    score=1.0,
                    source=self.name,
                    metadata={"mode": mode, "workspace": self.workspace},
                )
            )

        return RetrievalResult(
            provider=self.name,
            query=q.query,
            chunks=chunks,
            context=context_str,
            latency_ms=latency_ms,
            metadata={"mode": mode, "workspace": self.workspace},
        )

    async def delete_document(self, doc_id: str) -> None:
        rag = await self._get()
        await rag.adelete_by_doc_id(doc_id)
        logger.info(
            "lightrag_document_deleted",
            doc_id=doc_id,
            workspace=self.workspace,
        )
