"""Hybrid retriever combining PostgreSQL dense vector search and native Full-Text Search with RRF.

Implements the production standard:
- Dense retrieval: pgvector cosine similarity via PGVector
- Sparse retrieval: Native PostgreSQL full-text search (tsvector + ts_rank_cd)
- Fusion: Reciprocal Rank Fusion (RRF) combining dense and sparse rank scores
"""

import hashlib
import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import chain
from sqlalchemy import text

from app.core.config import settings
from app.core.database import engine
from app.core.rag_config import RAGConfig, get_rag_config
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def _compute_chunk_fingerprint(doc: Document) -> str:
    """Computes a unique deduplication key for a document chunk."""
    chunk_id = doc.metadata.get("id") or doc.metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    # Fallback to content hash + source
    source = doc.metadata.get("source", "")
    content = doc.page_content.strip()
    return hashlib.sha256(f"{source}:{content}".encode("utf-8")).hexdigest()


async def _dense_search(
    query: str,
    top_k: int,
    collection_name: str,
    filter_dict: dict[str, Any] | None = None,
) -> list[tuple[Document, float]]:
    """Performs dense vector similarity search via pgvector."""
    try:
        vector_store = get_vector_store(collection_name=collection_name)
        # Note: PGVector supports filter dictionaries over JSONB cmetadata
        results = await vector_store.asimilarity_search_with_score(
            query=query,
            k=top_k,
            filter=filter_dict,
        )
        return results
    except Exception as exc:
        logger.error("Dense vector search failed: %s", exc, exc_info=True)
        return []


async def _sparse_search(
    query: str,
    top_k: int,
    collection_name: str,
    filter_dict: dict[str, Any] | None = None,
) -> list[tuple[Document, float]]:
    """Performs sparse keyword full-text search using PostgreSQL tsvector and ts_rank_cd."""
    try:
        # Build query filtering over langchain_pg_embedding
        conditions = [
            "to_tsvector('english', document) @@ plainto_tsquery('english', :query)",
            "collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = :col_name)",
        ]
        params: dict[str, Any] = {
            "query": query,
            "col_name": collection_name,
            "limit": top_k,
        }

        # Apply metadata filters if provided (e.g. user_id or document_id)
        if filter_dict:
            for idx, (k, v) in enumerate(filter_dict.items()):
                param_key = f"filter_val_{idx}"
                param_field = f"filter_field_{idx}"
                conditions.append(f"cmetadata->>:{param_field} = :{param_key}")
                params[param_field] = str(k)
                params[param_key] = str(v)

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT id, document, cmetadata,
                   ts_rank_cd(to_tsvector('english', document), plainto_tsquery('english', :query)) AS rank
            FROM langchain_pg_embedding
            WHERE {where_clause}
            ORDER BY rank DESC
            LIMIT :limit;
        """

        async with engine.connect() as conn:
            result = await conn.execute(text(sql), params)
            rows = result.fetchall()

        sparse_docs: list[tuple[Document, float]] = []
        for row in rows:
            doc = Document(
                page_content=row.document,
                metadata=dict(row.cmetadata or {}),
            )
            # Store internal database row id in metadata
            doc.metadata["db_id"] = str(row.id)
            sparse_docs.append((doc, float(row.rank)))

        return sparse_docs
    except Exception as exc:
        logger.error("Sparse PostgreSQL FTS search failed: %s", exc, exc_info=True)
        return []


def apply_reciprocal_rank_fusion(
    dense_results: list[tuple[Document, float]],
    sparse_results: list[tuple[Document, float]],
    config: RAGConfig,
) -> list[Document]:
    """Combines dense and sparse ranked lists using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF(d) = (w_dense / (k + rank_dense)) + (w_sparse / (k + rank_sparse))
    """
    k = config.rrf_k
    w_dense = config.vector_weight
    w_sparse = config.sparse_weight

    rrf_scores: dict[str, float] = {}
    doc_registry: dict[str, Document] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}

    # Accumulate dense ranks (1-indexed)
    for rank, (doc, _) in enumerate(dense_results, 1):
        fingerprint = _compute_chunk_fingerprint(doc)
        doc_registry[fingerprint] = doc
        dense_ranks[fingerprint] = rank
        rrf_scores[fingerprint] = rrf_scores.get(fingerprint, 0.0) + (w_dense / (k + rank))

    # Accumulate sparse ranks (1-indexed)
    for rank, (doc, _) in enumerate(sparse_results, 1):
        fingerprint = _compute_chunk_fingerprint(doc)
        if fingerprint not in doc_registry:
            doc_registry[fingerprint] = doc
        sparse_ranks[fingerprint] = rank
        rrf_scores[fingerprint] = rrf_scores.get(fingerprint, 0.0) + (w_sparse / (k + rank))

    # Sort documents in descending order of fused RRF score
    sorted_fingerprints = sorted(
        rrf_scores.keys(),
        key=lambda fp: rrf_scores[fp],
        reverse=True,
    )

    fused_docs: list[Document] = []
    for fp in sorted_fingerprints[:config.top_k_final]:
        doc = doc_registry[fp]
        # Attach fusion telemetry to metadata
        doc.metadata["rrf_score"] = round(rrf_scores[fp], 6)
        doc.metadata["dense_rank"] = dense_ranks.get(fp)
        doc.metadata["sparse_rank"] = sparse_ranks.get(fp)
        fused_docs.append(doc)

    logger.debug(
        "RRF fused %d dense and %d sparse candidates into %d final chunks",
        len(dense_results),
        len(sparse_results),
        len(fused_docs),
    )
    return fused_docs


async def hybrid_search(
    query: str,
    config: RAGConfig | None = None,
    collection_name: str | None = None,
    filter_dict: dict[str, Any] | None = None,
) -> list[Document]:
    """Executes end-to-end hybrid search (dense pgvector + sparse PostgreSQL FTS) with RRF fusion.

    Args:
        query: User search query or question.
        config: Optional RAGConfig tuning precision vs recall, chunk sizes, and RRF weights.
        collection_name: Optional collection name override.
        filter_dict: Optional metadata filter dict (e.g. {'user_id': 'aditya'}).

    Returns:
        Ranked list of top-K fused Document objects.
    """
    rag_cfg = config or get_rag_config()
    target_collection = collection_name or settings.RAG_COLLECTION_NAME

    # Execute dense and sparse retrieval
    dense_candidates = await _dense_search(
        query=query,
        top_k=rag_cfg.top_k_dense,
        collection_name=target_collection,
        filter_dict=filter_dict,
    )
    sparse_candidates = await _sparse_search(
        query=query,
        top_k=rag_cfg.top_k_sparse,
        collection_name=target_collection,
        filter_dict=filter_dict,
    )

    # Fuse candidate lists
    fused_results = apply_reciprocal_rank_fusion(
        dense_results=dense_candidates,
        sparse_results=sparse_candidates,
        config=rag_cfg,
    )
    return fused_results


def create_hybrid_retriever(
    config: RAGConfig | None = None,
    collection_name: str | None = None,
    filter_dict: dict[str, Any] | None = None,
):
    """Creates a LangChain Runnable retriever that executes hybrid search."""
    rag_cfg = config or get_rag_config()

    @chain
    async def _retriever(query: str) -> list[Document]:
        return await hybrid_search(
            query=query,
            config=rag_cfg,
            collection_name=collection_name,
            filter_dict=filter_dict,
        )

    return _retriever
