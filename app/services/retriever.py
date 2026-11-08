"""Hybrid retriever combining PostgreSQL dense vector search and native Full-Text Search with RRF.

Implements the production standard:
- Dense retrieval: pgvector cosine similarity via PGVector
- Sparse retrieval: Native PostgreSQL full-text search (tsvector + ts_rank_cd)
- Fusion: Reciprocal Rank Fusion (RRF) combining dense and sparse rank scores
"""

import asyncio
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


def _visible_chunk_filter(filters):
    visible = {"$or": [{"_base_active": {"$eq": True}}, {"_base_active": {"$exists": False}}]}
    return {"$and": [filters, visible]} if filters else visible


def build_rag_filter(
    user_id: str | None = None,
    thread_id: str | None = None,
    thread_scoped_only: bool = False,
) -> dict[str, Any] | None:
    """Builds a structured metadata filter compatible with both PGVector and sparse search.

    If thread_id is provided and thread_scoped_only is False (default for chat turns),
    it allows retrieving both thread-scoped documents and user-global documents
    (documents where thread_id does not exist).
    If thread_id is None, it scopes retrieval to user_id (or returns None).
    """
    clauses: list[dict[str, Any]] = []
    if user_id:
        clauses.append({"user_id": {"$eq": user_id}})

    if thread_id:
        if thread_scoped_only:
            clauses.append({"thread_id": {"$eq": thread_id}})
        else:
            clauses.append({
                "$or": [
                    {"thread_id": {"$eq": thread_id}},
                    {"thread_id": {"$exists": False}},
                ]
            })

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _build_sparse_filter_clause(
    filter_spec: dict[str, Any],
    params: dict[str, Any],
    counter: list[int],
) -> str:
    """Recursively converts filter specifications into PostgreSQL JSONB SQL conditions."""
    if not filter_spec:
        return ""
    if "$and" in filter_spec:
        parts = [_build_sparse_filter_clause(item, params, counter) for item in filter_spec["$and"]]
        valid_parts = [p for p in parts if p]
        return f"({' AND '.join(valid_parts)})" if valid_parts else ""
    if "$or" in filter_spec:
        parts = [_build_sparse_filter_clause(item, params, counter) for item in filter_spec["$or"]]
        valid_parts = [p for p in parts if p]
        return f"({' OR '.join(valid_parts)})" if valid_parts else ""

    field_conds = []
    for field, val in filter_spec.items():
        if field.startswith("$"):
            continue
        if isinstance(val, dict):
            if "$eq" in val:
                c = counter[0]
                counter[0] += 1
                param_key = f"sparse_p_{c}"
                field_conds.append(f"cmetadata->>'{field}' = :{param_key}")
                params[param_key] = str(val["$eq"])
            elif "$exists" in val:
                if val["$exists"]:
                    field_conds.append(f"(cmetadata ? '{field}' AND cmetadata->>'{field}' IS NOT NULL)")
                else:
                    field_conds.append(f"(NOT (cmetadata ? '{field}') OR cmetadata->>'{field}' IS NULL)")
        else:
            c = counter[0]
            counter[0] += 1
            param_key = f"sparse_p_{c}"
            field_conds.append(f"cmetadata->>'{field}' = :{param_key}")
            params[param_key] = str(val)

    return " AND ".join(field_conds)


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
            filter=_visible_chunk_filter(filter_dict),
        )
        return results
    except Exception as exc:
        logger.error("Dense vector search failed: %s", exc, exc_info=True)
        raise RuntimeError("Dense retrieval is unavailable") from exc


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
            "COALESCE(cmetadata->>'_base_active', 'true') = 'true'",
            "to_tsvector('english', document) @@ plainto_tsquery('english', :query)",
            "collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = :col_name)",
        ]
        params: dict[str, Any] = {
            "query": query,
            "col_name": collection_name,
            "limit": top_k,
        }

        # Apply metadata filters if provided (supports both simple dicts and complex $and/$or clauses)
        if filter_dict:
            counter = [0]
            sparse_clause = _build_sparse_filter_clause(filter_dict, params, counter)
            if sparse_clause:
                conditions.append(sparse_clause)

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
        raise RuntimeError("Sparse retrieval is unavailable") from exc


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
    thread_id: str | None = None,
    thread_scoped_only: bool = False,
) -> list[Document]:
    """Executes end-to-end hybrid search (dense pgvector + sparse PostgreSQL FTS) with RRF fusion.

    Args:
        query: User search query or question.
        config: Optional RAGConfig tuning precision vs recall, chunk sizes, and RRF weights.
        collection_name: Optional collection name override.
        filter_dict: Optional metadata filter dict (e.g. {'user_id': 'aditya'}).
        thread_id: Optional thread ID to apply thread scoping (thread + global fallback).
        thread_scoped_only: If True, search strictly thread-scoped documents without global fallback.

    Returns:
        Ranked list of top-K fused Document objects.
    """
    rag_cfg = config or get_rag_config()
    target_collection = collection_name or settings.RAG_COLLECTION_NAME

    effective_filter = dict(filter_dict) if filter_dict else None
    if thread_id and (not effective_filter or "thread_id" not in effective_filter):
        user_id = effective_filter.get("user_id") if effective_filter else None
        effective_filter = build_rag_filter(
            user_id=user_id,
            thread_id=thread_id,
            thread_scoped_only=thread_scoped_only,
        )

    # Execute dense and sparse retrieval in parallel to minimise wall-clock latency
    dense_candidates, sparse_candidates = await asyncio.gather(
        _dense_search(
            query=query,
            top_k=rag_cfg.top_k_dense,
            collection_name=target_collection,
            filter_dict=effective_filter,
        ),
        _sparse_search(
            query=query,
            top_k=rag_cfg.top_k_sparse,
            collection_name=target_collection,
            filter_dict=effective_filter,
        ),
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
