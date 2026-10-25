"""PostgreSQL vector store service backed by langchain-postgres PGVector.

Stores Document objects in PostgreSQL with:
- Dense HNSW cosine distance vector index on embeddings (1536 dimensions)
- Sparse GIN index on to_tsvector for native full-text search (FTS)
"""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_postgres import PGVector
from sqlalchemy import text

from app.core.config import settings
from app.core.database import engine
from app.core.embedding import get_embeddings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton cache: one PGVector instance per collection name.
# PGVector holds no per-request state — it wraps the shared async SQLAlchemy
# engine — so reusing a cached instance across requests is safe and avoids the
# overhead of reconstructing the object (embedding model init, JSONB config,
# etc.) on every search call.
# ---------------------------------------------------------------------------
_vector_store_cache: dict[str, PGVector] = {}


def get_vector_store(collection_name: str | None = None) -> PGVector:
    """Returns a cached asynchronous PGVector instance for the given collection.

    The first call for a given collection name constructs and caches the
    instance; subsequent calls return the cached object instantly.

    Args:
        collection_name: Optional collection name override (defaults to settings.RAG_COLLECTION_NAME).
    """
    target_collection = collection_name or settings.RAG_COLLECTION_NAME

    if target_collection not in _vector_store_cache:
        embeddings = get_embeddings()
        _vector_store_cache[target_collection] = PGVector(
            embeddings=embeddings,
            collection_name=target_collection,
            connection=engine,
            async_mode=True,
            embedding_length=settings.EMBEDDING_DIMENSIONS,
            use_jsonb=True,
            create_extension=True,
        )
        logger.debug("PGVector instance created and cached for collection '%s'", target_collection)

    return _vector_store_cache[target_collection]


def get_vector_store_retriever(
    collection_name: str | None = None,
    search_type: str = "similarity",
    **search_kwargs: Any,
):
    """Returns a native VectorStoreRetriever wrapping the PGVector instance.

    Compatible with standard LangChain Runnable retriever workflows:
        retriever = get_vector_store_retriever(search_kwargs={"k": 4})
        docs = await retriever.ainvoke("query")
    """
    store = get_vector_store(collection_name=collection_name)
    return store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


async def setup_vector_store() -> None:
    """Ensures HNSW and GIN indexes exist on the underlying langchain_pg_embedding table."""
    try:
        # Trigger table initialization
        get_vector_store()
        # Ping collection creation if not yet present
        async with engine.begin() as conn:
            # Create extension if not present
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

            # Ensure HNSW index on vector embeddings for sub-millisecond cosine search
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_langchain_pg_embedding_hnsw 
                ON langchain_pg_embedding USING hnsw (embedding vector_cosine_ops);
            """))

            # Ensure GIN index for native PostgreSQL Full-Text Search on document content
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_langchain_pg_embedding_fts 
                ON langchain_pg_embedding USING GIN(to_tsvector('english', document));
            """))

        logger.info("PGVector HNSW and GIN indexes verified successfully.")
    except Exception as exc:
        logger.error("Failed to setup PGVector indexes: %s", exc, exc_info=True)


async def index_documents(
    documents: list[Document],
    collection_name: str | None = None,
) -> list[str]:
    """Asynchronously adds a list of Document objects to the vector store.

    Args:
        documents: List of langchain_core.documents.Document chunks.
        collection_name: Optional target collection.

    Returns:
        List of generated chunk IDs.
    """
    if not documents:
        return []

    store = get_vector_store(collection_name=collection_name)
    ids = await store.aadd_documents(documents)
    logger.info("Indexed %d documents into PGVector collection '%s'", len(ids), store.collection_name)
    return ids


async def delete_documents_by_metadata(
    key: str,
    value: Any,
    collection_name: str | None = None,
) -> int:
    """Deletes all chunks matching a specific metadata key-value pair.

    Args:
        key: The JSON metadata field name (e.g. 'document_id', 'source', 'user_id').
        value: The matching value.
        collection_name: Optional collection name to scope deletion.

    Returns:
        Number of deleted chunk records.
    """
    target_collection = collection_name or settings.RAG_COLLECTION_NAME
    stmt = text("""
        DELETE FROM langchain_pg_embedding
        WHERE cmetadata->>:key = :val
          AND collection_id = (
              SELECT uuid FROM langchain_pg_collection WHERE name = :col_name
          );
    """)

    async with engine.begin() as conn:
        result = await conn.execute(
            stmt,
            {"key": key, "val": str(value), "col_name": target_collection},
        )
        deleted_count = result.rowcount
        logger.info(
            "Deleted %d chunks matching cmetadata->>'%s' = '%s' from collection '%s'",
            deleted_count,
            key,
            value,
            target_collection,
        )
        return deleted_count
