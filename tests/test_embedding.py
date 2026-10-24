"""Unit tests for decoupled OpenAI embedding module."""

from unittest.mock import patch

from app.core.embedding import get_embeddings


def test_get_embeddings_fallback():
    # Without OPENAI_API_KEY, should fall back to DeterministicFakeEmbedding
    with patch("app.core.embedding.settings.OPENAI_API_KEY", None):
        get_embeddings.cache_clear()
        embedder = get_embeddings()
        vec = embedder.embed_query("Test query")
        assert len(vec) == 1536
        assert isinstance(vec[0], float)


def test_get_embeddings_dimensions():
    with patch("app.core.embedding.settings.OPENAI_API_KEY", None):
        get_embeddings.cache_clear()
        embedder = get_embeddings(dimensions=1536)
        docs = ["Doc 1", "Doc 2"]
        vectors = embedder.embed_documents(docs)
        assert len(vectors) == 2
        assert len(vectors[0]) == 1536
        assert len(vectors[1]) == 1536
