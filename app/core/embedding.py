"""OpenAI embedding model configuration and factory.

Decoupled strictly from conversational chat models:
- Embedding model: OpenAI text-embedding-3-small (1536 dimensions)
- Fallback: DeterministicFakeEmbedding for testing / development without API keys
"""

from functools import lru_cache
import logging

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1536


@lru_cache
def get_embeddings(
    model: str | None = None,
    dimensions: int | None = None,
) -> Embeddings:
    """Returns an Embeddings client instance.

    Uses OpenAI text-embedding-3-small when OPENAI_API_KEY is configured.
    Falls back to DeterministicFakeEmbedding for automated tests and offline development.

    Args:
        model: Optional model name override (defaults to settings.EMBEDDING_MODEL).
        dimensions: Vector dimensionality (defaults to 1536).
    """
    model_name = model or settings.EMBEDDING_MODEL or DEFAULT_EMBEDDING_MODEL
    dims = dimensions or settings.EMBEDDING_DIMENSIONS or DEFAULT_EMBEDDING_DIMENSIONS
    api_key = settings.OPENAI_API_KEY

    if api_key:
        logger.info("Initializing OpenAIEmbeddings with model=%s, dimensions=%d", model_name, dims)
        return OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
            dimensions=dims,
        )

    logger.warning(
        "OPENAI_API_KEY not configured. Falling back to DeterministicFakeEmbedding(size=%d) "
        "for development / testing.",
        dims,
    )
    return DeterministicFakeEmbedding(size=dims)
