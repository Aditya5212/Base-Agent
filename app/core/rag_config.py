"""RAG configuration, token chunking settings, and precision vs. recall presets."""

import enum
from typing import Any
from pydantic import BaseModel, Field


class RAGPreset(str, enum.Enum):
    """Preset profiles tuning the precision vs. recall tradeoff in RAG retrieval."""

    HIGH_PRECISION = "high_precision"
    BALANCED = "balanced"
    HIGH_RECALL = "high_recall"


# Pre-configured token boundaries and search depths
PRESET_CONFIGS: dict[RAGPreset, dict[str, Any]] = {
    RAGPreset.HIGH_PRECISION: {
        "chunk_size": 200,      # tokens (~800 characters)
        "chunk_overlap": 30,    # tokens
        "top_k_dense": 8,
        "top_k_sparse": 8,
        "top_k_final": 3,
        "vector_weight": 0.65,
        "sparse_weight": 0.35,
        "rrf_k": 60,
        "description": "High precision: smaller chunks and narrower retrieval for strict entity/fact lookup.",
    },
    RAGPreset.BALANCED: {
        "chunk_size": 500,      # tokens (~2000 characters)
        "chunk_overlap": 75,    # tokens
        "top_k_dense": 10,
        "top_k_sparse": 10,
        "top_k_final": 5,
        "vector_weight": 0.60,
        "sparse_weight": 0.40,
        "rrf_k": 60,
        "description": "Balanced: medium chunks and balanced weights for general conversational RAG.",
    },
    RAGPreset.HIGH_RECALL: {
        "chunk_size": 1000,     # tokens (~4000 characters)
        "chunk_overlap": 150,   # tokens
        "top_k_dense": 15,
        "top_k_sparse": 15,
        "top_k_final": 8,
        "vector_weight": 0.50,
        "sparse_weight": 0.50,
        "rrf_k": 60,
        "description": "High recall: larger context windows for document summarization and thematic synthesis.",
    },
}


class RAGConfig(BaseModel):
    """Configurable parameters governing document chunking and hybrid search."""

    preset: RAGPreset = Field(
        default=RAGPreset.BALANCED,
        description="Active precision vs. recall preset profile",
    )
    chunk_size: int = Field(
        default=500,
        ge=50,
        le=4000,
        description="Target chunk size in tokens (tiktoken cl100k_base)",
    )
    chunk_overlap: int = Field(
        default=75,
        ge=0,
        le=1000,
        description="Chunk overlap in tokens",
    )
    top_k_dense: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Candidate chunks retrieved via pgvector dense search",
    )
    top_k_sparse: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Candidate chunks retrieved via PostgreSQL FTS sparse search",
    )
    top_k_final: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Final number of re-ordered chunks after RRF fusion",
    )
    vector_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Relative weight for dense vector similarity in RRF",
    )
    sparse_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Relative weight for sparse FTS keyword match in RRF",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        le=200,
        description="Reciprocal Rank Fusion smoothing parameter k (standard is 60)",
    )


def get_rag_config(
    preset: RAGPreset | str = RAGPreset.BALANCED,
    **overrides: Any,
) -> RAGConfig:
    """Builds a RAGConfig populated from a preset with optional parameter overrides."""
    if isinstance(preset, str):
        try:
            resolved_preset = RAGPreset(preset.lower())
        except ValueError:
            resolved_preset = RAGPreset.BALANCED
    else:
        resolved_preset = preset

    base_values = dict(PRESET_CONFIGS[resolved_preset])
    base_values["preset"] = resolved_preset
    base_values.update(overrides)

    return RAGConfig(**base_values)
