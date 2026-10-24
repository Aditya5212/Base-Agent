"""Unit tests for hybrid retriever and Reciprocal Rank Fusion (RRF) logic."""

from langchain_core.documents import Document

from app.core.rag_config import get_rag_config
from app.services.retriever import apply_reciprocal_rank_fusion


def test_rrf_scoring_and_boost():
    cfg = get_rag_config("balanced")

    doc_a = Document(page_content="Common doc appearing in both dense and sparse", metadata={"id": "doc_a"})
    doc_b = Document(page_content="Doc appearing only in dense search", metadata={"id": "doc_b"})
    doc_c = Document(page_content="Doc appearing only in sparse search", metadata={"id": "doc_c"})

    dense_results = [
        (doc_a, 0.95),  # Rank 1 dense
        (doc_b, 0.85),  # Rank 2 dense
    ]

    sparse_results = [
        (doc_a, 0.90),  # Rank 1 sparse
        (doc_c, 0.70),  # Rank 2 sparse
    ]

    fused = apply_reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        config=cfg,
    )

    # Document A appears in both dense and sparse, so its RRF score must be highest
    assert len(fused) == 3
    assert fused[0].metadata["id"] == "doc_a"
    assert fused[0].metadata["rrf_score"] > fused[1].metadata["rrf_score"]
    assert fused[0].metadata["dense_rank"] == 1
    assert fused[0].metadata["sparse_rank"] == 1


def test_rrf_deduplication():
    cfg = get_rag_config("balanced", top_k_final=2)

    doc_1 = Document(page_content="Duplicate chunk text", metadata={"id": "doc_1"})
    doc_2 = Document(page_content="Unique chunk text", metadata={"id": "doc_2"})

    dense = [(doc_1, 0.9)]
    sparse = [(doc_1, 0.8), (doc_2, 0.7)]

    fused = apply_reciprocal_rank_fusion(dense, sparse, config=cfg)
    assert len(fused) == 2
    # Ensure doc_1 is not repeated
    unique_ids = {d.metadata["id"] for d in fused}
    assert len(unique_ids) == 2
