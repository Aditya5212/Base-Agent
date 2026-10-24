"""Unit tests for token-based Markdown text splitter."""

from langchain_core.documents import Document

from app.core.rag_config import RAGPreset, get_rag_config
from app.services.text_splitter import count_tokens, split_markdown_document


def test_count_tokens():
    text = "Hello world! This is a test sentence."
    tokens = count_tokens(text)
    assert tokens > 0
    assert count_tokens("") == 0


def test_split_markdown_headers_preserved():
    md = """# Architecture
Base Agent uses FastAPI and LangGraph.

## Database
Neon PostgreSQL provides pgvector and FTS.

### Vector Search
HNSW index enables low-latency cosine similarity.
"""
    doc = Document(page_content=md, metadata={"source": "arch.md", "user_id": "u1"})
    chunks = split_markdown_document(doc)

    assert len(chunks) >= 3
    # First chunk should have header_1
    assert any(c.metadata.get("header_1") == "Architecture" for c in chunks)
    # Child section should have header_1 and header_2
    assert any(c.metadata.get("header_2") == "Database" for c in chunks)
    # Every chunk must retain parent metadata
    for c in chunks:
        assert c.metadata["source"] == "arch.md"
        assert c.metadata["user_id"] == "u1"
        assert "chunk_index" in c.metadata
        assert "token_count" in c.metadata
        assert c.metadata["token_count"] > 0


def test_split_presets_bounds():
    long_text = ("This is a repetitive line of text designed to test token limits. " * 50)
    doc = Document(page_content=long_text, metadata={"source": "long.txt"})

    cfg_precision = get_rag_config(RAGPreset.HIGH_PRECISION)
    chunks_precision = split_markdown_document(doc, config=cfg_precision)

    cfg_recall = get_rag_config(RAGPreset.HIGH_RECALL)
    chunks_recall = split_markdown_document(doc, config=cfg_recall)

    # High precision (smaller chunks) should yield more chunks than high recall (larger chunks)
    assert len(chunks_precision) >= len(chunks_recall)
