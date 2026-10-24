"""Interactive script testing live RAG ingestion, PGVector + FTS hybrid search, and RRF.

Usage:
  uv run python -m scripts.test_rag_live
"""

import asyncio
import uuid

from langchain_core.documents import Document

from app.core.rag_config import RAGPreset, get_rag_config
from app.services.document_loader import load_document
from app.services.retriever import hybrid_search
from app.services.text_splitter import split_markdown_document
from app.services.vector_store import delete_documents_by_metadata, index_documents, setup_vector_store


SAMPLE_DOCUMENT = """# LangGraph Architecture Overview

LangGraph is a library designed by LangChain for building stateful, multi-actor applications with LLMs.

## Core Concepts
LangGraph models agent workflows as graphs:
- **State**: The shared memory or state object passed between nodes.
- **Nodes**: Python functions or runnables that receive the state and return updates.
- **Edges**: Direct transitions or conditional routing determining the next node to execute.

## Persistent Checkpointing
LangGraph supports durable persistence using checkpointers such as PostgreSQL AsyncPostgresSaver.
State snapshots are committed after every super-step, allowing threads to resume anytime.

# PostgreSQL and pgvector for Hybrid Search

Modern retrieval combines dense vector embeddings with sparse keyword search.
- **pgvector**: Stores high-dimensional embedding vectors (e.g. 1536-dim OpenAI text-embedding-3-small) using an HNSW index.
- **PostgreSQL Full-Text Search (FTS)**: Uses tsvector and GIN indexes to rank keyword matches with ts_rank_cd.
- **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse ranks to provide superior precision and recall.
"""


async def main():
    doc_id = f"doc-live-{uuid.uuid4().hex[:6]}"
    user_id = "aditya_live"
    print(f"=== Starting Live RAG & Hybrid Search Test [Doc ID: {doc_id}] ===")

    # Step 1: Ensure indexes
    print("\n1. Verifying PGVector and FTS indexes in PostgreSQL...")
    await setup_vector_store()

    # Step 2: Ingest via MarkItDown loader
    print("\n2. Loading sample document with MarkItDown...")
    doc = load_document(
        source=SAMPLE_DOCUMENT.encode("utf-8"),
        filename="langgraph_and_rag_guide.md",
        extra_metadata={"document_id": doc_id, "user_id": user_id},
    )
    print(f"Loaded document size: {len(doc.page_content)} characters.")

    # Step 3: Chunk using tiktoken
    print("\n3. Splitting into token-bounded chunks (Balanced preset)...")
    cfg = get_rag_config(RAGPreset.BALANCED)
    chunks = split_markdown_document(doc, config=cfg)
    print(f"Created {len(chunks)} chunks:")
    for idx, c in enumerate(chunks):
        headers = [v for k, v in c.metadata.items() if k.startswith("header_")]
        header_str = " > ".join(headers) if headers else "Root"
        print(f"  Chunk {idx} [{c.metadata['token_count']} tokens | Section: {header_str}]:")
        print(f"    Preview: {c.page_content[:70].strip()}...")

    # Step 4: Index into PGVector
    print("\n4. Indexing chunks into PGVector...")
    chunk_ids = await index_documents(chunks)
    print(f"Indexed {len(chunk_ids)} chunk embeddings into PostgreSQL.")

    # Step 5: Test Hybrid Search
    queries = [
        "What are nodes and edges in LangGraph?",
        "How does pgvector HNSW index work with PostgreSQL FTS?",
    ]

    for q in queries:
        print(f"\n5. Executing Hybrid Search (Dense + FTS + RRF) for: '{q}'")
        fused_results = await hybrid_search(
            query=q,
            config=cfg,
            filter_dict={"user_id": user_id},
        )
        print(f"Retrieved {len(fused_results)} fused chunks:")
        for r_idx, res in enumerate(fused_results, 1):
            print(f"  [{r_idx}] RRF Score: {res.metadata.get('rrf_score')} "
                  f"(Dense Rank: {res.metadata.get('dense_rank')}, Sparse Rank: {res.metadata.get('sparse_rank')})")
            print(f"      Text: {res.page_content[:100].strip()}...")

    # Step 6: Cleanup
    print("\n6. Cleaning up test document chunks from PostgreSQL...")
    deleted = await delete_documents_by_metadata("document_id", doc_id)
    print(f"Purged {deleted} chunks. Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
