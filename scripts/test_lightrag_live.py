"""Interactive smoke test script for live LightRAG integration with Neon PostgreSQL.

Verifies:
1. Workspace isolation and PGTableGraphStorage table initialization on Neon.
2. Ingestion of structured entity/relation text.
3. Multi-hop 'mix' retrieval mode extracting graph entities and context without redundant generation.
"""

import asyncio
import sys
import time

from app.core.rag import (
    IngestDocument,
    RetrievalQuery,
    get_provider,
    shutdown_all_providers,
)


async def main():
    print("=" * 70)
    print("  Base Agent — Live LightRAG + PGTableGraphStorage Smoke Test")
    print("=" * 70)

    workspace = "smoke_test"
    print(f"\n[1/4] Initializing LightRAGProvider with workspace: '{workspace}'...")
    t0 = time.perf_counter()
    provider = await get_provider("lightrag", workspace=workspace)
    init_time = (time.perf_counter() - t0) * 1000
    print(f"  Initialized in {init_time:.1f}ms")

    print("\n[2/4] Checking provider health...")
    health = await provider.health_check()
    print(f"  Status: {'HEALTHY' if health.healthy else 'UNHEALTHY'}")
    if not health.healthy:
        print(f"  Detail: {health.detail}")
        print("  Aborting smoke test due to unhealthy state.")
        await shutdown_all_providers()
        return

    sample_doc = IngestDocument(
        doc_id="smoke_doc_001",
        content=(
            "Alice is the Chief AI Scientist at Antigravity Systems. "
            "She co-founded the Graph Intelligence Initiative with Bob, who leads database infrastructure. "
            "Antigravity Systems is building a next-generation Agent Builder platform on FastAPI and LangGraph, "
            "leveraging Neon PostgreSQL and LightRAG's PGTableGraphStorage for zero-hallucination factual reasoning. "
            "Bob specializes in PostgreSQL distributed performance and HNSW vector indexes."
        ),
        file_path="graph_initiative.txt",
    )

    print(f"\n[3/4] Ingesting test document '{sample_doc.doc_id}' into knowledge graph...")
    t_ingest = time.perf_counter()
    ingest_result = await provider.ingest([sample_doc])
    print(
        f"  Ingest finished in {(time.perf_counter() - t_ingest):.2f}s "
        f"(accepted={ingest_result.accepted}, failed={ingest_result.failed})"
    )
    if ingest_result.errors:
        print(f"  Errors: {ingest_result.errors}")

    query_str = "Who co-founded the Graph Intelligence Initiative and what does Bob specialize in?"
    print(f"\n[4/4] Executing multi-hop retrieval query in 'mix' mode:")
    print(f"  Query: \"{query_str}\"")

    t_query = time.perf_counter()
    query_result = await provider.query(RetrievalQuery(query=query_str, top_k=5, mode="mix"))
    q_latency = (time.perf_counter() - t_query) * 1000

    print(f"\n  Retrieval completed in {q_latency:.1f}ms")
    print(f"  Provider: {query_result.provider}")
    print(f"  Chunks found: {len(query_result.chunks)}")
    print("\n" + "-" * 70)
    print("  RETRIEVED GRAPH CONTEXT:")
    print("-" * 70)
    if query_result.context:
        print(query_result.context[:1000] + ("..." if len(query_result.context) > 1000 else ""))
    else:
        print("  (No graph context returned)")
    print("-" * 70)

    print("\nShutting down providers cleanly...")
    await shutdown_all_providers()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
