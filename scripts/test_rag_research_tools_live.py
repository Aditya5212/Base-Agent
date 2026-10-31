"""Live interactive verification script for the Modular RAG Research Tool Suite.

Demonstrates:
1. Exploitation: hybrid_document_search (targeted single query) & graph_rag_search (LightRAG knowledge graph)
2. Exploration: multi_query_search (agent provides multiple diverse query angles executed in parallel)
3. Agentic Deep Research: Agent orchestrates multi-query search, hybrid search, and graph traversal
4. 3-Tier Modularity:
   - Chatbot Level: Pre-bound tool execution
   - Agent Node Level: resolve_agent_tools from AgentConfig
   - Workflow Level: ToolCallNode & RetrieveNode execution
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import sys

# Ensure repository root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sqlalchemy import text
from app.core.database import engine
from app.schemas.agent import AgentConfig, ToolSpec
from app.services.agent_service import resolve_agent_tools
from app.services.rag_tools import (
    get_user_rag_tool_map,
    get_user_rag_tools,
    make_graph_rag_tool,
    make_hybrid_search_tool,
    make_multi_query_search_tool,
)
from app.services.workflow.nodes.retrieve import build_retrieve_node
from app.services.workflow.nodes.tool_call import build_tool_call_node

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rag_tools_live")


async def main():
    print("=" * 80)
    print("🎯 BASE AGENT — LIVE MODULAR RAG RESEARCH TOOL SUITE VERIFICATION")
    print("=" * 80)

    # 1. Verify DB Connectivity
    print("\n[Step 1] Testing live PostgreSQL connection...")
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        print("✅ PostgreSQL connected successfully.")
    except Exception as exc:
        print(f"❌ Database connection failed: {exc}. Aborting.")
        return

    user_id = "root"

    # 2. Exploitation: Hybrid Document Search (Single query)
    print("\n" + "=" * 80)
    print("[Step 2] EXPLOITATION: Hybrid Document Search (Single Targeted Query)")
    print("=" * 80)
    hybrid_tool = make_hybrid_search_tool(user_id=user_id)
    print(f"Tool Name: {hybrid_tool.name}")
    print(f"Description: {hybrid_tool.description[:120]}...")
    
    query = "database connection pooling architecture"
    print(f"\nQuerying: '{query}' [preset='balanced', top_k=3]...")
    res_hybrid = await hybrid_tool.ainvoke({"query": query, "preset": "balanced", "top_k": 3})
    print("\n--- [Hybrid Search Result] ---")
    print(res_hybrid[:600] + ("..." if len(res_hybrid) > 600 else ""))

    # 3. Exploitation: Knowledge Graph Search (LightRAG)
    print("\n" + "=" * 80)
    print("[Step 3] EXPLOITATION: Knowledge Graph Traversal (LightRAG)")
    print("=" * 80)
    graph_tool = make_graph_rag_tool(user_id=user_id)
    print(f"Tool Name: {graph_tool.name}")
    print(f"Querying graph: '{query}' [mode='mix']...")
    res_graph = await graph_tool.ainvoke({"query": query, "mode": "mix", "top_k": 3})
    print("\n--- [Knowledge Graph Result] ---")
    print(res_graph[:600] + ("..." if len(res_graph) > 600 else ""))

    # 4. Exploration: Multi-Query Search (Agent-Provided Queries Array)
    print("\n" + "=" * 80)
    print("[Step 4] EXPLORATION: Multi-Query Search (Agent Provides Multiple Angles in Parallel)")
    print("=" * 80)
    mq_tool = make_multi_query_search_tool(user_id=user_id)
    print(f"Tool Name: {mq_tool.name}")
    agent_queries = [
        "database connection pooling architecture",
        "connection pool sizing and concurrency limits",
        "asyncpg vs psycopg3 connection pool tradeoffs",
    ]
    print(f"Agent executing parallel search with {len(agent_queries)} distinct queries:")
    for q in agent_queries:
        print(f"  • {q}")

    res_mq = await mq_tool.ainvoke({"queries": agent_queries, "preset": "balanced", "top_k": 5})
    print("\n--- [Multi-Query Search Result] ---")
    print(res_mq[:800] + ("..." if len(res_mq) > 800 else ""))

    # 5. Agentic Deep Research Orchestration Pattern
    print("\n" + "=" * 80)
    print("[Step 5] AGENTIC DEEP RESEARCH: Orchestrated by Agent via Multi-Query + Graph Tools")
    print("=" * 80)
    print("Demonstrating how an autonomous agent or multi-step workflow orchestrates deep research:")
    print("  1. Agent formulates exploratory query set across angles -> calls multi_query_search")
    print("  2. Agent queries entity relationship network -> calls graph_rag_search")
    print("  3. Agent synthesizes findings into a unified research report.")
    
    # Run in parallel as an agent would
    mq_task = mq_tool.ainvoke({"queries": agent_queries, "top_k": 4})
    graph_task = graph_tool.ainvoke({"query": query, "mode": "mix", "top_k": 3})
    doc_evidence, graph_evidence = await asyncio.gather(mq_task, graph_task)

    print("\n--- [Agentic Synthesis Dossier Preview] ---")
    print(f"Document Evidence Length: {len(doc_evidence)} chars")
    print(f"Graph Evidence Length: {len(graph_evidence)} chars")
    print("✅ Agent successfully orchestrated multi-query and graph tools concurrently.")

    # 6. Tier 2: Agent Node Level (resolve_agent_tools)
    print("\n" + "=" * 80)
    print("[Step 6] TIER 2: Agent Node Level Dynamic Tool Binding")
    print("=" * 80)
    agent_cfg = AgentConfig(
        tools=[
            ToolSpec(tool_id="all_rag_tools", enabled=True),
        ]
    )
    resolved_tools = resolve_agent_tools(agent_cfg, user_id=user_id)
    print(f"Resolved {len(resolved_tools)} tools for AgentConfig:")
    for t in resolved_tools:
        print(f" - {t.name}: {t.description[:80]}...")

    # 7. Tier 3: Workflow Engine Level (ToolCallNode & RetrieveNode)
    print("\n" + "=" * 80)
    print("[Step 7] TIER 3: Workflow Engine ToolCallNode & RetrieveNode Execution")
    print("=" * 80)
    tool_node_handler = build_tool_call_node(
        "node_rag_tool",
        {
            "tool_name": "multi_query_search",
            "input_mapping": {"queries": agent_queries},
            "output_variable": "workflow_mq_result",
        },
    )
    workflow_state = {
        "user_id": user_id,
        "variables": {},
        "node_history": [],
    }
    wf_res = await tool_node_handler(workflow_state)
    print(f"ToolCallNode executed successfully. Variables populated: {list(wf_res['variables'].keys())}")
    print(f"Node history: {wf_res['node_history']}")

    retrieve_node_handler = build_retrieve_node(
        "node_retrieve_mq",
        {
            "provider_name": "multi_query",
            "query_template": "{{variables.queries}}",
            "top_k": 3,
            "output_variable": "retrieved_context",
            "sources_variable": "retrieved_sources",
        },
    )
    wf_state_2 = {
        "user_id": user_id,
        "variables": {"queries": "connection pool architecture\npool sizing tradeoffs"},
        "node_history": [],
    }
    ret_res = await retrieve_node_handler(wf_state_2)
    print(f"RetrieveNode executed successfully. Variables populated: {list(ret_res['variables'].keys())}")
    print(f"Sources count: {len(ret_res['variables'].get('retrieved_sources', []))}")

    print("\n" + "=" * 80)
    print("🎉 ALL MODULAR RAG RESEARCH TOOLS & 3-TIER INTEGRATIONS VERIFIED LIVE!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
