"""Wipes all test data from the database and re-seeds the canonical root user.

Usage:
  python scripts/reset_db_and_seed_root.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from app.core.database import async_session_factory
from app.models.user import User
from app.services.agent_service import get_or_create_default_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TABLES_TO_TRUNCATE = [
    "workflow_runs",
    "workflows",
    "tools",
    "structured_schemas",
    "document_chunks",
    "documents",
    "thread_assets",
    "threads",
    "agents",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoints",
    "langchain_pg_embedding",
    "langchain_pg_collection",
    "lightrag_doc_chunks",
    "lightrag_doc_full",
    "lightrag_doc_status",
    "lightrag_entity_chunks",
    "lightrag_full_entities",
    "lightrag_full_relations",
    "lightrag_graph_edges",
    "lightrag_graph_nodes",
    "lightrag_llm_cache",
    "lightrag_relation_chunks",
    "lightrag_vdb_chunks_text_embedding_3_small_1536d",
    "lightrag_vdb_entity_text_embedding_3_small_1536d",
    "lightrag_vdb_relation_text_embedding_3_small_1536d",
    "taskiq_messages",
    "taskiq_results",
    "users",
]


async def reset_database():
    logger.info("Starting clean database reset...")

    async with async_session_factory() as session:
        # Check which tables exist in public schema
        r = await session.execute(
            text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
        )
        existing_tables = {row[0] for row in r.fetchall()}

        tables_to_clear = [t for t in TABLES_TO_TRUNCATE if t in existing_tables]

        if tables_to_clear:
            tables_str = ", ".join(f'"{t}"' for t in tables_to_clear)
            logger.info("Truncating tables with CASCADE: %s", tables_str)
            await session.execute(text(f"TRUNCATE TABLE {tables_str} CASCADE;"))
            await session.commit()
            logger.info("Successfully truncated %d tables.", len(tables_to_clear))

        # Re-seed canonical root user
        logger.info("Seeding canonical root user...")
        root_user = User(
            id="root",
            email="root@baseagent.ai",
            name="Administrator",
            role="root",
            is_active=True,
            metadata_json={"system": True, "description": "System superuser account"},
        )
        session.add(root_user)
        await session.commit()
        await session.refresh(root_user)
        logger.info("Root user seeded: id=%s, email=%s", root_user.id, root_user.email)

        # Seed canonical default agent for root
        logger.info("Seeding default agent for root...")
        default_agent = await get_or_create_default_agent(session, "root")
        logger.info(
            "Default agent seeded: id=%s, name=%s, is_default=%s",
            default_agent.id,
            default_agent.name,
            default_agent.is_default,
        )

        # Verify state
        logger.info("\n--- Post-Reset Verification ---")
        for t in ["users", "agents", "documents", "threads", "workflows", "tools", "structured_schemas", "checkpoints"]:
            if t in existing_tables:
                cnt_res = await session.execute(text(f'SELECT count(*) FROM "{t}"'))
                logger.info("  %s: %d row(s)", t, cnt_res.scalar())

    logger.info("\nDatabase reset complete! The system is clean and ready.")


if __name__ == "__main__":
    asyncio.run(reset_database())

