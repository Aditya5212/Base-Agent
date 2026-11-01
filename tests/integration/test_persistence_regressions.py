"""Requires --run-integration and an explicitly disposable TEST_DATABASE_URL."""
import asyncio
import uuid

import pytest
from fastapi import HTTPException
from langchain_core.documents import Document
from sqlalchemy import text

from app.core.database import AsyncSessionLocal
from app.schemas.workflow import WorkflowCreate, WorkflowRunRequest, WorkflowResumeRequest
from app.services.document_lock import document_lock
from app.services.memory import close_checkpointer_pool, setup_checkpointer
from app.services.retriever import hybrid_search
from app.services.vector_store import index_documents, delete_documents_by_metadata
from app.services import workflow_service as svc


@pytest.mark.asyncio
async def test_durable_tool_approval_resume_after_pool_restart_uses_snapshot():
    await close_checkpointer_pool()
    await setup_checkpointer()
    user = f"persist_{uuid.uuid4().hex}"
    payload = WorkflowCreate.model_validate({
        "name": "Approval persistence regression",
        "nodes": [{"id": "s", "type": "start"},
                  {"id": "t", "type": "tool_call", "config": {"tool_name": "calculator", "requires_approval": True, "input_mapping": {"expression": "2+2"}}},
                  {"id": "e", "type": "end", "config": {"response_variable": "tool_result"}}],
        "edges": [{"source": "s", "target": "t"}, {"source": "t", "target": "e"}],
    })
    run = None
    try:
        async with AsyncSessionLocal() as db:
            wf = await svc.create_workflow(db, user, payload)
            run = await svc.execute_workflow(db, wf.id, user, WorkflowRunRequest(thread_id="client-thread"))
            assert run.status == "INTERRUPTED"
            assert run.output_data is None
            assert run.execution_log[-1]["interrupts"][0]["value"]["tool"] == "calculator"
            wf.nodes = [*wf.nodes[:1], {**wf.nodes[1], "config": {**wf.nodes[1]["config"], "input_mapping": {"expression": "100+100"}}}, wf.nodes[2]]
            wf.version += 1
            await db.commit()
        await close_checkpointer_pool()
        await setup_checkpointer()

        async def resume():
            async with AsyncSessionLocal() as db:
                return await svc.resume_workflow_run(db, run.id, user, WorkflowResumeRequest(resume_data={"approved": True}))
        outcomes = await asyncio.gather(resume(), resume(), return_exceptions=True)
        success = [item for item in outcomes if not isinstance(item, Exception)]
        failure = [item for item in outcomes if isinstance(item, Exception)]
        assert len(success) == 1
        assert success[0].status == "COMPLETED"
        assert success[0].output_data == {"tool_result": 4}
        assert len(failure) == 1 and isinstance(failure[0], HTTPException) and failure[0].status_code == 400
    finally:
        if run is not None:
            from app.services.memory import get_checkpointer
            await (await get_checkpointer()).adelete_thread(run.checkpoint_id)
        await close_checkpointer_pool()


@pytest.mark.asyncio
async def test_document_lock_excludes_other_database_connections_and_releases():
    key = str(uuid.uuid4())
    async with document_lock(key) as first:
        assert first
        async with document_lock(key) as second:
            assert second is False
    async with document_lock(key) as after_release:
        assert after_release is True


@pytest.mark.asyncio
async def test_vector_retries_upsert_same_chunk_and_hybrid_search_is_scoped():
    owner = f"vector_{uuid.uuid4().hex}"
    doc_id, chunk_id = str(uuid.uuid4()), str(uuid.uuid4())
    chunk = Document(page_content="Cobalt otters study checkpoint persistence.",
        metadata={"document_id": doc_id, "chunk_id": chunk_id, "user_id": owner, "source": "regression.txt"})
    try:
        assert await index_documents([chunk]) == [chunk_id]
        assert await index_documents([chunk]) == [chunk_id]
        async with AsyncSessionLocal() as db:
            count = (await db.execute(text("SELECT count(*) FROM langchain_pg_embedding WHERE id=:id"), {"id": chunk_id})).scalar_one()
        assert count == 1
        found = await hybrid_search("cobalt otters", filter_dict={"user_id": owner})
        assert len(found) == 1
        assert await hybrid_search("cobalt otters", filter_dict={"user_id": owner + "_other"}) == []
    finally:
        await delete_documents_by_metadata("document_id", doc_id)
