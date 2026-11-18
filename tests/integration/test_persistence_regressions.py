"""Requires --run-integration and an explicitly disposable TEST_DATABASE_URL."""
import asyncio
import uuid

import pytest
from fastapi import HTTPException
from langchain_core.documents import Document
from sqlalchemy import text

from app.core.database import AsyncSessionLocal
from app.models.document import Document as DBDocument, DocumentStatus
from app.models.thread import Thread, ThreadAsset
from app.schemas.workflow import WorkflowCreate, WorkflowRunRequest, WorkflowResumeRequest
from app.services.document_lock import document_lock
from app.services.document_processor import process_document_pipeline
from app.services.memory import close_checkpointer_pool, setup_checkpointer
from app.services.retriever import hybrid_search
from app.services.thread_service import delete_thread_cascade, verify_thread_admission, get_or_create_user
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
        assert success[0].status == "PENDING"
        async with AsyncSessionLocal() as db:
            assert await svc.claim_workflow_run(db, run.id, "approval-worker")
            completed = await svc.execute_claimed_run(db, run.id, "approval-worker")
            assert completed.status == "COMPLETED"
            assert completed.output_data == {"tool_result": 4}
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


@pytest.mark.asyncio
async def test_concurrent_upload_admission_and_thread_deletion_locking():
    """Verifies row-level locking serialization between admission (FOR SHARE) and deletion (FOR UPDATE)."""
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    thread_id = f"thread_{uuid.uuid4().hex[:8]}"

    async with AsyncSessionLocal() as db:
        await get_or_create_user(db, user_id=user_id)
        thread = Thread(id=thread_id, user_id=user_id, title="Locking Integration Test")
        db.add(thread)
        await db.commit()

    upload_lock_acquired = asyncio.Event()
    delete_started = asyncio.Event()
    admission_can_proceed = asyncio.Event()

    async def simulate_upload_admission():
        async with AsyncSessionLocal() as db:
            # Step 1: Acquire FOR SHARE admission lock
            th = await verify_thread_admission(db, thread_id=thread_id, user_id=user_id, for_share=True)
            assert th.status == "ACTIVE"
            upload_lock_acquired.set()
            # Wait until deletion has attempted to run
            await admission_can_proceed.wait()
            # Insert document under the lock
            doc = DBDocument(
                id=uuid.uuid4(),
                user_id=user_id,
                thread_id=thread_id,
                file_name="admitted.md",
                r2_key=f"documents/{user_id}/threads/{thread_id}/admitted.md",
                mime_type="text/markdown",
                file_size=100,
                status=DocumentStatus.INDEXED,
            )
            db.add(doc)
            await db.commit()

    async def simulate_deletion():
        await upload_lock_acquired.wait()
        async with AsyncSessionLocal() as db:
            delete_started.set()
            with patch("app.services.r2_storage.r2_storage_service.delete_file", new_callable=AsyncMock), \
                 patch("app.services.vector_store.delete_documents_by_metadata", new_callable=AsyncMock), \
                 patch("app.services.memory.delete_thread_memory", new_callable=AsyncMock, return_value=True), \
                 patch("app.core.rag.factory.get_provider", new_callable=AsyncMock), \
                 patch("app.services.document_lock.document_lock") as mock_lock:

                mock_ctx = AsyncMock()
                mock_ctx.__aenter__.return_value = True
                mock_ctx.__aexit__.return_value = None
                mock_lock.return_value = mock_ctx

                # This call must block until simulate_upload_admission commits its doc
                await delete_thread_cascade(db, thread_id=thread_id, user_id=user_id)

    # Launch deletion concurrently while upload holds admission lock
    upload_task = asyncio.create_task(simulate_upload_admission())
    delete_task = asyncio.create_task(simulate_deletion())

    # Wait for deletion to start waiting for the lock
    await delete_started.wait()
    await asyncio.sleep(0.05)
    # Release upload to commit
    admission_can_proceed.set()

    await asyncio.gather(upload_task, delete_task)

    # After both finish: thread and its admitted document are cleanly purged
    async with AsyncSessionLocal() as db:
        remaining_th = await db.get(Thread, thread_id)
        assert remaining_th is None
        doc_count = (await db.execute(text("SELECT count(*) FROM documents WHERE thread_id=:tid"), {"tid": thread_id})).scalar_one()
        assert doc_count == 0


@pytest.mark.asyncio
async def test_ingestion_aborted_when_thread_marked_deleting_in_database():
    """Verifies background ingestion task aborts and marks document FAILED when parent thread is DELETING."""
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    thread_id = f"thread_{uuid.uuid4().hex[:8]}"
    doc_id = uuid.uuid4()
    r2_key = f"documents/{user_id}/threads/{thread_id}/aborted.txt"

    async with AsyncSessionLocal() as db:
        await get_or_create_user(db, user_id=user_id)
        thread = Thread(id=thread_id, user_id=user_id, title="Deleting Thread", metadata_json={"status": "DELETING"})
        db.add(thread)
        doc = DBDocument(
            id=doc_id,
            user_id=user_id,
            thread_id=thread_id,
            file_name="aborted.txt",
            r2_key=r2_key,
            mime_type="text/plain",
            file_size=50,
            status=DocumentStatus.PARSING,
        )
        db.add(doc)
        await db.commit()

    try:
        # Run process_document_pipeline directly
        res = await process_document_pipeline(
            document_id=str(doc_id),
            r2_key=r2_key,
            user_id=user_id,
            file_name="aborted.txt",
            file_bytes=b"Sample content for aborted ingestion.",
            preset="balanced",
            index_graph=False,
            thread_id=thread_id,
        )
        assert res == 0

        async with AsyncSessionLocal() as db:
            updated_doc = await db.get(DBDocument, doc_id)
            assert updated_doc is not None
            assert updated_doc.status == DocumentStatus.FAILED
            assert "Parent thread is being deleted" in (updated_doc.error_message or "")
    finally:
        async with AsyncSessionLocal() as db:
            await db.execute(text("DELETE FROM documents WHERE id=:id"), {"id": str(doc_id)})
            await db.execute(text("DELETE FROM threads WHERE id=:id"), {"id": thread_id})
            await db.commit()


@pytest.mark.asyncio
async def test_cleanup_failure_preserves_records_for_safe_retry_in_database():
    """Verifies failure during cascade deletion preserves DB records, enabling subsequent retry to succeed."""
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    thread_id = f"thread_{uuid.uuid4().hex[:8]}"
    doc_id = uuid.uuid4()
    asset_id = uuid.uuid4()

    async with AsyncSessionLocal() as db:
        await get_or_create_user(db, user_id=user_id)
        thread = Thread(id=thread_id, user_id=user_id, title="Retry Thread")
        db.add(thread)
        doc = DBDocument(
            id=doc_id,
            user_id=user_id,
            thread_id=thread_id,
            file_name="retry_doc.md",
            r2_key=f"documents/{user_id}/threads/{thread_id}/retry_doc.md",
            mime_type="text/markdown",
            file_size=120,
            status=DocumentStatus.INDEXED,
        )
        db.add(doc)
        asset = ThreadAsset(
            id=asset_id,
            thread_id=thread_id,
            user_id=user_id,
            asset_type="image",
            file_name="diagram.png",
            r2_key=f"threads/{thread_id}/diagram.png",
            mime_type="image/png",
            file_size=5000,
        )
        db.add(asset)
        await db.commit()

    # Attempt 1: R2 service failure
    async with AsyncSessionLocal() as db:
        with patch("app.services.r2_storage.r2_storage_service.delete_file", new_callable=AsyncMock) as mock_r2_del, \
             patch("app.services.vector_store.delete_documents_by_metadata", new_callable=AsyncMock), \
             patch("app.services.memory.delete_thread_memory", new_callable=AsyncMock, return_value=True), \
             patch("app.core.rag.factory.get_provider", new_callable=AsyncMock), \
             patch("app.services.document_lock.document_lock") as mock_lock:

            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = True
            mock_ctx.__aexit__.return_value = None
            mock_lock.return_value = mock_ctx
            mock_r2_del.side_effect = RuntimeError("R2 Service Unavailable")

            with pytest.raises(RuntimeError, match="Records preserved for retry"):
                await delete_thread_cascade(db, thread_id=thread_id, user_id=user_id)

    # Verify records were preserved and thread marked DELETING
    async with AsyncSessionLocal() as db:
        th = await db.get(Thread, thread_id)
        assert th is not None
        assert th.status == "DELETING"
        d = await db.get(DBDocument, doc_id)
        assert d is not None
        a = await db.get(ThreadAsset, asset_id)
        assert a is not None

    # Attempt 2: Retry succeeds
    async with AsyncSessionLocal() as db:
        with patch("app.services.r2_storage.r2_storage_service.delete_file", new_callable=AsyncMock, return_value=True), \
             patch("app.services.vector_store.delete_documents_by_metadata", new_callable=AsyncMock, return_value=1), \
             patch("app.services.memory.delete_thread_memory", new_callable=AsyncMock, return_value=True), \
             patch("app.core.rag.factory.get_provider", new_callable=AsyncMock), \
             patch("app.services.document_lock.document_lock") as mock_lock:

            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = True
            mock_ctx.__aexit__.return_value = None
            mock_lock.return_value = mock_ctx

            success = await delete_thread_cascade(db, thread_id=thread_id, user_id=user_id)
            assert success is True

    # Verify records are now completely removed
    async with AsyncSessionLocal() as db:
        th_final = await db.get(Thread, thread_id)
        assert th_final is None
        d_final = await db.get(DBDocument, doc_id)
        assert d_final is None
        a_final = await db.get(ThreadAsset, asset_id)
        assert a_final is None
