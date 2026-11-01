import uuid

import pytest
from langchain_core.documents import Document

from app.core.database import AsyncSessionLocal
from app.services.document_processor import _activate_generation
from app.services import document_processor as pipeline
from app.models.document import Document as StoredDocument, DocumentStatus
from sqlalchemy import select
from app.models.document import DocumentChunk
from unittest.mock import patch
from app.services.retriever import hybrid_search
from app.services.vector_store import index_documents, delete_documents_by_metadata


@pytest.mark.asyncio
async def test_staged_chunks_are_hidden_and_activation_rollback_preserves_old_index():
    doc_id, owner = str(uuid.uuid4()), "generation_" + uuid.uuid4().hex
    old = Document(page_content="Cobalt generation original.", metadata={"document_id": doc_id, "user_id": owner, "chunk_id": str(uuid.uuid4())})
    new = Document(page_content="Cobalt generation replacement.", metadata={"document_id": doc_id, "user_id": owner, "chunk_id": str(uuid.uuid4()), "_base_generation": "new", "_base_active": False})
    try:
        await index_documents([old])
        await index_documents([new])
        visible = await hybrid_search("cobalt generation", filter_dict={"user_id": owner})
        assert [d.page_content for d in visible] == [old.page_content]
        async with AsyncSessionLocal() as db:
            await _activate_generation(db, doc_id, "new")
            await db.rollback()  # Simulates failure while persisting inspection metadata.
        visible = await hybrid_search("cobalt generation", filter_dict={"user_id": owner})
        assert [d.page_content for d in visible] == [old.page_content]
        async with AsyncSessionLocal() as db:
            await _activate_generation(db, doc_id, "new")
            await db.commit()
        visible = await hybrid_search("cobalt generation", filter_dict={"user_id": owner})
        assert [d.page_content for d in visible] == [new.page_content]
    finally:
        await delete_documents_by_metadata("document_id", doc_id)


@pytest.mark.asyncio
async def test_ingestion_metadata_failure_preserves_visible_generation_and_retry_recovers():
    doc_id = uuid.uuid4()
    owner = "ingest_" + uuid.uuid4().hex
    source = "temporary-source.txt"
    first = b"Cobalt otters source first."
    second = b"Cobalt otters source replacement."
    async with AsyncSessionLocal() as db:
        db.add(StoredDocument(id=doc_id, user_id=owner, file_name=source, r2_key=source,
            mime_type="text/plain", file_size=len(first), status=DocumentStatus.PARSING,
            indexing_config={"preset": "balanced", "index_graph": False}))
        await db.commit()
    try:
        assert await pipeline.process_document_pipeline(str(doc_id), source, owner, source, file_bytes=first) == 1
        async with AsyncSessionLocal() as db:
            record = await db.get(StoredDocument, doc_id)
            old_generation = record.indexing_config["active_generation"]
            record.status = DocumentStatus.PARSING
            await db.commit()
        original_activate = pipeline._activate_generation
        async def failing_activate(*args):
            await original_activate(*args)
            raise ValueError("injected metadata commit failure")
        with patch.object(pipeline, "_activate_generation", failing_activate):
            with pytest.raises(ValueError, match="injected"):
                await pipeline.process_document_pipeline(str(doc_id), source, owner, source, file_bytes=second, is_reindex=True)
        visible = await hybrid_search("cobalt otters", filter_dict={"user_id": owner})
        assert len(visible) == 1 and "source first" in visible[0].page_content
        async with AsyncSessionLocal() as db:
            record = await db.get(StoredDocument, doc_id)
            assert record.status == DocumentStatus.FAILED
            assert record.indexing_config["active_generation"] == old_generation
            pending = record.indexing_config["pending_generation"]
        assert await pipeline.process_document_pipeline(str(doc_id), source, owner, source, file_bytes=second, is_reindex=True) == 1
        visible = await hybrid_search("cobalt otters", filter_dict={"user_id": owner})
        assert len(visible) == 1 and "source replacement" in visible[0].page_content
        async with AsyncSessionLocal() as db:
            record = await db.get(StoredDocument, doc_id)
            assert record.indexing_config["active_generation"] == pending
            chunks = list((await db.execute(select(DocumentChunk).where(DocumentChunk.document_id == doc_id))).scalars())
            assert len(chunks) == 1
    finally:
        await delete_documents_by_metadata("document_id", str(doc_id))
        async with AsyncSessionLocal() as db:
            record = await db.get(StoredDocument, doc_id)
            if record:
                await db.delete(record)
                await db.commit()
