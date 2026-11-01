from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import uuid

import pytest
from langchain_core.documents import Document as TextDocument

from app.models.document import DocumentStatus
from app.services import document_processor as pipeline
from app.services.tool_executors.database_executor import validate_readonly_sql, DatabaseToolExecutionError, DatabaseToolExecutor
from app.schemas.tool import DatabaseToolConfig


@asynccontextmanager
async def acquired_lock(doc_id):
    yield True


@asynccontextmanager
async def busy_lock(doc_id):
    yield False


def database(doc):
    added = []
    result = SimpleNamespace(scalar_one=lambda: 2)
    session = SimpleNamespace(get=AsyncMock(return_value=doc), execute=AsyncMock(return_value=result),
                              commit=AsyncMock(), add=added.append)
    @asynccontextmanager
    async def factory():
        yield session
    return factory, session, added


def record(graph=False):
    return SimpleNamespace(id=uuid.uuid4(), user_id="alice", r2_key="source", status=DocumentStatus.PARSING,
                           indexing_config={"preset": "high_precision", "index_graph": graph}, provider_status={})


@pytest.mark.asyncio
async def test_busy_or_deleted_document_never_runs_indexing():
    with patch.object(pipeline, "document_lock", busy_lock), patch.object(pipeline, "_index_document", AsyncMock()) as index:
        assert await pipeline.process_document_pipeline(str(uuid.uuid4()), "source", "alice", "file.txt") == 0
    index.assert_not_called()
    factory, _, _ = database(None)
    with patch.object(pipeline, "document_lock", acquired_lock), patch.object(pipeline, "async_session_factory", factory), patch.object(pipeline, "_index_document", AsyncMock()) as index:
        assert await pipeline.process_document_pipeline(str(uuid.uuid4()), "source", "alice", "file.txt") == 0
    index.assert_not_called()


@pytest.mark.asyncio
async def test_completed_duplicate_delivery_is_skipped():
    doc = record()
    doc.status = DocumentStatus.INDEXED
    factory, _, _ = database(doc)
    with patch.object(pipeline, "document_lock", acquired_lock), patch.object(pipeline, "async_session_factory", factory), patch.object(pipeline, "_index_document", AsyncMock()) as index:
        assert await pipeline.process_document_pipeline(str(doc.id), "source", "alice", "file.txt", is_reindex=True) == 2
    index.assert_not_called()


@pytest.mark.asyncio
async def test_stored_ingestion_preset_wins_over_queued_defaults():
    doc = record()
    factory, _, _ = database(doc)
    with patch.object(pipeline, "document_lock", acquired_lock), patch.object(pipeline, "async_session_factory", factory), patch.object(pipeline, "_index_document", AsyncMock(return_value=1)) as index:
        await pipeline.process_document_pipeline(str(doc.id), "source", "alice", "file.txt", preset="balanced")
    assert index.call_args.args[4]["preset"] == "high_precision"


@pytest.mark.asyncio
async def test_normal_upload_can_index_graph_and_track_provider_status():
    doc = record(graph=True)
    factory, _, added = database(doc)
    raw = TextDocument(page_content="document", metadata={"user_id": "alice"})
    provider = SimpleNamespace(ingest=AsyncMock(return_value=SimpleNamespace(failed=0)))
    with patch.object(pipeline, "document_lock", acquired_lock), patch.object(pipeline, "async_session_factory", factory), patch.object(pipeline, "load_document", return_value=raw), patch.object(pipeline, "split_markdown_document", side_effect=lambda *a, **kw: [TextDocument(page_content="chunk", metadata={})]), patch.object(pipeline, "index_documents", AsyncMock(return_value=["id"])) as index, patch("app.core.rag.factory.get_provider", AsyncMock(return_value=provider)):
        await pipeline.process_document_pipeline(str(doc.id), "source", "alice", "file.txt", file_bytes=b"data")
        assert doc.provider_status == {"hybrid": "INDEXED", "lightrag": "INDEXED"}
        first_id = index.call_args.args[0][0].metadata["chunk_id"]
        # A new attempt after activation stages a separate generation.
        doc.status = DocumentStatus.FAILED
        await pipeline.process_document_pipeline(str(doc.id), "source", "alice", "file.txt", file_bytes=b"data")
        assert index.call_args.args[0][0].metadata["chunk_id"] != first_id
    assert provider.ingest.await_count == 2
    assert len(added) == 2  # Each transaction also deletes its prior inspection rows.


@pytest.mark.asyncio
async def test_optional_graph_failure_keeps_hybrid_ready_but_reports_graph_failure():
    doc = record(True)
    factory, _, _ = database(doc)
    provider = SimpleNamespace(ingest=AsyncMock(side_effect=RuntimeError("graph unavailable")))
    raw = TextDocument(page_content="doc")
    with patch.object(pipeline, "document_lock", acquired_lock), patch.object(pipeline, "async_session_factory", factory), patch.object(pipeline, "load_document", return_value=raw), patch.object(pipeline, "split_markdown_document", return_value=[TextDocument(page_content="chunk")]), patch.object(pipeline, "index_documents", AsyncMock()), patch("app.core.rag.factory.get_provider", AsyncMock(return_value=provider)):
        await pipeline.process_document_pipeline(str(doc.id), "source", "alice", "file.txt", file_bytes=b"data")
    assert doc.status == DocumentStatus.INDEXED
    assert doc.provider_status["lightrag"] == "FAILED"


@pytest.mark.asyncio
async def test_parse_failure_preserves_prior_index():
    doc = record()
    factory, _, _ = database(doc)
    with patch.object(pipeline, "document_lock", acquired_lock), patch.object(pipeline, "async_session_factory", factory), patch.object(pipeline, "load_document", side_effect=ValueError("bad file")), patch.object(pipeline, "delete_documents_by_metadata", AsyncMock()) as purge:
        with pytest.raises(ValueError, match="bad file"):
            await pipeline.process_document_pipeline(str(doc.id), "source", "alice", "file.txt", file_bytes=b"bad", is_reindex=True)
    purge.assert_not_called()
    assert doc.status == DocumentStatus.FAILED


@pytest.mark.asyncio
async def test_document_lock_releases_on_the_same_physical_connection():
    from app.services import document_lock as locks
    connection = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalar=lambda: True)), commit=AsyncMock())
    @asynccontextmanager
    async def connect():
        yield connection
    with patch.object(locks, "_lock_engine", SimpleNamespace(connect=connect)):
        async with locks.document_lock("id") as acquired:
            assert acquired
    queries = [str(c.args[0]) for c in connection.execute.call_args_list]
    assert "pg_try_advisory_lock" in queries[0]
    assert "pg_advisory_unlock" in queries[1]
    assert connection.execute.call_args_list[0].args[1] == connection.execute.call_args_list[1].args[1]


def test_multiline_mutating_cte_is_not_classified_readonly():
    with pytest.raises(DatabaseToolExecutionError):
        validate_readonly_sql("WITH moved AS (DELETE\nFROM example RETURNING *) SELECT * FROM moved")


@pytest.mark.asyncio
async def test_sqlite_queries_and_no_create_side_effect(tmp_path):
    executor = DatabaseToolExecutor(DatabaseToolConfig(connection_string="sqlite:///:memory:", query_template="SELECT :n AS n"))
    assert await executor.execute({"n": 5}) == [{"n": 5}]
    missing = tmp_path / "missing.db"
    executor = DatabaseToolExecutor(DatabaseToolConfig(connection_string=f"sqlite:///{missing.as_posix()}", query_template="SELECT 1"))
    with pytest.raises(DatabaseToolExecutionError):
        await executor.execute({})
    assert not missing.exists()


@pytest.mark.asyncio
async def test_retrieval_outage_is_distinct_from_no_matches():
    from app.services.retriever import _dense_search
    with patch("app.services.retriever.get_vector_store", side_effect=RuntimeError("offline")):
        with pytest.raises(RuntimeError, match="unavailable"):
            await _dense_search("query", 3, "collection")
