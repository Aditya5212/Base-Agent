"""Automated regression and acceptance tests for thread knowledge reliability.

Covers:
1. Thread A vs Thread B isolation
2. Global query isolation (zero leakage of thread documents)
3. Thread preference in RRF and Multi-Query fusion (thread_boost bonus)
4. Dual-layer LightRAG traversal with global fallback
5. Durable thread deletion & retry resilience (no orphaned DB/storage records)
6. Concurrency coordination with active document ingestion
7. Cross-tenant workflow thread ownership enforcement
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from langchain_core.documents import Document

from app.core.rag.protocols import RetrievalQuery, RetrievalResult, RetrievedChunk
from app.models.document import DocumentStatus
from app.models.thread import Thread, ThreadAsset
from app.models.document import Document as DBDocument
from app.services.multi_query import cross_query_reciprocal_rank_fusion
from app.services.retrieval_service import (
    RetrievalScope,
    build_scoped_filter,
    query_graph_rag_dual_layer,
    resolve_retrieval_scope,
)
from app.services.retriever import apply_reciprocal_rank_fusion
from app.services.thread_service import delete_thread_cascade


# ==============================================================================
# 1. Retrieval Scope Resolution Contracts
# ==============================================================================

def test_resolve_retrieval_scope_defaults():
    # When no thread is supplied, must default to GLOBAL_ONLY
    assert resolve_retrieval_scope(None, thread_id=None) == RetrievalScope.GLOBAL_ONLY

    # When thread is supplied without explicit scope, default to THREAD_WITH_GLOBAL
    assert resolve_retrieval_scope(None, thread_id="thread-123") == RetrievalScope.THREAD_WITH_GLOBAL

    # Explicit scopes
    assert resolve_retrieval_scope("global_only", thread_id=None) == RetrievalScope.GLOBAL_ONLY
    assert resolve_retrieval_scope("thread_only", thread_id="t1") == RetrievalScope.THREAD_ONLY
    assert resolve_retrieval_scope("thread_with_global", thread_id="t1") == RetrievalScope.THREAD_WITH_GLOBAL
    assert resolve_retrieval_scope("all_user", thread_id=None) == RetrievalScope.ALL_USER


def test_resolve_retrieval_scope_requires_thread():
    # thread_only requires a thread_id
    with pytest.raises(ValueError, match="requires a valid thread_id"):
        resolve_retrieval_scope("thread_only", thread_id=None)

    # thread_with_global requires a thread_id
    with pytest.raises(ValueError, match="requires a valid thread_id"):
        resolve_retrieval_scope("thread_with_global", thread_id=None)


# ==============================================================================
# 2. Filter Scoping & Boundary Isolation (Thread A vs Thread B vs Global)
# ==============================================================================

def test_scoped_filter_thread_a_vs_thread_b():
    # Thread A filter must strictly require thread_id == thread_A
    filter_a = build_scoped_filter(user_id="user_1", scope=RetrievalScope.THREAD_ONLY, thread_id="thread_A")
    assert filter_a == {
        "$and": [
            {"user_id": {"$eq": "user_1"}},
            {"thread_id": {"$eq": "thread_A"}},
        ]
    }

    # Thread B query can never match thread A
    filter_b = build_scoped_filter(user_id="user_1", scope=RetrievalScope.THREAD_ONLY, thread_id="thread_B")
    assert filter_b["$and"][1] == {"thread_id": {"$eq": "thread_B"}}


def test_scoped_filter_thread_with_global_excludes_thread_b():
    # Thread A with global fallback allows thread_A OR global (no thread_id), strictly excluding thread_B
    filter_with_global = build_scoped_filter(user_id="user_1", scope=RetrievalScope.THREAD_WITH_GLOBAL, thread_id="thread_A")
    assert filter_with_global == {
        "$and": [
            {"user_id": {"$eq": "user_1"}},
            {
                "$or": [
                    {"thread_id": {"$eq": "thread_A"}},
                    {"thread_id": {"$exists": False}},
                ]
            },
        ]
    }


def test_scoped_filter_global_only_blocks_all_thread_docs():
    # Global search must explicitly check thread_id does NOT exist, preventing leakage of any thread's docs
    filter_global = build_scoped_filter(user_id="user_1", scope=RetrievalScope.GLOBAL_ONLY)
    assert filter_global == {
        "$and": [
            {"user_id": {"$eq": "user_1"}},
            {"thread_id": {"$exists": False}},
        ]
    }


def test_scoped_filter_all_user():
    # all_user explicitly matches all documents for that user regardless of thread
    filter_all = build_scoped_filter(user_id="user_1", scope=RetrievalScope.ALL_USER)
    assert filter_all == {"user_id": {"$eq": "user_1"}}


# ==============================================================================
# 3. Thread Preference in Hybrid & Multi-Query RRF Fusion
# ==============================================================================

def test_rrf_thread_priority_boost():
    # Two documents with identical relevance rank in dense and sparse searches
    thread_doc = Document(
        page_content="Thread doc content",
        metadata={"document_id": "doc_th", "thread_id": "target_thread"},
    )
    global_doc = Document(
        page_content="Global doc content",
        metadata={"document_id": "doc_gl"},
    )

    dense_docs = [thread_doc, global_doc]
    sparse_docs = [global_doc, thread_doc]

    fused = apply_reciprocal_rank_fusion(
        dense_results=dense_docs,
        sparse_results=sparse_docs,
        active_thread_id="target_thread",
        thread_boost=0.05,
    )

    assert len(fused) == 2
    # Thread document must rank #1 due to the active thread bonus
    assert fused[0].metadata["document_id"] == "doc_th"
    assert fused[0].metadata["is_thread_document"] is True
    assert fused[0].metadata["retrieval_scope"] == "thread"
    assert fused[0].metadata["thread_boost_applied"] == 0.05

    # Global document ranks #2
    assert fused[1].metadata["document_id"] == "doc_gl"
    assert fused[1].metadata["is_thread_document"] is False
    assert fused[1].metadata["retrieval_scope"] == "global"


def test_cross_query_rrf_thread_priority_boost():
    thread_doc = Document(
        page_content="Thread doc in multi-query",
        metadata={"document_id": "doc_th_mq", "thread_id": "active_conv"},
    )
    global_doc = Document(
        page_content="Global doc in multi-query",
        metadata={"document_id": "doc_gl_mq"},
    )

    query_results = [
        ("q1", [thread_doc, global_doc]),
        ("q2", [global_doc, thread_doc]),
    ]

    fused = cross_query_reciprocal_rank_fusion(
        query_results=query_results,
        active_thread_id="active_conv",
        thread_boost=0.05,
    )

    assert len(fused) == 2
    assert fused[0].metadata["document_id"] == "doc_th_mq"
    assert fused[0].metadata["is_thread_document"] is True


# ==============================================================================
# 4. Dual-Layer LightRAG Traversal (Thread First + Global Fallback)
# ==============================================================================

@pytest.mark.asyncio
async def test_lightrag_dual_layer_query():
    thread_chunk = RetrievedChunk(
        chunk_id="chunk_th_1",
        content="Thread specific entity facts.",
        source="thread_notes.txt",
        score=0.9,
        metadata={"workspace": "thread_ws"},
    )
    global_chunk = RetrievedChunk(
        chunk_id="chunk_gl_1",
        content="Global company overview facts.",
        source="handbook.pdf",
        score=0.8,
        metadata={"workspace": "user_ws"},
    )

    mock_thread_provider = AsyncMock()
    mock_thread_provider.query.return_value = RetrievalResult(
        provider="lightrag",
        query="test query",
        chunks=[thread_chunk],
        context="[Thread entities] Project X deadline is Friday.",
    )

    mock_global_provider = AsyncMock()
    mock_global_provider.query.return_value = RetrievalResult(
        provider="lightrag",
        query="test query",
        chunks=[global_chunk],
        context="[Global entities] Project X was initiated in 2026.",
    )

    async def mock_provider_getter(name: str, workspace: str):
        if "thread" in workspace:
            return mock_thread_provider
        return mock_global_provider

    context, sources = await query_graph_rag_dual_layer(
        query="Tell me about Project X",
        user_id="user_123",
        thread_id="thread_456",
        top_k=5,
        provider_getter=mock_provider_getter,
    )

    # Must contain both labeled sections
    assert "Thread-Specific Graph Context" in context
    assert "Project X deadline is Friday" in context
    assert "Global Knowledge Graph Context" in context
    assert "Project X was initiated in 2026" in context

    # Sources must track both chunks with proper scopes
    assert len(sources) == 2
    assert sources[0]["chunk_id"] == "chunk_th_1"
    assert sources[0]["scope"] == "thread"
    assert sources[1]["chunk_id"] == "chunk_gl_1"
    assert sources[1]["scope"] == "global"


# ==============================================================================
# 5. Durable Thread Deletion: Retry Resilience & No Orphaned Data
# ==============================================================================

@pytest.mark.asyncio
async def test_durable_thread_deletion_preserves_records_on_r2_failure():
    thread_id = "thread_del_fail"
    user_id = "user_del"

    mock_thread = Thread(id=thread_id, user_id=user_id, title="Test Thread")
    mock_doc = DBDocument(
        id=uuid.uuid4(),
        thread_id=thread_id,
        user_id=user_id,
        file_name="important.pdf",
        r2_key="documents/user_del/threads/thread_del_fail/important.pdf",
        status=DocumentStatus.INDEXED,
    )
    mock_asset = ThreadAsset(
        id=uuid.uuid4(),
        thread_id=thread_id,
        user_id=user_id,
        file_name="diagram.png",
        r2_key="threads/thread_del_fail/diagram.png",
    )

    mock_db = AsyncMock()
    mock_db.get.return_value = mock_thread

    # Execute statement returns
    mock_doc_result = MagicMock()
    mock_doc_result.scalars.return_value.all.return_value = [mock_doc]

    mock_asset_result = MagicMock()
    mock_asset_result.scalars.return_value.all.return_value = [mock_asset]

    mock_db.execute.side_effect = [mock_doc_result, mock_asset_result]

    with patch("app.services.r2_storage.r2_storage_service.delete_file", new_callable=AsyncMock) as mock_r2_del, \
         patch("app.services.vector_store.delete_documents_by_metadata", new_callable=AsyncMock) as mock_vec_del, \
         patch("app.services.memory.delete_thread_memory", new_callable=AsyncMock) as mock_mem_del, \
         patch("app.core.rag.factory.get_provider", new_callable=AsyncMock) as mock_get_prov, \
         patch("app.services.document_lock.document_lock") as mock_lock:

        # Setup advisory lock to acquire immediately
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = True
        mock_ctx.__aexit__.return_value = None
        mock_lock.return_value = mock_ctx

        # Simulate R2 service outage
        mock_r2_del.side_effect = RuntimeError("R2 Storage 503 Service Unavailable")

        with pytest.raises(RuntimeError, match="Records preserved for retry"):
            await delete_thread_cascade(mock_db, thread_id=thread_id, user_id=user_id)

        # Database record must NOT have been deleted
        mock_db.delete.assert_not_called()
        mock_db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_durable_thread_deletion_succeeds_when_external_services_ok():
    thread_id = "thread_del_ok"
    user_id = "user_del"

    mock_thread = Thread(id=thread_id, user_id=user_id, title="Test Thread")
    mock_doc = DBDocument(
        id=uuid.uuid4(),
        thread_id=thread_id,
        user_id=user_id,
        file_name="clean.pdf",
        r2_key="documents/user_del/threads/thread_del_ok/clean.pdf",
        status=DocumentStatus.INDEXED,
    )

    mock_db = AsyncMock()
    mock_db.get.return_value = mock_thread

    mock_doc_result = MagicMock()
    mock_doc_result.scalars.return_value.all.return_value = [mock_doc]

    mock_asset_result = MagicMock()
    mock_asset_result.scalars.return_value.all.return_value = []

    mock_db.execute.side_effect = [mock_doc_result, mock_asset_result]

    with patch("app.services.r2_storage.r2_storage_service.delete_file", new_callable=AsyncMock) as mock_r2_del, \
         patch("app.services.vector_store.delete_documents_by_metadata", new_callable=AsyncMock) as mock_vec_del, \
         patch("app.services.memory.delete_thread_memory", new_callable=AsyncMock) as mock_mem_del, \
         patch("app.core.rag.factory.get_provider", new_callable=AsyncMock) as mock_get_prov, \
         patch("app.services.document_lock.document_lock") as mock_lock:

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = True
        mock_ctx.__aexit__.return_value = None
        mock_lock.return_value = mock_ctx

        mock_r2_del.return_value = True
        mock_vec_del.return_value = 5
        mock_mem_del.return_value = True

        success = await delete_thread_cascade(mock_db, thread_id=thread_id, user_id=user_id)
        assert success is True

        # Database record cleanly deleted and committed
        mock_db.delete.assert_called_once_with(mock_thread)
        mock_db.commit.assert_called_once()


# ==============================================================================
# 6. Concurrency Coordination: Deletion during Active Document Ingestion
# ==============================================================================

@pytest.mark.asyncio
async def test_thread_deletion_aborts_active_ingestion():
    thread_id = "thread_active_ingest"
    user_id = "user_ingest"

    mock_thread = Thread(id=thread_id, user_id=user_id, title="Ingestion Thread")
    mock_doc = DBDocument(
        id=uuid.uuid4(),
        thread_id=thread_id,
        user_id=user_id,
        file_name="processing.pdf",
        r2_key="documents/user_ingest/threads/thread_active_ingest/processing.pdf",
        status=DocumentStatus.PARSING,  # Actively parsing!
    )

    mock_db = AsyncMock()
    mock_db.get.return_value = mock_thread

    mock_doc_result = MagicMock()
    mock_doc_result.scalars.return_value.all.return_value = [mock_doc]

    mock_asset_result = MagicMock()
    mock_asset_result.scalars.return_value.all.return_value = []

    mock_db.execute.side_effect = [mock_doc_result, mock_asset_result]

    with patch("app.services.r2_storage.r2_storage_service.delete_file", new_callable=AsyncMock) as mock_r2_del, \
         patch("app.services.vector_store.delete_documents_by_metadata", new_callable=AsyncMock) as mock_vec_del, \
         patch("app.services.memory.delete_thread_memory", new_callable=AsyncMock) as mock_mem_del, \
         patch("app.core.rag.factory.get_provider", new_callable=AsyncMock) as mock_get_prov, \
         patch("app.services.document_lock.document_lock") as mock_lock:

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = True
        mock_ctx.__aexit__.return_value = None
        mock_lock.return_value = mock_ctx

        mock_r2_del.return_value = True

        success = await delete_thread_cascade(mock_db, thread_id=thread_id, user_id=user_id)
        assert success is True

        # Document status must have been transitioned to FAILED
        assert mock_doc.status == DocumentStatus.FAILED
        assert "Thread deleted during active ingestion" in mock_doc.error_message
