import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document

from app.services.multi_query import multi_query_hybrid_search


@pytest.mark.asyncio
async def test_one_failed_query_preserves_successful_results():
    evidence = Document(page_content="available evidence", metadata={"chunk_id": "chunk-1"})
    with patch("app.services.multi_query.hybrid_search", AsyncMock(side_effect=[RuntimeError("provider unavailable"), [evidence]])):
        results = await multi_query_hybrid_search(["first", "second"], "alice")
    assert results == [evidence]


@pytest.mark.asyncio
async def test_cancelled_query_propagates_cancellation_instead_of_entering_fusion():
    with patch("app.services.multi_query.hybrid_search", AsyncMock(side_effect=[asyncio.CancelledError(), []])):
        with pytest.raises(asyncio.CancelledError):
            await multi_query_hybrid_search(["first", "second"], "alice")
