"""Tests for Taskiq broker and asynchronous background worker tasks."""

from unittest.mock import AsyncMock, patch
import pytest

from app.core.taskiq import broker, init_taskiq_broker, shutdown_taskiq_broker
from app.tasks.document_tasks import process_document_task


def test_taskiq_task_registered():
    """Verifies that process_document_task is properly registered with the broker."""
    assert "process_document_task" in broker.get_all_tasks()


@pytest.mark.asyncio
async def test_process_document_task_execution():
    """Verifies that process_document_task executes and delegates to process_document_pipeline."""
    with patch(
        "app.tasks.document_tasks.process_document_pipeline",
        new_callable=AsyncMock,
    ) as mock_pipeline:
        mock_pipeline.return_value = True

        result = await process_document_task(
            document_id="test-doc-123",
            r2_key="documents/root/test.pdf",
            user_id="root",
            file_name="test.pdf",
            preset="balanced",
            is_reindex=False,
        )

        assert result is True
        mock_pipeline.assert_awaited_once_with(
            document_id="test-doc-123",
            r2_key="documents/root/test.pdf",
            user_id="root",
            file_name="test.pdf",
            preset="balanced",
            file_bytes=None,
            is_reindex=False,
            thread_id=None,
        )


@pytest.mark.asyncio
async def test_broker_lifecycle_hooks():
    """Verifies that init_taskiq_broker and shutdown_taskiq_broker run cleanly."""
    with patch.object(broker, "startup", new_callable=AsyncMock) as mock_startup, \
         patch.object(broker, "shutdown", new_callable=AsyncMock) as mock_shutdown:
        await init_taskiq_broker()
        mock_startup.assert_awaited_once()

        await shutdown_taskiq_broker()
        mock_shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_document_task_kiq_dispatch():
    """Verifies that kiq dispatches through the broker kick interface correctly."""
    with patch.object(broker, "kick", new_callable=AsyncMock) as mock_kick:
        mock_kick.return_value = AsyncMock()
        await process_document_task.kiq(
            document_id="test-doc-kiq",
            r2_key="documents/root/test.pdf",
            user_id="root",
            file_name="test.pdf",
            preset="balanced",
            is_reindex=False,
        )
        mock_kick.assert_awaited_once()

