"""Background and distributed asynchronous worker tasks."""

from app.tasks.document_tasks import process_document_task

__all__ = ["process_document_task"]
