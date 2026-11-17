"""Workflow execution background tasks executed by Taskiq workers."""

from __future__ import annotations

import logging
from typing import Any
import uuid

from app.core.taskiq import broker
from app.core.database import AsyncSessionLocal
from app.schemas.workflow import WorkflowRunRequest
@broker.task(task_name="execute_workflow_task")
async def execute_workflow_task(
    workflow_id: str,
    user_id: str,
    input_data: dict[str, Any] | None = None,
    budget_seconds: int | None = None,
    thread_id: str | None = None,
    worker_id: str | None = None,
) -> str:
    """Taskiq worker task for executing a workflow run in the background with worker leasing."""
    from app.services.workflow_service import execute_workflow, claim_workflow_run

    w_id = uuid.UUID(workflow_id)
    payload = WorkflowRunRequest(
        input_data=input_data or {},
        budget_seconds=budget_seconds,
        thread_id=thread_id,
    )
    assigned_worker = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
    logger.info(
        "Taskiq worker executing workflow %s for user %s (worker_id=%s, budget=%s)",
        workflow_id,
        user_id,
        assigned_worker,
        budget_seconds,
    )
    async with AsyncSessionLocal() as session:
        run = await execute_workflow(session, w_id, user_id, payload)
        await claim_workflow_run(session, run.id, worker_id=assigned_worker, lease_seconds=60)
        return str(run.id)
