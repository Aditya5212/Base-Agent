"""Workflow execution background tasks executed by Taskiq workers."""

from __future__ import annotations

import logging
from typing import Any
import uuid

from app.core.taskiq import broker
from app.core.database import AsyncSessionLocal
from app.schemas.workflow import WorkflowRunRequest

logger = logging.getLogger(__name__)


@broker.task(task_name="execute_workflow_task")
async def execute_workflow_task(
    run_id: str | None = None,
    worker_id: str | None = None,
    workflow_id: str | None = None,
    user_id: str | None = None,
    input_data: dict[str, Any] | None = None,
    budget_seconds: int | None = None,
    thread_id: str | None = None,
) -> str:
    """Taskiq worker task for executing a workflow run in the background with worker leasing.

    Workers accept an existing run_id, claim its persisted work atomically, and execute
    under a periodic heartbeat and fencing token.
    """
    from app.services.workflow_service import (
        claim_workflow_run,
        create_workflow_run,
        execute_claimed_run,
    )

    assigned_worker = worker_id or f"worker-{uuid.uuid4().hex[:8]}"

    async with AsyncSessionLocal() as session:
        if run_id:
            r_uuid = uuid.UUID(run_id)
            logger.info(
                "Taskiq worker claiming run %s (worker_id=%s)",
                run_id,
                assigned_worker,
            )
            claimed = await claim_workflow_run(
                session=session,
                run_id=r_uuid,
                worker_id=assigned_worker,
                lease_seconds=30,
            )
            if not claimed:
                logger.warning(
                    "Worker %s could not claim workflow run %s (already claimed, fenced, or inactive)",
                    assigned_worker,
                    run_id,
                )
                return str(run_id)

            await execute_claimed_run(session, r_uuid, worker_id=assigned_worker)
            return str(run_id)

        # Backwards compatibility for invocations providing workflow_id directly
        if not workflow_id or not user_id:
            raise ValueError("Either run_id or both workflow_id and user_id must be provided")

        w_id = uuid.UUID(workflow_id)
        payload = WorkflowRunRequest(
            input_data=input_data or {},
            budget_seconds=budget_seconds,
            thread_id=thread_id,
        )
        logger.info(
            "Taskiq worker creating and executing workflow %s for user %s (worker_id=%s)",
            workflow_id,
            user_id,
            assigned_worker,
        )
        run, _ = await create_workflow_run(session, w_id, user_id, payload, queue_execution=False)
        claimed = await claim_workflow_run(session, run.id, worker_id=assigned_worker, lease_seconds=30)
        if not claimed:
            logger.warning("Could not claim newly created run %s", run.id)
            return str(run.id)

        await execute_claimed_run(session, run.id, worker_id=assigned_worker)
        return str(run.id)


@broker.task(task_name="reconcile_stale_workflows_task")
async def reconcile_stale_workflows_task(threshold_seconds: int = 0) -> list[str]:
    """Sweeps and recovers stalled workflow runs whose worker lease has expired."""
    from app.services.workflow_service import reconcile_stale_workflow_runs

    async with AsyncSessionLocal() as session:
        reconciled = await reconcile_stale_workflow_runs(session, threshold_seconds=threshold_seconds)
        return [str(r) for r in reconciled]


@broker.task(task_name="retry_pending_dispatches_task")
async def retry_pending_dispatches_task(limit: int = 50, max_retries: int = 5) -> int:
    """Retries pending or failed workflow run dispatches from transactional outbox."""
    from app.services.workflow_service import retry_pending_dispatches

    async with AsyncSessionLocal() as session:
        retried = await retry_pending_dispatches(session, limit=limit, max_retries=max_retries)
        return len(retried)
