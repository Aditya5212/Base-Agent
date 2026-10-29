"""Live verification script for Phase 2: Workflow Engine & Graph Compiler.

Tests end-to-end execution of a realistic multi-node workflow against the live database:
  START -> RETRIEVE -> CHAT -> CONDITIONAL -> (VIP_PATH or STD_PATH) -> END
"""

import asyncio
import logging
import os
from pathlib import Path
import sys
import uuid

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_workflow_live")

from app.core.database import async_session_factory
from app.schemas.workflow import WorkflowCreate
from app.services.workflow.compiler import WorkflowCompiler
from app.services.workflow_service import (
    create_workflow,
    delete_workflow,
    execute_workflow,
    get_workflow_run,
)
from app.schemas.workflow import WorkflowRunRequest


async def run_live_test():
    test_user_id = f"live_test_{uuid.uuid4().hex[:8]}"
    logger.info("Starting live workflow verification for user '%s'...", test_user_id)

    # 1. Define a realistic multi-step workflow
    workflow_payload = WorkflowCreate.model_validate({
        "name": "Live Support Inquiry Router",
        "description": "Extracts context, classifies inquiry, and generates routing output",
        "nodes": [
            {
                "id": "start",
                "name": "Inquiry Input",
                "type": "start",
                "config": {
                    "initial_variables": {
                        "user_tier": "vip",
                        "inquiry": "How do I upgrade my team subscription?",
                    }
                },
            },
            {
                "id": "retrieve_context",
                "name": "Knowledge Search",
                "type": "retrieve",
                "config": {
                    "query_template": "{{variables.inquiry}}",
                    "top_k": 2,
                    "output_variable": "kb_context",
                },
            },
            {
                "id": "evaluate_tier",
                "name": "Tier Evaluation Router",
                "type": "conditional",
                "config": {
                    "branches": [
                        {
                            "target_node_id": "vip_processing",
                            "condition_type": "expression",
                            "expression": "variables.user_tier == 'vip'",
                        }
                    ],
                    "default_target_node_id": "std_processing",
                },
            },
            {
                "id": "vip_processing",
                "name": "VIP Concierge Formatter",
                "type": "code",
                "config": {
                    "code": (
                        "sla_hours = 1\n"
                        "priority = 'URGENT'\n"
                        "routing_tag = 'concierge-vip'\n"
                        "result = {'sla': sla_hours, 'priority': priority, 'queue': routing_tag}"
                    ),
                    "output_variable": "routing_decision",
                },
            },
            {
                "id": "std_processing",
                "name": "Standard Queue Formatter",
                "type": "code",
                "config": {
                    "code": (
                        "sla_hours = 24\n"
                        "priority = 'NORMAL'\n"
                        "routing_tag = 'general-support'\n"
                        "result = {'sla': sla_hours, 'priority': priority, 'queue': routing_tag}"
                    ),
                    "output_variable": "routing_decision",
                },
            },
            {
                "id": "chat_response",
                "name": "Acknowledge Generation",
                "type": "chat",
                "config": {
                    "system_prompt": "You are a customer service assistant. Greet {{variables.user_tier}} user.",
                    "user_prompt_template": "Customer asked: {{variables.inquiry}}",
                    "output_variable": "customer_message",
                },
            },
            {
                "id": "end",
                "name": "Workflow Completion",
                "type": "end",
                "config": {
                    "response_variable": "routing_decision",
                },
            },
        ],
        "edges": [
            {"source": "start", "target": "retrieve_context"},
            {"source": "retrieve_context", "target": "chat_response"},
            {"source": "chat_response", "target": "evaluate_tier"},
            {"source": "evaluate_tier", "target": "vip_processing"},
            {"source": "evaluate_tier", "target": "std_processing"},
            {"source": "vip_processing", "target": "end"},
            {"source": "std_processing", "target": "end"},
        ],
        "recursion_limit": 25,
    })

    async with async_session_factory() as session:
        # Step A: Validate and Create in PostgreSQL
        logger.info("Persisting workflow to database...")
        workflow = await create_workflow(
            session=session,
            user_id=test_user_id,
            payload=workflow_payload,
        )
        logger.info("Workflow successfully created with ID: %s, version: %d", workflow.id, workflow.version)

        # Step B: Execute the workflow run
        logger.info("Executing workflow run with VIP tier input...")
        run = await execute_workflow(
            session=session,
            workflow_id=workflow.id,
            user_id=test_user_id,
            payload=WorkflowRunRequest(
                input_data={"user_tier": "vip", "inquiry": "How do I upgrade to enterprise?"}
            ),
        )

        logger.info("Execution finished with status: %s", run.status)
        logger.info("Execution run ID: %s", run.id)
        logger.info("Final output data: %s", run.output_data)
        logger.info("Step trace: %s", run.execution_log)

        # Step C: Assert expected outcomes
        assert run.status == "COMPLETED", f"Expected COMPLETED, got {run.status}"
        assert run.output_data is not None
        decision = run.output_data.get("routing_decision") or run.output_data
        assert decision.get("priority") == "URGENT"
        assert decision.get("queue") == "concierge-vip"

        # Verify step history visited VIP path and NOT STD path
        node_history = run.execution_log[0].get("node_history", [])
        logger.info("Node execution path: %s", " -> ".join(node_history))
        assert "vip_processing" in node_history
        assert "std_processing" not in node_history

        # Step D: Test second run with STANDARD tier input to verify branching
        logger.info("Executing second run with standard tier input...")
        std_run = await execute_workflow(
            session=session,
            workflow_id=workflow.id,
            user_id=test_user_id,
            payload=WorkflowRunRequest(
                input_data={"user_tier": "standard", "inquiry": "Where is my invoice?"}
            ),
        )
        assert std_run.status == "COMPLETED"
        std_decision = std_run.output_data.get("routing_decision") or std_run.output_data
        assert std_decision.get("priority") == "NORMAL"
        assert std_decision.get("queue") == "general-support"
        std_history = std_run.execution_log[0].get("node_history", [])
        logger.info("Standard branch path: %s", " -> ".join(std_history))
        assert "std_processing" in std_history
        assert "vip_processing" not in std_history

        # Step E: Cleanup
        logger.info("Cleaning up live test workflow...")
        await delete_workflow(session=session, workflow_id=workflow.id, user_id=test_user_id)
        logger.info("Live test workflow deleted cleanly.")

    print("\n=======================================================")
    print(" LIVE WORKFLOW ENGINE VERIFICATION PASSED SUCCESSFULLY ")
    print("=======================================================\n")


if __name__ == "__main__":
    asyncio.run(run_live_test())
