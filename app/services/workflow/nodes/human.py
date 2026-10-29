from __future__ import annotations

import logging
from typing import Any

from langgraph.types import interrupt

from app.schemas.workflow import HumanApprovalNodeConfig
from app.services.workflow.state import WorkflowState, render_template

logger = logging.getLogger(__name__)


def build_human_approval_node(node_id: str, config_dict: dict[str, Any]):
    """Builds a human-in-the-loop review node that pauses execution via LangGraph interrupt."""
    config = HumanApprovalNodeConfig.model_validate(config_dict or {})

    async def human_approval_handler(state: WorkflowState) -> dict[str, Any]:
        logger.debug("Executing HUMAN_APPROVAL node '%s'", node_id)
        rendered_message = render_template(config.message_template, state)

        # Trigger LangGraph interrupt and pause execution
        # When resumed with a Command(resume=...), this returns the resume payload
        decision = interrupt({
            "type": "human_approval",
            "node_id": node_id,
            "message": rendered_message,
            "timeout_seconds": config.timeout_seconds,
        })

        logger.info("Human approval node '%s' resumed with decision: %s", node_id, decision)

        variables = dict(state.get("variables") or {})
        history = list(state.get("node_history") or [])
        history.append(node_id)

        # Record approval result
        variables[config.approval_variable] = decision
        return {
            "variables": variables,
            "node_history": history,
        }

    return human_approval_handler

