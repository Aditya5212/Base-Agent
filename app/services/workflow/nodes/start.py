from __future__ import annotations

import logging
from typing import Any

from app.schemas.workflow import StartNodeConfig
from app.services.workflow.state import WorkflowState

logger = logging.getLogger(__name__)


def build_start_node(node_id: str, config_dict: dict[str, Any]):
    """Builds the START node handler for a workflow graph."""
    config = StartNodeConfig.model_validate(config_dict or {})

    async def start_handler(state: WorkflowState) -> dict[str, Any]:
        logger.debug("Executing START node '%s'", node_id)
        variables = dict(state.get("variables") or {})

        # Apply initial variables if not already set by runtime input
        for k, v in config.initial_variables.items():
            if k not in variables:
                variables[k] = v

        history = list(state.get("node_history") or [])
        history.append(node_id)

        return {
            "variables": variables,
            "node_history": history,
        }

    return start_handler

