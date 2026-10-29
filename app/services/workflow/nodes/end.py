from __future__ import annotations

import logging
from typing import Any

from app.schemas.workflow import EndNodeConfig
from app.services.workflow.state import WorkflowState

logger = logging.getLogger(__name__)


def build_end_node(node_id: str, config_dict: dict[str, Any]):
    """Builds the END node handler for a workflow graph."""
    config = EndNodeConfig.model_validate(config_dict or {})

    async def end_handler(state: WorkflowState) -> dict[str, Any]:
        logger.debug("Executing END node '%s'", node_id)
        variables = dict(state.get("variables") or {})

        # If response_variable is configured, ensure it is recorded
        if config.response_variable and config.response_variable not in variables:
            logger.warning(
                "Configured response_variable '%s' not found in variables",
                config.response_variable,
            )

        history = list(state.get("node_history") or [])
        history.append(node_id)

        return {
            "node_history": history,
        }

    return end_handler

