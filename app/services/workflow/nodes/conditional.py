from __future__ import annotations

import logging
import re
from typing import Any

from app.schemas.workflow import ConditionalNodeConfig
from app.services.workflow.state import (
    WorkflowState,
    evaluate_condition_expression,
    evaluate_variable_match,
    resolve_path_value,
)

logger = logging.getLogger(__name__)


def build_conditional_node(node_id: str, config_dict: dict[str, Any]):
    """Builds the passthrough handler for a conditional router node."""
    async def conditional_handler(state: WorkflowState) -> dict[str, Any]:
        logger.debug("Visiting CONDITIONAL router node '%s'", node_id)
        history = list(state.get("node_history") or [])
        history.append(node_id)
        return {"node_history": history}

    return conditional_handler


def build_conditional_router(node_id: str, config_dict: dict[str, Any]):
    """Builds the conditional branching decision function evaluated by LangGraph edges."""
    config = ConditionalNodeConfig.model_validate(config_dict or {})

    def router_fn(state: WorkflowState) -> str:
        logger.debug("Evaluating routing branches for CONDITIONAL node '%s'", node_id)

        for idx, branch in enumerate(config.branches):
            target = branch.target_node_id
            matched = False

            if branch.condition_type == "expression" and branch.expression:
                matched = evaluate_condition_expression(branch.expression, state)
            elif branch.condition_type == "variable_match" and branch.variable and branch.operator:
                matched = evaluate_variable_match(
                    variable_name=branch.variable,
                    operator_name=branch.operator,
                    expected_value=branch.value,
                    state=state,
                )
            elif branch.condition_type == "regex" and branch.variable and branch.expression:
                actual = str(resolve_path_value(branch.variable, state) or "")
                try:
                    matched = bool(re.search(branch.expression, actual))
                except Exception as rex:
                    logger.warning("Regex evaluation error on branch %d: %s", idx, rex)
                    matched = False

            if matched:
                logger.info(
                    "Conditional branch %d matched: routing from '%s' to '%s'",
                    idx,
                    node_id,
                    target,
                )
                return target

        logger.info(
            "No conditional branches matched for node '%s'. Falling back to default target '%s'",
            node_id,
            config.default_target_node_id,
        )
        return config.default_target_node_id

    return router_fn

