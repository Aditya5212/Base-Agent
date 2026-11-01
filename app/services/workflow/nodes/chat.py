from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.schemas.workflow import ChatNodeConfig
from app.services.workflow.middleware.pipeline import invoke_chat_model_with_middlewares
from app.services.workflow.state import WorkflowState, render_template

logger = logging.getLogger(__name__)


def build_chat_node(node_id: str, config_dict: dict[str, Any]):
    """Builds an LLM generation node handler with middleware integration."""
    config = ChatNodeConfig.model_validate(config_dict or {})

    async def chat_handler(state: WorkflowState) -> dict[str, Any]:
        logger.debug("Executing CHAT node '%s' [model=%s]", node_id, config.model_id)

        # 1. Render system and user prompts with variable interpolation
        rendered_system = render_template(config.system_prompt, state)
        system_msg = SystemMessage(content=rendered_system)

        llm_input_messages: list[BaseMessage] = [system_msg]
        new_messages: list[BaseMessage] = []

        # Existing conversation messages
        existing_msgs = list(state.get("messages") or [])
        llm_input_messages.extend(existing_msgs)

        # Optional user prompt template
        if config.user_prompt_template:
            rendered_user = render_template(config.user_prompt_template, state)
            user_msg = HumanMessage(content=rendered_user)
            llm_input_messages.append(user_msg)
            new_messages.append(user_msg)
        elif not existing_msgs:
            # If no existing history and no prompt template, use 'input' variable
            input_val = str(state.get("variables", {}).get("input", "")).strip()
            if input_val:
                user_msg = HumanMessage(content=input_val)
                llm_input_messages.append(user_msg)
                new_messages.append(user_msg)

        # 2. Invoke LLM wrapped with retry & fallback middlewares
        response_msg = await invoke_chat_model_with_middlewares(
            messages=llm_input_messages,
            model_id=config.model_id,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            fallback_model_id=config.fallback_model_id,
            max_retries=config.max_retries,
        )

        response_text = response_msg.content
        if isinstance(response_text, list):
            response_text = " ".join(
                b.get("text", "") for b in response_text if isinstance(b, dict) and "text" in b
            )

        # 3. Store response in state variables
        variables = dict(state.get("variables") or {})
        variables[config.output_variable] = response_text

        # 4. Update conversation memory if enabled
        history = list(state.get("node_history") or [])
        history.append(node_id)

        result: dict[str, Any] = {
            "variables": variables,
            "node_history": history,
        }

        if config.enable_memory:
            new_messages.append(response_msg)
            result["messages"] = new_messages

        return result

    return chat_handler

