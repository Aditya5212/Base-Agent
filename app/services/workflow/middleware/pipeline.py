from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.llm import get_llm

logger = logging.getLogger(__name__)


async def invoke_chat_model_with_middlewares(
    messages: list[BaseMessage],
    model_id: str = "deepseek-flash",
    temperature: float = 0.7,
    max_tokens: int | None = None,
    fallback_model_id: str | None = None,
    max_retries: int = 3,
) -> AIMessage:
    """Invokes chat LLM wrapped with ModelRetry and ModelFallback middlewares."""
    llm = get_llm(
        model_name=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # In mock fallback mode (e.g. testing without API key), return simulated response
    if llm is None:
        last_text = messages[-1].content if messages else "No prompt"
        if isinstance(last_text, list):
            last_text = " ".join(
                b.get("text", "") for b in last_text if isinstance(b, dict) and "text" in b
            )
        return AIMessage(
            content=f"[Mock Response from {model_id}] Received: {str(last_text)[:100]}"
        )

    # 1. Primary Model Execution with ModelRetryMiddleware (Exponential Backoff)
    try:
        retryer = AsyncRetrying(
            stop=stop_after_attempt(max(1, max_retries)),
            wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        async for attempt in retryer:
            with attempt:
                logger.debug(
                    "Invoking LLM '%s' (attempt %d/%d)",
                    model_id,
                    attempt.retry_state.attempt_number,
                    max_retries,
                )
                response = await llm.ainvoke(messages)
                return response
        raise RuntimeError("Model retry loop completed without a response")
    except Exception as primary_exc:
        logger.warning(
            "Primary model '%s' failed after %d attempts: %s",
            model_id,
            max_retries,
            str(primary_exc),
        )

        # 2. Failover to ModelFallbackMiddleware if configured
        if fallback_model_id and fallback_model_id != model_id:
            logger.info("Failing over to fallback model '%s'", fallback_model_id)
            fallback_llm = get_llm(
                model_name=fallback_model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if fallback_llm is not None:
                try:
                    fallback_response = await fallback_llm.ainvoke(messages)
                    return fallback_response
                except Exception as fb_exc:
                    logger.error(
                        "Fallback model '%s' also failed: %s",
                        fallback_model_id,
                        str(fb_exc),
                    )
                    raise fb_exc

        raise primary_exc

