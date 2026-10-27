"""DeepSeek LLM Configuration & Client Factory.

Model Specifications:
--------------------------------------------------------------------------------
1. deepseek-flash (Default / Base)
   - Underlying Version: DeepSeek-V4.1-Flash-1017
   - Base URL (OpenAI):   https://api.deepseek.com
   - Base URL (Anthropic):https://api.deepseek.com/anthropic
   - Context Length:      1M tokens (1,000,000)
   - Max Output:          384K tokens (384,000)
   - Features:            Vision, Tool Calls, JSON Output, Responses API
   - Concurrency Limit:   2500

2. deepseek-v4-pro (Advanced Reasoning)
   - Underlying Version: DeepSeek-V4-Pro-0813
   - Base URL (OpenAI):   https://api.deepseek.com
   - Context Length:      1M tokens (1,000,000)
   - Max Output:          384K tokens (384,000)
   - Features:            Advanced Reasoning, Tool Calls, JSON Output
   - Concurrency Limit:   500
--------------------------------------------------------------------------------
"""

import logging
from typing import Any

from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

# Supported DeepSeek model identifiers
DEEPSEEK_FLASH = "deepseek-flash"  # Version: DeepSeek-V4.1-Flash-1017 (Base/Default)
DEEPSEEK_PRO = "deepseek-v4-pro"   # Version: DeepSeek-V4-Pro-0813 (Advanced Reasoning)

# Backward-compatible references
MODEL_DEEPSEEK_FLASH = DEEPSEEK_FLASH
MODEL_DEEPSEEK_PRO = DEEPSEEK_PRO

DEFAULT_MODEL = DEEPSEEK_FLASH
DEFAULT_BASE_URL = "https://api.deepseek.com"

# Internal alias mapping to resolve version strings or short names to API model identifiers
_MODEL_ALIASES: dict[str, str] = {
    "deepseek-flash": DEEPSEEK_FLASH,
    "deepseek-v4.1-flash-1017": DEEPSEEK_FLASH,
    "flash": DEEPSEEK_FLASH,
    "deepseek-v4-pro": DEEPSEEK_PRO,
    "deepseek-v4-pro-0813": DEEPSEEK_PRO,
    "pro": DEEPSEEK_PRO,
}


def resolve_model_name(name_or_alias: str | None) -> str:
    """Normalizes model input (e.g. 'flash', 'DeepSeek-V4.1-Flash-1017') to the valid DeepSeek API name."""
    if not name_or_alias or not name_or_alias.strip():
        configured = settings.DEEPSEEK_MODEL or settings.LLM_MODEL
        if configured:
            return _MODEL_ALIASES.get(configured.strip().lower(), configured)
        return DEFAULT_MODEL

    normalized = name_or_alias.strip().lower()
    return _MODEL_ALIASES.get(normalized, name_or_alias)


def get_llm(
    model_name: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> ChatOpenAI | None:
    """Returns a configured ChatOpenAI client pointing to DeepSeek.

    Args:
        model_name: Model identifier or alias (e.g. 'deepseek-flash' or 'deepseek-v4-pro').
        temperature: Sampling temperature (defaults to settings.DEFAULT_TEMPERATURE).
        max_tokens: Maximum tokens in generated completion.
        base_url: API endpoint URL (defaults to settings.DEEPSEEK_BASE_URL).
        api_key: DeepSeek API key (defaults to settings.effective_llm_api_key).
        extra_body: Optional extra payload (e.g. thinking mode parameters).

    Returns:
        ChatOpenAI instance, or None if no API key is configured (enabling mock development mode).
    """
    resolved_api_key = settings.effective_llm_api_key if api_key is None else api_key
    if not resolved_api_key or not resolved_api_key.strip():
        logger.warning("No DEEPSEEK_API_KEY configured. Running in mock fallback mode.")
        return None

    canonical_model = resolve_model_name(model_name)
    resolved_base_url = (
        base_url
        or settings.DEEPSEEK_BASE_URL
        or settings.LLM_BASE_URL
        or DEFAULT_BASE_URL
    )
    resolved_temp = (
        temperature if temperature is not None else settings.DEFAULT_TEMPERATURE
    )

    client_kwargs: dict[str, Any] = {
        "model": canonical_model,
        "api_key": resolved_api_key,
        "base_url": resolved_base_url,
        "temperature": resolved_temp,
    }

    if max_tokens is not None:
        client_kwargs["max_tokens"] = max_tokens

    if extra_body:
        client_kwargs["extra_body"] = extra_body


    logger.debug(
        "Initializing DeepSeek ChatOpenAI [model=%s, base_url=%s, temperature=%s]",
        canonical_model,
        resolved_base_url,
        resolved_temp,
    )

    return ChatOpenAI(**client_kwargs)
