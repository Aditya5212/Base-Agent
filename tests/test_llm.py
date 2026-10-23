from app.core.llm import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEEPSEEK_FLASH,
    DEEPSEEK_PRO,
    get_llm,
    resolve_model_name,
)


def test_deepseek_models_constants():
    assert DEEPSEEK_FLASH == "deepseek-flash"
    assert DEEPSEEK_PRO == "deepseek-v4-pro"
    assert DEFAULT_MODEL == DEEPSEEK_FLASH
    assert DEFAULT_BASE_URL == "https://api.deepseek.com"


def test_resolve_model_name_aliases():
    # Alias and version string to canonical API model name
    assert resolve_model_name("deepseek-flash") == DEEPSEEK_FLASH
    assert resolve_model_name("flash") == DEEPSEEK_FLASH
    assert resolve_model_name("DeepSeek-V4.1-Flash-1017") == DEEPSEEK_FLASH
    assert resolve_model_name("deepseek-v4.1-flash-1017") == DEEPSEEK_FLASH

    assert resolve_model_name("deepseek-v4-pro") == DEEPSEEK_PRO
    assert resolve_model_name("pro") == DEEPSEEK_PRO
    assert resolve_model_name("DeepSeek-V4-Pro-0813") == DEEPSEEK_PRO
    assert resolve_model_name("deepseek-v4-pro-0813") == DEEPSEEK_PRO

    # None returns default
    assert resolve_model_name(None) == DEEPSEEK_FLASH


def test_get_llm_without_api_key():
    # Passing empty string ensures None is returned even if env has a key
    assert get_llm(api_key="") is None


def test_get_llm_with_api_key_default():
    llm = get_llm(api_key="mock-deepseek-key")
    assert llm is not None
    assert llm.model_name == DEEPSEEK_FLASH
    assert llm.openai_api_base == "https://api.deepseek.com"


def test_get_llm_with_pro_model_alias():
    llm = get_llm(model_name="deepseek-v4-pro", api_key="mock-deepseek-key")
    assert llm is not None
    assert llm.model_name == DEEPSEEK_PRO
    assert llm.openai_api_base == "https://api.deepseek.com"
