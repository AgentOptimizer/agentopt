"""
Model factory for creating LangChain model objects from string names.
Handles provider detection and fallback chains (native → LiteLLM → OpenRouter).
"""

from typing import Any, Union, List, Dict
import os
from dotenv import load_dotenv

load_dotenv()


def _openrouter_fallback(model_name: str) -> Any:
    """Create a ChatOpenAI pointed at OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
    )


def _litellm_fallback(model_name: str) -> Any:
    """Create a ChatOpenAI pointed at a LiteLLM proxy server."""
    api_key = os.getenv("LITELLM_API_KEY")
    base_url = os.getenv("LITELLM_API_BASE")
    if not api_key or not base_url:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
    )


def _proxy_fallback(model_name: str) -> Any:
    """Try LiteLLM first, then OpenRouter."""
    return _litellm_fallback(model_name) or _openrouter_fallback(model_name)


def _no_key_error(provider: str, model_name: str) -> ValueError:
    """Standard error when no API key or proxy is available."""
    return ValueError(
        f"No API key found for '{model_name}'. "
        f"Set {provider} or LITELLM_API_KEY+LITELLM_API_BASE or OPENROUTER_API_KEY."
    )


def create_model_from_string(model_name: str) -> Any:
    """
    Create a LangChain model object from a string name.

    Automatically detects provider based on prefix/name:
    - "openai/" or "gpt-*" / "o1-*" / "o3-*" → OpenAI
    - "anthropic/" or "claude*" → Anthropic
    - "google/" or "gemini*" → Google Gemini
    - "bedrock/" → AWS Bedrock
    - "litellm/" → LiteLLM proxy
    - Otherwise → LiteLLM proxy → OpenRouter

    Fallback chain: native API key → LiteLLM proxy → OpenRouter

    Args:
        model_name: Model name string (e.g., "openai/gpt-4o", "bedrock/anthropic.claude-3-haiku-20240307-v1:0")

    Returns:
        LangChain model object
    """
    # ---- Explicit prefix checks (order matters: most specific first) ----

    # --- AWS Bedrock ---
    if model_name.startswith("bedrock/"):
        clean_name = model_name.removeprefix("bedrock/")
        aws_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_region = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION"))
        if aws_key and aws_region:
            try:
                from langchain_aws import ChatBedrockConverse

                return ChatBedrockConverse(
                    model=clean_name,
                    region_name=aws_region,
                )
            except ImportError:
                pass
        fallback = _proxy_fallback(model_name)
        if fallback:
            return fallback
        raise _no_key_error(
            "AWS_ACCESS_KEY_ID + AWS_DEFAULT_REGION", model_name
        )

    # --- LiteLLM explicit prefix ---
    if model_name.startswith("litellm/"):
        clean_name = model_name.removeprefix("litellm/")
        fallback = _litellm_fallback(clean_name)
        if fallback:
            return fallback
        raise ValueError(
            f"LiteLLM requested for '{model_name}' but LITELLM_API_KEY and "
            "LITELLM_API_BASE are not both set."
        )

    # --- OpenAI ---
    if model_name.startswith("openai/"):
        clean_name = model_name.removeprefix("openai/")
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=clean_name, api_key=openai_key)
        fallback = _proxy_fallback(model_name)
        if fallback:
            return fallback
        raise _no_key_error("OPENAI_API_KEY", model_name)

    # --- Google / Gemini ---
    if model_name.startswith("google/") or "gemini" in model_name.lower():
        clean_name = model_name.removeprefix("google/")
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                return ChatGoogleGenerativeAI(
                    model=clean_name, google_api_key=google_key
                )
            except ImportError:
                pass
        fallback = _proxy_fallback(model_name)
        if fallback:
            return fallback
        raise _no_key_error("GOOGLE_API_KEY", model_name)

    # --- Anthropic / Claude ---
    if model_name.startswith("anthropic/") or "claude" in model_name.lower():
        clean_name = model_name.removeprefix("anthropic/")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                from langchain_anthropic import ChatAnthropic

                return ChatAnthropic(model=clean_name, api_key=anthropic_key)
            except ImportError:
                pass
        fallback = _proxy_fallback(model_name)
        if fallback:
            return fallback
        raise _no_key_error("ANTHROPIC_API_KEY", model_name)

    # --- Default: try native OpenAI, then proxy fallbacks ---
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, api_key=openai_key)

    fallback = _proxy_fallback(model_name)
    if fallback:
        return fallback
    raise _no_key_error("OPENAI_API_KEY", model_name)


def normalize_models(
    models: Union[
        Dict[str, List[Union[str, Any]]], Dict[str, List[str]], Dict[str, List[Any]]
    ],
    convert_strings: bool = False,
) -> Dict[str, List[Any]]:
    """
    Normalize models input.

    Args:
        models: Dictionary mapping attribute paths to list of model names (strings) or model objects
        convert_strings: If True, convert string names to LangChain model objects.
                        If False (default), pass strings through for framework-specific conversion.

    Returns:
        Dictionary mapping attribute paths to list of model objects or strings
    """
    normalized = {}
    for attr_path, model_list in models.items():
        normalized_list = []
        for model in model_list:
            if isinstance(model, str) and convert_strings:
                # Convert string to LangChain model object
                normalized_list.append(create_model_from_string(model))
            else:
                # Pass through as-is (string or model object)
                normalized_list.append(model)
        normalized[attr_path] = normalized_list
    return normalized
