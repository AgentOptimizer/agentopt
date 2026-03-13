"""Shared constants for model proxy and model selection."""

import os
from typing import Dict, List, Optional, Tuple

# Model name fields to check on LLM objects, in priority order.
MODEL_FIELDS = ("model", "model_name", "model_id")


# Mapping of model provider prefixes to native environment variables.
PROVIDER_API_KEYS: Dict[str, str] = {
    "anthropic/": "ANTHROPIC_API_KEY",
    "openai/": "OPENAI_API_KEY",
    "google/": "GOOGLE_API_KEY",
    "gemini/": "GEMINI_API_KEY",
    "mistral/": "MISTRAL_API_KEY",
    "cohere/": "COHERE_API_KEY",
    "groq/": "GROQ_API_KEY",
    "together_ai/": "TOGETHER_API_KEY",
    "openrouter/": "OPENROUTER_API_KEY",
    "deepseek/": "DEEPSEEK_API_KEY",
    "fireworks_ai/": "FIREWORKS_API_KEY",
    "replicate/": "REPLICATE_API_TOKEN",
    "huggingface/": "HUGGINGFACE_API_KEY",
    "bedrock/": "AWS_ACCESS_KEY_ID",
    "vertex_ai/": "GOOGLE_APPLICATION_CREDENTIALS",
    "azure/": "AZURE_API_KEY",
    "litellm/": "LITELLM_API_KEY",
}

# Heuristic: bare model name patterns → provider.
_BARE_NAME_HEURISTICS: List[Tuple[Tuple[str, ...], str, str]] = [
    # (prefixes_or_substrings, provider, env_var)
    (("gpt-", "o1-", "o3-", "o4-"), "openai", "OPENAI_API_KEY"),
    (("claude",), "anthropic", "ANTHROPIC_API_KEY"),
    (("gemini",), "google", "GOOGLE_API_KEY"),
]


def detect_provider(model_name: str) -> Tuple[str, str, Optional[str]]:
    """Detect provider from model name, strip prefix, resolve API key.

    Returns ``(provider, clean_name, api_key)``.

    Resolution order:
      1. Explicit prefix (e.g. ``"openai/gpt-4o"`` → ``("openai", "gpt-4o", key)``).
         Checks the native API key first; falls back to ``OPENROUTER_API_KEY``.
      2. Bare-name heuristics (e.g. ``"gpt-4o"`` → openai, ``"claude-..."`` → anthropic).
      3. Default — returns ``("default", model_name, OPENAI_API_KEY)``.
    """
    # 1. Explicit prefix match.
    for prefix, env_var in PROVIDER_API_KEYS.items():
        if model_name.startswith(prefix):
            provider = prefix.rstrip("/")
            clean_name = model_name[len(prefix) :]
            api_key = os.getenv(env_var)
            if api_key:
                return provider, clean_name, api_key
            # Fallback to OpenRouter when native key is missing.
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_key:
                return "openrouter", clean_name, openrouter_key
            return provider, clean_name, None

    # 2. Bare-name heuristics.
    lower = model_name.lower()
    for patterns, provider, env_var in _BARE_NAME_HEURISTICS:
        if any(lower.startswith(p) for p in patterns):
            api_key = os.getenv(env_var)
            if api_key:
                return provider, model_name, api_key
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_key:
                return "openrouter", model_name, openrouter_key
            return provider, model_name, None

    # 3. Default — try OpenAI key.
    return "default", model_name, os.getenv("OPENAI_API_KEY")


def no_key_error(provider_env: str, model_name: str) -> ValueError:
    """Standard error when no API key or proxy is available."""
    return ValueError(
        f"No API key found for '{model_name}'. "
        f"Set {provider_env} or OPENROUTER_API_KEY."
    )


def _check_api_key(model_name: str) -> Optional[str]:
    """Check if the required API key for a model is set in the environment.

    Args:
        model_name: Model name string (e.g. "anthropic/claude-3-haiku-20240307").

    Returns:
        An error message string if the key is missing, or None if the key is present
        (or the provider is not recognized).
    """
    for prefix, env_var in PROVIDER_API_KEYS.items():
        if model_name.startswith(prefix):
            if not os.environ.get(env_var):
                return (
                    f"Model '{model_name}' requires the {env_var} environment variable, "
                    f"but it is not set. Please set {env_var} or remove this model "
                    f"from the candidate list."
                )
            return None
    return None


def validate_model_candidates(
    model_candidates: Dict,
) -> Tuple[List[str], List[str]]:
    """Validate API keys for all model candidates before evaluation.

    Args:
        model_candidates: Dictionary mapping ModelProxy to list of model names/objects.

    Returns:
        Tuple of (warnings, errors) — warnings for missing keys, errors is empty
        (reserved for future hard failures).
    """
    warnings: List[str] = []
    seen = set()

    for proxy, candidates in model_candidates.items():
        for candidate in candidates:
            name = candidate if isinstance(candidate, str) else None
            if name is None:
                for field in MODEL_FIELDS:
                    if hasattr(candidate, field):
                        name = str(getattr(candidate, field))
                        break
            if name is None or name in seen:
                continue
            seen.add(name)

            msg = _check_api_key(name)
            if msg is not None:
                warnings.append(msg)

    return warnings, []
