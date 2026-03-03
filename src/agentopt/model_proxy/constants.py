"""Shared constants for model proxy and model selection."""

import os
from typing import Dict, List, Optional, Tuple

# Model name fields to check on LLM objects, in priority order.
MODEL_FIELDS = ("model", "model_name", "model_id")


# Mapping of model provider prefixes to required environment variables.
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
}


def check_api_key(model_name: str) -> Optional[str]:
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

            msg = check_api_key(name)
            if msg is not None:
                warnings.append(msg)

    return warnings, []
