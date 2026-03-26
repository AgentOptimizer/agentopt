"""Shared Bedrock utilities for all benchmarks.

Provides make_llm(), model mappings, and pricing for the 10 Bedrock
application inference profile models.
"""

from __future__ import annotations

import os
from typing import Any


# ---------------------------------------------------------------------------
# Application inference profile mappings
# ---------------------------------------------------------------------------

_PROFILE_PROVIDERS = {
    "58ii6j0n0zhw": "anthropic",    # Claude 3 Haiku
    "4ax1twcuwbfk": "anthropic",    # Claude Haiku 4.5
    "vqhud2pxz4wy": "anthropic",    # Claude Opus 4.6
    "z3ze3bovw9th": "deepseek",     # DeepSeek R1
    "fkpdj71utboq": "openai",       # gpt-oss-20b
    "d9uiuyipu5b2": "openai",       # gpt-oss-120b
    "nrqbxznvrt7p": "moonshotai",   # Kimi K2.5
    "uj2ujdo7k1qe": "mistral",      # Ministral 3 8B
    "d6kuf8xcphsl": "qwen",         # Qwen3 32B
    "a6jppcyeu4ms": "qwen",         # Qwen3 Next 80B A3B
}

_PROFILE_DISPLAY_NAMES = {
    "58ii6j0n0zhw": "Claude 3 Haiku",
    "4ax1twcuwbfk": "Claude Haiku 4.5",
    "vqhud2pxz4wy": "Claude Opus 4.6",
    "z3ze3bovw9th": "DeepSeek R1",
    "fkpdj71utboq": "gpt-oss-20b",
    "d9uiuyipu5b2": "gpt-oss-120b",
    "nrqbxznvrt7p": "Kimi K2.5",
    "uj2ujdo7k1qe": "Ministral 3 8B",
    "d6kuf8xcphsl": "Qwen3 32B",
    "a6jppcyeu4ms": "Qwen3 Next 80B A3B",
}

# Reverse mapping: display name -> full ARN
_DISPLAY_NAME_TO_ARN = {
    v: f"arn:aws:bedrock:us-east-1:920736616554:application-inference-profile/{k}"
    for k, v in _PROFILE_DISPLAY_NAMES.items()
}

DEFAULT_MODELS = [
    "Claude 3 Haiku",
    "Claude Haiku 4.5",
    "Claude Opus 4.6",
    "DeepSeek R1",
    "gpt-oss-20b",
    "gpt-oss-120b",
    "Kimi K2.5",
    "Ministral 3 8B",
    "Qwen3 32B",
    "Qwen3 Next 80B A3B",
]

# Models known to NOT support native tool/function calling on Bedrock.
NO_TOOL_CALLING_MODELS = {
    "DeepSeek R1",
    "Ministral 3 8B",
}


# ---------------------------------------------------------------------------
# Pricing ($/MTok) — Bedrock on-demand
# ---------------------------------------------------------------------------

_BASE_PRICES = {
    "Claude 3 Haiku": {"input_price": 0.25, "output_price": 1.25},
    "Claude Haiku 4.5": {"input_price": 0.80, "output_price": 4.00},
    "Claude Opus 4.6": {"input_price": 5.00, "output_price": 25.00},
    "DeepSeek R1": {"input_price": 1.35, "output_price": 5.40},
    "gpt-oss-20b": {"input_price": 0.22, "output_price": 0.88},
    "gpt-oss-120b": {"input_price": 1.20, "output_price": 4.80},
    "Kimi K2.5": {"input_price": 0.35, "output_price": 1.40},
    "Ministral 3 8B": {"input_price": 0.04, "output_price": 0.04},
    "Qwen3 32B": {"input_price": 0.17, "output_price": 0.85},
    "Qwen3 Next 80B A3B": {"input_price": 0.25, "output_price": 1.25},
}

# Build BEDROCK_PRICES with both display names and ARN keys
BEDROCK_PRICES: dict[str, dict[str, float]] = dict(_BASE_PRICES)
for _pid, _display in _PROFILE_DISPLAY_NAMES.items():
    if _display in _BASE_PRICES:
        _arn = f"arn:aws:bedrock:us-east-1:920736616554:application-inference-profile/{_pid}"
        BEDROCK_PRICES[_arn] = _BASE_PRICES[_display]


# ---------------------------------------------------------------------------
# Display name helper
# ---------------------------------------------------------------------------

def display_name(model: str) -> str:
    """Convert a model string to a human-readable display name."""
    if "application-inference-profile/" in model:
        profile_id = model.rsplit("/", 1)[-1]
        return _PROFILE_DISPLAY_NAMES.get(profile_id, model)
    if model.startswith("bedrock/"):
        return model[len("bedrock/"):]
    return model


def supports_tool_calling(model: str) -> bool:
    """Return True if the model supports native tool/function calling."""
    name = display_name(model)
    return name not in NO_TOOL_CALLING_MODELS


# ---------------------------------------------------------------------------
# Provider inference fallback
# ---------------------------------------------------------------------------

_PROVIDER_KEYWORDS = {
    "anthropic": ["anthropic", "claude"],
    "meta": ["meta", "llama"],
    "mistral": ["mistral"],
    "amazon": ["amazon", "nova", "titan"],
    "deepseek": ["deepseek"],
    "openai": ["gpt", "openai"],
    "moonshotai": ["kimi", "moonshot"],
    "qwen": ["qwen"],
}


def _infer_provider(model_id: str) -> str | None:
    """Best-effort provider guess from the model ID string."""
    model_lower = model_id.lower()
    for provider, keywords in _PROVIDER_KEYWORDS.items():
        if any(kw in model_lower for kw in keywords):
            return provider
    return None


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def make_llm(model: str, temperature: float = 0.0) -> Any:
    """Create a LangChain chat model for Bedrock.

    Supports display names (e.g. "Claude Opus 4.6"), bedrock/* prefixed
    models, and direct model IDs.
    """
    # Resolve display name to ARN
    if model in _DISPLAY_NAME_TO_ARN:
        arn = _DISPLAY_NAME_TO_ARN[model]
        profile_id = arn.rsplit("/", 1)[-1]
        provider = _PROFILE_PROVIDERS.get(profile_id)
        from langchain_aws import ChatBedrockConverse
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        kwargs: dict[str, Any] = {
            "model": arn, "region_name": region, "temperature": temperature,
        }
        if provider:
            kwargs["provider"] = provider
        return ChatBedrockConverse(**kwargs)

    # bedrock/ prefix
    if model.startswith("bedrock/"):
        model_id = model[len("bedrock/"):]
        from langchain_aws import ChatBedrockConverse
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        kwargs = {"model": model_id, "region_name": region, "temperature": temperature}
        if model_id.startswith("arn:"):
            profile_id = model_id.rsplit("/", 1)[-1]
            provider = _PROFILE_PROVIDERS.get(profile_id) or _infer_provider(model_id)
            if provider:
                kwargs["provider"] = provider
        return ChatBedrockConverse(**kwargs)

    # Direct model ID
    bare = model.split("/")[-1] if "/" in model else model
    if bare.startswith("us.") or bare.startswith("arn:"):
        from langchain_aws import ChatBedrockConverse
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        return ChatBedrockConverse(model=bare, region_name=region, temperature=temperature)
    elif bare.startswith("claude") or model.startswith("anthropic/"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=bare, temperature=temperature)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=bare, temperature=temperature)


# ---------------------------------------------------------------------------
# Selector imports
# ---------------------------------------------------------------------------

def get_selectors() -> dict[str, Any]:
    """Return the selector name -> class mapping."""
    from agentopt import (
        BruteForceModelSelector,
        HillClimbingModelSelector,
        ArmEliminationModelSelector,
        HyperbandModelSelector,
        RandomSearchModelSelector,
    )
    try:
        from agentopt import BayesianOptimizationModelSelector
    except ImportError:
        BayesianOptimizationModelSelector = None

    selectors: dict[str, Any] = {
        "brute_force": BruteForceModelSelector,
        "hill_climbing": HillClimbingModelSelector,
        "arm_elimination": ArmEliminationModelSelector,
        "random_search": RandomSearchModelSelector,
        "hyperband": HyperbandModelSelector,
    }
    if BayesianOptimizationModelSelector:
        selectors["bayesian_optimization"] = BayesianOptimizationModelSelector
    return selectors


def add_common_cli_args(parser) -> None:
    """Add CLI arguments shared by all benchmarks."""
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Models to evaluate by display name",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Run all 10 default Bedrock models",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max samples to evaluate",
    )
    parser.add_argument(
        "--selector", choices=sorted(get_selectors().keys()), default="brute_force",
        help="Model selector algorithm (default: brute_force)",
    )
    parser.add_argument(
        "--mode", choices=["langgraph", "raw"], default="langgraph",
        help="Execution mode (default: langgraph)",
    )
    parser.add_argument("--parallel", action="store_true", help="Parallel evaluation")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--csv", type=str, default=None, help="Save results to CSV")
    parser.add_argument("--plot", type=str, default=None, help="Save plot to file")
    parser.add_argument(
        "--sample-fraction", type=float, default=0.5,
        help="Fraction for random_search (default: 0.5)",
    )
    parser.add_argument(
        "--reduction-factor", type=float, default=3.0,
        help="Reduction factor for hyperband (default: 3.0)",
    )
    parser.add_argument("--verbose", action="store_true", help="Detailed logging")


def resolve_models(args) -> list[str]:
    """Resolve --models / --all-models into a list of model display names."""
    if args.all_models:
        return list(DEFAULT_MODELS)
    if args.models:
        return args.models
    # Default: first two models for quick testing
    return DEFAULT_MODELS[:2]


def build_selector_kwargs(args) -> dict:
    """Build extra kwargs for selector from CLI args."""
    kwargs: dict = {}
    if args.selector == "random_search":
        kwargs["sample_fraction"] = args.sample_fraction
    if args.selector == "hyperband":
        kwargs["reduction_factor"] = args.reduction_factor
    return kwargs
