"""Shared utilities for all benchmarks.

Provides make_llm(), model helpers, and pricing for OpenRouter models.
Uses langchain-openai's ChatOpenAI with OpenRouter's OpenAI-compatible API.
"""

from __future__ import annotations

import os
from typing import Any


# ---------------------------------------------------------------------------
# OpenRouter configuration
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

# Models known to NOT support native tool/function calling.
# Update this set as you add models. OpenRouter docs list FC support per model.
NO_TOOL_CALLING_MODELS: set[str] = set()


def display_name(model: str) -> str:
    """Convert a model string to a human-readable display name.

    For OpenRouter models like 'anthropic/claude-sonnet-4', returns as-is
    (already human-readable). Strips 'openrouter/' prefix if present.
    """
    if model.startswith("openrouter/"):
        return model[len("openrouter/"):]
    return model


def supports_tool_calling(model: str) -> bool:
    """Return True if the model supports native tool/function calling."""
    name = display_name(model)
    return name not in NO_TOOL_CALLING_MODELS


# ---------------------------------------------------------------------------
# Content extraction helpers
# ---------------------------------------------------------------------------

def extract_text_content(content: Any) -> str:
    """Extract text from a model response content field.

    Handles:
    - Plain strings (most models)
    - Lists of dicts with {"type": "text", "text": "..."} blocks
    - Strips reasoning/thinking blocks (if any model returns them)
    - Returns empty string if no text content found
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") in ("reasoning_content", "thinking"):
                    continue  # skip reasoning blocks
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts) if text_parts else ""
    return str(content) if content else ""


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def make_llm(model: str, temperature: float = 0.0, **kwargs) -> Any:
    """Create a LangChain ChatOpenAI pointing at OpenRouter.

    Args:
        model: OpenRouter model ID (e.g. 'anthropic/claude-sonnet-4',
               'openai/gpt-4o', 'google/gemini-2.5-flash').
        temperature: Sampling temperature (default 0.0).
        **kwargs: Extra kwargs passed to ChatOpenAI.

    Requires OPENROUTER_API_KEY env var (or pass api_key in kwargs).

    By default, disables reasoning/thinking mode to match Bedrock behavior.
    gpt-oss models require reasoning, so they use 'low' instead of 'none'.
    Pass reasoning_effort=<value> in kwargs to override.
    """
    from langchain_openai import ChatOpenAI

    api_key = kwargs.pop("api_key", None) or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Set it in your environment or .env file."
        )

    # Disable reasoning by default to match Bedrock behavior.
    # gpt-oss models mandate reasoning — use 'low' as minimum.
    model_kwargs = kwargs.pop("model_kwargs", {})
    if "reasoning_effort" not in model_kwargs:
        if model.startswith("openai/gpt-oss"):
            model_kwargs["reasoning_effort"] = "low"
        else:
            model_kwargs["reasoning_effort"] = "none"

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE_URL,
        model_kwargs=model_kwargs,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Selector imports
# ---------------------------------------------------------------------------

def get_selectors() -> dict[str, Any]:
    """Return the selector name -> class mapping."""
    from agentopt import (
        BruteForceModelSelector,
        HillClimbingModelSelector,
        ArmEliminationModelSelector,
        RandomSearchModelSelector,
    )
    try:
        from agentopt import BayesianOptimizationModelSelector
    except ImportError:
        BayesianOptimizationModelSelector = None
    try:
        from agentopt import EpsilonLUCBModelSelector
    except ImportError:
        EpsilonLUCBModelSelector = None
    try:
        from agentopt import ThresholdBanditSEModelSelector
    except ImportError:
        ThresholdBanditSEModelSelector = None

    selectors: dict[str, Any] = {
        "brute_force": BruteForceModelSelector,
        "hill_climbing": HillClimbingModelSelector,
        "arm_elimination": ArmEliminationModelSelector,
        "random_search": RandomSearchModelSelector,
    }
    if BayesianOptimizationModelSelector:
        selectors["bayesian_optimization"] = BayesianOptimizationModelSelector
    if EpsilonLUCBModelSelector:
        selectors["epsilon_lucb"] = EpsilonLUCBModelSelector
    if ThresholdBanditSEModelSelector:
        selectors["threshold_se"] = ThresholdBanditSEModelSelector
    return selectors


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def add_common_cli_args(parser) -> None:
    """Add CLI arguments shared by all benchmarks."""
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Models to evaluate (OpenRouter model IDs, e.g. anthropic/claude-sonnet-4)",
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
    parser.add_argument("--verbose", action="store_true", help="Detailed logging")
    parser.add_argument(
        "--max-concurrent", type=int, default=20,
        help="Max concurrent requests per model (default: 20)",
    )
    parser.add_argument(
        "--per-combo", action="store_true",
        help="Use per-combo concurrency (all combos run simultaneously)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL file for per-sample results",
    )


def resolve_models(args) -> list[str]:
    """Resolve --models into a list of OpenRouter model IDs."""
    if args.models:
        return args.models
    raise ValueError(
        "No models specified. Use --models to provide OpenRouter model IDs, e.g.:\n"
        "  --models anthropic/claude-sonnet-4 openai/gpt-4o google/gemini-2.5-flash"
    )


def build_selector_kwargs(args) -> dict:
    """Build extra kwargs for selector from CLI args."""
    kwargs: dict = {}
    if args.selector == "random_search":
        kwargs["sample_fraction"] = args.sample_fraction
    return kwargs
