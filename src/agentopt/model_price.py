"""
Token pricing for LLM models.

Prices are loaded from ``model_price.json`` (per million tokens in USD).
The JSON file uses OpenRouter model naming: ``"provider/model-name"``.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple


def _find_price_file() -> Optional[Path]:
    """Locate ``model_price.json``.

    Search order:
      1. Current working directory.
      2. Walk up from this module's file to find the project root.
    """
    name = "model_price.json"

    # 1. CWD
    cwd_path = Path.cwd() / name
    if cwd_path.is_file():
        return cwd_path

    # 2. Walk up from this file's directory.
    current = Path(__file__).resolve().parent
    for _ in range(10):
        candidate = current / name
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent

    return None


def _load_prices() -> Dict[str, Tuple[float, float]]:
    path = _find_price_file()
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: (v["input_price"], v["output_price"]) for k, v in data.items()}


MODEL_PRICES: Dict[str, Tuple[float, float]] = _load_prices()


def get_model_price(
    model_name: str, custom_prices: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Optional[Tuple[float, float]]:
    """Look up (input_$/MTok, output_$/MTok) for a model.

    Checks *custom_prices* first (if provided), then the built-in table.
    Returns ``None`` if the model is not found.
    """
    if custom_prices and model_name in custom_prices:
        return custom_prices[model_name]
    if model_name in MODEL_PRICES:
        return MODEL_PRICES[model_name]
    # Fallback: bare name → suffix match (e.g. "gpt-4o" matches "openai/gpt-4o")
    suffix = "/" + model_name
    for key in MODEL_PRICES:
        if key.endswith(suffix):
            return MODEL_PRICES[key]
    return None


def compute_price(
    input_tokens: Dict[str, int],
    output_tokens: Dict[str, int],
    custom_prices: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Optional[float]:
    """Compute total cost in USD from per-model token dicts.

    Returns ``None`` if any model involved has unknown pricing.
    """
    total = 0.0
    all_models = set(input_tokens) | set(output_tokens)
    for model in all_models:
        price = get_model_price(model, custom_prices=custom_prices)
        if price is None:
            return None
        inp_price_per_mtok, out_price_per_mtok = price
        inp = input_tokens.get(model, 0)
        out = output_tokens.get(model, 0)
        total += inp * inp_price_per_mtok / 1_000_000
        total += out * out_price_per_mtok / 1_000_000
    return total
