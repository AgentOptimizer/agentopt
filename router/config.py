"""Router configuration."""

from dataclasses import dataclass, field
from typing import List


# Canonical combo orderings — MUST match scores.json keys exactly.
# 1-tuple benchmarks: 9 models, sorted alphabetically.
MODELS_1TUPLE = [
    "agent=Claude 3 Haiku",
    "agent=Claude Haiku 4.5",
    "agent=Claude Opus 4.6",
    "agent=Kimi K2.5",
    "agent=Ministral 3 8B",
    "agent=Qwen3 32B",
    "agent=Qwen3 Next 80B A3B",
    "agent=gpt-oss-120b",
    "agent=gpt-oss-20b",
]

# 2-tuple benchmarks: 81 combos (9×9), sorted alphabetically.
_ROLE_MODELS = [
    "Claude 3 Haiku",
    "Claude Haiku 4.5",
    "Claude Opus 4.6",
    "Kimi K2.5",
    "Ministral 3 8B",
    "Qwen3 32B",
    "Qwen3 Next 80B A3B",
    "gpt-oss-120b",
    "gpt-oss-20b",
]

# HotpotQA: planner × solver
COMBOS_HOTPOTQA = [
    f"planner={p} + solver={s}"
    for p in _ROLE_MODELS
    for s in _ROLE_MODELS
]

# MathQA: answer × critic
COMBOS_MATHQA = [
    f"answer={a} + critic={c}"
    for a in _ROLE_MODELS
    for c in _ROLE_MODELS
]

# Per-combo cost in dollars (total cost across all samples, from brute_force_results.csv).
# Used for cost-aware evaluation only — not in training.
COMBO_COSTS = {
    # GPQA (from cached_results/gpqa/brute_force_results.csv)
    "agent=Claude 3 Haiku": 0.0560,
    "agent=Claude Haiku 4.5": 0.5169,
    "agent=Claude Opus 4.6": 2.4692,
    "agent=Kimi K2.5": 1.1346,
    "agent=Ministral 3 8B": 0.0071,
    "agent=Qwen3 32B": 0.0732,
    "agent=Qwen3 Next 80B A3B": 0.1329,
    "agent=gpt-oss-120b": 0.1901,
    "agent=gpt-oss-20b": 0.1303,
}

# Benchmark token mapping
BENCHMARK_TOKENS = {
    "gpqa": "[GPQA]",
    "bfcl": "[BFCL]",
    "hotpotqa": "[HOTPOTQA]",
    "mathqa": "[MATHQA]",
}


@dataclass
class RouterConfig:
    """Configuration for the BERT router."""

    # Model
    model_name: str = "answerdotai/ModernBERT-large"
    max_length: int = 8192

    # Heads (hierarchical: all heads output n_models dimensions)
    n_models: int = 9    # number of candidate models per role
    agg: str = "max"     # 2-tuple marginal aggregation: "max" or "mean"

    # Training
    freeze_encoder: bool = False  # if True, only train linear heads
    epochs: int = 5
    batch_size: int = 16       # effective batch size
    micro_batch: int = 2       # actual batch per forward pass (gradient accumulation)
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    test_size: float = 0.2

    # Data
    scores_path: str = "router/scores.json"
    output_dir: str = "router/checkpoints"

    # Benchmarks to train on
    benchmarks: List[str] = field(
        default_factory=lambda: ["gpqa", "bfcl", "hotpotqa", "mathqa"]
    )
