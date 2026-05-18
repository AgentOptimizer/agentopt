#!/usr/bin/env python3
"""
Offline Selector Simulator v2
==============================
Loads brute-force JSONL (from --output flag) and replays selector decision
logic without any API calls. Works across all benchmarks.

Supports all 7 selectors from the new agentopt package:
  - random_search
  - arm_elimination
  - epsilon_lucb
  - threshold_se
  - hill_climbing
  - bayesian_optimization
  - lm_proposal (proposal only — skips LLM call, picks random or best-guess)

Usage:
    python offline_selector_sim_v2.py \
        --jsonl agentopt/results/gpqa_10sample_bf.jsonl

    python offline_selector_sim_v2.py \
        --jsonl agentopt/results/mathqa_10sample_bf.jsonl \
        --selectors random_search,arm_elimination --seeds 20
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Bedrock pricing ($/MTok) — keyed by ARN suffix (profile ID)
# Updated to official rates as of March 2026
# ---------------------------------------------------------------------------
_BEDROCK_PRICES_BY_SUFFIX: Dict[str, Tuple[float, float]] = {
    "58ii6j0n0zhw": (0.25, 1.25),     # Claude 3 Haiku
    "4ax1twcuwbfk": (1.00, 5.00),     # Claude Haiku 4.5
    "vqhud2pxz4wy": (5.00, 25.00),    # Claude Opus 4.6
    "fkpdj71utboq": (0.07, 0.30),     # gpt-oss-20b
    "d9uiuyipu5b2": (0.15, 0.60),     # gpt-oss-120b
    "nrqbxznvrt7p": (0.60, 3.00),     # Kimi K2.5
    "uj2ujdo7k1qe": (0.15, 0.15),     # Ministral 3 8B
    "d6kuf8xcphsl": (0.15, 0.60),     # Qwen3 32B
    "a6jppcyeu4ms": (0.15, 1.20),     # Qwen3 Next 80B A3B
}


def _compute_sample_cost(input_tokens: Dict[str, int],
                         output_tokens: Dict[str, int]) -> float:
    """Compute cost in USD from token dicts using Bedrock pricing."""
    total = 0.0
    for key, count in input_tokens.items():
        suffix = key.rsplit("/", 1)[-1] if "/" in key else key
        prices = _BEDROCK_PRICES_BY_SUFFIX.get(suffix)
        if prices:
            total += count * prices[0] / 1_000_000
    for key, count in output_tokens.items():
        suffix = key.rsplit("/", 1)[-1] if "/" in key else key
        prices = _BEDROCK_PRICES_BY_SUFFIX.get(suffix)
        if prices:
            total += count * prices[1] / 1_000_000
    return total


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    score: float
    latency_seconds: float
    input_tokens: Dict[str, int]
    output_tokens: Dict[str, int]
    cost: float = 0.0


# Lookup: model_name -> {datapoint_index -> SampleResult}
LookupTable = Dict[str, Dict[int, SampleResult]]


def load_jsonl(path: str) -> Tuple[List[str], List[int], LookupTable]:
    """Load brute-force JSONL into a lookup table.

    Returns (model_names_sorted, datapoint_indices_sorted, table).
    """
    table: LookupTable = {}
    models: Set[str] = set()
    datapoints: Set[int] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            combo = json.loads(line)
            model = combo["model_name"]
            models.add(model)

            for dp in combo.get("datapoint_results", []):
                dp_idx = dp["datapoint_index"]
                score = dp["score"]
                latency = dp["latency_seconds"]
                input_tokens = dp.get("input_tokens", {})
                output_tokens = dp.get("output_tokens", {})
                cost = _compute_sample_cost(input_tokens, output_tokens)

                if model not in table:
                    table[model] = {}
                table[model][dp_idx] = SampleResult(
                    score=score,
                    latency_seconds=latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                )
                datapoints.add(dp_idx)

    return sorted(models), sorted(datapoints), table


# ---------------------------------------------------------------------------
# Stats helpers (matching base.py exactly)
# ---------------------------------------------------------------------------

def _compute_stats(scores: List[float]) -> Tuple[float, float]:
    """Return (mean, sample_std). Matches BaseModelSelector._compute_stats."""
    n = len(scores)
    if n == 0:
        return 0.0, 0.5
    mean = sum(scores) / n
    if n < 2:
        return mean, 0.5
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
    return mean, math.sqrt(variance)


def _is_dominated(scores_i: List[float], scores_j: List[float],
                   confidence: float = 1.0) -> bool:
    """Return True if arm i is statistically dominated by arm j."""
    n_i, n_j = len(scores_i), len(scores_j)
    if n_i == 0 or n_j == 0:
        return False
    mu_i, std_i = _compute_stats(scores_i)
    mu_j, std_j = _compute_stats(scores_j)
    se_i = std_i / math.sqrt(n_i)
    se_j = std_j / math.sqrt(n_j)
    return mu_i + confidence * se_i < mu_j - confidence * se_j


def _confidence_bounds(scores: List[float], confidence: float = 1.96
                       ) -> Tuple[float, float]:
    """Return (LCB, UCB) for a list of scores."""
    mu, std = _compute_stats(scores)
    n = len(scores)
    if n == 0:
        return 0.0, 1.0
    se = std / math.sqrt(n)
    return mu - confidence * se, mu + confidence * se


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------

@dataclass
class ModelSummary:
    model_name: str
    accuracy: float
    latency_seconds: float
    cost: float
    n_samples_evaluated: int
    is_best: bool = False


@dataclass
class SimulationResult:
    selector: str
    seed: Optional[int]
    params: Dict[str, Any]
    best_model: Optional[str]
    best_accuracy: float
    total_evaluations: int
    total_cost: float
    models_tested: int
    found_true_best: bool
    compute_time_seconds: float = 0.0
    model_results: List[ModelSummary] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Brute-force ground truth
# ---------------------------------------------------------------------------

def compute_ground_truth(models: List[str], datapoints: List[int],
                         table: LookupTable) -> Tuple[str, float]:
    """Compute brute-force best model (accuracy, tiebreak latency then cost)."""
    best_name = None
    best_acc = float("-inf")
    best_lat = float("inf")
    best_cost = float("inf")
    tol = 1e-9

    for model in models:
        samples = table.get(model, {})
        available = [samples[dp] for dp in datapoints if dp in samples]
        if not available:
            continue
        acc = sum(s.score for s in available) / len(available)
        lat = sum(s.latency_seconds for s in available) / len(available)
        cost = sum(s.cost for s in available)

        if acc > best_acc + tol:
            best_name, best_acc, best_lat, best_cost = model, acc, lat, cost
        elif abs(acc - best_acc) <= tol and lat < best_lat - tol:
            best_name, best_acc, best_lat, best_cost = model, acc, lat, cost
        elif (abs(acc - best_acc) <= tol and abs(lat - best_lat) <= tol
              and cost < best_cost - tol):
            best_name, best_acc, best_lat, best_cost = model, acc, lat, cost

    return best_name, best_acc


def _pick_best(evaluated: Dict[int, Tuple[float, float, float]],
               models: List[str]) -> Tuple[Optional[str], float]:
    """Pick best from evaluated models. Returns (name, accuracy)."""
    best_name = None
    best_acc = float("-inf")
    best_lat = float("inf")
    tol = 1e-9
    for idx, (acc, lat, cost) in evaluated.items():
        if acc > best_acc + tol:
            best_name, best_acc, best_lat = models[idx], acc, lat
        elif abs(acc - best_acc) <= tol and lat < best_lat - tol:
            best_name, best_acc, best_lat = models[idx], acc, lat
    return best_name, best_acc


def _evaluate_model_full(idx: int, models: List[str], datapoints: List[int],
                         table: LookupTable) -> Tuple[float, float, float, int, float]:
    """Evaluate a model on all datapoints. Returns (acc, lat, cost, n_eval, compute_time)."""
    model = models[idx]
    samples = table.get(model, {})
    available = [samples[dp] for dp in datapoints if dp in samples]
    n_eval = len(available)
    acc = sum(s.score for s in available) / n_eval if n_eval else 0.0
    lat = sum(s.latency_seconds for s in available) / n_eval if n_eval else 0.0
    cost = sum(s.cost for s in available)
    ct = sum(s.latency_seconds for s in available)
    return acc, lat, cost, n_eval, ct


# ---------------------------------------------------------------------------
# Selector: Random Search
# ---------------------------------------------------------------------------

def simulate_random_search(
    models: List[str], datapoints: List[int], table: LookupTable,
    sample_fraction: float = 0.25, seed: int = 42,
) -> SimulationResult:
    n_total = len(models)
    sample_size = max(1, min(n_total, math.ceil(n_total * sample_fraction)))

    rng = random.Random(seed)
    sampled = sorted(rng.sample(range(n_total), sample_size)) if sample_size < n_total else list(range(n_total))

    total_evals = 0
    total_cost = 0.0
    compute_time = 0.0
    model_results = []
    best_name, best_acc, best_lat = None, float("-inf"), float("inf")
    tol = 1e-9

    for idx in sampled:
        acc, lat, cost, n_eval, ct = _evaluate_model_full(idx, models, datapoints, table)
        total_evals += n_eval
        total_cost += cost
        compute_time += ct
        model_results.append(ModelSummary(models[idx], acc, lat, cost, n_eval))
        if acc > best_acc + tol or (abs(acc - best_acc) <= tol and lat < best_lat - tol):
            best_name, best_acc, best_lat = models[idx], acc, lat

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    gt_name, _ = compute_ground_truth(models, datapoints, table)
    return SimulationResult("random_search", seed,
        {"sample_fraction": sample_fraction},
        best_name, best_acc, total_evals, total_cost,
        len(sampled), best_name == gt_name, compute_time, model_results)


# ---------------------------------------------------------------------------
# Selector: Arm Elimination
# ---------------------------------------------------------------------------

def simulate_arm_elimination(
    models: List[str], datapoints: List[int], table: LookupTable,
    n_initial: Optional[int] = None, growth_factor: float = 2.0,
    confidence: float = 1.0, seed: Optional[int] = None,
) -> SimulationResult:
    n_samples = len(datapoints)
    if n_initial is None:
        n_initial = max(1, n_samples // 10)

    dp_order = list(datapoints)
    if seed is not None:
        random.Random(seed).shuffle(dp_order)

    n_models = len(models)
    combo_scores: Dict[int, List[float]] = {i: [] for i in range(n_models)}
    combo_latencies: Dict[int, List[float]] = {i: [] for i in range(n_models)}
    combo_costs: Dict[int, float] = {i: 0.0 for i in range(n_models)}
    active: Set[int] = set(range(n_models))
    total_evals = 0
    compute_time = 0.0

    offset = 0
    batch_size = n_initial

    while active and offset < n_samples:
        batch_end = min(offset + batch_size, n_samples)
        batch_dps = dp_order[offset:batch_end]

        for idx in sorted(active):
            samples = table.get(models[idx], {})
            for dp in batch_dps:
                if dp in samples:
                    s = samples[dp]
                    combo_scores[idx].append(s.score)
                    combo_latencies[idx].append(s.latency_seconds)
                    combo_costs[idx] += s.cost
                    compute_time += s.latency_seconds
                    total_evals += 1

        newly_eliminated = set()
        for i in active:
            for j in active:
                if i != j and _is_dominated(combo_scores[i], combo_scores[j], confidence):
                    newly_eliminated.add(i)
                    break
        active -= newly_eliminated

        if len(active) <= 1:
            break
        offset = batch_end
        batch_size = max(1, int(batch_size * growth_factor))

    model_results = []
    best_name, best_acc, best_lat = None, float("-inf"), float("inf")
    tol = 1e-9
    for idx in range(n_models):
        scores = combo_scores[idx]
        lats = combo_latencies[idx]
        acc = sum(scores) / len(scores) if scores else 0.0
        lat = sum(lats) / len(lats) if lats else 0.0
        cost = combo_costs[idx]
        model_results.append(ModelSummary(models[idx], acc, lat, cost, len(scores)))
        if acc > best_acc + tol or (abs(acc - best_acc) <= tol and lat < best_lat - tol):
            best_name, best_acc, best_lat = models[idx], acc, lat

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    gt_name, _ = compute_ground_truth(models, datapoints, table)
    return SimulationResult("arm_elimination", seed,
        {"n_initial": n_initial, "growth_factor": growth_factor, "confidence": confidence},
        best_name, best_acc, total_evals, sum(combo_costs.values()),
        n_models, best_name == gt_name, compute_time, model_results)


# ---------------------------------------------------------------------------
# Selector: Epsilon-LUCB
# ---------------------------------------------------------------------------

def simulate_epsilon_lucb(
    models: List[str], datapoints: List[int], table: LookupTable,
    epsilon: float = 0.01, confidence: float = 1.96,
    seed: Optional[int] = None,
) -> SimulationResult:
    """Simulate Epsilon-LUCB: focus on top arm and closest challenger."""
    n_samples = len(datapoints)
    n_models = len(models)

    dp_order = list(datapoints)
    if seed is not None:
        random.Random(seed).shuffle(dp_order)

    combo_scores: Dict[int, List[float]] = {i: [] for i in range(n_models)}
    combo_costs: Dict[int, float] = {i: 0.0 for i in range(n_models)}
    total_evals = 0
    compute_time = 0.0

    # Initial: give each model 1 sample
    for idx in range(n_models):
        if dp_order:
            dp = dp_order[0]
            samples = table.get(models[idx], {})
            if dp in samples:
                s = samples[dp]
                combo_scores[idx].append(s.score)
                combo_costs[idx] += s.cost
                compute_time += s.latency_seconds
                total_evals += 1

    # Iterate through remaining samples
    for dp_i in range(1, n_samples):
        dp = dp_order[dp_i]

        # Find top arm (highest mean) and challenger (highest UCB excluding top)
        means = [(idx, sum(combo_scores[idx]) / len(combo_scores[idx])
                  if combo_scores[idx] else 0.0) for idx in range(n_models)]
        means.sort(key=lambda x: x[1], reverse=True)
        top_idx = means[0][0]
        top_lcb = _confidence_bounds(combo_scores[top_idx], confidence)[0]

        # Find challenger: arm with highest UCB that isn't top
        best_challenger_idx = None
        best_challenger_ucb = float("-inf")
        for idx in range(n_models):
            if idx == top_idx:
                continue
            _, ucb = _confidence_bounds(combo_scores[idx], confidence)
            if ucb > best_challenger_ucb:
                best_challenger_ucb = ucb
                best_challenger_idx = idx

        # Check stopping condition
        if best_challenger_idx is not None and top_lcb - best_challenger_ucb >= epsilon:
            break

        # Evaluate top and challenger on this sample
        for idx in [top_idx, best_challenger_idx]:
            if idx is None:
                continue
            samples = table.get(models[idx], {})
            if dp in samples:
                s = samples[dp]
                combo_scores[idx].append(s.score)
                combo_costs[idx] += s.cost
                compute_time += s.latency_seconds
                total_evals += 1

    model_results = []
    best_name, best_acc, best_lat = None, float("-inf"), float("inf")
    tol = 1e-9
    for idx in range(n_models):
        scores = combo_scores[idx]
        acc = sum(scores) / len(scores) if scores else 0.0
        samples = table.get(models[idx], {})
        available = [samples[dp] for dp in datapoints if dp in samples]
        lat = sum(s.latency_seconds for s in available) / len(available) if available else 0.0
        cost = combo_costs[idx]
        model_results.append(ModelSummary(models[idx], acc, lat, cost, len(scores)))
        if acc > best_acc + tol or (abs(acc - best_acc) <= tol and lat < best_lat - tol):
            best_name, best_acc, best_lat = models[idx], acc, lat

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    gt_name, _ = compute_ground_truth(models, datapoints, table)
    return SimulationResult("epsilon_lucb", seed,
        {"epsilon": epsilon, "confidence": confidence},
        best_name, best_acc, total_evals, sum(combo_costs.values()),
        n_models, best_name == gt_name, compute_time, model_results)


# ---------------------------------------------------------------------------
# Selector: Threshold Successive Elimination
# ---------------------------------------------------------------------------

def simulate_threshold_se(
    models: List[str], datapoints: List[int], table: LookupTable,
    threshold: float = 0.5, confidence: float = 1.96,
    seed: Optional[int] = None,
) -> SimulationResult:
    """Simulate Threshold SE: classify models above/below threshold."""
    n_samples = len(datapoints)
    n_models = len(models)

    dp_order = list(datapoints)
    if seed is not None:
        random.Random(seed).shuffle(dp_order)

    combo_scores: Dict[int, List[float]] = {i: [] for i in range(n_models)}
    combo_costs: Dict[int, float] = {i: 0.0 for i in range(n_models)}
    active: Set[int] = set(range(n_models))
    total_evals = 0
    compute_time = 0.0

    for dp_i in range(n_samples):
        dp = dp_order[dp_i]
        if not active:
            break

        for idx in sorted(active):
            samples = table.get(models[idx], {})
            if dp in samples:
                s = samples[dp]
                combo_scores[idx].append(s.score)
                combo_costs[idx] += s.cost
                compute_time += s.latency_seconds
                total_evals += 1

        # Eliminate models whose CI is entirely on one side of threshold
        newly_eliminated = set()
        for idx in active:
            if len(combo_scores[idx]) < 2:
                continue
            lcb, ucb = _confidence_bounds(combo_scores[idx], confidence)
            if ucb < threshold or lcb > threshold:
                newly_eliminated.add(idx)

        active -= newly_eliminated
        if not active:
            break

    model_results = []
    best_name, best_acc, best_lat = None, float("-inf"), float("inf")
    tol = 1e-9
    for idx in range(n_models):
        scores = combo_scores[idx]
        acc = sum(scores) / len(scores) if scores else 0.0
        samples = table.get(models[idx], {})
        available = [samples[dp] for dp in datapoints if dp in samples]
        lat = sum(s.latency_seconds for s in available) / len(available) if available else 0.0
        cost = combo_costs[idx]
        model_results.append(ModelSummary(models[idx], acc, lat, cost, len(scores)))
        if acc > best_acc + tol or (abs(acc - best_acc) <= tol and lat < best_lat - tol):
            best_name, best_acc, best_lat = models[idx], acc, lat

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    gt_name, _ = compute_ground_truth(models, datapoints, table)
    return SimulationResult("threshold_se", seed,
        {"threshold": threshold, "confidence": confidence},
        best_name, best_acc, total_evals, sum(combo_costs.values()),
        n_models, best_name == gt_name, compute_time, model_results)


# ---------------------------------------------------------------------------
# Selector: Hill Climbing (topology-aware, matches library exactly)
# ---------------------------------------------------------------------------

# Quality ranking for display names (best → worst), matching model_topology.py
_QUALITY_RANKING_DISPLAY: List[str] = [
    "Claude Opus 4.6",
    "gpt-oss-120b",
    "Kimi K2.5",
    "Qwen3 Next 80B A3B",
    "Claude Haiku 4.5",
    "gpt-oss-20b",
    "Qwen3 32B",
    "Claude 3 Haiku",
    "Ministral 3 8B",
]

# Speed ranking for display names (fastest → slowest), matching model_topology.py
_SPEED_RANKING_DISPLAY: List[str] = [
    "Ministral 3 8B",
    "Qwen3 Next 80B A3B",
    "Qwen3 32B",
    "Claude 3 Haiku",
    "Kimi K2.5",
    "gpt-oss-120b",
    "Claude Haiku 4.5",
    "Claude Opus 4.6",
    "gpt-oss-20b",
]


def _parse_combo_name(name: str) -> List[Tuple[str, str]]:
    """Parse 'role=Model + role=Model' into [(role, model), ...]."""
    parts = name.split(" + ")
    result = []
    for p in parts:
        eq = p.index("=")
        result.append((p[:eq], p[eq+1:]))
    return result


def _build_combo_name(positions: List[Tuple[str, str]]) -> str:
    """Build combo name from [(role, model), ...]."""
    return " + ".join(f"{role}={model}" for role, model in positions)


def _sort_by_ranking(candidates: List[str], ranking: List[str]) -> List[str]:
    """Sort candidates by ranking order. Unknown models appended at end."""
    known = [(ranking.index(c), c) for c in candidates if c in ranking]
    unknown = [c for c in candidates if c not in ranking]
    known.sort()
    return [c for _, c in known] + unknown


def _get_higher_quality_neighbor(current: str, candidates: List[str]) -> Optional[str]:
    """Return the next higher-quality candidate (one step up), or None if at top."""
    sorted_cands = _sort_by_ranking(candidates, _QUALITY_RANKING_DISPLAY)
    if current not in sorted_cands:
        return None
    idx = sorted_cands.index(current)
    if idx == 0:
        return None  # already highest quality
    return sorted_cands[idx - 1]


def _get_faster_neighbor(current: str, candidates: List[str]) -> Optional[str]:
    """Return the next faster candidate (one step up in speed), or None if fastest."""
    sorted_cands = _sort_by_ranking(candidates, _SPEED_RANKING_DISPLAY)
    if current not in sorted_cands:
        return None
    idx = sorted_cands.index(current)
    if idx == 0:
        return None  # already fastest
    return sorted_cands[idx - 1]


def _generate_neighbors_hc(
    combo_name: str, models: List[str], base_models: List[str],
    seen: Set[int], rng: random.Random, improve_quality: bool,
) -> List[int]:
    """Generate unseen neighbors that differ by one node (one direction only).

    Matches library's _generate_neighbors: for each node, get ONE neighbor
    (either quality-up or speed-up), skip if already seen.
    """
    positions = _parse_combo_name(combo_name)
    node_indices = list(range(len(positions)))
    rng.shuffle(node_indices)

    neighbors = []
    for pos_idx in node_indices:
        role, current_model = positions[pos_idx]
        if improve_quality:
            neighbor_model = _get_higher_quality_neighbor(current_model, base_models)
        else:
            neighbor_model = _get_faster_neighbor(current_model, base_models)

        if neighbor_model is None:
            continue

        new_positions = list(positions)
        new_positions[pos_idx] = (role, neighbor_model)
        new_name = _build_combo_name(new_positions)
        if new_name in models:
            new_idx = models.index(new_name)
            if new_idx not in seen:
                neighbors.append(new_idx)

    return neighbors


def _get_neighbors_hc(
    combo_idx: int, models: List[str], base_models: List[str],
    seen: Set[int], rng: random.Random, accuracy: float,
) -> List[int]:
    """Get neighbors with quality/speed fallback logic (matches library _get_neighbors).

    If accuracy < 1.0: try quality-up first, fall back to speed-up.
    If accuracy >= 1.0: try speed-up first, fall back to quality-up.
    """
    combo_name = models[combo_idx]
    if accuracy < 1.0:
        neighbors = _generate_neighbors_hc(combo_name, models, base_models, seen, rng, improve_quality=True)
        if not neighbors:
            neighbors = _generate_neighbors_hc(combo_name, models, base_models, seen, rng, improve_quality=False)
    else:
        neighbors = _generate_neighbors_hc(combo_name, models, base_models, seen, rng, improve_quality=False)
        if not neighbors:
            neighbors = _generate_neighbors_hc(combo_name, models, base_models, seen, rng, improve_quality=True)
    return neighbors


def simulate_hill_climbing(
    models: List[str], datapoints: List[int], table: LookupTable,
    seed: int = 42, num_restarts: int = 3, max_iterations: int = 20,
    patience: int = 3,
) -> SimulationResult:
    """Simulate topology-aware hill climbing matching the library exactly.

    - Quality-first neighbor generation with speed fallback
    - One step per node per direction (not ±1 both ways)
    - Patience-based convergence (3 no-improvement iterations)
    - Shared seen set across restarts
    """
    n_models = len(models)
    rng = random.Random(seed)

    # Extract unique base model names from combo names
    base_models_set: Set[str] = set()
    for m in models:
        for _, model in _parse_combo_name(m):
            base_models_set.add(model)
    base_models = sorted(base_models_set)

    evaluated: Dict[int, Tuple[float, float, float]] = {}
    total_evals = 0
    total_cost = 0.0
    compute_time = 0.0
    model_n_evals: Dict[int, int] = {}
    seen: Set[int] = set()  # shared across restarts, like library

    def eval_model(idx):
        nonlocal total_evals, total_cost, compute_time
        if idx in evaluated:
            return evaluated[idx]
        acc, lat, cost, n_eval, ct = _evaluate_model_full(idx, models, datapoints, table)
        evaluated[idx] = (acc, lat, cost)
        total_evals += n_eval
        total_cost += cost
        compute_time += ct
        model_n_evals[idx] = n_eval
        return (acc, lat, cost)

    global_best_idx = None
    global_best_acc = float("-inf")
    global_best_lat = float("inf")
    tol = 1e-9

    for restart in range(num_restarts):
        # Pick random unseen start
        unseen = [i for i in range(n_models) if i not in seen]
        if not unseen:
            break
        current = rng.choice(unseen)
        seen.add(current)
        current_acc, current_lat, _ = eval_model(current)

        best_restart_idx = current
        best_restart_acc = current_acc
        best_restart_lat = current_lat
        no_improve_count = 0

        for iteration in range(max_iterations):
            # Get neighbors (quality-first with speed fallback)
            neighbors = _get_neighbors_hc(current, models, base_models, seen, rng, current_acc)
            if not neighbors:
                break

            # Evaluate all neighbors, find best improving one
            best_neighbor = None
            best_n_acc = float("-inf")
            best_n_lat = float("inf")
            for n_idx in neighbors:
                seen.add(n_idx)
                n_acc, n_lat, _ = eval_model(n_idx)
                if n_acc > best_n_acc + tol or (abs(n_acc - best_n_acc) <= tol and n_lat < best_n_lat):
                    best_neighbor = n_idx
                    best_n_acc = n_acc
                    best_n_lat = n_lat

            # Check if best neighbor improves over current
            improves = (
                best_neighbor is not None
                and (best_n_acc > current_acc + tol
                     or (abs(best_n_acc - current_acc) <= tol and best_n_lat < current_lat))
            )

            if improves:
                current = best_neighbor
                current_acc = best_n_acc
                current_lat = best_n_lat
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Update restart best
            if current_acc > best_restart_acc + tol or (
                abs(current_acc - best_restart_acc) <= tol and current_lat < best_restart_lat
            ):
                best_restart_idx = current
                best_restart_acc = current_acc
                best_restart_lat = current_lat

            if no_improve_count >= patience:
                break

        # Update global best
        if best_restart_acc > global_best_acc + tol or (
            abs(best_restart_acc - global_best_acc) <= tol and best_restart_lat < global_best_lat
        ):
            global_best_idx = best_restart_idx
            global_best_acc = best_restart_acc
            global_best_lat = best_restart_lat

    best_name = models[global_best_idx] if global_best_idx is not None else models[0]
    best_acc = global_best_acc if global_best_idx is not None else 0.0

    model_results = []
    for idx in range(n_models):
        if idx in evaluated:
            acc, lat, cost = evaluated[idx]
            n_eval = model_n_evals[idx]
        else:
            acc, lat, cost, n_eval = 0.0, 0.0, 0.0, 0
        model_results.append(ModelSummary(models[idx], acc, lat, cost, n_eval,
                                          is_best=(models[idx] == best_name)))

    gt_name, _ = compute_ground_truth(models, datapoints, table)
    return SimulationResult("hill_climbing", seed,
        {"num_restarts": num_restarts, "max_iterations": max_iterations, "patience": patience},
        best_name, best_acc, total_evals, total_cost,
        len(evaluated), best_name == gt_name, compute_time, model_results)


# ---------------------------------------------------------------------------
# Selector: Bayesian Optimization
# ---------------------------------------------------------------------------

def _check_botorch():
    try:
        import torch  # noqa: F401
        from botorch.models.gp_regression_mixed import MixedSingleTaskGP  # noqa: F401
        return True
    except ImportError:
        return False


def simulate_bayesian_optimization(
    models: List[str], datapoints: List[int], table: LookupTable,
    n_initial_random: Optional[int] = None, n_iterations: Optional[int] = None,
    seed: int = 42,
) -> SimulationResult:
    import torch
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.gp_regression_mixed import MixedSingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    n_models = len(models)
    if n_initial_random is None:
        n_initial_random = min(4, n_models)
    if n_iterations is None:
        n_iterations = max(0, int(0.2 * n_models))

    rng = random.Random(seed)
    evaluated: Set[int] = set()
    X_list, Y_list = [], []
    model_data: Dict[int, Tuple[float, float, float, int, float]] = {}
    total_evals = 0

    def eval_model(idx):
        nonlocal total_evals
        acc, lat, cost, n_eval, ct = _evaluate_model_full(idx, models, datapoints, table)
        model_data[idx] = (acc, lat, cost, n_eval, ct)
        total_evals += n_eval
        return acc

    # Initial random
    pool = list(range(n_models))
    rng.shuffle(pool)
    for idx in pool[:n_initial_random]:
        evaluated.add(idx)
        acc = eval_model(idx)
        X_list.append([idx])
        Y_list.append(acc)

    # BO loop
    for _ in range(n_iterations):
        unseen = [c for c in range(n_models) if c not in evaluated]
        if not unseen:
            break

        if len(X_list) < 2:
            idx = rng.choice(unseen)
        else:
            train_X = torch.tensor(X_list, dtype=torch.float64)
            train_Y = torch.tensor(Y_list, dtype=torch.float64).unsqueeze(-1)
            model_gp = MixedSingleTaskGP(train_X=train_X, train_Y=train_Y, cat_dims=[0])
            mll = ExactMarginalLogLikelihood(model_gp.likelihood, model_gp)
            fit_gpytorch_mll(mll)
            cand_X = torch.tensor([[c] for c in unseen], dtype=torch.float64)
            acq = LogExpectedImprovement(model=model_gp, best_f=train_Y.max().item())
            with torch.no_grad():
                ei = acq(cand_X.unsqueeze(1))
            idx = unseen[ei.argmax().item()]

        evaluated.add(idx)
        acc = eval_model(idx)
        X_list.append([idx])
        Y_list.append(acc)

    model_results = []
    best_name, best_acc, best_lat = None, float("-inf"), float("inf")
    tol = 1e-9
    for idx in range(n_models):
        if idx in model_data:
            acc, lat, cost, n_eval, ct = model_data[idx]
        else:
            acc, lat, cost, n_eval = 0.0, 0.0, 0.0, 0
        model_results.append(ModelSummary(models[idx], acc, lat, cost, n_eval))
        if idx in evaluated:
            if acc > best_acc + tol or (abs(acc - best_acc) <= tol and lat < best_lat - tol):
                best_name, best_acc, best_lat = models[idx], acc, lat

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    gt_name, _ = compute_ground_truth(models, datapoints, table)
    total_cost = sum(d[2] for d in model_data.values())
    compute_time = sum(d[4] for d in model_data.values())

    return SimulationResult("bayesian_optimization", seed,
        {"n_initial_random": n_initial_random, "n_iterations": n_iterations},
        best_name, best_acc, total_evals, total_cost,
        len(evaluated), best_name == gt_name, compute_time, model_results)


# ---------------------------------------------------------------------------
# Selector: LM Proposal (live LLM call + offline eval lookup)
# ---------------------------------------------------------------------------

# Display name -> (input $/MTok, output $/MTok)
_DISPLAY_PRICES = {
    "Claude 3 Haiku": (0.25, 1.25),
    "Claude Haiku 4.5": (1.00, 5.00),
    "Claude Opus 4.6": (5.00, 25.00),
    "gpt-oss-20b": (0.07, 0.30),
    "gpt-oss-120b": (0.15, 0.60),
    "Kimi K2.5": (0.60, 3.00),
    "Ministral 3 8B": (0.15, 0.15),
    "Qwen3 32B": (0.15, 0.60),
    "Qwen3 Next 80B A3B": (0.15, 1.20),
}


def _extract_nodes_and_candidates(models: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    """Extract node names and per-node candidate lists from combo model names.

    E.g. 'answer=Opus + critic=Haiku' -> nodes=['answer','critic'],
         candidates={'answer': ['Opus','Haiku',...], 'critic': [...]}
    """
    node_candidates: Dict[str, set] = {}
    for model_name in models:
        parts = model_name.split(" + ")
        for part in parts:
            if "=" in part:
                node, candidate = part.split("=", 1)
                if node not in node_candidates:
                    node_candidates[node] = set()
                node_candidates[node].add(candidate)
            else:
                # 1-tuple: single node called "agent"
                if "agent" not in node_candidates:
                    node_candidates["agent"] = set()
                node_candidates["agent"].add(part)

    nodes = sorted(node_candidates.keys())
    candidates = {n: sorted(node_candidates[n]) for n in nodes}
    return nodes, candidates


def _load_dataset_previews(benchmark: str, limit: int = 200) -> List[Dict[str, Any]]:
    """Load actual dataset samples for preview. Returns list of {input, expected}."""
    base = os.path.dirname(os.path.abspath(__file__))
    previews = []

    if benchmark == "gpqa":
        path = os.path.join(base, "benchmarks/GPQA/data/gpqa_diamond.jsonl")
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                row = json.loads(line)
                choices = {"A": row["Correct Answer"], "B": row["Incorrect Answer 1"],
                           "C": row["Incorrect Answer 2"], "D": row["Incorrect Answer 3"]}
                q = (f"{row['Question']}\n\nChoose one of the following:\n"
                     + "\n".join(f"  ({k}) {v}" for k, v in choices.items())
                     + "\n\nAnswer with ONLY the letter (A, B, C, or D).")
                previews.append({"input": q, "expected": "A"})  # always A since we put correct in A

    elif benchmark == "bfcl":
        path = os.path.join(base, "benchmarks/BFCL/data/BFCL_v3_multi_turn_base.json")
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                entry = json.loads(line)
                # Show first turn's user message as preview
                turns = entry.get("question", [])
                first_msg = turns[0][0]["content"] if turns and turns[0] else "N/A"
                previews.append({
                    "input": first_msg,
                    "expected": f"(multi-turn function calling, {len(turns)} turns)",
                })

    elif benchmark == "hotpotqa":
        path = os.path.join(base, "benchmarks/HotpotQA/data/hotpot_dev_distractor_v1.json")
        with open(path) as f:
            data = json.load(f)
        for i, entry in enumerate(data[:limit]):
            q = entry["question"]
            previews.append({"input": q, "expected": entry["answer"]})

    elif benchmark == "mathqa":
        try:
            from datasets import load_dataset
            ds = load_dataset("allenai/math_qa", split="test")
            for i, row in enumerate(ds):
                if i >= limit:
                    break
                previews.append({
                    "input": f"{row['Problem']}\n\nOptions: {row['options']}",
                    "expected": row["correct"],
                })
        except Exception as e:
            print(f"  Warning: could not load MathQA dataset: {e}")

    return previews


def _build_lm_proposal_prompt(
    nodes: List[str],
    candidates: Dict[str, List[str]],
    benchmark_name: str,
    dataset_preview: List[Dict[str, Any]],
) -> str:
    """Build the prompt for the LM proposer (matches real LMProposalModelSelector)."""
    # Build nodes info with pricing
    nodes_info = []
    for node in nodes:
        node_entry = {"node_name": node, "candidates": []}
        for c in candidates[node]:
            cand = {"name": c}
            if c in _DISPLAY_PRICES:
                cand["input_price_per_mtok"] = _DISPLAY_PRICES[c][0]
                cand["output_price_per_mtok"] = _DISPLAY_PRICES[c][1]
            node_entry["candidates"].append(cand)
        nodes_info.append(node_entry)

    # Build response example
    example = {
        "combination": {node: candidates[node][0] for node in nodes},
        "reasoning": "Your explanation here.",
    }

    sections = [
        "# Task\n"
        "You are an expert AI model selector. You will be given a multi-agent "
        "workflow where each node can use one of several candidate LLMs. "
        "Your job is to select the best combination of models for the nodes.\n",

        "# The objective to target when selecting the model combination:\n"
        "maximize accuracy and then minimize latency and cost\n",

        f"# Benchmark: {benchmark_name}\n",

        "# Agent Pipeline\n"
        "The agent has the following nodes and each can be assigned one of its candidate models.\n"
        f"```json\n{json.dumps(nodes_info, indent=2)}\n```\n",

        "# Dataset Preview\n"
        "Below are sample inputs and their expected outputs. Use these to understand "
        "the task complexity and choose models accordingly.\n"
        f"```json\n{json.dumps(dataset_preview, indent=2)}\n```\n",

        "# Response Format\n"
        "Respond with a JSON object like this example:\n"
        f"```json\n{json.dumps(example)}\n```\n",

        "# Constraints\n"
        "- Each key in `combination` must be a node name from the pipeline above.\n"
        "- Each value must be a candidate model name from that node's candidates list.\n"
        "- All nodes must be included.\n"
        "- Return exactly one combination.\n",
    ]

    return "\n".join(sections)


def _parse_lm_proposal_response(
    text: str, nodes: List[str], candidates: Dict[str, List[str]],
    models: List[str],
) -> Optional[str]:
    """Parse LLM response and return matching model_name from models list, or None."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    combination = payload.get("combination", {})
    if set(combination.keys()) != set(nodes):
        return None

    # Validate all candidates exist
    for node in nodes:
        if combination[node] not in candidates[node]:
            return None

    # Build the model_name string to match against our models list
    if len(nodes) == 1 and nodes[0] == "agent":
        target = f"agent={combination['agent']}"
    else:
        parts = [f"{node}={combination[node]}" for node in sorted(nodes)]
        target = " + ".join(parts)

    # Find exact match
    if target in models:
        return target

    # Try flexible matching
    for model in models:
        match = True
        for node in nodes:
            if f"{node}={combination[node]}" not in model:
                match = False
                break
        if match:
            return model

    return None


_DATASET_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def simulate_lm_proposal(
    models: List[str], datapoints: List[int], table: LookupTable,
    seed: int = 42, preview_size: int = 10,
    proposer_model: str = "gpt-4.1",
    benchmark: str = "auto",
) -> SimulationResult:
    """Simulate LM Proposal: call an LLM to propose the best combo, then look up its score.

    Uses the first ``preview_size`` dataset samples as preview (deterministic),
    temperature=0.0, and max 3 retries — matching the agentopt package exactly.
    Fully deterministic: identical result regardless of seed.
    """

    # Extract node structure
    nodes, candidates = _extract_nodes_and_candidates(models)

    # Auto-detect benchmark
    if benchmark == "auto":
        for model in models:
            if "answer=" in model and "critic=" in model:
                benchmark = "mathqa"
                break
            elif "planner=" in model and "solver=" in model:
                benchmark = "hotpotqa"
                break
            elif "agent=" in model:
                benchmark = "gpqa"  # default for 1-tuple; overridden by CLI
                break

    benchmark_descriptions = {
        "gpqa": "GPQA Diamond (graduate-level science multiple choice questions)",
        "bfcl": "BFCL v3 Multi-Turn (multi-turn function calling with backend APIs)",
        "hotpotqa": "HotpotQA (multi-hop question answering with planner+solver agents)",
        "mathqa": "MathQA (self-reflective math with answer+critic agents)",
    }
    benchmark_name = benchmark_descriptions.get(benchmark, benchmark)

    # Load dataset (cached across seeds)
    if benchmark not in _DATASET_CACHE:
        print(f"  Loading {benchmark} dataset for previews...")
        _DATASET_CACHE[benchmark] = _load_dataset_previews(benchmark, limit=200)
    all_samples = _DATASET_CACHE[benchmark]

    if not all_samples:
        print(f"  Warning: no dataset samples loaded for {benchmark}")
        # Fallback to first model
        proposed_model = models[0]
    else:
        # Use first N samples (deterministic, matches agentopt package)
        dataset_preview = all_samples[:preview_size]

        prompt = _build_lm_proposal_prompt(nodes, candidates, benchmark_name, dataset_preview)

        # Call the proposer LLM with retry on rate limits
        import time as _time
        proposed_model = None
        try:
            from openai import OpenAI
            client = OpenAI()
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert model-selection assistant. "
                        "Analyze the agent pipeline, candidate models, and dataset, "
                        "then return a single JSON object with your recommended "
                        "model combination."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=proposer_model,
                        temperature=0.0,
                        response_format={"type": "json_object"},
                        messages=messages,
                    )
                    raw = response.choices[0].message.content or ""
                    proposed_model = _parse_lm_proposal_response(raw, nodes, candidates, models)
                    break
                except Exception as api_err:
                    if "429" in str(api_err) or "rate_limit" in str(api_err):
                        wait = 2 ** attempt * 3  # 3, 6, 12, 24, 48s
                        _time.sleep(wait)
                    else:
                        print(f"  LM Proposal error: {api_err}")
                        break
        except Exception as e:
            print(f"  LM Proposal error: {e}")

    if proposed_model is None:
        proposed_model = models[0]
        print(f"  LM Proposal: failed to parse response, falling back to {proposed_model}")

    # Look up the proposed combo's score from brute force data
    acc, lat, cost, n_eval, ct = _evaluate_model_full(
        models.index(proposed_model), models, datapoints, table
    )

    model_results = [ModelSummary(proposed_model, acc, lat, cost, n_eval, is_best=True)]

    gt_name, _ = compute_ground_truth(models, datapoints, table)
    return SimulationResult(
        "lm_proposal", seed,
        {"proposer_model": proposer_model, "preview_size": preview_size},
        proposed_model, acc, n_eval, cost,
        1, proposed_model == gt_name, ct, model_results,
    )


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_multi_seed(selector_fn, models, datapoints, table,
                   n_seeds, base_seed=0, **kwargs):
    results = []
    for i in range(n_seeds):
        result = selector_fn(models, datapoints, table, seed=base_seed + i, **kwargs)
        results.append(result)
    return results


def summarize_multi_seed(results, gt_name, gt_acc, models=None,
                         datapoints=None, table=None):
    if models is not None and datapoints is not None and table is not None:
        true_accs = {}
        for model in models:
            samples = table.get(model, {})
            available = [samples[dp] for dp in datapoints if dp in samples]
            true_accs[model] = sum(s.score for s in available) / len(available) if available else 0.0
        accuracies = [true_accs.get(r.best_model, 0.0) for r in results]
    else:
        accuracies = [r.best_accuracy for r in results]

    evals = [r.total_evaluations for r in results]
    costs = [r.total_cost for r in results]
    found_best = [r.found_true_best for r in results]
    compute_times = [r.compute_time_seconds for r in results]
    n = len(results)
    mean_acc = sum(accuracies) / n
    std_acc = (sum((a - mean_acc)**2 for a in accuracies) / max(n - 1, 1)) ** 0.5

    return {
        "selector": results[0].selector,
        "n_seeds": n,
        "ground_truth_best": gt_name,
        "ground_truth_accuracy": gt_acc,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "min_accuracy": min(accuracies),
        "max_accuracy": max(accuracies),
        "found_true_best_pct": sum(found_best) / n * 100,
        "mean_evaluations": sum(evals) / n,
        "mean_cost": sum(costs) / n,
        "mean_compute_time": sum(compute_times) / n,
        "params": results[0].params,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_single_result(result, gt_name, gt_acc):
    print(f"\n{'─'*60}")
    print(f"  {result.selector}  (seed={result.seed})")
    print(f"  Params: {result.params}")
    print(f"{'─'*60}")
    print(f"  {'Model':<50} {'Acc':>7} {'Samples':>8} {'Cost':>10}")
    print(f"  {'─'*50} {'─'*7} {'─'*8} {'─'*10}")
    for mr in sorted(result.model_results, key=lambda x: (-x.accuracy, x.latency_seconds)):
        marker = " *" if mr.is_best else ""
        print(f"  {mr.model_name:<50} {mr.accuracy:>7.3f} {mr.n_samples_evaluated:>8}"
              f" ${mr.cost:>8.4f}{marker}")
    print(f"\n  Best: {result.best_model}  ({result.best_accuracy:.3f})")
    print(f"  Total evaluations: {result.total_evaluations}")
    print(f"  Total cost: ${result.total_cost:.4f}")
    print(f"  Found true best ({gt_name}): {'YES' if result.found_true_best else 'NO'}")


def print_summary(summary):
    print(f"\n{'='*60}")
    print(f"  {summary['selector']}  —  {summary['n_seeds']} seeds")
    print(f"  Params: {summary['params']}")
    print(f"{'='*60}")
    print(f"  Ground truth best: {summary['ground_truth_best']}"
          f"  ({summary['ground_truth_accuracy']:.3f})")
    print(f"  Mean accuracy found:  {summary['mean_accuracy']:.4f}"
          f"  +/- {summary['std_accuracy']:.4f}")
    print(f"  Accuracy range:       [{summary['min_accuracy']:.4f},"
          f" {summary['max_accuracy']:.4f}]")
    print(f"  Found true best:      {summary['found_true_best_pct']:.1f}%"
          f"  ({int(summary['found_true_best_pct'] * summary['n_seeds'] / 100)}"
          f"/{summary['n_seeds']})")
    print(f"  Mean evaluations:     {summary['mean_evaluations']:.0f}")
    print(f"  Mean total cost:      ${summary['mean_cost']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Selector: Matrix UCB (plain)
# ---------------------------------------------------------------------------

def simulate_matrix_ucb(
    models: List[str], datapoints: List[int], table: LookupTable,
    a: float = 1.0, observation_budget_fraction: float = 1.0,
    seed: int = 42,
) -> SimulationResult:
    """Offline simulation of MatrixUCBModelSelector.

    Replays the UCB cell-selection logic from matrix_ucb.py using the lookup
    table instead of live API calls. Each step picks the combo with the highest
    UCB (mean + sqrt(a/count)), then evaluates a batch of unseen datapoints
    for that combo.

    observation_budget_fraction: fraction of available cells to observe before
    stopping (default 1.0 = full grid = same as brute force).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n_combos = len(models)
    n_dp = len(datapoints)
    max_cells = 20  # match default max_concurrent

    # Build availability mask: True if cell exists in lookup table
    available = np.zeros((n_combos, n_dp), dtype=bool)
    for i, model_name in enumerate(models):
        model_data = table.get(model_name, {})
        for j, dp_id in enumerate(datapoints):
            if dp_id in model_data:
                available[i, j] = True
    n_available = int(available.sum())

    # Budget: stop after observing this many cells
    budget = max(1, int(math.ceil(observation_budget_fraction * n_available))) if observation_budget_fraction < 1.0 else n_available

    # observed tracks scores; unavailable cells use -inf sentinel (excluded from UCB)
    observed = np.full((n_combos, n_dp), np.nan, dtype=np.float64)
    cell_costs: Dict[Tuple[int, int], float] = {}
    total_evals = 0
    total_cost = 0.0

    # Per-combo count of available datapoints (for "fully observed" check)
    available_per_combo = available.sum(axis=1)

    while True:
        filled = int(np.sum(~np.isnan(observed)))
        if filled >= budget:
            break

        # UCB selection (same as _ucb_plain_next_batch)
        bounds = np.full(n_combos, np.inf, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            mus = np.nanmean(observed, axis=1)
        counts = np.sum(~np.isnan(observed), axis=1)
        mask = counts > 0
        bounds[mask] = mus[mask] + np.sqrt(a / counts[mask])
        fully_observed = counts >= available_per_combo
        bounds[fully_observed] = -np.inf
        if bool(np.all(fully_observed)):
            break

        best_combo = int(np.argmax(bounds))
        # Only pick from available AND unobserved cells
        unobserved_dp = np.where(np.isnan(observed[best_combo]) & available[best_combo])[0]
        if len(unobserved_dp) == 0:
            break

        remaining = budget - total_evals
        k = min(max_cells, len(unobserved_dp), remaining)
        if k <= 0:
            break
        pick = rng.permutation(len(unobserved_dp))[:k]
        dp_indices = unobserved_dp[pick]

        model_name = models[best_combo]
        model_data = table.get(model_name, {})
        for dp_local_idx in dp_indices:
            if total_evals >= budget:
                break
            dp_id = datapoints[dp_local_idx]
            sr = model_data[dp_id]
            observed[best_combo, dp_local_idx] = sr.score
            cell_costs[(best_combo, dp_local_idx)] = sr.cost
            total_evals += 1
            total_cost += sr.cost

    # Find best model from observed data
    best_name = None
    best_acc = float("-inf")
    best_lat = float("inf")
    tol = 1e-9
    model_results = []
    for i, model_name in enumerate(models):
        valid = ~np.isnan(observed[i])
        if not np.any(valid):
            continue
        acc = float(np.nanmean(observed[i]))
        model_data = table.get(model_name, {})
        lats = [model_data[datapoints[j]].latency_seconds
                for j in range(n_dp) if not np.isnan(observed[i, j]) and datapoints[j] in model_data]
        lat = sum(lats) / len(lats) if lats else 0.0
        cost_i = sum(cell_costs.get((i, j), 0.0) for j in range(n_dp))
        n_eval_i = int(np.sum(valid))
        model_results.append(ModelSummary(model_name, acc, lat, cost_i, n_eval_i))
        if acc > best_acc + tol or (abs(acc - best_acc) <= tol and lat < best_lat - tol):
            best_name, best_acc, best_lat = model_name, acc, lat

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    # Compute TRUE full-dataset accuracy of the selected combo
    # (not the partial-observation estimate which is biased at low budgets)
    if best_name is not None:
        model_data = table.get(best_name, {})
        all_scores = [model_data[dp].score for dp in datapoints if dp in model_data]
        true_best_acc = sum(all_scores) / len(all_scores) if all_scores else 0.0
    else:
        true_best_acc = 0.0

    gt_name, _ = compute_ground_truth(models, datapoints, table)
    return SimulationResult(
        "matrix_ucb", seed, {"a": a, "observation_budget_fraction": observation_budget_fraction},
        best_name, true_best_acc, total_evals, total_cost,
        len(model_results), best_name == gt_name, 0.0, model_results,
    )


# ---------------------------------------------------------------------------
# Selector: Matrix UCB with Low-Rank Factorization
# ---------------------------------------------------------------------------

def simulate_matrix_ucb_lrf(
    models: List[str], datapoints: List[int], table: LookupTable,
    rank: int = 1, ensemble_size: int = 64,
    warmup_percentage: float = 0.05,
    regularizer_weight: float = 0.1, drop_probability: float = 0.05,
    iterations: int = 10, eta: float = 5.0,
    observation_budget_fraction: float = 1.0,
    seed: int = 42,
) -> SimulationResult:
    """Offline simulation of MatrixUCBLRFModelSelector.

    Replays the LRF cell-selection logic using the lookup table. During warmup
    (until warmup_percentage of cells are filled), picks random cells. After
    warmup, uses ensemble ALS factorization to estimate unseen cells and picks
    the combo+datapoint with highest uncertainty.

    observation_budget_fraction: fraction of available cells to observe before
    stopping (default 1.0 = full grid = same as brute force).
    """
    import numpy as np
    import torch
    from agentopt.model_selection.matrix_ucb_factorization import Factorization

    n_combos = len(models)
    n_dp = len(datapoints)
    max_cells = 20

    # Build availability mask: True if cell exists in lookup table
    available_np = np.zeros((n_combos, n_dp), dtype=bool)
    for i, model_name in enumerate(models):
        model_data = table.get(model_name, {})
        for j, dp_id in enumerate(datapoints):
            if dp_id in model_data:
                available_np[i, j] = True
    available_t = torch.from_numpy(available_np)
    n_available = int(available_t.sum().item())
    available_per_combo = available_t.sum(dim=1)

    # Budget: stop after observing this many cells
    budget = max(1, int(math.ceil(observation_budget_fraction * n_available))) if observation_budget_fraction < 1.0 else n_available

    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    observed_t = torch.full((n_combos, n_dp), float("nan"), dtype=torch.float64)
    cell_costs: Dict[Tuple[int, int], float] = {}
    total_evals = 0
    total_cost = 0.0

    def _observe_cell(combo_i, dp_local_i):
        nonlocal total_evals, total_cost
        if not available_t[combo_i, dp_local_i]:
            return  # skip unavailable cells
        if total_evals >= budget:
            return  # budget exhausted
        model_name = models[combo_i]
        dp_id = datapoints[dp_local_i]
        model_data = table.get(model_name, {})
        sr = model_data[dp_id]
        observed_t[combo_i, dp_local_i] = sr.score
        cell_costs[(combo_i, dp_local_i)] = sr.cost
        total_evals += 1
        total_cost += sr.cost

    while True:
        filled = int((~observed_t.isnan()).sum().item())
        if filled >= budget:
            break
        mc_step = min(max_cells, budget - filled)
        if mc_step <= 0:
            break

        obs_frac = filled / n_available if n_available > 0 else 1.0

        if obs_frac < warmup_percentage:
            # Warmup: random available+unobserved cells
            candidates = available_t & observed_t.isnan()
            combo_idx, dp_idx = torch.where(candidates)
            n_cand = combo_idx.numel()
            k = min(mc_step, n_cand)
            if k <= 0:
                break
            perm = torch.randperm(n_cand)[:k]
            for p in perm:
                _observe_cell(int(combo_idx[p]), int(dp_idx[p]))
        else:
            # LRF + UCB
            fac = Factorization(
                n_combos, n_dp, rank, ensemble_size,
                regularizer_weight=regularizer_weight,
                drop_probability=drop_probability,
            ).to(dtype=torch.float64)
            fac.fit(observed_t, iterations=iterations)
            matrix_approx = fac()
            entry_mus = matrix_approx.mean(0)
            entry_stds = matrix_approx.std(0)
            entry_stds = torch.nan_to_num(entry_stds, nan=0.0, posinf=0.0, neginf=0.0)
            entry_ucb = entry_mus + eta * entry_stds

            observed_mask = ~observed_t.isnan()
            entry_ucb = entry_ucb.clone()
            entry_ucb[observed_mask] = observed_t[observed_mask]
            combo_ucb = entry_ucb.mean(1)

            counts = observed_mask.sum(1)
            bounds = torch.full((n_combos,), float("inf"), dtype=torch.float64)
            valid = counts > 0
            bounds[valid] = combo_ucb[valid]
            fully_observed = counts >= available_per_combo
            bounds[fully_observed] = float("-inf")
            if bool(fully_observed.all()):
                break

            best_combo = int(torch.argmax(bounds).item())
            # Only pick from available AND unobserved cells
            combo_entry_stds = entry_stds[best_combo].clone()
            unavail_or_obs = observed_mask[best_combo] | ~available_t[best_combo]
            combo_entry_stds[unavail_or_obs] = -1.0
            n_unobs = int((available_t[best_combo] & ~observed_mask[best_combo]).sum().item())
            k = min(mc_step, max(n_unobs, 0))
            if k <= 0:
                break
            _, top_dp = torch.topk(combo_entry_stds, k, largest=True)

            for dp_local_i in top_dp.tolist():
                _observe_cell(best_combo, dp_local_i)

    # Find best model
    best_name = None
    best_acc = float("-inf")
    best_lat = float("inf")
    tol = 1e-9
    model_results = []

    for i, model_name in enumerate(models):
        valid = ~observed_t[i].isnan()
        if not valid.any():
            continue
        acc = float(observed_t[i][valid].mean().item())
        model_data = table.get(model_name, {})
        lats = [model_data[datapoints[j]].latency_seconds
                for j in range(n_dp) if not observed_t[i, j].isnan() and datapoints[j] in model_data]
        lat = sum(lats) / len(lats) if lats else 0.0
        cost_i = sum(cell_costs.get((i, j), 0.0) for j in range(n_dp))
        n_eval_i = int(valid.sum().item())
        model_results.append(ModelSummary(model_name, acc, lat, cost_i, n_eval_i))
        if acc > best_acc + tol or (abs(acc - best_acc) <= tol and lat < best_lat - tol):
            best_name, best_acc, best_lat = model_name, acc, lat

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    # Compute TRUE full-dataset accuracy of the selected combo
    # (not the partial-observation estimate which is biased at low budgets)
    if best_name is not None:
        model_data = table.get(best_name, {})
        all_scores = [model_data[dp].score for dp in datapoints if dp in model_data]
        true_best_acc = sum(all_scores) / len(all_scores) if all_scores else 0.0
    else:
        true_best_acc = 0.0

    gt_name, _ = compute_ground_truth(models, datapoints, table)
    return SimulationResult(
        "matrix_ucb_lrf", seed,
        {"rank": rank, "ensemble_size": ensemble_size,
         "warmup_percentage": warmup_percentage, "eta": eta,
         "observation_budget_fraction": observation_budget_fraction},
        best_name, true_best_acc, total_evals, total_cost,
        len(model_results), best_name == gt_name, 0.0, model_results,
    )


ALL_SELECTORS = [
    "random_search", "arm_elimination", "epsilon_lucb",
    "threshold_se", "hill_climbing", "bayesian_optimization",
    "lm_proposal", "matrix_ucb", "matrix_ucb_lrf",
]


def main():
    parser = argparse.ArgumentParser(
        description="Offline selector simulator v2 — replay selectors on brute-force JSONL data",
    )
    parser.add_argument("--jsonl", required=True, help="Path to brute-force JSONL file")
    parser.add_argument("--selectors", default="all",
                        help=f"Comma-separated: {','.join(ALL_SELECTORS)},all")
    parser.add_argument("--seeds", type=int, default=1, help="Number of random seeds (default 1)")
    parser.add_argument("--base-seed", type=int, default=42, help="Starting seed (default 42)")
    parser.add_argument("--output", default=None, help="Path to write summary CSV")

    # Selector-specific params
    parser.add_argument("--rs-fraction", type=float, default=0.25)
    parser.add_argument("--arm-confidence", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hc-restarts", type=int, default=3, help="Hill climbing num_restarts (default 3)")
    parser.add_argument("--proposer-model", type=str, default="gpt-4.1", help="LM Proposal proposer model (default gpt-4.1)")
    parser.add_argument("--preview-size", type=int, default=10, help="LM Proposal preview sample count (default 10)")
    parser.add_argument("--benchmark", type=str, default="auto",
                        choices=["auto", "gpqa", "bfcl", "hotpotqa", "mathqa"],
                        help="Benchmark name for LM Proposal dataset loading (default auto-detect)")

    args = parser.parse_args()

    print(f"Loading: {args.jsonl}")
    models, datapoints, table = load_jsonl(args.jsonl)
    print(f"  Models: {len(models)}")
    print(f"  Samples: {len(datapoints)}")
    print(f"  Total entries: {sum(len(v) for v in table.values())}")

    gt_name, gt_acc = compute_ground_truth(models, datapoints, table)
    print(f"\n  Ground truth best: {gt_name}  ({gt_acc:.4f})")

    bf_evals = sum(len(v) for v in table.values())
    bf_cost = sum(s.cost for m in table.values() for s in m.values())
    print(f"  Brute force: {bf_evals} evaluations, ${bf_cost:.4f} total cost")

    if args.selectors == "all":
        selectors = list(ALL_SELECTORS)
        if not _check_botorch():
            selectors.remove("bayesian_optimization")
            print("\n  (Skipping bayesian_optimization — torch/botorch not installed)")
        try:
            from openai import OpenAI
            import os
            if not os.environ.get("OPENAI_API_KEY"):
                selectors.remove("lm_proposal")
                print("\n  (Skipping lm_proposal — OPENAI_API_KEY not set)")
        except ImportError:
            selectors.remove("lm_proposal")
            print("\n  (Skipping lm_proposal — openai not installed)")
    else:
        selectors = [s.strip() for s in args.selectors.split(",")]

    summaries = []

    for sel in selectors:
        if sel == "random_search":
            fn = simulate_random_search
            kwargs = {"sample_fraction": args.rs_fraction}
        elif sel == "arm_elimination":
            fn = simulate_arm_elimination
            kwargs = {"confidence": args.arm_confidence}
        elif sel == "epsilon_lucb":
            fn = simulate_epsilon_lucb
            kwargs = {"epsilon": args.epsilon}
        elif sel == "threshold_se":
            fn = simulate_threshold_se
            kwargs = {"threshold": args.threshold}
        elif sel == "hill_climbing":
            fn = simulate_hill_climbing
            kwargs = {"num_restarts": args.hc_restarts}
        elif sel == "bayesian_optimization":
            if not _check_botorch():
                print(f"\n  Skipping {sel} — torch/botorch not installed")
                continue
            fn = simulate_bayesian_optimization
            kwargs = {}
        elif sel == "lm_proposal":
            fn = simulate_lm_proposal
            kwargs = {"proposer_model": args.proposer_model, "preview_size": args.preview_size,
                      "benchmark": args.benchmark}
        else:
            print(f"\n  Unknown selector: {sel}")
            continue

        if args.seeds > 1:
            results = run_multi_seed(fn, models, datapoints, table,
                                     n_seeds=args.seeds, base_seed=args.base_seed, **kwargs)
            summary = summarize_multi_seed(results, gt_name, gt_acc, models, datapoints, table)
            print_summary(summary)
            summaries.append(summary)
        else:
            result = fn(models, datapoints, table, seed=args.base_seed, **kwargs)
            print_single_result(result, gt_name, gt_acc)
            summaries.append(summarize_multi_seed([result], gt_name, gt_acc, models, datapoints, table))

    if args.output and summaries:
        import csv as csv_mod
        fields = ["selector", "n_seeds", "ground_truth_best", "ground_truth_accuracy",
                  "mean_accuracy", "std_accuracy", "found_true_best_pct",
                  "mean_evaluations", "mean_cost", "params"]
        with open(args.output, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s in summaries:
                row = {k: s[k] for k in fields}
                row["params"] = json.dumps(row["params"])
                writer.writerow(row)
        print(f"\nSummary CSV written to: {args.output}")


if __name__ == "__main__":
    main()
