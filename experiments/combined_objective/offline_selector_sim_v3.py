#!/usr/bin/env python3
"""
Offline Selector Simulator v3 — Combined Objective
====================================================
Fork of v2 that changes the selector objective from pure accuracy to:

    objective = accuracy - lambda_cost * cost - lambda_latency * latency

where cost and latency are PER-SAMPLE averages (same scale across benchmarks
when normalized). When lambda_cost=0 and lambda_latency=0, this degrades to
pure accuracy maximization (identical to v2).

Usage:
    python combined_objective/offline_selector_sim_v3.py \
        --jsonl agentopt/results/gpqa_200_bf.jsonl \
        --selectors all --seeds 50 \
        --lambda-cost 0.1 --lambda-latency 0.05
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
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
except ImportError:
    pass

# Allow unpickling SampleResult objects serialized by v2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Bedrock pricing ($/MTok) — keyed by ARN suffix (profile ID)
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


LookupTable = Dict[str, Dict[int, SampleResult]]


def load_jsonl(path: str) -> Tuple[List[str], List[int], LookupTable]:
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


def load_pickle(path: str) -> Tuple[List[str], List[int], LookupTable]:
    """Load from pickle lookup table (from cache_selector_sim.py).

    Pickle contains SampleResult objects from v2 — same fields as v3.
    """
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)

    models = data["model_names"]
    datapoints = data["datapoints"]
    raw_table = data["table"]

    table: LookupTable = {}
    for model_name, dp_dict in raw_table.items():
        table[model_name] = {}
        for dp_idx, sr in dp_dict.items():
            table[model_name][dp_idx] = SampleResult(
                score=sr.score,
                latency_seconds=sr.latency_seconds,
                input_tokens=getattr(sr, "input_tokens", {}),
                output_tokens=getattr(sr, "output_tokens", {}),
                cost=sr.cost,
            )

    return models, datapoints, table


# ---------------------------------------------------------------------------
# Combined objective with min-max normalization
# ---------------------------------------------------------------------------

@dataclass
class NormStats:
    """Min-max normalization stats computed once from the full dataset."""
    cost_min: float = 0.0
    cost_max: float = 1.0
    latency_min: float = 0.0
    latency_max: float = 1.0

    @property
    def cost_range(self) -> float:
        r = self.cost_max - self.cost_min
        return r if r > 1e-12 else 1.0

    @property
    def latency_range(self) -> float:
        r = self.latency_max - self.latency_min
        return r if r > 1e-12 else 1.0


def compute_norm_stats(table: LookupTable, datapoints: List[int]) -> NormStats:
    """Compute min/max cost and latency across all models and datapoints."""
    all_costs = []
    all_lats = []
    for model_data in table.values():
        for dp in datapoints:
            if dp in model_data:
                sr = model_data[dp]
                all_costs.append(sr.cost)
                all_lats.append(sr.latency_seconds)
    if not all_costs:
        return NormStats()
    return NormStats(
        cost_min=min(all_costs), cost_max=max(all_costs),
        latency_min=min(all_lats), latency_max=max(all_lats),
    )


_NORM: Optional[NormStats] = None


def set_norm_stats(table: LookupTable, datapoints: List[int]) -> NormStats:
    """Compute and set the global normalization stats. Call once after loading data."""
    global _NORM
    _NORM = compute_norm_stats(table, datapoints)
    return _NORM


def compute_sample_objective(sr: SampleResult, lambda_cost: float,
                             lambda_latency: float) -> float:
    """Compute per-sample objective with min-max normalized cost/latency.

    objective = score - λ_cost * norm_cost - λ_latency * norm_latency

    Where norm_cost = (cost - min) / (max - min), putting it in [0, 1].
    Lambdas are now interpretable: 0.1 = "10% as important as accuracy".
    Uses global _NORM (set by set_norm_stats). If not set, uses raw values.
    """
    if _NORM is None:
        return sr.score - lambda_cost * sr.cost - lambda_latency * sr.latency_seconds
    norm_cost = (sr.cost - _NORM.cost_min) / _NORM.cost_range
    norm_lat = (sr.latency_seconds - _NORM.latency_min) / _NORM.latency_range
    return sr.score - lambda_cost * norm_cost - lambda_latency * norm_lat


def compute_model_objective(model_name: str, datapoints: List[int],
                            table: LookupTable, lambda_cost: float,
                            lambda_latency: float) -> float:
    """Compute mean objective for a model across all available datapoints."""
    samples = table.get(model_name, {})
    available = [samples[dp] for dp in datapoints if dp in samples]
    if not available:
        return float("-inf")
    objectives = [compute_sample_objective(s, lambda_cost, lambda_latency) for s in available]
    return sum(objectives) / len(objectives)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _compute_stats(scores: List[float]) -> Tuple[float, float]:
    n = len(scores)
    if n == 0:
        return 0.0, 0.5
    mean = sum(scores) / n
    if n < 2:
        return mean, 0.5
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
    return mean, math.sqrt(variance)


def _is_dominated(objectives_i: List[float], objectives_j: List[float],
                   confidence: float = 1.0) -> bool:
    """Return True if arm i is statistically dominated by arm j (on objective)."""
    n_i, n_j = len(objectives_i), len(objectives_j)
    if n_i == 0 or n_j == 0:
        return False
    mu_i, std_i = _compute_stats(objectives_i)
    mu_j, std_j = _compute_stats(objectives_j)
    se_i = std_i / math.sqrt(n_i)
    se_j = std_j / math.sqrt(n_j)
    return mu_i + confidence * se_i < mu_j - confidence * se_j


def _confidence_bounds(scores: List[float], confidence: float = 1.96
                       ) -> Tuple[float, float]:
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
    objective: float
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
    best_objective: float
    total_evaluations: int
    total_cost: float
    models_tested: int
    found_true_best: bool
    compute_time_seconds: float = 0.0
    model_results: List[ModelSummary] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Brute-force ground truth (by combined objective)
# ---------------------------------------------------------------------------

def compute_ground_truth(models: List[str], datapoints: List[int],
                         table: LookupTable, lambda_cost: float = 0.0,
                         lambda_latency: float = 0.0) -> Tuple[str, float, float]:
    """Compute brute-force best model by combined objective.

    Returns (best_name, best_accuracy, best_objective).
    """
    best_name = None
    best_obj = float("-inf")
    best_acc = 0.0
    tol = 1e-9

    for model in models:
        samples = table.get(model, {})
        available = [samples[dp] for dp in datapoints if dp in samples]
        if not available:
            continue
        acc = sum(s.score for s in available) / len(available)
        obj = sum(compute_sample_objective(s, lambda_cost, lambda_latency)
                  for s in available) / len(available)

        if obj > best_obj + tol:
            best_name, best_obj, best_acc = model, obj, acc

    return best_name, best_acc, best_obj


def _evaluate_model_full(idx: int, models: List[str], datapoints: List[int],
                         table: LookupTable, lambda_cost: float = 0.0,
                         lambda_latency: float = 0.0
                         ) -> Tuple[float, float, float, float, int, float]:
    """Evaluate a model on all datapoints.

    Returns (accuracy, objective, latency, cost, n_eval, compute_time).
    """
    model = models[idx]
    samples = table.get(model, {})
    available = [samples[dp] for dp in datapoints if dp in samples]
    n_eval = len(available)
    acc = sum(s.score for s in available) / n_eval if n_eval else 0.0
    obj = sum(compute_sample_objective(s, lambda_cost, lambda_latency)
              for s in available) / n_eval if n_eval else float("-inf")
    lat = sum(s.latency_seconds for s in available) / n_eval if n_eval else 0.0
    cost = sum(s.cost for s in available)
    ct = sum(s.latency_seconds for s in available)
    return acc, obj, lat, cost, n_eval, ct


# ---------------------------------------------------------------------------
# Selector: Brute Force (reference baseline)
# ---------------------------------------------------------------------------

def simulate_brute_force(
    models: List[str], datapoints: List[int], table: LookupTable,
    lambda_cost: float = 0.0, lambda_latency: float = 0.0,
    seed: int = 42,
) -> SimulationResult:
    """Evaluate all models on all datapoints — the reference upper bound."""
    total_evals = 0
    total_cost = 0.0
    compute_time = 0.0
    model_results = []
    best_name = None
    best_obj = float("-inf")
    best_acc = 0.0
    tol = 1e-9

    for idx in range(len(models)):
        acc, obj, lat, cost, n_eval, ct = _evaluate_model_full(
            idx, models, datapoints, table, lambda_cost, lambda_latency)
        total_evals += n_eval
        total_cost += cost
        compute_time += ct
        model_results.append(ModelSummary(models[idx], acc, obj, lat, cost, n_eval))
        if obj > best_obj + tol:
            best_name, best_obj, best_acc = models[idx], obj, acc

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    return SimulationResult("brute_force", seed,
        {"lambda_cost": lambda_cost, "lambda_latency": lambda_latency},
        best_name, best_acc, best_obj, total_evals, total_cost,
        len(models), True, compute_time, model_results)


# ---------------------------------------------------------------------------
# Selector: Random Search
# ---------------------------------------------------------------------------

def simulate_random_search(
    models: List[str], datapoints: List[int], table: LookupTable,
    lambda_cost: float = 0.0, lambda_latency: float = 0.0,
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
    best_name = None
    best_obj = float("-inf")
    best_acc = 0.0
    tol = 1e-9

    for idx in sampled:
        acc, obj, lat, cost, n_eval, ct = _evaluate_model_full(
            idx, models, datapoints, table, lambda_cost, lambda_latency)
        total_evals += n_eval
        total_cost += cost
        compute_time += ct
        model_results.append(ModelSummary(models[idx], acc, obj, lat, cost, n_eval))
        if obj > best_obj + tol:
            best_name, best_obj, best_acc = models[idx], obj, acc

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    gt_name, _, _ = compute_ground_truth(models, datapoints, table, lambda_cost, lambda_latency)
    return SimulationResult("random_search", seed,
        {"sample_fraction": sample_fraction,
         "lambda_cost": lambda_cost, "lambda_latency": lambda_latency},
        best_name, best_acc, best_obj, total_evals, total_cost,
        len(sampled), best_name == gt_name, compute_time, model_results)


# ---------------------------------------------------------------------------
# Selector: Arm Elimination
# ---------------------------------------------------------------------------

def simulate_arm_elimination(
    models: List[str], datapoints: List[int], table: LookupTable,
    lambda_cost: float = 0.0, lambda_latency: float = 0.0,
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
    combo_objectives: Dict[int, List[float]] = {i: [] for i in range(n_models)}
    combo_latencies: Dict[int, List[float]] = {i: [] for i in range(n_models)}
    combo_costs: Dict[int, float] = {i: 0.0 for i in range(n_models)}
    combo_scores: Dict[int, List[float]] = {i: [] for i in range(n_models)}
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
                    obj = compute_sample_objective(s, lambda_cost, lambda_latency)
                    combo_objectives[idx].append(obj)
                    combo_scores[idx].append(s.score)
                    combo_latencies[idx].append(s.latency_seconds)
                    combo_costs[idx] += s.cost
                    compute_time += s.latency_seconds
                    total_evals += 1

        newly_eliminated = set()
        for i in active:
            for j in active:
                if i != j and _is_dominated(combo_objectives[i], combo_objectives[j], confidence):
                    newly_eliminated.add(i)
                    break
        active -= newly_eliminated

        if len(active) <= 1:
            break
        offset = batch_end
        batch_size = max(1, int(batch_size * growth_factor))

    model_results = []
    best_name = None
    best_obj = float("-inf")
    best_acc = 0.0
    tol = 1e-9
    for idx in range(n_models):
        objs = combo_objectives[idx]
        scores = combo_scores[idx]
        lats = combo_latencies[idx]
        obj = sum(objs) / len(objs) if objs else float("-inf")
        acc = sum(scores) / len(scores) if scores else 0.0
        lat = sum(lats) / len(lats) if lats else 0.0
        cost = combo_costs[idx]
        model_results.append(ModelSummary(models[idx], acc, obj, lat, cost, len(objs)))
        if obj > best_obj + tol:
            best_name, best_obj, best_acc = models[idx], obj, acc

    # Report true full-dataset accuracy/objective of selected config (not partial estimate)
    if best_name is not None:
        best_idx = models.index(best_name)
        true_acc, true_obj, _, _, _, _ = _evaluate_model_full(
            best_idx, models, datapoints, table, lambda_cost, lambda_latency)
        best_acc = true_acc
        best_obj = true_obj

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    gt_name, _, _ = compute_ground_truth(models, datapoints, table, lambda_cost, lambda_latency)
    return SimulationResult("arm_elimination", seed,
        {"n_initial": n_initial, "growth_factor": growth_factor, "confidence": confidence,
         "lambda_cost": lambda_cost, "lambda_latency": lambda_latency},
        best_name, best_acc, best_obj, total_evals, sum(combo_costs.values()),
        n_models, best_name == gt_name, compute_time, model_results)


# ---------------------------------------------------------------------------
# Selector: Epsilon-LUCB
# ---------------------------------------------------------------------------

def simulate_epsilon_lucb(
    models: List[str], datapoints: List[int], table: LookupTable,
    lambda_cost: float = 0.0, lambda_latency: float = 0.0,
    epsilon: float = 0.01, confidence: float = 1.96,
    seed: Optional[int] = None,
) -> SimulationResult:
    n_samples = len(datapoints)
    n_models = len(models)

    dp_order = list(datapoints)
    if seed is not None:
        random.Random(seed).shuffle(dp_order)

    combo_objectives: Dict[int, List[float]] = {i: [] for i in range(n_models)}
    combo_costs: Dict[int, float] = {i: 0.0 for i in range(n_models)}
    combo_scores: Dict[int, List[float]] = {i: [] for i in range(n_models)}
    total_evals = 0
    compute_time = 0.0

    # Initial: give each model 1 sample
    for idx in range(n_models):
        if dp_order:
            dp = dp_order[0]
            samples = table.get(models[idx], {})
            if dp in samples:
                s = samples[dp]
                obj = compute_sample_objective(s, lambda_cost, lambda_latency)
                combo_objectives[idx].append(obj)
                combo_scores[idx].append(s.score)
                combo_costs[idx] += s.cost
                compute_time += s.latency_seconds
                total_evals += 1

    for dp_i in range(1, n_samples):
        dp = dp_order[dp_i]

        # Find top arm (highest mean objective) and challenger
        means = [(idx, sum(combo_objectives[idx]) / len(combo_objectives[idx])
                  if combo_objectives[idx] else float("-inf")) for idx in range(n_models)]
        means.sort(key=lambda x: x[1], reverse=True)
        top_idx = means[0][0]
        top_lcb = _confidence_bounds(combo_objectives[top_idx], confidence)[0]

        best_challenger_idx = None
        best_challenger_ucb = float("-inf")
        for idx in range(n_models):
            if idx == top_idx:
                continue
            _, ucb = _confidence_bounds(combo_objectives[idx], confidence)
            if ucb > best_challenger_ucb:
                best_challenger_ucb = ucb
                best_challenger_idx = idx

        if best_challenger_idx is not None and top_lcb - best_challenger_ucb >= epsilon:
            break

        for idx in [top_idx, best_challenger_idx]:
            if idx is None:
                continue
            samples = table.get(models[idx], {})
            if dp in samples:
                s = samples[dp]
                obj = compute_sample_objective(s, lambda_cost, lambda_latency)
                combo_objectives[idx].append(obj)
                combo_scores[idx].append(s.score)
                combo_costs[idx] += s.cost
                compute_time += s.latency_seconds
                total_evals += 1

    model_results = []
    best_name = None
    best_obj = float("-inf")
    best_acc = 0.0
    tol = 1e-9
    for idx in range(n_models):
        objs = combo_objectives[idx]
        scores = combo_scores[idx]
        obj = sum(objs) / len(objs) if objs else float("-inf")
        acc = sum(scores) / len(scores) if scores else 0.0
        samples = table.get(models[idx], {})
        available = [samples[dp] for dp in datapoints if dp in samples]
        lat = sum(s.latency_seconds for s in available) / len(available) if available else 0.0
        cost = combo_costs[idx]
        model_results.append(ModelSummary(models[idx], acc, obj, lat, cost, len(objs)))
        if obj > best_obj + tol:
            best_name, best_obj, best_acc = models[idx], obj, acc

    # Report true full-dataset accuracy/objective of selected config (not partial estimate)
    if best_name is not None:
        best_idx = models.index(best_name)
        true_acc, true_obj, _, _, _, _ = _evaluate_model_full(
            best_idx, models, datapoints, table, lambda_cost, lambda_latency)
        best_acc = true_acc
        best_obj = true_obj

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    gt_name, _, _ = compute_ground_truth(models, datapoints, table, lambda_cost, lambda_latency)
    return SimulationResult("epsilon_lucb", seed,
        {"epsilon": epsilon, "confidence": confidence,
         "lambda_cost": lambda_cost, "lambda_latency": lambda_latency},
        best_name, best_acc, best_obj, total_evals, sum(combo_costs.values()),
        n_models, best_name == gt_name, compute_time, model_results)


# ---------------------------------------------------------------------------
# Selector: Threshold Successive Elimination
# ---------------------------------------------------------------------------

def simulate_threshold_se(
    models: List[str], datapoints: List[int], table: LookupTable,
    lambda_cost: float = 0.0, lambda_latency: float = 0.0,
    threshold: float = 0.5, confidence: float = 1.96,
    seed: Optional[int] = None,
) -> SimulationResult:
    n_samples = len(datapoints)
    n_models = len(models)

    dp_order = list(datapoints)
    if seed is not None:
        random.Random(seed).shuffle(dp_order)

    combo_objectives: Dict[int, List[float]] = {i: [] for i in range(n_models)}
    combo_costs: Dict[int, float] = {i: 0.0 for i in range(n_models)}
    combo_scores: Dict[int, List[float]] = {i: [] for i in range(n_models)}
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
                obj = compute_sample_objective(s, lambda_cost, lambda_latency)
                combo_objectives[idx].append(obj)
                combo_scores[idx].append(s.score)
                combo_costs[idx] += s.cost
                compute_time += s.latency_seconds
                total_evals += 1

        newly_eliminated = set()
        for idx in active:
            if len(combo_objectives[idx]) < 2:
                continue
            lcb, ucb = _confidence_bounds(combo_objectives[idx], confidence)
            if ucb < threshold or lcb > threshold:
                newly_eliminated.add(idx)

        active -= newly_eliminated
        if not active:
            break

    model_results = []
    best_name = None
    best_obj = float("-inf")
    best_acc = 0.0
    tol = 1e-9
    for idx in range(n_models):
        objs = combo_objectives[idx]
        scores = combo_scores[idx]
        obj = sum(objs) / len(objs) if objs else float("-inf")
        acc = sum(scores) / len(scores) if scores else 0.0
        samples = table.get(models[idx], {})
        available = [samples[dp] for dp in datapoints if dp in samples]
        lat = sum(s.latency_seconds for s in available) / len(available) if available else 0.0
        cost = combo_costs[idx]
        model_results.append(ModelSummary(models[idx], acc, obj, lat, cost, len(objs)))
        if obj > best_obj + tol:
            best_name, best_obj, best_acc = models[idx], obj, acc

    # Report true full-dataset accuracy/objective of selected config (not partial estimate)
    if best_name is not None:
        best_idx = models.index(best_name)
        true_acc, true_obj, _, _, _, _ = _evaluate_model_full(
            best_idx, models, datapoints, table, lambda_cost, lambda_latency)
        best_acc = true_acc
        best_obj = true_obj

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    gt_name, _, _ = compute_ground_truth(models, datapoints, table, lambda_cost, lambda_latency)
    return SimulationResult("threshold_se", seed,
        {"threshold": threshold, "confidence": confidence,
         "lambda_cost": lambda_cost, "lambda_latency": lambda_latency},
        best_name, best_acc, best_obj, total_evals, sum(combo_costs.values()),
        n_models, best_name == gt_name, compute_time, model_results)


# ---------------------------------------------------------------------------
# Selector: Hill Climbing
# ---------------------------------------------------------------------------

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
    parts = name.split(" + ")
    result = []
    for p in parts:
        eq = p.index("=")
        result.append((p[:eq], p[eq+1:]))
    return result


def _build_combo_name(positions: List[Tuple[str, str]]) -> str:
    return " + ".join(f"{role}={model}" for role, model in positions)


def _sort_by_ranking(candidates: List[str], ranking: List[str]) -> List[str]:
    known = [(ranking.index(c), c) for c in candidates if c in ranking]
    unknown = [c for c in candidates if c not in ranking]
    known.sort()
    return [c for _, c in known] + unknown


def _get_higher_quality_neighbor(current: str, candidates: List[str]) -> Optional[str]:
    sorted_cands = _sort_by_ranking(candidates, _QUALITY_RANKING_DISPLAY)
    if current not in sorted_cands:
        return None
    idx = sorted_cands.index(current)
    if idx == 0:
        return None
    return sorted_cands[idx - 1]


def _get_faster_neighbor(current: str, candidates: List[str]) -> Optional[str]:
    sorted_cands = _sort_by_ranking(candidates, _SPEED_RANKING_DISPLAY)
    if current not in sorted_cands:
        return None
    idx = sorted_cands.index(current)
    if idx == 0:
        return None
    return sorted_cands[idx - 1]


def _generate_neighbors_hc(
    combo_name: str, models: List[str], base_models: List[str],
    seen: Set[int], rng: random.Random, improve_quality: bool,
) -> List[int]:
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
    seen: Set[int], rng: random.Random, objective: float,
) -> List[int]:
    """Get neighbors. Uses objective >= 1.0 threshold for speed-first mode."""
    combo_name = models[combo_idx]
    if objective < 1.0:
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
    lambda_cost: float = 0.0, lambda_latency: float = 0.0,
    seed: int = 42, num_restarts: int = 3, max_iterations: int = 20,
    patience: int = 3,
) -> SimulationResult:
    n_models = len(models)
    rng = random.Random(seed)

    base_models_set: Set[str] = set()
    for m in models:
        for _, model in _parse_combo_name(m):
            base_models_set.add(model)
    base_models = sorted(base_models_set)

    evaluated: Dict[int, Tuple[float, float, float, float]] = {}  # idx -> (acc, obj, lat, cost)
    total_evals = 0
    total_cost = 0.0
    compute_time = 0.0
    model_n_evals: Dict[int, int] = {}
    seen: Set[int] = set()

    def eval_model(idx):
        nonlocal total_evals, total_cost, compute_time
        if idx in evaluated:
            return evaluated[idx]
        acc, obj, lat, cost, n_eval, ct = _evaluate_model_full(
            idx, models, datapoints, table, lambda_cost, lambda_latency)
        evaluated[idx] = (acc, obj, lat, cost)
        total_evals += n_eval
        total_cost += cost
        compute_time += ct
        model_n_evals[idx] = n_eval
        return (acc, obj, lat, cost)

    global_best_idx = None
    global_best_obj = float("-inf")
    tol = 1e-9

    for restart in range(num_restarts):
        unseen = [i for i in range(n_models) if i not in seen]
        if not unseen:
            break
        current = rng.choice(unseen)
        seen.add(current)
        current_acc, current_obj, current_lat, _ = eval_model(current)

        best_restart_idx = current
        best_restart_obj = current_obj
        no_improve_count = 0

        for iteration in range(max_iterations):
            neighbors = _get_neighbors_hc(current, models, base_models, seen, rng, current_obj)
            if not neighbors:
                break

            best_neighbor = None
            best_n_obj = float("-inf")
            for n_idx in neighbors:
                seen.add(n_idx)
                n_acc, n_obj, n_lat, _ = eval_model(n_idx)
                if n_obj > best_n_obj + tol:
                    best_neighbor = n_idx
                    best_n_obj = n_obj

            improves = best_neighbor is not None and best_n_obj > current_obj + tol

            if improves:
                current = best_neighbor
                current_obj = best_n_obj
                no_improve_count = 0
            else:
                no_improve_count += 1

            if current_obj > best_restart_obj + tol:
                best_restart_idx = current
                best_restart_obj = current_obj

            if no_improve_count >= patience:
                break

        if best_restart_obj > global_best_obj + tol:
            global_best_idx = best_restart_idx
            global_best_obj = best_restart_obj

    best_name = models[global_best_idx] if global_best_idx is not None else models[0]
    best_acc = evaluated[global_best_idx][0] if global_best_idx is not None and global_best_idx in evaluated else 0.0
    best_obj = global_best_obj if global_best_idx is not None else float("-inf")

    model_results = []
    for idx in range(n_models):
        if idx in evaluated:
            acc, obj, lat, cost = evaluated[idx]
            n_eval = model_n_evals[idx]
        else:
            acc, obj, lat, cost, n_eval = 0.0, float("-inf"), 0.0, 0.0, 0
        model_results.append(ModelSummary(models[idx], acc, obj, lat, cost, n_eval,
                                          is_best=(models[idx] == best_name)))

    gt_name, _, _ = compute_ground_truth(models, datapoints, table, lambda_cost, lambda_latency)
    return SimulationResult("hill_climbing", seed,
        {"num_restarts": num_restarts, "max_iterations": max_iterations, "patience": patience,
         "lambda_cost": lambda_cost, "lambda_latency": lambda_latency},
        best_name, best_acc, best_obj, total_evals, total_cost,
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
    lambda_cost: float = 0.0, lambda_latency: float = 0.0,
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
    model_data: Dict[int, Tuple[float, float, float, float, int, float]] = {}
    total_evals = 0

    def eval_model(idx):
        nonlocal total_evals
        acc, obj, lat, cost, n_eval, ct = _evaluate_model_full(
            idx, models, datapoints, table, lambda_cost, lambda_latency)
        model_data[idx] = (acc, obj, lat, cost, n_eval, ct)
        total_evals += n_eval
        return obj

    pool = list(range(n_models))
    rng.shuffle(pool)
    for idx in pool[:n_initial_random]:
        evaluated.add(idx)
        obj = eval_model(idx)
        X_list.append([idx])
        Y_list.append(obj)

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
        obj = eval_model(idx)
        X_list.append([idx])
        Y_list.append(obj)

    model_results = []
    best_name = None
    best_obj = float("-inf")
    best_acc = 0.0
    tol = 1e-9
    for idx in range(n_models):
        if idx in model_data:
            acc, obj, lat, cost, n_eval, ct = model_data[idx]
        else:
            acc, obj, lat, cost, n_eval = 0.0, float("-inf"), 0.0, 0.0, 0
        model_results.append(ModelSummary(models[idx], acc, obj, lat, cost, n_eval))
        if idx in evaluated:
            if obj > best_obj + tol:
                best_name, best_obj, best_acc = models[idx], obj, acc

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    gt_name, _, _ = compute_ground_truth(models, datapoints, table, lambda_cost, lambda_latency)
    total_cost = sum(d[3] for d in model_data.values())
    compute_time = sum(d[5] for d in model_data.values())

    return SimulationResult("bayesian_optimization", seed,
        {"n_initial_random": n_initial_random, "n_iterations": n_iterations,
         "lambda_cost": lambda_cost, "lambda_latency": lambda_latency},
        best_name, best_acc, best_obj, total_evals, total_cost,
        len(evaluated), best_name == gt_name, compute_time, model_results)


# ---------------------------------------------------------------------------
# Selector: LM Proposal
# ---------------------------------------------------------------------------

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
                if "agent" not in node_candidates:
                    node_candidates["agent"] = set()
                node_candidates["agent"].add(part)

    nodes = sorted(node_candidates.keys())
    candidates = {n: sorted(node_candidates[n]) for n in nodes}
    return nodes, candidates


def _load_dataset_previews(benchmark: str, limit: int = 200) -> List[Dict[str, Any]]:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
                previews.append({"input": q, "expected": "A"})

    elif benchmark == "bfcl":
        path = os.path.join(base, "benchmarks/BFCL/data/BFCL_v3_multi_turn_base.json")
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                entry = json.loads(line)
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
    lambda_cost: float = 0.0,
    lambda_latency: float = 0.0,
) -> str:
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

    example = {
        "combination": {node: candidates[node][0] for node in nodes},
        "reasoning": "Your explanation here.",
    }

    # Build objective description based on lambdas
    if lambda_cost == 0.0 and lambda_latency == 0.0:
        objective_text = "maximize accuracy and then minimize latency and cost"
    else:
        parts = ["maximize accuracy"]
        if lambda_cost > 0:
            parts.append(f"penalize cost (weight={lambda_cost})")
        if lambda_latency > 0:
            parts.append(f"penalize latency (weight={lambda_latency})")
        objective_text = ", ".join(parts)
        objective_text += (
            f"\n\nFormally: objective = accuracy - {lambda_cost} * per_sample_cost"
            f" - {lambda_latency} * per_sample_latency_seconds"
        )

    sections = [
        "# Task\n"
        "You are an expert AI model selector. You will be given a multi-agent "
        "workflow where each node can use one of several candidate LLMs. "
        "Your job is to select the best combination of models for the nodes.\n",

        f"# The objective to target when selecting the model combination:\n"
        f"{objective_text}\n",

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
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    combination = payload.get("combination", {})
    if set(combination.keys()) != set(nodes):
        return None

    for node in nodes:
        if combination[node] not in candidates[node]:
            return None

    if len(nodes) == 1 and nodes[0] == "agent":
        target = f"agent={combination['agent']}"
    else:
        parts = [f"{node}={combination[node]}" for node in sorted(nodes)]
        target = " + ".join(parts)

    if target in models:
        return target

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
    lambda_cost: float = 0.0, lambda_latency: float = 0.0,
    seed: int = 42, preview_size: int = 10,
    proposer_model: str = "gpt-4.1",
    benchmark: str = "auto",
) -> SimulationResult:
    nodes, candidates = _extract_nodes_and_candidates(models)

    if benchmark == "auto":
        for model in models:
            if "answer=" in model and "critic=" in model:
                benchmark = "mathqa"
                break
            elif "planner=" in model and "solver=" in model:
                benchmark = "hotpotqa"
                break
            elif "agent=" in model:
                benchmark = "gpqa"
                break

    benchmark_descriptions = {
        "gpqa": "GPQA Diamond (graduate-level science multiple choice questions)",
        "bfcl": "BFCL v3 Multi-Turn (multi-turn function calling with backend APIs)",
        "hotpotqa": "HotpotQA (multi-hop question answering with planner+solver agents)",
        "mathqa": "MathQA (self-reflective math with answer+critic agents)",
    }
    benchmark_name = benchmark_descriptions.get(benchmark, benchmark)

    if benchmark not in _DATASET_CACHE:
        print(f"  Loading {benchmark} dataset for previews...")
        _DATASET_CACHE[benchmark] = _load_dataset_previews(benchmark, limit=200)
    all_samples = _DATASET_CACHE[benchmark]

    if not all_samples:
        print(f"  Warning: no dataset samples loaded for {benchmark}")
        proposed_model = models[0]
    else:
        dataset_preview = all_samples[:preview_size]

        prompt = _build_lm_proposal_prompt(
            nodes, candidates, benchmark_name, dataset_preview,
            lambda_cost, lambda_latency)

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
                        wait = 2 ** attempt * 3
                        _time.sleep(wait)
                    else:
                        print(f"  LM Proposal error: {api_err}")
                        break
        except Exception as e:
            print(f"  LM Proposal error: {e}")

    if proposed_model is None:
        proposed_model = models[0]
        print(f"  LM Proposal: failed to parse response, falling back to {proposed_model}")

    acc, obj, lat, cost, n_eval, ct = _evaluate_model_full(
        models.index(proposed_model), models, datapoints, table,
        lambda_cost, lambda_latency)

    model_results = [ModelSummary(proposed_model, acc, obj, lat, cost, n_eval, is_best=True)]

    gt_name, _, _ = compute_ground_truth(models, datapoints, table, lambda_cost, lambda_latency)
    return SimulationResult(
        "lm_proposal", seed,
        {"proposer_model": proposer_model, "preview_size": preview_size,
         "lambda_cost": lambda_cost, "lambda_latency": lambda_latency},
        proposed_model, acc, obj, n_eval, cost,
        1, proposed_model == gt_name, ct, model_results,
    )


# ---------------------------------------------------------------------------
# Selector: Matrix UCB (plain)
# ---------------------------------------------------------------------------

def simulate_matrix_ucb(
    models: List[str], datapoints: List[int], table: LookupTable,
    lambda_cost: float = 0.0, lambda_latency: float = 0.0,
    a: float = 1.0, observation_budget_fraction: float = 1.0,
    seed: int = 42,
) -> SimulationResult:
    import numpy as np
    rng = np.random.default_rng(seed)
    n_combos = len(models)
    n_dp = len(datapoints)
    max_cells = 20

    available = np.zeros((n_combos, n_dp), dtype=bool)
    for i, model_name in enumerate(models):
        model_data = table.get(model_name, {})
        for j, dp_id in enumerate(datapoints):
            if dp_id in model_data:
                available[i, j] = True
    n_available = int(available.sum())

    budget = max(1, int(math.ceil(observation_budget_fraction * n_available))) if observation_budget_fraction < 1.0 else n_available

    observed = np.full((n_combos, n_dp), np.nan, dtype=np.float64)
    cell_costs: Dict[Tuple[int, int], float] = {}
    total_evals = 0
    total_cost = 0.0

    available_per_combo = available.sum(axis=1)

    while True:
        filled = int(np.sum(~np.isnan(observed)))
        if filled >= budget:
            break

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
            obj = compute_sample_objective(sr, lambda_cost, lambda_latency)
            observed[best_combo, dp_local_idx] = obj
            cell_costs[(best_combo, dp_local_idx)] = sr.cost
            total_evals += 1
            total_cost += sr.cost

    best_name = None
    best_obj = float("-inf")
    best_acc = 0.0
    tol = 1e-9
    model_results = []
    for i, model_name in enumerate(models):
        valid = ~np.isnan(observed[i])
        if not np.any(valid):
            continue
        obj_mean = float(np.nanmean(observed[i]))
        model_data_i = table.get(model_name, {})
        lats = [model_data_i[datapoints[j]].latency_seconds
                for j in range(n_dp) if not np.isnan(observed[i, j]) and datapoints[j] in model_data_i]
        lat = sum(lats) / len(lats) if lats else 0.0
        scores = [model_data_i[datapoints[j]].score
                  for j in range(n_dp) if not np.isnan(observed[i, j]) and datapoints[j] in model_data_i]
        acc = sum(scores) / len(scores) if scores else 0.0
        cost_i = sum(cell_costs.get((i, j), 0.0) for j in range(n_dp))
        n_eval_i = int(np.sum(valid))
        model_results.append(ModelSummary(model_name, acc, obj_mean, lat, cost_i, n_eval_i))
        if obj_mean > best_obj + tol:
            best_name, best_obj, best_acc = model_name, obj_mean, acc

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    # True full-dataset objective of selected combo
    if best_name is not None:
        true_best_obj = compute_model_objective(
            best_name, datapoints, table, lambda_cost, lambda_latency)
        model_data_best = table.get(best_name, {})
        all_scores = [model_data_best[dp].score for dp in datapoints if dp in model_data_best]
        true_best_acc = sum(all_scores) / len(all_scores) if all_scores else 0.0
    else:
        true_best_obj = float("-inf")
        true_best_acc = 0.0

    gt_name, _, _ = compute_ground_truth(models, datapoints, table, lambda_cost, lambda_latency)
    return SimulationResult(
        "matrix_ucb", seed,
        {"a": a, "observation_budget_fraction": observation_budget_fraction,
         "lambda_cost": lambda_cost, "lambda_latency": lambda_latency},
        best_name, true_best_acc, true_best_obj, total_evals, total_cost,
        len(model_results), best_name == gt_name, 0.0, model_results,
    )


# ---------------------------------------------------------------------------
# Selector: Matrix UCB with Low-Rank Factorization
# ---------------------------------------------------------------------------

def simulate_matrix_ucb_lrf(
    models: List[str], datapoints: List[int], table: LookupTable,
    lambda_cost: float = 0.0, lambda_latency: float = 0.0,
    rank: int = 1, ensemble_size: int = 64,
    warmup_percentage: float = 0.05,
    regularizer_weight: float = 0.1, drop_probability: float = 0.05,
    iterations: int = 10, eta: float = 5.0,
    observation_budget_fraction: float = 1.0,
    seed: int = 42,
) -> SimulationResult:
    import numpy as np
    import torch
    from agentopt.model_selection.matrix_ucb_factorization import Factorization

    n_combos = len(models)
    n_dp = len(datapoints)
    max_cells = 20

    available_np = np.zeros((n_combos, n_dp), dtype=bool)
    for i, model_name in enumerate(models):
        model_data = table.get(model_name, {})
        for j, dp_id in enumerate(datapoints):
            if dp_id in model_data:
                available_np[i, j] = True
    available_t = torch.from_numpy(available_np)
    n_available = int(available_t.sum().item())
    available_per_combo = available_t.sum(dim=1)

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
            return
        if total_evals >= budget:
            return
        model_name = models[combo_i]
        dp_id = datapoints[dp_local_i]
        model_data = table.get(model_name, {})
        sr = model_data[dp_id]
        obj = compute_sample_objective(sr, lambda_cost, lambda_latency)
        observed_t[combo_i, dp_local_i] = obj
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

    best_name = None
    best_obj = float("-inf")
    best_acc = 0.0
    tol = 1e-9
    model_results = []

    for i, model_name in enumerate(models):
        valid = ~observed_t[i].isnan()
        if not valid.any():
            continue
        obj_mean = float(observed_t[i][valid].mean().item())
        model_data_i = table.get(model_name, {})
        lats = [model_data_i[datapoints[j]].latency_seconds
                for j in range(n_dp) if not observed_t[i, j].isnan() and datapoints[j] in model_data_i]
        lat = sum(lats) / len(lats) if lats else 0.0
        scores = [model_data_i[datapoints[j]].score
                  for j in range(n_dp) if not observed_t[i, j].isnan() and datapoints[j] in model_data_i]
        acc = sum(scores) / len(scores) if scores else 0.0
        cost_i = sum(cell_costs.get((i, j), 0.0) for j in range(n_dp))
        n_eval_i = int(valid.sum().item())
        model_results.append(ModelSummary(model_name, acc, obj_mean, lat, cost_i, n_eval_i))
        if obj_mean > best_obj + tol:
            best_name, best_obj, best_acc = model_name, obj_mean, acc

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    # True full-dataset objective of selected combo
    if best_name is not None:
        true_best_obj = compute_model_objective(
            best_name, datapoints, table, lambda_cost, lambda_latency)
        model_data_best = table.get(best_name, {})
        all_scores = [model_data_best[dp].score for dp in datapoints if dp in model_data_best]
        true_best_acc = sum(all_scores) / len(all_scores) if all_scores else 0.0
    else:
        true_best_obj = float("-inf")
        true_best_acc = 0.0

    gt_name, _, _ = compute_ground_truth(models, datapoints, table, lambda_cost, lambda_latency)
    return SimulationResult(
        "matrix_ucb_lrf", seed,
        {"rank": rank, "ensemble_size": ensemble_size,
         "warmup_percentage": warmup_percentage, "eta": eta,
         "observation_budget_fraction": observation_budget_fraction,
         "lambda_cost": lambda_cost, "lambda_latency": lambda_latency},
        best_name, true_best_acc, true_best_obj, total_evals, total_cost,
        len(model_results), best_name == gt_name, 0.0, model_results,
    )


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

ALL_SELECTORS = [
    "brute_force", "random_search", "arm_elimination", "epsilon_lucb",
    "threshold_se", "hill_climbing", "bayesian_optimization",
    "lm_proposal", "matrix_ucb", "matrix_ucb_lrf",
]


def run_multi_seed(selector_fn, models, datapoints, table,
                   n_seeds, base_seed=0, **kwargs):
    results = []
    for i in range(n_seeds):
        result = selector_fn(models, datapoints, table, seed=base_seed + i, **kwargs)
        results.append(result)
    return results


def summarize_multi_seed(results, gt_name, gt_acc, gt_obj,
                         models=None, datapoints=None, table=None,
                         lambda_cost=0.0, lambda_latency=0.0):
    if models is not None and datapoints is not None and table is not None:
        true_accs = {}
        true_objs = {}
        for model in models:
            samples = table.get(model, {})
            available = [samples[dp] for dp in datapoints if dp in samples]
            true_accs[model] = sum(s.score for s in available) / len(available) if available else 0.0
            true_objs[model] = (sum(compute_sample_objective(s, lambda_cost, lambda_latency)
                                    for s in available) / len(available)) if available else float("-inf")
        accuracies = [true_accs.get(r.best_model, 0.0) for r in results]
        objectives = [true_objs.get(r.best_model, float("-inf")) for r in results]
    else:
        accuracies = [r.best_accuracy for r in results]
        objectives = [r.best_objective for r in results]

    evals = [r.total_evaluations for r in results]
    costs = [r.total_cost for r in results]
    found_best = [r.found_true_best for r in results]
    compute_times = [r.compute_time_seconds for r in results]
    n = len(results)
    mean_acc = sum(accuracies) / n
    std_acc = (sum((a - mean_acc)**2 for a in accuracies) / max(n - 1, 1)) ** 0.5
    mean_obj = sum(objectives) / n
    std_obj = (sum((o - mean_obj)**2 for o in objectives) / max(n - 1, 1)) ** 0.5

    return {
        "selector": results[0].selector,
        "n_seeds": n,
        "ground_truth_best": gt_name,
        "ground_truth_accuracy": gt_acc,
        "ground_truth_objective": gt_obj,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_objective": mean_obj,
        "std_objective": std_obj,
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
    print(f"  {'Model':<50} {'Acc':>7} {'Obj':>9} {'Samples':>8} {'Cost':>10}")
    print(f"  {'─'*50} {'─'*7} {'─'*9} {'─'*8} {'─'*10}")
    for mr in sorted(result.model_results, key=lambda x: (-x.objective, x.latency_seconds)):
        marker = " *" if mr.is_best else ""
        print(f"  {mr.model_name:<50} {mr.accuracy:>7.3f} {mr.objective:>9.4f}"
              f" {mr.n_samples_evaluated:>8} ${mr.cost:>8.4f}{marker}")
    print(f"\n  Best: {result.best_model}  (acc={result.best_accuracy:.3f}, obj={result.best_objective:.4f})")
    print(f"  Total evaluations: {result.total_evaluations}")
    print(f"  Total cost: ${result.total_cost:.4f}")
    print(f"  Found true best ({gt_name}): {'YES' if result.found_true_best else 'NO'}")


def print_summary(summary):
    print(f"\n{'='*60}")
    print(f"  {summary['selector']}  —  {summary['n_seeds']} seeds")
    print(f"  Params: {summary['params']}")
    print(f"{'='*60}")
    print(f"  Ground truth best: {summary['ground_truth_best']}"
          f"  (acc={summary['ground_truth_accuracy']:.3f},"
          f" obj={summary['ground_truth_objective']:.4f})")
    print(f"  Mean accuracy found:   {summary['mean_accuracy']:.4f}"
          f"  +/- {summary['std_accuracy']:.4f}")
    print(f"  Mean objective found:  {summary['mean_objective']:.4f}"
          f"  +/- {summary['std_objective']:.4f}")
    print(f"  Accuracy range:        [{summary['min_accuracy']:.4f},"
          f" {summary['max_accuracy']:.4f}]")
    print(f"  Found true best:       {summary['found_true_best_pct']:.1f}%"
          f"  ({int(summary['found_true_best_pct'] * summary['n_seeds'] / 100)}"
          f"/{summary['n_seeds']})")
    print(f"  Mean evaluations:      {summary['mean_evaluations']:.0f}")
    print(f"  Mean total cost:       ${summary['mean_cost']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Offline selector simulator v3 — combined objective (accuracy - λ*cost - λ*latency)",
    )
    parser.add_argument("--jsonl", default=None, help="Path to brute-force JSONL file")
    parser.add_argument("--pickle", default=None, help="Path to pickle lookup table (.pkl)")
    parser.add_argument("--selectors", default="all",
                        help=f"Comma-separated: {','.join(ALL_SELECTORS)},all")
    parser.add_argument("--seeds", type=int, default=1, help="Number of random seeds (default 1)")
    parser.add_argument("--base-seed", type=int, default=42, help="Starting seed (default 42)")
    parser.add_argument("--output", default=None, help="Path to write summary CSV")

    # Combined objective params
    parser.add_argument("--lambda-cost", type=float, default=0.0,
                        help="Weight for cost penalty (default 0.0 = pure accuracy)")
    parser.add_argument("--lambda-latency", type=float, default=0.0,
                        help="Weight for latency penalty (default 0.0 = pure accuracy)")

    # Selector-specific params
    parser.add_argument("--rs-fraction", type=float, default=0.25)
    parser.add_argument("--arm-confidence", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hc-restarts", type=int, default=3)
    parser.add_argument("--proposer-model", type=str, default="gpt-4.1")
    parser.add_argument("--preview-size", type=int, default=10)
    parser.add_argument("--benchmark", type=str, default="auto",
                        choices=["auto", "gpqa", "bfcl", "hotpotqa", "mathqa"])

    # Matrix UCB params
    parser.add_argument("--ucb-budget", type=float, default=0.2,
                        help="Matrix UCB observation budget fraction (default 0.2)")
    parser.add_argument("--lrf-ensemble", type=int, default=8,
                        help="Matrix UCB-LRF ensemble size (default 8)")

    args = parser.parse_args()

    lc = args.lambda_cost
    ll = args.lambda_latency

    if args.pickle:
        source = args.pickle
        print(f"Loading pickle: {source}")
        models, datapoints, table = load_pickle(source)
    elif args.jsonl:
        source = args.jsonl
        print(f"Loading JSONL: {source}")
        models, datapoints, table = load_jsonl(source)
    else:
        parser.error("One of --jsonl or --pickle is required")
    print(f"  Models: {len(models)}")
    print(f"  Samples: {len(datapoints)}")
    print(f"  Total entries: {sum(len(v) for v in table.values())}")
    print(f"  Lambda cost: {lc}, Lambda latency: {ll}")

    norm = set_norm_stats(table, datapoints)
    print(f"  Normalization: cost [{norm.cost_min:.4f}, {norm.cost_max:.4f}],"
          f" latency [{norm.latency_min:.2f}s, {norm.latency_max:.2f}s]")

    gt_name, gt_acc, gt_obj = compute_ground_truth(models, datapoints, table, lc, ll)
    print(f"\n  Ground truth best: {gt_name}  (acc={gt_acc:.4f}, obj={gt_obj:.4f})")

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
            import os as _os
            if not _os.environ.get("OPENAI_API_KEY"):
                selectors.remove("lm_proposal")
                print("\n  (Skipping lm_proposal — OPENAI_API_KEY not set)")
        except ImportError:
            selectors.remove("lm_proposal")
            print("\n  (Skipping lm_proposal — openai not installed)")
    else:
        selectors = [s.strip() for s in args.selectors.split(",")]

    summaries = []

    for sel in selectors:
        if sel == "brute_force":
            fn = simulate_brute_force
            kwargs = {"lambda_cost": lc, "lambda_latency": ll}
        elif sel == "random_search":
            fn = simulate_random_search
            kwargs = {"sample_fraction": args.rs_fraction,
                      "lambda_cost": lc, "lambda_latency": ll}
        elif sel == "arm_elimination":
            fn = simulate_arm_elimination
            kwargs = {"confidence": args.arm_confidence,
                      "lambda_cost": lc, "lambda_latency": ll}
        elif sel == "epsilon_lucb":
            fn = simulate_epsilon_lucb
            kwargs = {"epsilon": args.epsilon,
                      "lambda_cost": lc, "lambda_latency": ll}
        elif sel == "threshold_se":
            fn = simulate_threshold_se
            kwargs = {"threshold": args.threshold,
                      "lambda_cost": lc, "lambda_latency": ll}
        elif sel == "hill_climbing":
            fn = simulate_hill_climbing
            kwargs = {"num_restarts": args.hc_restarts,
                      "lambda_cost": lc, "lambda_latency": ll}
        elif sel == "bayesian_optimization":
            if not _check_botorch():
                print(f"\n  Skipping {sel} — torch/botorch not installed")
                continue
            fn = simulate_bayesian_optimization
            kwargs = {"lambda_cost": lc, "lambda_latency": ll}
        elif sel == "lm_proposal":
            fn = simulate_lm_proposal
            kwargs = {"proposer_model": args.proposer_model,
                      "preview_size": args.preview_size,
                      "benchmark": args.benchmark,
                      "lambda_cost": lc, "lambda_latency": ll}
        elif sel == "matrix_ucb":
            fn = simulate_matrix_ucb
            kwargs = {"observation_budget_fraction": args.ucb_budget,
                      "lambda_cost": lc, "lambda_latency": ll}
        elif sel == "matrix_ucb_lrf":
            fn = simulate_matrix_ucb_lrf
            kwargs = {"observation_budget_fraction": args.ucb_budget,
                      "ensemble_size": args.lrf_ensemble,
                      "lambda_cost": lc, "lambda_latency": ll}
        else:
            print(f"\n  Unknown selector: {sel}")
            continue

        if args.seeds > 1:
            results = run_multi_seed(fn, models, datapoints, table,
                                     n_seeds=args.seeds, base_seed=args.base_seed, **kwargs)
            summary = summarize_multi_seed(results, gt_name, gt_acc, gt_obj,
                                           models, datapoints, table, lc, ll)
            print_summary(summary)
            summaries.append(summary)
        else:
            result = fn(models, datapoints, table, seed=args.base_seed, **kwargs)
            print_single_result(result, gt_name, gt_acc)
            summaries.append(summarize_multi_seed(
                [result], gt_name, gt_acc, gt_obj, models, datapoints, table, lc, ll))

    if args.output and summaries:
        import csv as csv_mod
        fields = ["selector", "n_seeds", "ground_truth_best",
                  "ground_truth_accuracy", "ground_truth_objective",
                  "mean_accuracy", "std_accuracy",
                  "mean_objective", "std_objective",
                  "found_true_best_pct",
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
