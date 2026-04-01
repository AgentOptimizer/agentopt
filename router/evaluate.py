"""Evaluation metrics for the BERT router.

Metrics per benchmark and overall:
- Selection accuracy: does argmax(predicted) match argmax(actual)?
- Top-3 accuracy: is the oracle best in the top 3 predicted?
- Mean selected score: average actual score of the router's pick
- Oracle score: average of the best available score per sample
- Mean regret: oracle_score - mean_selected_score
- Cost of selected: average cost of the router's pick (if cost data available)

All metrics respect the mask — only combos with actual scores are
considered when computing argmax, top-k, etc.

Baselines:
- Random: pick a random combo (among those with scores)
- Most expensive: always pick the highest-cost combo
- Cheapest: always pick the lowest-cost combo
- Majority: always pick the combo with highest average score in train set
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .config import COMBO_COSTS, MODELS_1TUPLE, COMBOS_HOTPOTQA, COMBOS_MATHQA


def _combo_list_for_head(head: int, benchmark: str) -> List[str]:
    """Return the combo name list for a given head/benchmark."""
    if head == 0:
        return MODELS_1TUPLE
    elif benchmark == "hotpotqa":
        return COMBOS_HOTPOTQA
    else:
        return COMBOS_MATHQA


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    train_samples: Optional[List[dict]] = None,
) -> Dict:
    """Evaluate the model on a dataloader.

    Args:
        model: BERTRouter model
        dataloader: test DataLoader
        device: torch device
        train_samples: if provided, compute majority baseline from train data

    Returns:
        Dict with per-benchmark and overall metrics.
    """
    model.eval()

    # Collect predictions and targets
    records = []  # list of (benchmark, head, pred_vec, score_vec, mask_vec)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        head = batch["head"].to(device)
        scores = batch["scores"]       # keep on CPU
        mask = batch["mask"]           # keep on CPU
        benchmarks = batch["benchmark"]

        predictions = model(input_ids, attention_mask, head).cpu()

        for i in range(len(benchmarks)):
            records.append({
                "benchmark": benchmarks[i],
                "head": head[i].item(),
                "pred": predictions[i],
                "scores": scores[i],
                "mask": mask[i],
            })

    # Group by benchmark
    by_bench = defaultdict(list)
    for r in records:
        by_bench[r["benchmark"]].append(r)

    # Compute majority baseline from training data
    majority_combo = {}
    if train_samples:
        majority_combo = _compute_majority(train_samples)

    # Compute metrics per benchmark
    all_metrics = {}
    all_sel_scores = []
    all_oracle_scores = []
    all_correct = []
    all_top3 = []

    for bench in sorted(by_bench.keys()):
        bench_records = by_bench[bench]
        m = _compute_benchmark_metrics(bench, bench_records, majority_combo.get(bench))
        all_metrics[bench] = m

        all_sel_scores.extend(m["_selected_scores"])
        all_oracle_scores.extend(m["_oracle_scores"])
        all_correct.extend(m["_correct"])
        all_top3.extend(m["_top3"])

    # Overall metrics
    n = len(all_correct)
    overall_sel_acc = sum(all_correct) / n if n else 0
    overall_top3 = sum(all_top3) / n if n else 0
    overall_mean_sel = sum(all_sel_scores) / n if n else 0
    overall_oracle = sum(all_oracle_scores) / n if n else 0

    all_metrics["overall"] = {
        "selection_accuracy": overall_sel_acc,
        "top3_accuracy": overall_top3,
        "mean_selected_score": overall_mean_sel,
        "oracle_score": overall_oracle,
        "mean_regret": overall_oracle - overall_mean_sel,
        "n_samples": n,
    }

    # Clean up internal lists
    for bench in list(all_metrics.keys()):
        for k in ("_selected_scores", "_oracle_scores", "_correct", "_top3"):
            all_metrics[bench].pop(k, None)

    return all_metrics


def _compute_benchmark_metrics(
    benchmark: str,
    records: list,
    majority_idx: Optional[int] = None,
) -> dict:
    """Compute metrics for a single benchmark."""
    correct = []
    top3 = []
    selected_scores = []
    oracle_scores = []
    random_scores = []

    rng = random.Random(42)

    for r in records:
        pred = r["pred"]
        scores = r["scores"]
        mask = r["mask"]
        head = r["head"]

        n_combos = 9 if head == 0 else 81

        # Only consider combos that have scores
        valid_indices = torch.where(mask[:n_combos])[0]
        if len(valid_indices) == 0:
            continue

        valid_scores = scores[valid_indices]
        valid_preds = pred[valid_indices]

        # Oracle: best actual score among valid combos
        oracle_val = valid_scores.max().item()
        oracle_idx_in_valid = valid_scores.argmax().item()
        oracle_idx = valid_indices[oracle_idx_in_valid].item()

        # Router's pick: highest predicted score among valid combos
        selected_idx_in_valid = valid_preds.argmax().item()
        selected_idx = valid_indices[selected_idx_in_valid].item()
        selected_score = scores[selected_idx].item()

        # Top-3: are any of the top 3 predicted the oracle best?
        k = min(3, len(valid_indices))
        top_k_in_valid = valid_preds.topk(k).indices
        top_k_indices = valid_indices[top_k_in_valid]
        is_top3 = oracle_idx in top_k_indices.tolist()

        correct.append(1 if selected_idx == oracle_idx else 0)
        top3.append(1 if is_top3 else 0)
        selected_scores.append(selected_score)
        oracle_scores.append(oracle_val)

        # Random baseline
        rand_idx = valid_indices[rng.randint(0, len(valid_indices) - 1)].item()
        random_scores.append(scores[rand_idx].item())

    n = len(correct)
    sel_acc = sum(correct) / n if n else 0
    top3_acc = sum(top3) / n if n else 0
    mean_sel = sum(selected_scores) / n if n else 0
    mean_oracle = sum(oracle_scores) / n if n else 0
    mean_random = sum(random_scores) / n if n else 0

    return {
        "selection_accuracy": sel_acc,
        "top3_accuracy": top3_acc,
        "mean_selected_score": mean_sel,
        "oracle_score": mean_oracle,
        "mean_regret": mean_oracle - mean_sel,
        "random_baseline_score": mean_random,
        "n_samples": n,
        # Internal — will be removed by caller
        "_selected_scores": selected_scores,
        "_oracle_scores": oracle_scores,
        "_correct": correct,
        "_top3": top3,
    }


def _compute_majority(train_samples: List[dict]) -> Dict[str, int]:
    """Compute majority baseline: combo index with highest avg score in train.

    Returns: {benchmark: combo_index}
    """
    # Accumulate scores per combo per benchmark
    by_bench = defaultdict(lambda: defaultdict(list))
    for s in train_samples:
        bench = s["benchmark"]
        scores = s["scores"]
        mask = s["mask"]
        n = 9 if s["head"] == 0 else 81
        for i in range(n):
            if mask[i]:
                by_bench[bench][i].append(scores[i].item())

    result = {}
    for bench, combo_scores in by_bench.items():
        best_idx = max(combo_scores, key=lambda i: sum(combo_scores[i]) / len(combo_scores[i]))
        result[bench] = best_idx
    return result


def run_baselines(
    test_samples: List[dict],
    train_samples: Optional[List[dict]] = None,
) -> Dict:
    """Run baseline selection strategies on test samples.

    Baselines:
    - random: uniform random among valid combos
    - cheapest: lowest cost combo (1-tuple only, needs COMBO_COSTS)
    - most_expensive: highest cost combo (1-tuple only)
    - majority: combo with highest avg score in train set
    """
    rng = random.Random(42)
    majority_map = _compute_majority(train_samples) if train_samples else {}

    by_bench = defaultdict(list)
    for s in test_samples:
        by_bench[s["benchmark"]].append(s)

    results = {}
    for bench in sorted(by_bench.keys()):
        samples = by_bench[bench]
        head = samples[0]["head"]
        n_combos = 9 if head == 0 else 81
        combo_list = _combo_list_for_head(head, bench)

        baselines = {
            "random": [],
            "majority": [],
        }
        oracle_scores = []

        # Cost-based baselines only for 1-tuple
        if head == 0:
            baselines["cheapest"] = []
            baselines["most_expensive"] = []
            # Find cheapest/most expensive indices
            costs = [COMBO_COSTS.get(combo_list[i], 0) for i in range(n_combos)]
            cheapest_idx = min(range(n_combos), key=lambda i: costs[i])
            expensive_idx = max(range(n_combos), key=lambda i: costs[i])

        maj_idx = majority_map.get(bench)

        for s in samples:
            scores = s["scores"]
            mask = s["mask"]
            valid = torch.where(mask[:n_combos])[0]
            if len(valid) == 0:
                continue

            oracle = scores[valid].max().item()
            oracle_scores.append(oracle)

            # Random
            r_idx = valid[rng.randint(0, len(valid) - 1)].item()
            baselines["random"].append(scores[r_idx].item())

            # Majority
            if maj_idx is not None and mask[maj_idx]:
                baselines["majority"].append(scores[maj_idx].item())
            else:
                baselines["majority"].append(scores[r_idx].item())

            # Cost-based (1-tuple)
            if head == 0:
                if mask[cheapest_idx]:
                    baselines["cheapest"].append(scores[cheapest_idx].item())
                else:
                    baselines["cheapest"].append(scores[r_idx].item())
                if mask[expensive_idx]:
                    baselines["most_expensive"].append(scores[expensive_idx].item())
                else:
                    baselines["most_expensive"].append(scores[r_idx].item())

        n = len(oracle_scores)
        mean_oracle = sum(oracle_scores) / n if n else 0
        bench_results = {"oracle": mean_oracle, "n_samples": n}

        for name, sel_scores in baselines.items():
            mean_sel = sum(sel_scores) / n if n else 0
            bench_results[name] = {
                "mean_score": mean_sel,
                "regret": mean_oracle - mean_sel,
            }

        results[bench] = bench_results

    return results
