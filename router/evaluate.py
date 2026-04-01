"""Evaluation metrics for the BERT router (hierarchical version).

For 1-tuple benchmarks (GPQA, BFCL):
    Router predicts 9 model scores → pick argmax → look up actual score.

For 2-tuple benchmarks (HotpotQA, MathQA):
    Router predicts role1 scores (9) + role2 scores (9) independently.
    Pick role1 = argmax(pred[0:9]), role2 = argmax(pred[9:18]).
    Combo index = role1 * 9 + role2.
    Look up actual combo score from original 81-dim scores.

Metrics per benchmark and overall:
- Selection accuracy: does the router's combo match the oracle best?
- Top-3 accuracy: is the oracle best reachable from top-3 role1 × top-3 role2?
- Mean selected score: average actual score of the router's pick
- Oracle score: average of the best available score per sample
- Mean regret: oracle_score - mean_selected_score

All metrics respect the mask — only combos with actual scores are
considered when computing argmax, top-k, etc.
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
    n_models: int = 9,
    train_samples: Optional[List[dict]] = None,
) -> Dict:
    """Evaluate the model on a dataloader.

    Args:
        model: BERTRouter model
        dataloader: test DataLoader
        device: torch device
        n_models: number of models per role
        train_samples: if provided, compute majority baseline from train data

    Returns:
        Dict with per-benchmark and overall metrics.
    """
    model.eval()

    # Collect predictions and targets
    records = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        head = batch["head"].to(device)
        combo_scores = batch["combo_scores"]   # (batch, 81) — original scores
        combo_mask = batch["combo_mask"]       # (batch, 81)
        benchmarks = batch["benchmark"]

        predictions = model(input_ids, attention_mask, head).cpu()  # (batch, 18)

        for i in range(len(benchmarks)):
            records.append({
                "benchmark": benchmarks[i],
                "head": head[i].item(),
                "pred": predictions[i],           # (18,)
                "combo_scores": combo_scores[i],  # (81,)
                "combo_mask": combo_mask[i],      # (81,)
            })

    # Group by benchmark
    by_bench = defaultdict(list)
    for r in records:
        by_bench[r["benchmark"]].append(r)

    # Compute metrics per benchmark
    all_metrics = {}
    all_sel_scores = []
    all_oracle_scores = []
    all_correct = []
    all_top3 = []

    for bench in sorted(by_bench.keys()):
        bench_records = by_bench[bench]
        m = _compute_benchmark_metrics(bench, bench_records, n_models)
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
    n_models: int = 9,
) -> dict:
    """Compute metrics for a single benchmark."""
    correct = []
    top3 = []
    selected_scores = []
    oracle_scores = []
    random_scores = []

    rng = random.Random(42)

    for r in records:
        pred = r["pred"]           # (18,)
        combo_scores = r["combo_scores"]  # (81,)
        combo_mask = r["combo_mask"]      # (81,)
        head = r["head"]

        if head == 0:
            # 1-tuple: same as before — pick from 9 models
            n_combos = n_models
            valid_indices = torch.where(combo_mask[:n_combos])[0]
            if len(valid_indices) == 0:
                continue

            valid_scores = combo_scores[valid_indices]
            valid_preds = pred[valid_indices]  # pred[0:9] for 1-tuple

            # Oracle
            oracle_val = valid_scores.max().item()
            oracle_idx_in_valid = valid_scores.argmax().item()
            oracle_idx = valid_indices[oracle_idx_in_valid].item()

            # Router's pick
            selected_idx_in_valid = valid_preds.argmax().item()
            selected_idx = valid_indices[selected_idx_in_valid].item()
            selected_score = combo_scores[selected_idx].item()

            # Top-3
            k = min(3, len(valid_indices))
            top_k_in_valid = valid_preds.topk(k).indices
            top_k_indices = valid_indices[top_k_in_valid]
            is_top3 = oracle_idx in top_k_indices.tolist()

            # Random baseline
            rand_idx = valid_indices[rng.randint(0, len(valid_indices) - 1)].item()
            random_scores.append(combo_scores[rand_idx].item())

        else:
            # 2-tuple hierarchical: pick role1 and role2 independently
            n_combos = n_models * n_models

            # Find valid combos for oracle computation
            valid_combo_indices = torch.where(combo_mask[:n_combos])[0]
            if len(valid_combo_indices) == 0:
                continue

            # Oracle: best actual combo score
            oracle_val = combo_scores[valid_combo_indices].max().item()
            oracle_combo_idx = valid_combo_indices[
                combo_scores[valid_combo_indices].argmax()
            ].item()

            # Determine which role1/role2 models have at least one valid combo
            combo_mask_2d = combo_mask[:n_combos].view(n_models, n_models)

            # Role 1: pick from models that have at least one valid combo
            role1_valid = combo_mask_2d.any(dim=1)  # (9,) — True if row has any valid
            role1_pred = pred[:n_models]
            valid_r1 = torch.where(role1_valid)[0]

            # Role 2: pick from models that have at least one valid combo
            role2_valid = combo_mask_2d.any(dim=0)  # (9,) — True if col has any valid
            role2_pred = pred[n_models:n_models * 2]
            valid_r2 = torch.where(role2_valid)[0]

            if len(valid_r1) == 0 or len(valid_r2) == 0:
                continue

            # Router picks
            r1_pick = valid_r1[role1_pred[valid_r1].argmax()].item()
            r2_pick = valid_r2[role2_pred[valid_r2].argmax()].item()
            selected_idx = r1_pick * n_models + r2_pick

            # If this specific combo doesn't have a score, fall back to
            # the best available combo with this role1 pick
            if combo_mask[selected_idx]:
                selected_score = combo_scores[selected_idx].item()
            else:
                # Find best available combo with role1=r1_pick
                row = combo_scores[r1_pick * n_models:(r1_pick + 1) * n_models]
                row_mask = combo_mask[r1_pick * n_models:(r1_pick + 1) * n_models]
                if row_mask.any():
                    best_in_row = row_mask.float() * row + (~row_mask).float() * (-1e9)
                    selected_idx = r1_pick * n_models + best_in_row.argmax().item()
                    selected_score = combo_scores[selected_idx].item()
                else:
                    # Fallback: pick any valid combo
                    selected_idx = valid_combo_indices[0].item()
                    selected_score = combo_scores[selected_idx].item()

            # Top-3 for hierarchical: can oracle be reached by any combo
            # in top-3 role1 × top-3 role2?
            k1 = min(3, len(valid_r1))
            k2 = min(3, len(valid_r2))
            top_r1 = valid_r1[role1_pred[valid_r1].topk(k1).indices]
            top_r2 = valid_r2[role2_pred[valid_r2].topk(k2).indices]

            # Check if oracle combo is reachable
            oracle_r1 = oracle_combo_idx // n_models
            oracle_r2 = oracle_combo_idx % n_models
            is_top3 = (oracle_r1 in top_r1.tolist()) and (oracle_r2 in top_r2.tolist())

            # Random baseline
            rand_combo = valid_combo_indices[
                rng.randint(0, len(valid_combo_indices) - 1)
            ].item()
            random_scores.append(combo_scores[rand_combo].item())

            correct.append(1 if selected_idx == oracle_combo_idx else 0)
            top3.append(1 if is_top3 else 0)
            selected_scores.append(selected_score)
            oracle_scores.append(oracle_val)
            continue

        # 1-tuple appends (after the if head==0 block above)
        correct.append(1 if selected_idx == oracle_idx else 0)
        top3.append(1 if is_top3 else 0)
        selected_scores.append(selected_score)
        oracle_scores.append(oracle_val)

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
