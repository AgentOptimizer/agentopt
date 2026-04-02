"""Evaluation metrics for the BERT router (classification version).

Primary metric: selection_accuracy — does the router pick the same model that
brute force model selection (accuracy > latency > cost tiebreaker) picks?

For 1-tuple benchmarks (GPQA, BFCL):
    Router predicts 9 logits → pick argmax → compare to tiebreaker label.

For 2-tuple benchmarks (HotpotQA, MathQA):
    Pass 1: Router predicts role1 logits from query → pick role1 = argmax
    Pass 2: Router predicts role2 logits from query + role1_output → pick role2
    Combo index = role1 * 9 + role2.
    Compare to tiebreaker-selected best combo.

Metrics per benchmark and overall:
- Selection accuracy: does the router's pick match the tiebreaker label?
- Score-match accuracy: does the router's pick achieve the oracle best score? (lenient)
- Top-3 accuracy: is the tiebreaker label reachable from top-3 picks?
- Mean selected score: average actual score of the router's pick
- Oracle score: average of the best available score per sample
- Mean regret: oracle_score - mean_selected_score

All metrics respect the mask — only combos with actual scores are
considered when computing argmax, top-k, etc.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .config import (
    BENCHMARK_TOKENS,
    MODELS_1TUPLE,
    COMBOS_HOTPOTQA,
    COMBOS_MATHQA,
    _ROLE_MODELS,
)


def _load_role1_outputs() -> Dict:
    """Load role1 outputs for two-pass evaluation."""
    path = os.path.join(os.path.dirname(__file__), "role1_outputs.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _load_labels() -> Dict:
    """Load tiebreaker labels for evaluation."""
    path = os.path.join(os.path.dirname(__file__), "labels.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    tokenizer=None,
    n_models: int = 9,
) -> Dict:
    """Evaluate the model on a dataloader.

    For 2-tuple benchmarks, performs two-pass evaluation:
    1. Predict role1 from query (head=1 examples)
    2. Encode query + role1 output -> predict role2

    Args:
        model: BERTRouter model
        dataloader: test DataLoader
        device: torch device
        tokenizer: needed for encoding role2 input in two-pass eval
        n_models: number of models per role

    Returns:
        Dict with per-benchmark and overall metrics.
    """
    model.eval()
    role1_outputs = _load_role1_outputs()
    all_labels = _load_labels()

    # First pass: collect all predictions and group by (benchmark, sample_idx)
    records_1tuple = []
    records_2tuple = {}

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        heads = batch["head"]
        labels = batch["labels"]
        combo_scores = batch["combo_scores"]
        combo_masks = batch["combo_mask"]
        benchmarks = batch["benchmark"]
        sample_idxs = batch["sample_idx"]

        logits = model(input_ids, attention_mask).cpu()  # (batch, 9)

        for i in range(len(benchmarks)):
            head = heads[i].item()

            if head == 0:
                records_1tuple.append({
                    "benchmark": benchmarks[i],
                    "sample_idx": sample_idxs[i],
                    "logits": logits[i],
                    "label": labels[i].item(),
                    "combo_scores": combo_scores[i],
                    "combo_mask": combo_masks[i],
                })
            elif head == 1:
                key = (benchmarks[i], sample_idxs[i])
                records_2tuple[key] = {
                    "benchmark": benchmarks[i],
                    "sample_idx": sample_idxs[i],
                    "role1_logits": logits[i],
                    "label": labels[i].item(),  # role1 tiebreaker label
                    "combo_scores": combo_scores[i],
                    "combo_mask": combo_masks[i],
                }

    # Second pass for 2-tuple: pick role1, then encode query+output, predict role2
    records_2tuple_eval = []
    max_length = 2048

    for key, rec in records_2tuple.items():
        bench = rec["benchmark"]
        sample_idx = rec["sample_idx"]
        role1_logits = rec["role1_logits"]  # (9,)
        combo_scores = rec["combo_scores"]  # (81,)
        combo_mask = rec["combo_mask"]  # (81,)

        # Get tiebreaker labels for this sample
        bench_labels = all_labels.get(bench, {})
        sample_label = bench_labels.get(str(sample_idx), {})
        role1_tiebreaker = sample_label.get("role1", -1) if isinstance(sample_label, dict) else -1
        role2_labels = sample_label.get("role2", {}) if isinstance(sample_label, dict) else {}

        # Pick role1 model (from valid models only)
        combo_mask_2d = combo_mask[:n_models * n_models].view(n_models, n_models)
        role1_valid = combo_mask_2d.any(dim=1)
        valid_r1 = torch.where(role1_valid)[0]

        if len(valid_r1) == 0:
            continue

        r1_pick = valid_r1[role1_logits[valid_r1].argmax()].item()

        # Get role1 output text
        bench_outputs = role1_outputs.get(bench, {})
        sample_outputs = bench_outputs.get(str(sample_idx), {})
        r1_model_name = _ROLE_MODELS[r1_pick]
        r1_output = sample_outputs.get(r1_model_name, "")

        # Encode query + role1 output for role2 prediction
        if tokenizer is not None and r1_output:
            token = BENCHMARK_TOKENS[bench]
            role2_text = f"{token} [SEP] {r1_output}"

            enc = tokenizer(
                role2_text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            r2_input_ids = enc["input_ids"].to(device)
            r2_attention_mask = enc["attention_mask"].to(device)
            role2_logits = model(r2_input_ids, r2_attention_mask).cpu().squeeze(0)
        else:
            role2_logits = role1_logits

        # Get the tiebreaker label for this role1's best role2
        r2_tiebreaker = role2_labels.get(str(r1_pick), -1)

        # Overall tiebreaker combo
        tiebreaker_combo = role1_tiebreaker * n_models + role2_labels.get(str(role1_tiebreaker), 0) if role1_tiebreaker >= 0 else -1

        records_2tuple_eval.append({
            "benchmark": bench,
            "role1_logits": role1_logits,
            "role2_logits": role2_logits,
            "r1_pick": r1_pick,
            "r1_tiebreaker": role1_tiebreaker,
            "r2_tiebreaker": r2_tiebreaker,
            "tiebreaker_combo": tiebreaker_combo,
            "combo_scores": combo_scores,
            "combo_mask": combo_mask,
        })

    # Compute metrics
    by_bench = defaultdict(lambda: {"1tuple": [], "2tuple": []})
    for r in records_1tuple:
        by_bench[r["benchmark"]]["1tuple"].append(r)
    for r in records_2tuple_eval:
        by_bench[r["benchmark"]]["2tuple"].append(r)

    all_metrics = {}
    all_sel_scores = []
    all_oracle_scores = []
    all_correct = []
    all_score_match = []
    all_top3 = []

    for bench in sorted(by_bench.keys()):
        bench_data = by_bench[bench]
        m = _compute_benchmark_metrics(bench, bench_data, n_models)
        all_metrics[bench] = m

        all_sel_scores.extend(m["_selected_scores"])
        all_oracle_scores.extend(m["_oracle_scores"])
        all_correct.extend(m["_correct"])
        all_score_match.extend(m["_score_match"])
        all_top3.extend(m["_top3"])

    # Overall metrics
    n = len(all_correct)
    overall_sel_acc = sum(all_correct) / n if n else 0
    overall_score_match = sum(all_score_match) / n if n else 0
    overall_top3 = sum(all_top3) / n if n else 0
    overall_mean_sel = sum(all_sel_scores) / n if n else 0
    overall_oracle = sum(all_oracle_scores) / n if n else 0

    all_metrics["overall"] = {
        "selection_accuracy": overall_sel_acc,
        "score_match_accuracy": overall_score_match,
        "top3_accuracy": overall_top3,
        "mean_selected_score": overall_mean_sel,
        "oracle_score": overall_oracle,
        "mean_regret": overall_oracle - overall_mean_sel,
        "n_samples": n,
    }

    # Clean up internal lists
    for bench in list(all_metrics.keys()):
        for k in ("_selected_scores", "_oracle_scores", "_correct", "_score_match", "_top3"):
            all_metrics[bench].pop(k, None)

    return all_metrics


def _compute_benchmark_metrics(
    benchmark: str,
    bench_data: dict,
    n_models: int = 9,
) -> dict:
    """Compute metrics for a single benchmark."""
    correct = []       # strict: matches tiebreaker label
    score_match = []   # lenient: achieves same score as oracle
    top3 = []
    selected_scores = []
    oracle_scores = []
    random_scores = []

    rng = random.Random(42)

    # 1-tuple records
    for r in bench_data["1tuple"]:
        logits = r["logits"]  # (9,)
        label = r["label"]   # tiebreaker-selected best model
        combo_scores = r["combo_scores"]  # (81,)
        combo_mask = r["combo_mask"]  # (81,)

        n_combos = n_models
        valid_indices = torch.where(combo_mask[:n_combos])[0]
        if len(valid_indices) == 0:
            continue

        valid_scores = combo_scores[valid_indices]
        valid_logits = logits[valid_indices]

        oracle_val = valid_scores.max().item()

        # Router's pick
        selected_idx_in_valid = valid_logits.argmax().item()
        selected_idx = valid_indices[selected_idx_in_valid].item()
        selected_score = combo_scores[selected_idx].item()

        # Strict: does selected match tiebreaker label?
        is_correct = 1 if selected_idx == label else 0

        # Lenient: does selected achieve oracle score?
        is_score_match = 1 if abs(selected_score - oracle_val) < 1e-6 else 0

        # Top-3: is the tiebreaker label in top-3 picks?
        k = min(3, len(valid_indices))
        top_k_in_valid = valid_logits.topk(k).indices
        top_k_indices = valid_indices[top_k_in_valid]
        is_top3 = 1 if label in top_k_indices.tolist() else 0

        rand_idx = valid_indices[rng.randint(0, len(valid_indices) - 1)].item()
        random_scores.append(combo_scores[rand_idx].item())

        correct.append(is_correct)
        score_match.append(is_score_match)
        top3.append(is_top3)
        selected_scores.append(selected_score)
        oracle_scores.append(oracle_val)

    # 2-tuple records (two-pass)
    for r in bench_data["2tuple"]:
        role1_logits = r["role1_logits"]  # (9,)
        role2_logits = r["role2_logits"]  # (9,)
        r1_pick = r["r1_pick"]
        r1_tiebreaker = r["r1_tiebreaker"]
        tiebreaker_combo = r["tiebreaker_combo"]
        combo_scores = r["combo_scores"]  # (81,)
        combo_mask = r["combo_mask"]  # (81,)

        n_combos = n_models * n_models
        valid_combo_indices = torch.where(combo_mask[:n_combos])[0]
        if len(valid_combo_indices) == 0:
            continue

        # Oracle
        oracle_val = combo_scores[valid_combo_indices].max().item()

        # Role2 pick
        combo_mask_2d = combo_mask[:n_combos].view(n_models, n_models)
        role2_valid = combo_mask_2d[r1_pick]
        valid_r2 = torch.where(role2_valid)[0]

        if len(valid_r2) == 0:
            selected_idx = valid_combo_indices[0].item()
            selected_score = combo_scores[selected_idx].item()
        else:
            r2_pick = valid_r2[role2_logits[valid_r2].argmax()].item()
            selected_idx = r1_pick * n_models + r2_pick

            if combo_mask[selected_idx]:
                selected_score = combo_scores[selected_idx].item()
            else:
                row = combo_scores[r1_pick * n_models:(r1_pick + 1) * n_models]
                row_mask = combo_mask[r1_pick * n_models:(r1_pick + 1) * n_models]
                if row_mask.any():
                    best_in_row = row_mask.float() * row + (~row_mask).float() * (-1e9)
                    selected_idx = r1_pick * n_models + best_in_row.argmax().item()
                    selected_score = combo_scores[selected_idx].item()
                else:
                    selected_idx = valid_combo_indices[0].item()
                    selected_score = combo_scores[selected_idx].item()

        # Strict: does selected combo match tiebreaker combo?
        is_correct = 1 if selected_idx == tiebreaker_combo else 0

        # Lenient: does selected achieve oracle score?
        is_score_match = 1 if abs(selected_score - oracle_val) < 1e-6 else 0

        # Top-3: check if tiebreaker combo is reachable from top-3 role1 x top-3 role2
        role1_valid_mask = combo_mask_2d.any(dim=1)
        valid_r1 = torch.where(role1_valid_mask)[0]

        k1 = min(3, len(valid_r1))
        k2 = min(3, len(valid_r2)) if len(valid_r2) > 0 else 0
        top_r1 = valid_r1[role1_logits[valid_r1].topk(k1).indices]

        if k2 > 0 and tiebreaker_combo >= 0:
            top_r2 = valid_r2[role2_logits[valid_r2].topk(k2).indices]
            is_top3 = False
            for r1_cand in top_r1.tolist():
                for r2_cand in top_r2.tolist():
                    cand_idx = r1_cand * n_models + r2_cand
                    if cand_idx == tiebreaker_combo:
                        is_top3 = True
                        break
                if is_top3:
                    break
        else:
            is_top3 = False

        # Random baseline
        rand_combo = valid_combo_indices[
            rng.randint(0, len(valid_combo_indices) - 1)
        ].item()
        random_scores.append(combo_scores[rand_combo].item())

        correct.append(is_correct)
        score_match.append(is_score_match)
        top3.append(1 if is_top3 else 0)
        selected_scores.append(selected_score)
        oracle_scores.append(oracle_val)

    n = len(correct)
    sel_acc = sum(correct) / n if n else 0
    score_match_acc = sum(score_match) / n if n else 0
    top3_acc = sum(top3) / n if n else 0
    mean_sel = sum(selected_scores) / n if n else 0
    mean_oracle = sum(oracle_scores) / n if n else 0
    mean_random = sum(random_scores) / n if n else 0

    return {
        "selection_accuracy": sel_acc,
        "score_match_accuracy": score_match_acc,
        "top3_accuracy": top3_acc,
        "mean_selected_score": mean_sel,
        "oracle_score": mean_oracle,
        "mean_regret": mean_oracle - mean_sel,
        "random_baseline_score": mean_random,
        "n_samples": n,
        "_selected_scores": selected_scores,
        "_oracle_scores": oracle_scores,
        "_correct": correct,
        "_score_match": score_match,
        "_top3": top3,
    }
