#!/usr/bin/env python3
"""
Extract per-sample tiebreaker labels from pickle lookup tables.

Uses the brute force tiebreaker logic (accuracy > latency > cost) to determine
the ONE correct model/combo per sample. This replaces score-vector regression
with classification — the router learns to pick the same model that brute force
model selection would pick.

Output: router/labels.json with structure:
{
  "gpqa": {"0": 2, "1": 3, ...},           // sample_idx → best model index (0-8)
  "bfcl": {"0": 2, "1": 2, ...},
  "hotpotqa": {
    "0": {
      "role1": 4,                            // best role1 from overall best combo
      "role2": {"0": 2, "1": 5, "4": 2, ...} // per-role1: best role2 index
    }
  },
  "mathqa": { same as hotpotqa }
}

Usage:
    cd agentopt/
    python -m router.extract_labels
"""

import json
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(ROOT)  # finalagentopt/
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
# Pickle files reference SampleResult from offline_selector_sim_v2
sys.path.insert(0, PARENT)

from router.config import MODELS_1TUPLE, _ROLE_MODELS

PICKLE_DIR = os.path.join(ROOT, "results", "cache_db_results")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "labels.json")


def _find_best(candidates):
    """Apply accuracy > latency > cost tiebreaker.

    Args:
        candidates: list of (index, score, latency, cost) tuples

    Returns:
        index of the best candidate
    """
    tol = 1e-9
    best_idx = None
    best_acc = float("-inf")
    best_lat = float("inf")
    best_cost = float("inf")

    for idx, score, latency, cost in candidates:
        if score > best_acc + tol:
            best_idx = idx
            best_acc = score
            best_lat = latency
            best_cost = cost
        elif abs(score - best_acc) <= tol and latency < best_lat - tol:
            best_idx = idx
            best_acc = score
            best_lat = latency
            best_cost = cost
        elif (abs(score - best_acc) <= tol
              and abs(latency - best_lat) <= tol
              and cost < best_cost - tol):
            best_idx = idx
            best_acc = score
            best_lat = latency
            best_cost = cost

    return best_idx


def _load_pickle(benchmark):
    """Load pickle lookup table."""
    pkl_path = os.path.join(PICKLE_DIR, f"{benchmark}_lookup.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _extract_1tuple_labels(benchmark, combo_list):
    """Extract labels for 1-tuple benchmarks (GPQA, BFCL).

    Returns: {sample_idx_str: best_model_index}
    """
    data = _load_pickle(benchmark)
    table = data["table"]       # {model_name: {dp_index: SampleResult}}
    datapoints = data["datapoints"]

    # Build name-to-index mapping
    name_to_idx = {name: i for i, name in enumerate(combo_list)}

    labels = {}
    tie_counts = []

    for dp in datapoints:
        candidates = []
        for model_name, model_data in table.items():
            if dp not in model_data:
                continue
            sr = model_data[dp]
            idx = name_to_idx.get(model_name)
            if idx is None:
                continue
            cost = sr.cost if hasattr(sr, 'cost') and sr.cost is not None else float("inf")
            latency = sr.latency_seconds if hasattr(sr, 'latency_seconds') else float("inf")
            candidates.append((idx, sr.score, latency, cost))

        if not candidates:
            continue

        best_idx = _find_best(candidates)
        labels[str(dp)] = best_idx

        # Count ties for stats
        best_score = max(c[1] for c in candidates)
        n_tied = sum(1 for c in candidates if abs(c[1] - best_score) < 1e-9)
        tie_counts.append(n_tied)

    avg_ties = sum(tie_counts) / len(tie_counts) if tie_counts else 0
    print(f"  {benchmark}: {len(labels)} samples, avg {avg_ties:.1f} models tied at top score")

    # Print label distribution
    from collections import Counter
    dist = Counter(labels.values())
    for idx in sorted(dist):
        name = combo_list[idx]
        print(f"    [{idx}] {name}: {dist[idx]} samples ({dist[idx]/len(labels)*100:.1f}%)")

    return labels


def _extract_2tuple_labels(benchmark, combo_list):
    """Extract labels for 2-tuple benchmarks (HotpotQA, MathQA).

    Returns: {
        sample_idx_str: {
            "role1": int,  // best role1 index (0-8)
            "role2": {     // per-role1: best role2 index
                "0": int, "1": int, ...
            }
        }
    }
    """
    data = _load_pickle(benchmark)
    table = data["table"]
    datapoints = data["datapoints"]

    n_models = 9
    # combo_list is ordered: role1_0*role2_0, role1_0*role2_1, ..., role1_8*role2_8
    # combo index = role1_idx * 9 + role2_idx

    labels = {}

    for dp in datapoints:
        # Collect all combo results for this sample
        combo_results = {}  # {combo_idx: (score, latency, cost)}
        for model_name, model_data in table.items():
            if dp not in model_data:
                continue
            try:
                combo_idx = combo_list.index(model_name)
            except ValueError:
                continue
            sr = model_data[dp]
            cost = sr.cost if hasattr(sr, 'cost') and sr.cost is not None else float("inf")
            latency = sr.latency_seconds if hasattr(sr, 'latency_seconds') else float("inf")
            combo_results[combo_idx] = (sr.score, latency, cost)

        if not combo_results:
            continue

        # Overall best combo (accuracy > latency > cost)
        candidates = [(idx, s, l, c) for idx, (s, l, c) in combo_results.items()]
        best_combo_idx = _find_best(candidates)
        best_role1 = best_combo_idx // n_models

        # Per-role1: best role2
        role2_labels = {}
        for r1 in range(n_models):
            row_candidates = []
            for r2 in range(n_models):
                cidx = r1 * n_models + r2
                if cidx in combo_results:
                    s, l, c = combo_results[cidx]
                    row_candidates.append((r2, s, l, c))
            if row_candidates:
                best_r2 = _find_best(row_candidates)
                role2_labels[str(r1)] = best_r2

        labels[str(dp)] = {
            "role1": best_role1,
            "role2": role2_labels,
        }

    print(f"  {benchmark}: {len(labels)} samples")

    # Role1 label distribution
    from collections import Counter
    r1_dist = Counter(v["role1"] for v in labels.values())
    for idx in sorted(r1_dist):
        name = _ROLE_MODELS[idx]
        print(f"    role1=[{idx}] {name}: {r1_dist[idx]} samples ({r1_dist[idx]/len(labels)*100:.1f}%)")

    return labels


def extract_all():
    """Extract tiebreaker labels for all benchmarks."""
    all_labels = {}

    print("Extracting 1-tuple labels...")
    all_labels["gpqa"] = _extract_1tuple_labels("gpqa", MODELS_1TUPLE)
    all_labels["bfcl"] = _extract_1tuple_labels("bfcl", MODELS_1TUPLE)

    print("\nExtracting 2-tuple labels...")
    from router.config import COMBOS_HOTPOTQA, COMBOS_MATHQA
    all_labels["hotpotqa"] = _extract_2tuple_labels("hotpotqa", COMBOS_HOTPOTQA)
    all_labels["mathqa"] = _extract_2tuple_labels("mathqa", COMBOS_MATHQA)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_labels, f)
    print(f"\nLabels saved to {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")


if __name__ == "__main__":
    extract_all()
