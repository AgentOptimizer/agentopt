#!/usr/bin/env python3
"""
Extract per-sample scores from pickle lookup tables for all 4 benchmarks.

The pickle files at results/cache_db_results/<benchmark>_lookup.pkl are the
authoritative source of per-sample scores. They were built by
cache_selector_sim.py which replayed cache.db through the actual eval
functions at the time the benchmarks were run.

We do NOT re-replay from cache.db because the GPQA dataset file changed
after the original run (re-downloaded from HuggingFace with different row
ordering), causing choice shuffling to differ. The pickle scores are frozen
from the correct original run.

Output: router/scores.json with structure:
{
  "gpqa": {"0": {"agent=Claude Opus 4.6": 1.0, ...}, ...},
  "bfcl": {"0": {"agent=Claude Opus 4.6": 1.0, ...}, ...},
  "hotpotqa": {"0": {"planner=X + solver=Y": 0.67, ...}, ...},
  "mathqa": {"0": {"answer=X + critic=Y": 1.0, ...}, ...},
}

Usage:
    cd agentopt/
    python -m router.extract_scores
    python -m router.extract_scores --benchmark gpqa   # single benchmark
"""

import argparse
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

PICKLE_DIR = os.path.join(ROOT, "results", "cache_db_results")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "scores.json")


def _load_pickle_scores(benchmark):
    """Load per-sample scores from a benchmark's pickle lookup table.

    Returns: {sample_index: {model_name: score}}
    """
    pkl_path = os.path.join(PICKLE_DIR, f"{benchmark}_lookup.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Pickle not found: {pkl_path}\n"
            f"Run cache_selector_sim.py --benchmark {benchmark} first."
        )

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    table = data["table"]       # {model_name: {dp_index: SampleResult}}
    datapoints = data["datapoints"]  # list of sample indices

    scores = {}  # {sample_index: {model_name: score}}
    for model_name, model_data in table.items():
        for dp in datapoints:
            if dp not in model_data:
                continue
            if dp not in scores:
                scores[dp] = {}
            scores[dp][model_name] = model_data[dp].score

    return scores


def extract_all(benchmarks=None):
    """Extract scores for all (or specified) benchmarks."""
    all_benchmarks = benchmarks or ["gpqa", "bfcl", "hotpotqa", "mathqa"]

    # Load existing scores if we're only running a subset
    all_scores = {}
    if os.path.exists(OUTPUT_PATH) and benchmarks is not None:
        with open(OUTPUT_PATH) as f:
            all_scores = json.load(f)

    for bench in all_benchmarks:
        print(f"\n{'='*60}")
        print(f"  Extracting scores: {bench}")
        print(f"{'='*60}")

        scores = _load_pickle_scores(bench)

        # Convert int keys to strings for JSON
        all_scores[bench] = {str(k): v for k, v in scores.items()}

        # Print coverage
        n_samples = len(scores)
        if scores:
            all_names = set()
            for s in scores.values():
                all_names.update(s.keys())
            n_models = len(all_names)
            full = sum(1 for s in scores.values() if len(s) == n_models)
            partial = sum(1 for s in scores.values() if len(s) < n_models)
            min_cov = min(len(s) for s in scores.values())
            max_cov = max(len(s) for s in scores.values())

            print(f"  Coverage: {n_samples} samples × {n_models} models/combos")
            print(f"  Full coverage: {full}/{n_samples} samples")
            if partial > 0:
                print(f"  Partial coverage: {partial} samples "
                      f"(min={min_cov}, max={max_cov})")

            # Verify known accuracies
            for name in sorted(all_names):
                model_scores = [scores[dp][name]
                                for dp in scores if name in scores[dp]]
                acc = sum(model_scores) / len(model_scores)
                print(f"    {name}: {acc*100:.2f}% ({len(model_scores)} samples)")
        else:
            print("  WARNING: No scores extracted!")

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_scores, f)
    print(f"\nScores saved to {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")

    # Final summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for bench in all_benchmarks:
        data = all_scores.get(bench, {})
        n_samples = len(data)
        if data:
            all_names = set()
            for s in data.values():
                all_names.update(s.keys())
            n_models = len(all_names)
            full = sum(1 for s in data.values() if len(s) == n_models)
            expected_models = 9 if bench in ("gpqa", "bfcl") else 81
            print(f"  {bench:>10}: {n_samples:>3} samples × {n_models:>2} models "
                  f"({full}/{n_samples} full) "
                  f"[expected: {n_samples}×{expected_models}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract per-sample scores from pickle lookup tables"
    )
    parser.add_argument(
        "--benchmark", "-b",
        choices=["gpqa", "bfcl", "hotpotqa", "mathqa"],
        help="Single benchmark to extract (default: all)",
    )
    args = parser.parse_args()

    benchmarks = [args.benchmark] if args.benchmark else None
    extract_all(benchmarks)
