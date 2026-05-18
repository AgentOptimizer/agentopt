#!/usr/bin/env python3
"""
Cache-Powered Selector Simulator
=================================
Runs brute force once from cache.db (instant — all API calls are cached),
extracts per-sample scores, then simulates all selectors with 50 seeds.

NO JSONL files involved. Source of truth is cache.db.

Usage:
    python cache_selector_sim.py --benchmark gpqa --seeds 50
    python cache_selector_sim.py --benchmark bfcl --seeds 50
    python cache_selector_sim.py --benchmark hotpotqa --seeds 50
    python cache_selector_sim.py --benchmark mathqa --seeds 50
"""

import argparse
import asyncio
import os
import pickle
import sys
import time

# Add agentopt to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agentopt"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agentopt", "src"))

from dotenv import load_dotenv
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "agentopt", ".env"))

# Import simulation functions from offline_selector_sim_v2
from offline_selector_sim_v2 import (
    SampleResult,
    LookupTable,
    compute_ground_truth,
    simulate_random_search,
    simulate_arm_elimination,
    simulate_epsilon_lucb,
    simulate_threshold_se,
    simulate_hill_climbing,
    run_multi_seed,
    summarize_multi_seed,
    print_summary,
    _compute_sample_cost,
    _check_botorch,
    ALL_SELECTORS,
)

try:
    from offline_selector_sim_v2 import simulate_bayesian_optimization
except ImportError:
    simulate_bayesian_optimization = None

try:
    from offline_selector_sim_v2 import simulate_lm_proposal
except ImportError:
    simulate_lm_proposal = None

try:
    from offline_selector_sim_v2 import simulate_matrix_ucb
except ImportError:
    simulate_matrix_ucb = None

try:
    from offline_selector_sim_v2 import simulate_matrix_ucb_lrf
except ImportError:
    simulate_matrix_ucb_lrf = None

from benchmarks.common import make_llm, display_name

# DEFAULT_MODELS and BEDROCK_PRICES were previously re-exported from
# benchmarks.common; they now live in other_benchmarks/mathqa/MAS_math_qa.py.
# Inlined here to avoid a cross-package import dance.
DEFAULT_MODELS = [
    "Claude 3 Haiku",
    "Claude Haiku 4.5",
    "Claude Opus 4.6",
    "gpt-oss-20b",
    "gpt-oss-120b",
    "Kimi K2.5",
    "Ministral 3 8B",
    "Qwen3 32B",
    "Qwen3 Next 80B A3B",
]

_BASE_PRICES = {
    "Claude 3 Haiku": {"input_price": 0.25, "output_price": 1.25},
    "Claude Haiku 4.5": {"input_price": 0.80, "output_price": 4.00},
    "Claude Opus 4.6": {"input_price": 5.00, "output_price": 25.00},
    "gpt-oss-20b": {"input_price": 0.22, "output_price": 0.88},
    "gpt-oss-120b": {"input_price": 1.20, "output_price": 4.80},
    "Kimi K2.5": {"input_price": 0.35, "output_price": 1.40},
    "Ministral 3 8B": {"input_price": 0.04, "output_price": 0.04},
    "Qwen3 32B": {"input_price": 0.17, "output_price": 0.85},
    "Qwen3 Next 80B A3B": {"input_price": 0.25, "output_price": 1.25},
}

_PROFILE_DISPLAY_NAMES = {
    "58ii6j0n0zhw": "Claude 3 Haiku",
    "4ax1twcuwbfk": "Claude Haiku 4.5",
    "vqhud2pxz4wy": "Claude Opus 4.6",
    "nrqbxznvrt7p": "Kimi K2.5",
    "uj2ujdo7k1qe": "Ministral 3 8B",
    "d6kuf8xcphsl": "Qwen3 32B",
    "a6jppcyeu4ms": "Qwen3 Next 80B A3B",
    "d9uiuyipu5b2": "gpt-oss-120b",
    "fkpdj71utboq": "gpt-oss-20b",
}

BEDROCK_PRICES = dict(_BASE_PRICES)
for _pid, _display in _PROFILE_DISPLAY_NAMES.items():
    if _display in _BASE_PRICES:
        _arn = (
            "arn:aws:bedrock:us-east-1:920736616554:"
            f"application-inference-profile/{_pid}"
        )
        BEDROCK_PRICES[_arn] = _BASE_PRICES[_display]

# Absolute path to the cache directory (inside agentopt/)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "agentopt", ".agentopt_cache")


def _compute_cost_from_tokens(input_tokens, output_tokens):
    """Compute cost from token dicts using Bedrock pricing."""
    return _compute_sample_cost(input_tokens, output_tokens)


def run_brute_force_from_cache(benchmark, models, limit=None):
    """Run brute force using cache.db and return LookupTable.

    Returns (model_names, datapoint_indices, table).
    """
    print(f"\n{'='*60}")
    print(f"  Running brute force from cache.db ({benchmark})")
    print(f"  Models: {len(models)}")
    print(f"{'='*60}\n")

    if benchmark == "gpqa":
        return _run_gpqa_bf(models)
    elif benchmark == "gpqa_main":
        return _run_gpqa_main_bf(models)
    elif benchmark == "bfcl":
        return _run_bfcl_bf(models, limit)
    elif benchmark == "hotpotqa":
        return _run_hotpotqa_bf(models, limit)
    elif benchmark == "mathqa":
        return _run_mathqa_bf(models, limit)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def _run_gpqa_bf(models):
    """Run GPQA brute force from cache and build lookup table."""
    from benchmarks.GPQA.eval import load_gpqa_dataset, gpqa_eval_fn, _gpqa_agent_fn_langgraph
    from agentopt.proxy.tracker import LLMTracker
    from agentopt import BruteForceModelSelector

    dataset = load_gpqa_dataset()
    print(f"  Dataset: {len(dataset)} samples")

    agent_fn = _gpqa_agent_fn_langgraph
    eval_fn = gpqa_eval_fn

    model_candidates = {"agent": list(models)}

    tracker = LLMTracker(cache=True, cache_dir=CACHE_DIR)
    tracker.start()

    selector = BruteForceModelSelector(
        agent_fn=agent_fn,
        models=model_candidates,
        eval_fn=eval_fn,
        dataset=dataset,
        model_prices=BEDROCK_PRICES,
        tracker=tracker,
    )

    t0 = time.time()
    results = selector.select_best(parallel=True, max_concurrent=20)
    elapsed = time.time() - t0
    print(f"  Brute force completed in {elapsed:.1f}s")

    tracker.stop()

    # Build lookup table from results
    return _results_to_lookup(results)


def _run_gpqa_main_bf(models):
    """Run GPQA Main brute force from cache via the raw 'direct' agent wrapper.

    The 448 GPQA Main cache entries were produced by
    ``_gpqa_agent_fn_raw(agent_type="direct")`` over OpenRouter (not by the
    LangGraph wrapper used for Diamond). Using a different agent wrapper would
    change the canonical request body and miss every cache key.
    """
    from benchmarks.GPQA.eval import load_gpqa_dataset, gpqa_eval_fn, _gpqa_agent_fn_raw
    from agentopt.proxy.tracker import LLMTracker
    from agentopt import BruteForceModelSelector

    dataset = load_gpqa_dataset(config="gpqa_main")
    print(f"  Dataset: {len(dataset)} samples (gpqa_main)")

    agent_fn = _gpqa_agent_fn_raw  # agent_type defaults to "direct"
    eval_fn = gpqa_eval_fn

    model_candidates = {"agent": list(models)}

    tracker = LLMTracker(cache=True, cache_dir=CACHE_DIR)
    tracker.start()

    selector = BruteForceModelSelector(
        agent_fn=agent_fn,
        models=model_candidates,
        eval_fn=eval_fn,
        dataset=dataset,
        model_prices=BEDROCK_PRICES,
        tracker=tracker,
    )

    t0 = time.time()
    results = selector.select_best(parallel=True, max_concurrent=20)
    elapsed = time.time() - t0
    print(f"  Brute force completed in {elapsed:.1f}s")

    tracker.stop()
    return _results_to_lookup(results)


def _run_bfcl_bf(models, limit=None):
    """Run BFCL brute force from cache and build lookup table."""
    from benchmarks.BFCL.bfcl_multi_turn import load_bfcl_dataset, bfcl_agent_fn, bfcl_eval_fn
    from agentopt.proxy.tracker import LLMTracker
    from agentopt import BruteForceModelSelector

    dataset = load_bfcl_dataset(limit=limit or 200)
    print(f"  Dataset: {len(dataset)} samples")

    agent_fn = bfcl_agent_fn
    eval_fn = bfcl_eval_fn

    model_candidates = {
        "agent": list(models)
    }

    tracker = LLMTracker(cache=True, cache_dir=CACHE_DIR)
    tracker.start()

    selector = BruteForceModelSelector(
        agent_fn=agent_fn,
        models=model_candidates,
        eval_fn=eval_fn,
        dataset=dataset,
        model_prices=BEDROCK_PRICES,
        tracker=tracker,
    )

    t0 = time.time()
    results = selector.select_best(parallel=True, max_concurrent=20)
    elapsed = time.time() - t0
    print(f"  Brute force completed in {elapsed:.1f}s")

    tracker.stop()
    return _results_to_lookup(results)


def _run_hotpotqa_bf(models, limit=None):
    """Run HotpotQA brute force from cache and build lookup table."""
    from benchmarks.HotpotQA.eval import (
        load_hotpotqa_distractor, hotpot_f1,
        _hotpotqa_agent_fn_langgraph,
    )
    from agentopt.proxy.tracker import LLMTracker
    from agentopt import BruteForceModelSelector

    dataset_path = os.path.join(
        os.path.dirname(__file__),
        "other_benchmarks/hotpot_qa/hotpot_dev_distractor_v1.json"
    )
    dataset = load_hotpotqa_distractor(dataset_path, limit=limit or 200)
    print(f"  Dataset: {len(dataset)} samples")

    agent_fn = _hotpotqa_agent_fn_langgraph
    eval_fn = hotpot_f1

    # HotpotQA is 2-tuple: planner + solver
    model_candidates = {
        "planner": list(models),
        "solver": list(models),
    }

    tracker = LLMTracker(cache=True, cache_dir=CACHE_DIR)
    tracker.start()

    selector = BruteForceModelSelector(
        agent_fn=agent_fn,
        models=model_candidates,
        eval_fn=eval_fn,
        dataset=dataset,
        model_prices=BEDROCK_PRICES,
        tracker=tracker,
    )

    t0 = time.time()
    results = selector.select_best(parallel=True, max_concurrent=20)
    elapsed = time.time() - t0
    print(f"  Brute force completed in {elapsed:.1f}s")

    tracker.stop()
    return _results_to_lookup(results)


def _run_mathqa_bf(models, limit=None):
    """Run MathQA brute force from cache and build lookup table."""
    from benchmarks.MathQA.eval import (
        load_math_qa, eval_fn as mathqa_eval_fn,
        _mathqa_agent_fn_langgraph,
    )
    from agentopt.proxy.tracker import LLMTracker
    from agentopt import BruteForceModelSelector

    # load_math_qa returns (train, test) — use train_split=1.0 to get all as train
    train_data, _ = load_math_qa(train_split=1.0, max_samples=limit or 200)
    dataset = train_data
    print(f"  Dataset: {len(dataset)} samples")

    agent_fn = _mathqa_agent_fn_langgraph
    eval_fn = mathqa_eval_fn

    # MathQA is 2-tuple: answer + critic
    model_candidates = {
        "answer": list(models),
        "critic": list(models),
    }

    tracker = LLMTracker(cache=True, cache_dir=CACHE_DIR)
    tracker.start()

    selector = BruteForceModelSelector(
        agent_fn=agent_fn,
        models=model_candidates,
        eval_fn=eval_fn,
        dataset=dataset,
        model_prices=BEDROCK_PRICES,
        tracker=tracker,
    )

    t0 = time.time()
    results = selector.select_best(parallel=True, max_concurrent=20)
    elapsed = time.time() - t0
    print(f"  Brute force completed in {elapsed:.1f}s")

    tracker.stop()
    return _results_to_lookup(results)


def _results_to_lookup(results):
    """Convert SelectionResults to (model_names, datapoint_indices, LookupTable, server_latencies).

    server_latencies: {model_name: [server_latency_ms_per_sample, ...]}
    """
    table = {}
    all_models = set()
    all_dps = set()
    server_latencies = {}  # model -> list of server_latency_ms values

    for model_result in results.results:
        model_name = model_result.model_name
        all_models.add(model_name)
        table[model_name] = {}
        server_latencies[model_name] = []

        if model_result.datapoint_results:
            for dp in model_result.datapoint_results:
                dp_idx = dp.datapoint_index
                all_dps.add(dp_idx)

                input_tokens = dp.input_tokens or {}
                output_tokens = dp.output_tokens or {}
                cost = _compute_cost_from_tokens(input_tokens, output_tokens)

                table[model_name][dp_idx] = SampleResult(
                    score=dp.score,
                    latency_seconds=dp.latency_seconds,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                )

                if dp.server_latency_ms is not None:
                    server_latencies[model_name].append(dp.server_latency_ms)

    return sorted(all_models), sorted(all_dps), table, server_latencies


def run_selectors(models, datapoints, table, n_seeds=50, base_seed=42,
                  selectors_list=None, server_latencies=None, benchmark="gpqa"):
    """Run all selectors with n_seeds and print results."""
    gt_name, gt_acc = compute_ground_truth(models, datapoints, table)

    bf_evals = sum(len(v) for v in table.values())
    bf_cost = sum(s.cost for m in table.values() for s in m.values())

    print(f"\n  Ground truth best: {gt_name}  ({gt_acc:.4f})")
    print(f"  Brute force: {bf_evals} evaluations, ${bf_cost:.4f} total cost")

    # Compute per-model server latency averages
    model_server_lat = {}
    if server_latencies:
        for model, lats in server_latencies.items():
            if lats:
                model_server_lat[model] = sum(lats) / len(lats) / 1000.0  # ms -> seconds
            else:
                model_server_lat[model] = None

    # Print brute force per-model results
    print(f"\n  {'='*90}")
    print(f"  BRUTE FORCE RESULTS (per model)")
    print(f"  {'='*90}")
    has_server_lat = bool(model_server_lat)
    if has_server_lat:
        print(f"  {'Rank':<5} {'Model':<35} {'Accuracy':>9} {'Server Lat':>11} {'Wall Lat':>10} {'Cost':>10}")
        print(f"  {'-'*5} {'-'*35} {'-'*9} {'-'*11} {'-'*10} {'-'*10}")
    else:
        print(f"  {'Rank':<5} {'Model':<35} {'Accuracy':>9} {'Latency':>10} {'Cost':>10}")
        print(f"  {'-'*5} {'-'*35} {'-'*9} {'-'*10} {'-'*10}")

    model_stats = []
    for model in models:
        samples = table.get(model, {})
        available = [samples[dp] for dp in datapoints if dp in samples]
        if not available:
            continue
        acc = sum(s.score for s in available) / len(available)
        lat = sum(s.latency_seconds for s in available) / len(available)
        cost = sum(s.cost for s in available)
        srv_lat = model_server_lat.get(model)
        model_stats.append((model, acc, lat, cost, srv_lat))

    model_stats.sort(key=lambda x: (-x[1], x[2]))
    for rank, (model, acc, lat, cost, srv_lat) in enumerate(model_stats, 1):
        marker = " <<<" if model == gt_name else ""
        if has_server_lat and srv_lat is not None:
            print(f"  {rank:<5} {model:<35} {acc:>8.2%} {srv_lat:>10.2f}s {lat:>9.2f}s ${cost:>9.4f}{marker}")
        else:
            print(f"  {rank:<5} {model:<35} {acc:>8.2%} {lat:>9.2f}s ${cost:>9.4f}{marker}")
    print()

    if selectors_list is None:
        selectors_list = [
            "random_search", "arm_elimination", "epsilon_lucb",
            "threshold_se", "hill_climbing",
        ]
        if _check_botorch():
            selectors_list.append("bayesian_optimization")
        else:
            print("\n  (Skipping bayesian_optimization — torch/botorch not installed)")
        if simulate_lm_proposal is not None:
            try:
                from openai import OpenAI
                if os.environ.get("OPENAI_API_KEY"):
                    selectors_list.append("lm_proposal")
                else:
                    print("\n  (Skipping lm_proposal — OPENAI_API_KEY not set)")
            except ImportError:
                print("\n  (Skipping lm_proposal — openai not installed)")
        if simulate_matrix_ucb is not None:
            selectors_list.append("matrix_ucb")
        if simulate_matrix_ucb_lrf is not None:
            selectors_list.append("matrix_ucb_lrf")

    selector_fns = {
        "random_search": (simulate_random_search, {"sample_fraction": 0.25}),
        "arm_elimination": (simulate_arm_elimination, {"confidence": 1.0}),
        "epsilon_lucb": (simulate_epsilon_lucb, {"epsilon": 0.01}),
        "threshold_se": (simulate_threshold_se, {"threshold": 0.75}),
        "hill_climbing": (simulate_hill_climbing, {"num_restarts": 3}),
    }
    if simulate_bayesian_optimization is not None:
        selector_fns["bayesian_optimization"] = (simulate_bayesian_optimization, {})
    if simulate_lm_proposal is not None:
        selector_fns["lm_proposal"] = (simulate_lm_proposal, {"proposer_model": "gpt-4.1", "preview_size": 10, "benchmark": benchmark})
    if simulate_matrix_ucb is not None:
        selector_fns["matrix_ucb"] = (simulate_matrix_ucb, {"a": 1.0})
    if simulate_matrix_ucb_lrf is not None:
        selector_fns["matrix_ucb_lrf"] = (simulate_matrix_ucb_lrf, {"rank": 1, "ensemble_size": 8, "warmup_percentage": 0.05, "eta": 5.0})

    all_summaries = []

    for sel_name in selectors_list:
        if sel_name not in selector_fns:
            print(f"\n  Skipping {sel_name} — not available")
            continue

        fn, kwargs = selector_fns[sel_name]

        t0 = time.time()
        results = run_multi_seed(fn, models, datapoints, table,
                                 n_seeds=n_seeds, base_seed=base_seed, **kwargs)
        elapsed = time.time() - t0

        summary = summarize_multi_seed(results, gt_name, gt_acc,
                                        models=models, datapoints=datapoints, table=table)
        print_summary(summary)
        print(f"  Simulation time:      {elapsed:.1f}s")
        all_summaries.append(summary)

    # Print comparison table
    print(f"\n\n{'='*80}")
    print(f"  COMPARISON TABLE  ({n_seeds} seeds)")
    print(f"{'='*80}")
    print(f"  {'Selector':<25} {'Found Best%':>11} {'Mean Acc':>9} {'Mean Evals':>11} {'Mean Cost':>10} {'Cost Savings':>12}")
    print(f"  {'-'*25} {'-'*11} {'-'*9} {'-'*11} {'-'*10} {'-'*12}")
    print(f"  {'brute_force (ref)':<25} {'100.0%':>11} {gt_acc:>9.4f} {bf_evals:>11} ${bf_cost:>9.4f} {'0.0%':>12}")
    print(f"  {'-'*25} {'-'*11} {'-'*9} {'-'*11} {'-'*10} {'-'*12}")

    for s in all_summaries:
        savings = (1 - s['mean_cost'] / bf_cost) * 100 if bf_cost > 0 else 0
        print(f"  {s['selector']:<25} {s['found_true_best_pct']:>10.1f}% {s['mean_accuracy']:>9.4f}"
              f" {s['mean_evaluations']:>11.0f} ${s['mean_cost']:>9.4f} {savings:>11.1f}%")

    return all_summaries


def _table_path(benchmark):
    """Return the default pickle path for a benchmark's lookup table."""
    d = os.path.join(os.path.dirname(__file__), "agentopt", "results", "cache_db_results")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{benchmark}_lookup.pkl")


def save_table(benchmark, model_names, datapoints, table, server_latencies):
    """Save lookup table to pickle for instant reload."""
    path = _table_path(benchmark)
    with open(path, "wb") as f:
        pickle.dump({
            "model_names": model_names,
            "datapoints": datapoints,
            "table": table,
            "server_latencies": server_latencies,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved lookup table to {path}")


def load_table(benchmark):
    """Load lookup table from pickle. Returns None if not found."""
    path = _table_path(benchmark)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  Loaded lookup table from {path}")
    return data["model_names"], data["datapoints"], data["table"], data["server_latencies"]


def main():
    parser = argparse.ArgumentParser(
        description="Cache-powered selector simulator — runs from cache.db, no JSONL",
    )
    parser.add_argument("--benchmark", required=True,
                        choices=["gpqa", "bfcl", "hotpotqa", "mathqa"],
                        help="Benchmark to simulate")
    parser.add_argument("--seeds", type=int, default=50, help="Number of random seeds (default 50)")
    parser.add_argument("--base-seed", type=int, default=42, help="Starting seed (default 42)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--selectors", default="all",
                        help="Comma-separated selector names, or 'all'")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild lookup table from cache.db (ignores saved pickle)")

    args = parser.parse_args()

    models = list(DEFAULT_MODELS)  # 9 models (no DeepSeek)

    # Step 1: Try loading saved lookup table, otherwise build from cache.db
    loaded = None
    if not args.rebuild:
        loaded = load_table(args.benchmark)

    if loaded is not None:
        model_names, datapoints, table, server_latencies = loaded
    else:
        model_names, datapoints, table, server_latencies = run_brute_force_from_cache(
            args.benchmark, models, limit=args.limit
        )
        # Auto-save for next time
        save_table(args.benchmark, model_names, datapoints, table, server_latencies)

    print(f"\n  Lookup table:")
    print(f"    Models: {len(model_names)}")
    print(f"    Datapoints: {len(datapoints)}")
    print(f"    Total entries: {sum(len(v) for v in table.values())}")

    # Step 2: Run selectors
    if args.selectors == "all":
        sel_list = None  # uses default
    else:
        sel_list = [s.strip() for s in args.selectors.split(",")]

    run_selectors(model_names, datapoints, table,
                  n_seeds=args.seeds, base_seed=args.base_seed,
                  selectors_list=sel_list,
                  server_latencies=server_latencies,
                  benchmark=args.benchmark)


if __name__ == "__main__":
    main()
