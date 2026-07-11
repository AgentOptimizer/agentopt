#!/usr/bin/env python3
"""Run ALL selectors at lambda_cost=0.1, lambda_latency=0.1 across all 4 benchmarks, 50 seeds."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from offline_selector_sim_v3 import (
    load_pickle, set_norm_stats, compute_ground_truth,
    simulate_brute_force, simulate_random_search, simulate_arm_elimination,
    simulate_hill_climbing, simulate_matrix_ucb, simulate_epsilon_lucb,
    simulate_threshold_se,
)

try:
    from offline_selector_sim_v3 import simulate_bayesian_optimization
except:
    simulate_bayesian_optimization = None

PICKLE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cache_db_results")
BENCHMARKS = ["gpqa", "bfcl", "hotpotqa", "mathqa"]
LAMBDA_COST = 0.1
LAMBDA_LATENCY = 0.1
N_SEEDS = 50

SELECTORS = [
    ("Hill Climbing", simulate_hill_climbing, {"num_restarts": 3, "max_iterations": 20, "patience": 3}),
    ("Arm Elimination", simulate_arm_elimination, {"confidence": 1.0}),
    ("Random Search", simulate_random_search, {"sample_fraction": 0.25}),
    ("Matrix UCB-E (β=0.2)", simulate_matrix_ucb, {"a": 1.0, "observation_budget_fraction": 0.2}),
    ("Matrix UCB-E (β=0.5)", simulate_matrix_ucb, {"a": 1.0, "observation_budget_fraction": 0.5}),
]

if simulate_bayesian_optimization:
    SELECTORS.append(("Bayesian Opt", simulate_bayesian_optimization, {}))

results = {}

for bench in BENCHMARKS:
    print(f"\n{'='*60}")
    print(f"  {bench.upper()}")
    print(f"{'='*60}")
    
    pkl_path = os.path.join(PICKLE_DIR, f"{bench}_lookup.pkl")
    models, datapoints, table = load_pickle(pkl_path)
    set_norm_stats(table, datapoints)
    
    gt_name, gt_acc, gt_obj = compute_ground_truth(models, datapoints, table, LAMBDA_COST, LAMBDA_LATENCY)
    print(f"  Ground truth: {gt_name}")
    print(f"    acc={gt_acc:.4f}, J={gt_obj:.4f}")
    
    # Brute force reference
    bf = simulate_brute_force(models, datapoints, table, LAMBDA_COST, LAMBDA_LATENCY)
    bf_evals = bf.total_evaluations
    results[(bench, "Brute Force")] = {
        "acc": bf.best_accuracy, "obj": bf.best_objective, 
        "savings": 0.0, "found_best": 100.0
    }
    print(f"  Brute force: acc={bf.best_accuracy:.4f}, J={bf.best_objective:.4f}, evals={bf_evals}")
    
    for sel_name, sel_fn, sel_kwargs in SELECTORS:
        accs, objs, evals_list, found = [], [], [], 0
        for seed in range(N_SEEDS):
            r = sel_fn(models, datapoints, table, LAMBDA_COST, LAMBDA_LATENCY, seed=seed, **sel_kwargs)
            accs.append(r.best_accuracy)
            objs.append(r.best_objective)
            evals_list.append(r.total_evaluations)
            if r.found_true_best:
                found += 1
        
        mean_acc = sum(accs) / len(accs)
        mean_obj = sum(objs) / len(objs)
        mean_evals = sum(evals_list) / len(evals_list)
        savings = (1 - mean_evals / bf_evals) * 100
        found_pct = found / N_SEEDS * 100
        
        results[(bench, sel_name)] = {
            "acc": mean_acc, "obj": mean_obj, 
            "savings": savings, "found_best": found_pct
        }
        print(f"  {sel_name}: acc={mean_acc:.4f}, J={mean_obj:.4f}, savings={savings:.1f}%, found_best={found_pct:.0f}%")

# Print summary table
print(f"\n\n{'='*80}")
print(f"SUMMARY TABLE (λ_cost={LAMBDA_COST}, λ_latency={LAMBDA_LATENCY}, {N_SEEDS} seeds)")
print(f"{'='*80}")
print(f"{'Benchmark':<12} {'Selector':<22} {'Accuracy':>10} {'J(c)':>10} {'Savings':>10} {'Found%':>8}")
print("-" * 80)
for bench in BENCHMARKS:
    for sel_name in ["Brute Force"] + [s[0] for s in SELECTORS]:
        key = (bench, sel_name)
        if key in results:
            r = results[key]
            print(f"{bench:<12} {sel_name:<22} {r['acc']:>10.4f} {r['obj']:>10.4f} {r['savings']:>9.1f}% {r['found_best']:>7.0f}%")
    print()
