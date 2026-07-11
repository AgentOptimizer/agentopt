#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from offline_selector_sim_v3 import (
    load_pickle, set_norm_stats, compute_ground_truth,
    simulate_brute_force, simulate_random_search, simulate_arm_elimination,
    simulate_hill_climbing, simulate_matrix_ucb, simulate_bayesian_optimization,
)

PICKLE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cache_db_results")
BENCHMARKS = [
    ("gpqa", "GPQA"),
    ("bfcl", "BFCL"),
    ("hotpotqa", "HotpotQA"),
    ("mathqa", "MathQA"),
]
LC, LL = 0.1, 0.1
N = 50

SELECTORS = [
    ("Brute Force", simulate_brute_force, {}),
    ("Hill Climbing", simulate_hill_climbing, {"num_restarts": 3, "max_iterations": 20, "patience": 3}),
    ("Arm Elimination", simulate_arm_elimination, {"confidence": 1.0}),
    ("Random Search", simulate_random_search, {"sample_fraction": 0.25}),
    ("Matrix UCB-E (β=0.2)", simulate_matrix_ucb, {"a": 1.0, "observation_budget_fraction": 0.2}),
    ("Matrix UCB-E (β=0.5)", simulate_matrix_ucb, {"a": 1.0, "observation_budget_fraction": 0.5}),
    ("Bayesian Opt", simulate_bayesian_optimization, {}),
]

for bench_key, bench_label in BENCHMARKS:
    pkl = os.path.join(PICKLE_DIR, f"{bench_key}_lookup.pkl")
    models, dps, table = load_pickle(pkl)
    set_norm_stats(table, dps)
    gt_name, gt_acc, gt_obj = compute_ground_truth(models, dps, table, LC, LL)
    
    bf = simulate_brute_force(models, dps, table, LC, LL)
    bf_evals = bf.total_evaluations

    rows = []
    for name, fn, kw in SELECTORS:
        accs, objs, evals_l, found = [], [], [], 0
        for seed in range(N):
            r = fn(models, dps, table, LC, LL, seed=seed, **kw)
            accs.append(r.best_accuracy)
            objs.append(r.best_objective)
            evals_l.append(r.total_evaluations)
            if r.found_true_best:
                found += 1
        ma = sum(accs)/N
        mj = sum(objs)/N
        me = sum(evals_l)/N
        sav = (1 - me/bf_evals)*100
        fp = found/N*100
        rows.append((name, ma, mj, sav, fp))
    
    rows.sort(key=lambda x: x[1], reverse=True)
    
    print(f"  {bench_label}  (J-optimal: {gt_name}, acc={gt_acc:.4f}, J={gt_obj:.4f})")
    print(f"  {'─'*82}")
    print(f"  {'Selector':<24} {'Accuracy':>10} {'J(c)':>10} {'Savings':>10} {'Found Best':>12}")
    print(f"  {'─'*82}")
    for name, ma, mj, sav, fp in rows:
        print(f"  {name:<24} {ma:>10.4f} {mj:>10.4f} {sav:>9.1f}% {fp:>10.0f}%")
    print()
