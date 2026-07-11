# Benchmark Experiments

Benchmark evaluation and model selection experiments for 9 LLM configurations across 4 benchmarks.

## Quick Start: Offline Selector Simulation

All brute force results are pre-computed in pickle lookup tables. Run selector simulations with zero API calls:

```bash
# From experiments/ directory

# Pure accuracy selectors (v2)
python offline_selector_sim_v2.py \
    --pickle results/cache_db_results/gpqa_lookup.pkl \
    --selectors all --seeds 50

# Combined objective selectors (v3)
python combined_objective/offline_selector_sim_v3.py \
    --pickle results/cache_db_results/gpqa_lookup.pkl \
    --selectors all --seeds 50 \
    --lambda-cost 0.1 --lambda-latency 0.1

# Run all benchmarks at once
cd combined_objective && python run_all_selectors.py
```

## Structure

```
experiments/
├── benchmarks/              # Benchmark implementations
│   ├── common.py            # Shared utilities (make_llm, CLI args, extract_text_content)
│   ├── GPQA/                # Graduate-level QA (198 diamond + 448 main samples)
│   ├── BFCL/                # Multi-turn function calling (200 samples)
│   ├── HotpotQA/            # Multi-hop QA (200 samples from 7405)
│   └── MathQA/              # Self-reflective math (200 samples)
├── results/
│   └── cache_db_results/    # Pickle lookup tables (frozen brute force results)
│       ├── gpqa_lookup.pkl      # 9 models x 198 samples
│       ├── bfcl_lookup.pkl      # 9 models x 200 samples
│       ├── hotpotqa_lookup.pkl  # 81 combos x 200 samples
│       └── mathqa_lookup.pkl    # 81 combos x 200 samples
├── cached_results/          # Pre-computed CSVs and LaTeX tables
│   ├── gpqa/                # Brute force + selector results
│   ├── bfcl/
│   ├── hotpotqa/
│   ├── mathqa/
│   └── multiobjective/      # LaTeX tables for J(c) analysis
├── combined_objective/      # Combined objective analysis
│   ├── offline_selector_sim_v3.py   # Main simulator (J = acc - λ*cost - λ*lat)
│   ├── run_all_selectors.py         # Run all selectors across all benchmarks
│   └── print_results.py             # Pretty-print results
├── offline_selector_sim_v2.py       # Pure accuracy simulator
├── cache_selector_sim.py            # Build lookup tables from cache.db
└── aggregate_results.py             # JSONL aggregation utility
```

## Benchmarks

| Benchmark | Samples | Combos | Architecture | Best Config |
|-----------|---------|--------|-------------|-------------|
| GPQA | 198 | 9 (1-tuple) | Direct QA | Claude Opus 4.6 (74.75%) |
| BFCL | 200 | 9 (1-tuple) | Multi-turn FC | 3-way tie at 70% |
| HotpotQA | 200 | 81 (2-tuple) | Planner + Solver | Mini 8B + Opus (74.27%) |
| MathQA | 200 | 81 (2-tuple) | Answer + Critic | Opus + Haiku 4.5 (98.84%) |

## Combined Objective

```
J(c) = accuracy - λ_cost * NormCost - λ_latency * NormLatency
```

Min-max normalization computed per-benchmark across all samples and models. At λ=0, reduces to pure accuracy.

Key finding: **Matrix UCB-E (β=0.5)** matches brute force on 3/4 benchmarks with 50% budget savings.

## 9 Models

Claude 3 Haiku, Claude Haiku 4.5, Claude Opus 4.6, gpt-oss-20b, gpt-oss-120b, Kimi K2.5, Ministral 3 8B, Qwen3 32B, Qwen3 Next 80B A3B

## Dependencies

```bash
pip install numpy botorch gpytorch  # For Bayesian Optimization selector
pip install datasets                 # For MathQA data loading
```

Pickle lookup tables only require Python stdlib + numpy.
