# Benchmark Results

We evaluated **9 models on AWS Bedrock** across **4 benchmarks** using LangGraph-based agents, then ran 8 model selection algorithms to measure how efficiently each finds the best model without exhaustive search. All results use 200 samples per benchmark with brute force ground truth.

!!! info "Reproducibility"
    Raw per-sample JSONL data, aggregated results, and summary CSVs are available in the [`results/`](https://github.com/AgentOptimizer/agentopt/tree/final-benchmark-results/results) directory on the `final-benchmark-results` branch.

## Models

All models accessed via AWS Bedrock Application Inference Profiles (on-demand pricing, March 2026).

| Model | Provider | Input $/MTok | Output $/MTok |
|:------|:---------|:-------------|:--------------|
| Claude Opus 4.6 | Anthropic | $5.00 | $25.00 |
| Claude Haiku 4.5 | Anthropic | $1.00 | $5.00 |
| Claude 3 Haiku | Anthropic | $0.25 | $1.25 |
| gpt-oss-120b | OpenAI | $0.15 | $0.60 |
| gpt-oss-20b | OpenAI | $0.07 | $0.30 |
| Kimi K2.5 | Moonshot AI | $0.60 | $3.00 |
| Qwen3 Next 80B A3B | Qwen | $0.15 | $1.20 |
| Qwen3 32B | Qwen | $0.15 | $0.60 |
| Ministral 3 8B | Mistral | $0.15 | $0.15 |

## Cross-Benchmark Summary

| Benchmark | Tuple | Samples | Combos | Best Combo | Accuracy | BF Cost | Arm Elim Savings |
|:----------|:------|:--------|:-------|:-----------|:---------|:--------|:-----------------|
| GPQA Diamond | 1-tuple | 198 | 9 | Claude Opus 4.6 | **80.30%** | $4.13 | 49% |
| BFCL Multi-Turn | 1-tuple | 200 | 9 | Claude Opus 4.6 | **72.00%** | $85.42 | 11% |
| HotpotQA | 2-tuple | 199 | 81 | Ministral + Opus | **74.78%** | $51.48 | 64% |
| MathQA | 2-tuple | 200 | 81 | Opus + Qwen3 Next | **98.83%** | $113.01 | 46% |

---

## GPQA Diamond

**Graduate-level science QA** — 198 multiple-choice questions from the GPQA Diamond dataset. Single-agent architecture: one LLM answers directly.

### Model Results

| Rank | Model | Accuracy | Avg Latency (s) | Cost |
|:-----|:------|:---------|:-----------------|:-----|
| 1 | Claude Opus 4.6 | **80.30%** | 86.4 | $2.43 |
| 2 | Kimi K2.5 | 72.02% | 97.2 | $0.72 |
| 3 | gpt-oss-120b | 68.02% | 88.5 | $0.19 |
| 4 | Claude Haiku 4.5 | 60.51% | 83.1 | $0.51 |
| 5 | gpt-oss-20b | 52.02% | 85.7 | $0.13 |
| 6 | Qwen3 Next 80B A3B | 51.04% | 90.0 | $0.06 |
| 7 | Qwen3 32B | 46.67% | 88.3 | $0.04 |
| 8 | Claude 3 Haiku | 37.31% | 80.1 | $0.06 |
| 9 | Ministral 3 8B | 36.87% | 84.7 | $0.01 |

### Selector Comparison (50 seeds)

| Selector | Find Rate | Mean Accuracy | Evaluations | Cost | Savings |
|:---------|:----------|:--------------|:------------|:-----|:--------|
| Brute Force | 100% | 80.30% | 1,759 | $4.13 | -- |
| LM Proposal | 100% | 80.30% | 198 | $2.43 | 41% |
| Arm Elimination | 98% | 80.14% | 444 | $2.10 | **49%** |
| Hill Climbing (3) | 92% | 79.64% | 1,552 | $3.75 | 9% |
| Epsilon LUCB | 90% | 79.47% | 361 | $2.32 | 44% |
| Hill Climbing (1) | 80% | 78.65% | 884 | $2.78 | 33% |
| Bayesian Opt | 56% | 75.41% | 976 | $2.32 | 44% |
| Random Search | 36% | 70.53% | 587 | $1.53 | 63% |
| Threshold SE | 36% | 65.88% | 294 | $0.26 | 94% |

---

## BFCL Multi-Turn

**Multi-turn function calling** — 200 samples from the Berkeley Function Calling Leaderboard (BFCL v3). Each sample has multiple turns with tool-calling loops. Models that don't support native function calling (Qwen3 32B, Kimi K2.5, Ministral 3 8B) use a text-based prompting fallback.

!!! note "Comparison with official BFCL leaderboard"
    Our evaluation uses a **live LangGraph agent** that executes tool calls against real backend state machines, whereas the official BFCL leaderboard uses static response matching. Our accuracy numbers are not directly comparable to the leaderboard — they reflect end-to-end agent performance including tool execution, state management, and multi-step reasoning.

### Model Results

| Rank | Model | Accuracy | Avg Latency (s) | Cost |
|:-----|:------|:---------|:-----------------|:-----|
| 1 | Claude Opus 4.6 | **72.00%** | 222.8 | $60.78 |
| 2 | Qwen3 Next 80B A3B | 71.00% | 226.3 | $1.87 |
| 3 | Kimi K2.5 | 68.50% | 228.2 | $3.86 |
| 4 | Claude Haiku 4.5 | 65.00% | 208.7 | $11.97 |
| 5 | gpt-oss-120b | 61.00% | 208.9 | $1.13 |
| 6 | Qwen3 32B | 50.00% | 211.3 | $0.97 |
| 7 | Claude 3 Haiku | 45.00% | 205.2 | $3.42 |
| 8 | gpt-oss-20b | 39.00% | 204.2 | $0.44 |
| 9 | Ministral 3 8B | 33.50% | 213.8 | $0.98 |

### Selector Comparison (50 seeds)

| Selector | Find Rate | Mean Accuracy | Evaluations | Cost | Savings |
|:---------|:----------|:--------------|:------------|:-----|:--------|
| Brute Force | 100% | 72.00% | 1,800 | $85.42 | -- |
| Arm Elimination | 100% | 72.00% | 922 | $76.33 | **11%** |
| Hill Climbing (3) | 94% | 71.94% | 1,652 | $80.38 | 6% |
| Epsilon LUCB | 60% | 71.33% | 407 | $42.60 | 50% |
| Bayesian Opt | 56% | 70.61% | 1,000 | $50.98 | 40% |
| Hill Climbing (1) | 24% | 70.89% | 808 | $27.49 | 68% |
| Random Search | 36% | 67.99% | 600 | $31.62 | 63% |
| Threshold SE | 12% | 57.52% | 285 | $6.45 | 92% |
| LM Proposal | 0% | 45.00% | 200 | $3.42 | 96% |

---

## HotpotQA

**Multi-hop question answering** — 199 samples from the HotpotQA distractor setting. Two-agent architecture: a **planner** proposes search steps, and a **solver** executes them with tool access. 81 model combinations (9 planners x 9 solvers).

### Top 15 Combos

| Rank | Planner | Solver | Accuracy | Cost |
|:-----|:--------|:-------|:---------|:-----|
| 1 | Ministral 3 8B | Claude Opus 4.6 | **74.78%** | $2.71 |
| 2 | Qwen3 32B | Claude Opus 4.6 | 72.97% | $2.67 |
| 3 | Claude 3 Haiku | Claude Opus 4.6 | 72.58% | $2.67 |
| 4 | Qwen3 Next 80B A3B | Claude Opus 4.6 | 72.32% | $2.66 |
| 5 | Qwen3 Next 80B A3B | gpt-oss-120b | 71.29% | $0.13 |
| 6 | Kimi K2.5 | Claude Opus 4.6 | 71.09% | $2.53 |
| 7 | Qwen3 32B | gpt-oss-120b | 70.47% | $0.12 |
| 8 | Qwen3 32B | Qwen3 Next 80B A3B | 69.12% | $0.11 |
| 9 | Claude 3 Haiku | Qwen3 Next 80B A3B | 68.60% | $0.15 |
| 10 | Ministral 3 8B | gpt-oss-120b | 68.56% | $0.12 |
| 11 | Claude 3 Haiku | gpt-oss-120b | 68.44% | $0.16 |
| 12 | Kimi K2.5 | Qwen3 Next 80B A3B | 68.41% | $0.28 |
| 13 | Qwen3 Next 80B A3B | Qwen3 Next 80B A3B | 68.29% | $0.12 |
| 14 | Qwen3 32B | gpt-oss-20b | 67.75% | $0.09 |
| 15 | Ministral 3 8B | Qwen3 Next 80B A3B | 67.46% | $0.11 |

### Bottom 5 Combos

| Rank | Planner | Solver | Accuracy | Cost |
|:-----|:--------|:-------|:---------|:-----|
| 77 | Claude Opus 4.6 | gpt-oss-20b | 31.31% | $2.01 |
| 78 | Qwen3 Next 80B A3B | Claude Haiku 4.5 | 30.85% | $0.71 |
| 79 | Claude Opus 4.6 | Claude Haiku 4.5 | 30.81% | $2.01 |
| 80 | Claude Haiku 4.5 | Claude Haiku 4.5 | 26.57% | $0.79 |
| 81 | Qwen3 32B | Claude Haiku 4.5 | 25.11% | $0.72 |

!!! warning "Capability as Liability"
    **Claude Opus 4.6 as planner achieves only ~32% accuracy** regardless of solver — the worst planner in the benchmark. Opus is "too smart" for the planner role: it calls `terminate()` and answers directly instead of delegating to the solver. The solver is never invoked. Meanwhile, the cheapest model (Ministral 3 8B) as planner with Opus as solver achieves the **best accuracy at 74.78%**. This demonstrates that stronger models can underperform in multi-agent architectures when the role requires delegation, not direct answering.

??? note "Full 81 Combo Results"

    | Rank | Planner | Solver | Accuracy | Cost | Note |
    |:-----|:--------|:-------|:---------|:-----|:-----|
    | 1 | Ministral 3 8B | Claude Opus 4.6 | 74.78% | $2.71 | |
    | 2 | Qwen3 32B | Claude Opus 4.6 | 72.97% | $2.67 | |
    | 3 | Claude 3 Haiku | Claude Opus 4.6 | 72.58% | $2.67 | |
    | 4 | Qwen3 Next 80B A3B | Claude Opus 4.6 | 72.32% | $2.66 | |
    | 5 | Qwen3 Next 80B A3B | gpt-oss-120b | 71.29% | $0.13 | |
    | 6 | Kimi K2.5 | Claude Opus 4.6 | 71.09% | $2.53 | |
    | 7 | Qwen3 32B | gpt-oss-120b | 70.47% | $0.12 | |
    | 8 | Qwen3 32B | Qwen3 Next 80B A3B | 69.12% | $0.11 | |
    | 9 | Claude 3 Haiku | Qwen3 Next 80B A3B | 68.60% | $0.15 | |
    | 10 | Ministral 3 8B | gpt-oss-120b | 68.56% | $0.12 | |
    | 11 | Claude 3 Haiku | gpt-oss-120b | 68.44% | $0.16 | |
    | 12 | Kimi K2.5 | Qwen3 Next 80B A3B | 68.41% | $0.28 | |
    | 13 | Qwen3 Next 80B A3B | Qwen3 Next 80B A3B | 68.29% | $0.12 | |
    | 14 | Qwen3 32B | gpt-oss-20b | 67.75% | $0.09 | |
    | 15 | Ministral 3 8B | Qwen3 Next 80B A3B | 67.46% | $0.11 | |
    | 16 | Kimi K2.5 | gpt-oss-120b | 67.35% | $0.28 | |
    | 17 | Ministral 3 8B | gpt-oss-20b | 66.99% | $0.07 | |
    | 18 | gpt-oss-120b | gpt-oss-120b | 66.84% | $0.10 | |
    | 19 | Kimi K2.5 | Claude 3 Haiku | 66.72% | $0.32 | |
    | 20 | gpt-oss-120b | Claude Opus 4.6 | 66.52% | $2.72 | |
    | 21 | gpt-oss-120b | Qwen3 Next 80B A3B | 66.38% | $0.09 | |
    | 22 | Claude 3 Haiku | gpt-oss-20b | 66.36% | $0.13 | |
    | 23 | Claude 3 Haiku | Ministral 3 8B | 65.79% | $0.10 | |
    | 24 | Kimi K2.5 | gpt-oss-20b | 65.37% | $0.27 | |
    | 25 | Claude 3 Haiku | Claude 3 Haiku | 65.13% | $0.20 | |
    | 26 | Ministral 3 8B | Claude 3 Haiku | 64.39% | $0.08 | |
    | 27 | Claude 3 Haiku | Kimi K2.5 | 64.17% | $0.28 | |
    | 28 | gpt-oss-120b | gpt-oss-20b | 63.80% | $0.05 | |
    | 29 | Claude 3 Haiku | Qwen3 32B | 63.75% | $0.10 | |
    | 30 | gpt-oss-20b | Claude Opus 4.6 | 63.30% | $2.53 | |
    | 31 | Ministral 3 8B | Ministral 3 8B | 62.20% | $0.09 | |
    | 32 | Ministral 3 8B | Kimi K2.5 | 62.14% | $0.22 | |
    | 33 | Kimi K2.5 | Kimi K2.5 | 61.96% | $0.45 | |
    | 34 | gpt-oss-120b | Claude 3 Haiku | 61.84% | $0.10 | |
    | 35 | gpt-oss-120b | Ministral 3 8B | 61.74% | $0.07 | |
    | 36 | Qwen3 32B | Claude 3 Haiku | 61.73% | $0.08 | |
    | 37 | gpt-oss-120b | Claude Haiku 4.5 | 61.53% | $0.61 | |
    | 38 | Qwen3 32B | Ministral 3 8B | 61.48% | $0.05 | |
    | 39 | Kimi K2.5 | Ministral 3 8B | 61.32% | $0.27 | |
    | 40 | gpt-oss-120b | Kimi K2.5 | 61.30% | $0.18 | |
    | 41 | Qwen3 Next 80B A3B | gpt-oss-20b | 61.04% | $0.06 | |
    | 42 | gpt-oss-20b | gpt-oss-120b | 60.43% | $0.07 | |
    | 43 | Ministral 3 8B | Qwen3 32B | 60.26% | $0.05 | |
    | 44 | gpt-oss-120b | Qwen3 32B | 60.17% | $0.07 | |
    | 45 | gpt-oss-20b | Qwen3 Next 80B A3B | 59.95% | $0.04 | |
    | 46 | Qwen3 Next 80B A3B | Claude 3 Haiku | 59.68% | $0.10 | |
    | 47 | Qwen3 32B | Kimi K2.5 | 58.30% | $0.22 | |
    | 48 | Qwen3 Next 80B A3B | Ministral 3 8B | 57.26% | $0.05 | |
    | 49 | Qwen3 32B | Qwen3 32B | 54.21% | $0.11 | |
    | 50 | Claude 3 Haiku | Claude Haiku 4.5 | 54.13% | $0.72 | |
    | 51 | Qwen3 Next 80B A3B | Kimi K2.5 | 53.38% | $0.24 | |
    | 52 | gpt-oss-20b | gpt-oss-20b | 52.69% | $0.05 | |
    | 53 | gpt-oss-20b | Claude 3 Haiku | 52.60% | $0.08 | |
    | 54 | Qwen3 Next 80B A3B | Qwen3 32B | 52.56% | $0.08 | |
    | 55 | gpt-oss-20b | Kimi K2.5 | 52.19% | $0.18 | |
    | 56 | gpt-oss-20b | Ministral 3 8B | 50.42% | $0.04 | |
    | 57 | gpt-oss-20b | Qwen3 32B | 49.90% | $0.04 | |
    | 58 | Ministral 3 8B | Claude Haiku 4.5 | 49.54% | $0.65 | |
    | 59 | Kimi K2.5 | Claude Haiku 4.5 | 47.97% | $0.88 | |
    | 60 | Qwen3 Next 80B A3B | Claude Haiku 4.5 | 44.81% | $0.67 | |
    | 61 | gpt-oss-20b | Claude Haiku 4.5 | 42.54% | $0.52 | |
    | 62 | Claude Haiku 4.5 | Claude Opus 4.6 | 42.39% | $2.50 | |
    | 63 | Claude Haiku 4.5 | gpt-oss-120b | 40.75% | $0.61 | |
    | 64 | Claude Haiku 4.5 | gpt-oss-20b | 38.95% | $0.50 | |
    | 65 | Claude Haiku 4.5 | Qwen3 Next 80B A3B | 38.72% | $0.56 | |
    | 66 | Claude Haiku 4.5 | Ministral 3 8B | 36.51% | $0.49 | |
    | 67 | Claude Haiku 4.5 | Kimi K2.5 | 36.43% | $0.69 | |
    | 68 | Claude Haiku 4.5 | Qwen3 32B | 35.09% | $0.54 | |
    | 69 | Claude Haiku 4.5 | Claude 3 Haiku | 34.98% | $0.51 | |
    | 70 | Claude Opus 4.6 | Claude Opus 4.6 | 32.70% | $2.00 | role2 never called |
    | 71 | Claude Opus 4.6 | Kimi K2.5 | 32.44% | $2.01 | role2 never called |
    | 72 | Claude Opus 4.6 | Qwen3 Next 80B A3B | 32.05% | $2.01 | role2 never called |
    | 73 | Claude Opus 4.6 | gpt-oss-120b | 32.00% | $2.01 | role2 never called |
    | 74 | Claude Opus 4.6 | Ministral 3 8B | 31.80% | $2.01 | role2 never called |
    | 75 | Claude Opus 4.6 | Claude 3 Haiku | 31.80% | $2.01 | role2 never called |
    | 76 | Claude Opus 4.6 | Qwen3 32B | 31.52% | $2.01 | role2 never called |
    | 77 | Claude Opus 4.6 | gpt-oss-20b | 31.31% | $2.01 | role2 never called |
    | 78 | Qwen3 Next 80B A3B | Claude Haiku 4.5 | 30.85% | $0.71 | |
    | 79 | Claude Opus 4.6 | Claude Haiku 4.5 | 30.81% | $2.01 | role2 never called |
    | 80 | Claude Haiku 4.5 | Claude Haiku 4.5 | 26.57% | $0.79 | |
    | 81 | Qwen3 32B | Claude Haiku 4.5 | 25.11% | $0.72 | |

### Selector Comparison (50 seeds)

| Selector | Find Rate | Mean Accuracy | Evaluations | Cost | Savings |
|:---------|:----------|:--------------|:------------|:-----|:--------|
| Brute Force | 100% | 74.78% | 16,108 | $51.48 | -- |
| Arm Elimination | 90% | 74.12% | 4,654 | $18.49 | **64%** |
| Hill Climbing (3) | 44% | 73.38% | 5,031 | $19.21 | 63% |
| Random Search | 30% | 72.34% | 4,176 | $13.26 | 74% |
| Hill Climbing (1) | 24% | 68.54% | 1,881 | $8.35 | 84% |
| Epsilon LUCB | 14% | 69.96% | 477 | $1.86 | 96% |
| Bayesian Opt | 8% | 72.78% | 3,979 | $12.13 | 76% |
| Threshold SE | 2% | 63.62% | 1,926 | $3.50 | 93% |
| LM Proposal | 0% | 34.41% | 199 | $1.86 | 96% |

---

## MathQA

**Self-reflective math reasoning** — 200 samples from the MathQA dataset. Two-agent architecture: an **answer model** solves problems, and a **critic model** checks the work. If the critic rejects, the answer model retries (up to 3 iterations). 81 model combinations (9 answer models x 9 critics).

### Top 15 Combos

| Rank | Answer Model | Critic Model | Accuracy | Cost |
|:-----|:-------------|:-------------|:---------|:-----|
| 1 | Claude Opus 4.6 | Qwen3 Next 80B A3B | **98.83%** | $5.89 |
| 2 | Claude Opus 4.6 | Ministral 3 8B | 98.73% | $5.31 |
| 3 | Claude Opus 4.6 | Claude Haiku 4.5 | 98.27% | $6.09 |
| 4 | Claude Opus 4.6 | Qwen3 32B | 97.79% | $6.42 |
| 5 | Claude Opus 4.6 | Claude Opus 4.6 | 97.77% | $6.95 |
| 6 | Claude Opus 4.6 | gpt-oss-120b | 97.73% | $6.14 |
| 7 | Claude Opus 4.6 | Claude 3 Haiku | 97.26% | $5.26 |
| 8 | Claude Opus 4.6 | Kimi K2.5 | 97.25% | $6.66 |
| 9 | Claude Opus 4.6 | gpt-oss-20b | 97.13% | $6.10 |
| 10 | Claude Haiku 4.5 | Ministral 3 8B | 94.47% | $2.59 |
| 11 | Claude Haiku 4.5 | Claude Haiku 4.5 | 94.00% | $3.17 |
| 12 | Claude Haiku 4.5 | Claude Opus 4.6 | 94.00% | $3.89 |
| 13 | Claude Haiku 4.5 | Qwen3 Next 80B A3B | 94.00% | $2.50 |
| 14 | Ministral 3 8B | Claude 3 Haiku | 93.98% | $0.05 |
| 15 | Claude Haiku 4.5 | Kimi K2.5 | 93.97% | $2.92 |

### Bottom 5 Combos

| Rank | Answer Model | Critic Model | Accuracy | Cost |
|:-----|:-------------|:-------------|:---------|:-----|
| 77 | Claude 3 Haiku | gpt-oss-20b | 72.96% | $0.31 |
| 78 | Kimi K2.5 | Qwen3 32B | 72.77% | $0.67 |
| 79 | Claude 3 Haiku | Qwen3 Next 80B A3B | 68.94% | $0.36 |
| 80 | Claude 3 Haiku | Qwen3 32B | 63.86% | $0.27 |
| 81 | Claude 3 Haiku | Claude 3 Haiku | 59.88% | $0.30 |

??? note "Full 81 Combo Results"

    | Rank | Answer Model | Critic Model | Accuracy | Cost | Note |
    |:-----|:-------------|:-------------|:---------|:-----|:-----|
    | 1 | Claude Opus 4.6 | Qwen3 Next 80B A3B | 98.83% | $5.89 | |
    | 2 | Claude Opus 4.6 | Ministral 3 8B | 98.73% | $5.31 | |
    | 3 | Claude Opus 4.6 | Claude Haiku 4.5 | 98.27% | $6.09 | |
    | 4 | Claude Opus 4.6 | Qwen3 32B | 97.79% | $6.42 | |
    | 5 | Claude Opus 4.6 | Claude Opus 4.6 | 97.77% | $6.95 | |
    | 6 | Claude Opus 4.6 | gpt-oss-120b | 97.73% | $6.14 | |
    | 7 | Claude Opus 4.6 | Claude 3 Haiku | 97.26% | $5.26 | |
    | 8 | Claude Opus 4.6 | Kimi K2.5 | 97.25% | $6.66 | |
    | 9 | Claude Opus 4.6 | gpt-oss-20b | 97.13% | $6.10 | |
    | 10 | Claude Haiku 4.5 | Ministral 3 8B | 94.47% | $2.59 | |
    | 11 | Claude Haiku 4.5 | Claude Haiku 4.5 | 94.00% | $3.17 | |
    | 12 | Claude Haiku 4.5 | Claude Opus 4.6 | 94.00% | $3.89 | |
    | 13 | Claude Haiku 4.5 | Qwen3 Next 80B A3B | 94.00% | $2.50 | |
    | 14 | Ministral 3 8B | Claude 3 Haiku | 93.98% | $0.05 | |
    | 15 | Claude Haiku 4.5 | Kimi K2.5 | 93.97% | $2.92 | |
    | 16 | Claude Haiku 4.5 | gpt-oss-120b | 93.93% | $2.63 | |
    | 17 | Claude Haiku 4.5 | Qwen3 32B | 93.93% | $2.66 | |
    | 18 | Claude Haiku 4.5 | gpt-oss-20b | 93.40% | $2.59 | |
    | 19 | Claude Haiku 4.5 | Claude 3 Haiku | 93.27% | $2.48 | |
    | 20 | Qwen3 Next 80B A3B | Qwen3 Next 80B A3B | 92.27% | $0.12 | |
    | 21 | Ministral 3 8B | Ministral 3 8B | 91.73% | $0.03 | |
    | 22 | Qwen3 Next 80B A3B | Ministral 3 8B | 91.47% | $0.06 | |
    | 23 | gpt-oss-120b | Claude 3 Haiku | 91.40% | $0.12 | |
    | 24 | Qwen3 Next 80B A3B | gpt-oss-120b | 90.93% | $0.07 | |
    | 25 | Qwen3 Next 80B A3B | Claude 3 Haiku | 90.60% | $0.06 | |
    | 26 | Ministral 3 8B | gpt-oss-120b | 90.40% | $0.04 | |
    | 27 | Qwen3 Next 80B A3B | Claude Opus 4.6 | 90.33% | $1.46 | |
    | 28 | gpt-oss-120b | gpt-oss-120b | 90.27% | $0.09 | |
    | 29 | Kimi K2.5 | Ministral 3 8B | 90.13% | $0.40 | |
    | 30 | Ministral 3 8B | Claude Opus 4.6 | 90.07% | $0.26 | |
    | 31 | Ministral 3 8B | Qwen3 Next 80B A3B | 89.80% | $0.03 | |
    | 32 | gpt-oss-120b | Ministral 3 8B | 89.73% | $0.04 | |
    | 33 | Qwen3 Next 80B A3B | Claude Haiku 4.5 | 89.73% | $0.65 | |
    | 34 | Qwen3 Next 80B A3B | Kimi K2.5 | 89.33% | $0.36 | |
    | 35 | Kimi K2.5 | Claude 3 Haiku | 89.13% | $0.42 | |
    | 36 | Qwen3 Next 80B A3B | gpt-oss-20b | 89.07% | $0.07 | |
    | 37 | gpt-oss-120b | gpt-oss-20b | 88.73% | $0.06 | |
    | 38 | Qwen3 Next 80B A3B | Qwen3 32B | 88.60% | $0.06 | |
    | 39 | Ministral 3 8B | gpt-oss-20b | 88.33% | $0.04 | |
    | 40 | Ministral 3 8B | Qwen3 32B | 88.33% | $0.03 | |
    | 41 | gpt-oss-120b | Claude Opus 4.6 | 87.47% | $1.66 | |
    | 42 | Kimi K2.5 | gpt-oss-120b | 87.40% | $0.44 | |
    | 43 | gpt-oss-120b | Qwen3 Next 80B A3B | 86.73% | $0.05 | |
    | 44 | Ministral 3 8B | Claude Haiku 4.5 | 86.67% | $0.35 | |
    | 45 | Ministral 3 8B | Kimi K2.5 | 86.27% | $0.17 | |
    | 46 | gpt-oss-120b | Claude Haiku 4.5 | 85.87% | $0.57 | |
    | 47 | gpt-oss-120b | Kimi K2.5 | 85.07% | $0.31 | |
    | 48 | gpt-oss-20b | Ministral 3 8B | 85.07% | $0.03 | |
    | 49 | gpt-oss-120b | Qwen3 32B | 85.00% | $0.05 | |
    | 50 | Kimi K2.5 | Claude Opus 4.6 | 82.53% | $2.14 | |
    | 51 | gpt-oss-20b | gpt-oss-120b | 82.40% | $0.05 | |
    | 52 | Kimi K2.5 | gpt-oss-20b | 82.33% | $0.46 | |
    | 53 | gpt-oss-20b | Claude 3 Haiku | 82.27% | $0.04 | |
    | 54 | Kimi K2.5 | Claude Haiku 4.5 | 81.53% | $0.90 | |
    | 55 | gpt-oss-20b | gpt-oss-20b | 81.13% | $0.04 | |
    | 56 | Kimi K2.5 | Qwen3 Next 80B A3B | 80.93% | $0.50 | |
    | 57 | Kimi K2.5 | Claude Opus 4.6 | 80.40% | $2.14 | |
    | 58 | gpt-oss-20b | Claude Opus 4.6 | 80.33% | $0.95 | |
    | 59 | Kimi K2.5 | Claude 3 Haiku | 80.27% | $0.42 | |
    | 60 | gpt-oss-20b | Qwen3 Next 80B A3B | 80.07% | $0.03 | |
    | 61 | Qwen3 32B | Ministral 3 8B | 79.87% | $0.03 | |
    | 62 | gpt-oss-20b | Claude Haiku 4.5 | 79.80% | $0.33 | |
    | 63 | Kimi K2.5 | Kimi K2.5 | 79.00% | $0.67 | |
    | 64 | Qwen3 32B | Claude 3 Haiku | 78.87% | $0.04 | |
    | 65 | gpt-oss-20b | Kimi K2.5 | 78.53% | $0.17 | |
    | 66 | Qwen3 32B | gpt-oss-120b | 78.00% | $0.04 | |
    | 67 | Qwen3 32B | Claude Opus 4.6 | 77.87% | $1.46 | |
    | 68 | Qwen3 32B | gpt-oss-20b | 77.80% | $0.03 | |
    | 69 | Qwen3 Next 80B A3B | Claude Opus 4.6 | 77.60% | $1.46 | |
    | 70 | Qwen3 32B | Claude Haiku 4.5 | 77.53% | $0.46 | |
    | 71 | Qwen3 32B | Qwen3 Next 80B A3B | 77.53% | $0.04 | |
    | 72 | Qwen3 32B | Qwen3 32B | 77.47% | $0.04 | |
    | 73 | gpt-oss-20b | Qwen3 32B | 77.47% | $0.03 | |
    | 74 | Qwen3 32B | Claude Haiku 4.5 | 77.40% | $0.46 | |
    | 75 | Qwen3 32B | Kimi K2.5 | 77.27% | $0.21 | |
    | 76 | Claude 3 Haiku | Kimi K2.5 | 74.53% | $0.29 | |
    | 77 | Claude 3 Haiku | gpt-oss-20b | 72.96% | $0.31 | |
    | 78 | Kimi K2.5 | Qwen3 32B | 72.77% | $0.67 | |
    | 79 | Claude 3 Haiku | Qwen3 Next 80B A3B | 68.94% | $0.36 | |
    | 80 | Claude 3 Haiku | Qwen3 32B | 63.86% | $0.27 | |
    | 81 | Claude 3 Haiku | Claude 3 Haiku | 59.88% | $0.30 | |

### Selector Comparison (50 seeds)

| Selector | Find Rate | Mean Accuracy | Evaluations | Cost | Savings |
|:---------|:----------|:--------------|:------------|:-----|:--------|
| Brute Force | 100% | 98.83% | 14,855 | $113.01 | -- |
| Arm Elimination | 96% | 98.80% | 3,632 | $61.22 | **46%** |
| Hill Climbing (3) | 72% | 97.81% | 4,058 | $45.72 | 60% |
| Random Search | 28% | 98.04% | 3,850 | $28.83 | 74% |
| Hill Climbing (1) | 28% | 93.53% | 1,610 | $18.26 | 84% |
| Bayesian Opt | 4% | 95.39% | 3,608 | $31.05 | 73% |
| Epsilon LUCB | 0% | 97.46% | 443 | $5.55 | 95% |
| LM Proposal | 0% | 96.87% | 149 | $5.15 | 95% |
| Threshold SE | 0% | 77.23% | 369 | $1.95 | 98% |
