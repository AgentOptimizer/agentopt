# Benchmark Results

We evaluated **9 models on AWS Bedrock** across **4 benchmarks** using LangGraph-based agents, then ran 8 model selection algorithms to measure how efficiently each finds the best model without exhaustive search. All results use 198–200 samples per benchmark with brute force ground truth. Selector comparisons were run with 50 random seeds.

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
| HotpotQA | 2-tuple | 199 | 81 | planner=Ministral 3 8B + solver=Claude Opus 4.6 | **74.78%** | $51.48 | 64% |
| MathQA | 2-tuple | 200 | 81 | answer=Claude Opus 4.6 + critic=Qwen3 Next 80B A3B | **98.83%** | $113.01 | 46% |

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

### Selector Comparison

| Selector | Find Rate | Mean Accuracy | Evaluations | Cost | Savings |
|:---------|:----------|:--------------|:------------|:-----|:--------|
| Brute Force | 100% | 80.30% | 1,759 | $4.13 | -- |
| LM Proposal | 100% | 80.30% | 198 | $2.43 | 41% |
| Arm Elimination | 98% | 80.14% | 444 | $2.10 | **49%** |
| Hill Climbing | 92% | 79.64% | 1,552 | $3.75 | 9% |
| Epsilon LUCB | 90% | 79.47% | 361 | $2.32 | 44% |
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

### Selector Comparison

| Selector | Find Rate | Mean Accuracy | Evaluations | Cost | Savings |
|:---------|:----------|:--------------|:------------|:-----|:--------|
| Brute Force | 100% | 72.00% | 1,800 | $85.42 | -- |
| Arm Elimination | 100% | 72.00% | 922 | $76.33 | **11%** |
| Hill Climbing | 94% | 71.94% | 1,652 | $80.38 | 6% |
| Epsilon LUCB | 60% | 71.33% | 407 | $42.60 | 50% |
| Bayesian Opt | 56% | 70.61% | 1,000 | $50.98 | 40% |
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

### Bottom 15 Combos

| Rank | Planner | Solver | Accuracy | Cost |
|:-----|:--------|:-------|:---------|:-----|
| 67 | Claude Haiku 4.5 | Qwen3 32B | 35.64% | $0.46 |
| 68 | Claude Haiku 4.5 | Claude 3 Haiku | 34.10% | $0.48 |
| 69 | Ministral 3 8B | Claude Haiku 4.5 | 33.95% | $0.74 |
| 70 | Claude Opus 4.6 | Claude Opus 4.6 | 32.70% | $2.00 |
| 71 | Claude Opus 4.6 | Kimi K2.5 | 32.44% | $2.01 |
| 72 | Claude Opus 4.6 | Qwen3 Next 80B A3B | 32.05% | $2.01 |
| 73 | Claude Opus 4.6 | gpt-oss-120b | 32.00% | $2.01 |
| 74 | Claude Opus 4.6 | Ministral 3 8B | 31.80% | $2.01 |
| 75 | Claude Opus 4.6 | Claude 3 Haiku | 31.80% | $2.01 |
| 76 | Claude Opus 4.6 | Qwen3 32B | 31.52% | $2.01 |
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
    | 16 | gpt-oss-120b | Claude Opus 4.6 | 67.03% | $1.57 | |
    | 17 | Kimi K2.5 | Ministral 3 8B | 66.79% | $0.27 | |
    | 18 | Qwen3 Next 80B A3B | gpt-oss-20b | 66.79% | $0.09 | |
    | 19 | Kimi K2.5 | gpt-oss-120b | 66.76% | $0.29 | |
    | 20 | Ministral 3 8B | gpt-oss-20b | 66.55% | $0.09 | |
    | 21 | Kimi K2.5 | gpt-oss-20b | 66.49% | $0.25 | |
    | 22 | Claude 3 Haiku | gpt-oss-20b | 65.10% | $0.13 | |
    | 23 | Claude 3 Haiku | Ministral 3 8B | 64.41% | $0.14 | |
    | 24 | Qwen3 Next 80B A3B | Kimi K2.5 | 64.28% | $0.27 | |
    | 25 | Ministral 3 8B | Kimi K2.5 | 64.03% | $0.26 | |
    | 26 | Qwen3 32B | Kimi K2.5 | 63.77% | $0.26 | |
    | 27 | gpt-oss-120b | Qwen3 Next 80B A3B | 63.62% | $0.09 | |
    | 28 | Qwen3 Next 80B A3B | Ministral 3 8B | 63.29% | $0.10 | |
    | 29 | Claude 3 Haiku | Kimi K2.5 | 62.86% | $0.31 | |
    | 30 | gpt-oss-120b | Claude Haiku 4.5 | 62.38% | $0.36 | |
    | 31 | Ministral 3 8B | Ministral 3 8B | 62.20% | $0.09 | |
    | 32 | Qwen3 32B | Ministral 3 8B | 62.09% | $0.09 | |
    | 33 | Kimi K2.5 | Kimi K2.5 | 61.96% | $0.45 | |
    | 34 | gpt-oss-120b | Kimi K2.5 | 61.15% | $0.17 | |
    | 35 | gpt-oss-120b | Claude 3 Haiku | 60.89% | $0.12 | |
    | 36 | gpt-oss-120b | Ministral 3 8B | 60.64% | $0.09 | |
    | 37 | gpt-oss-120b | gpt-oss-120b | 60.51% | $0.10 | |
    | 38 | gpt-oss-120b | gpt-oss-20b | 59.10% | $0.08 | |
    | 39 | gpt-oss-120b | Qwen3 32B | 58.54% | $0.09 | |
    | 40 | Kimi K2.5 | Claude 3 Haiku | 57.18% | $0.32 | |
    | 41 | Claude 3 Haiku | Qwen3 32B | 56.28% | $0.15 | |
    | 42 | Kimi K2.5 | Qwen3 32B | 55.72% | $0.27 | |
    | 43 | Ministral 3 8B | Qwen3 32B | 55.30% | $0.11 | |
    | 44 | gpt-oss-20b | Claude Opus 4.6 | 55.13% | $0.90 | |
    | 45 | gpt-oss-20b | Ministral 3 8B | 54.99% | $0.05 | |
    | 46 | Qwen3 Next 80B A3B | Qwen3 32B | 54.88% | $0.11 | |
    | 47 | gpt-oss-20b | Kimi K2.5 | 54.69% | $0.11 | |
    | 48 | gpt-oss-20b | gpt-oss-120b | 54.26% | $0.06 | |
    | 49 | Qwen3 32B | Qwen3 32B | 54.21% | $0.11 | |
    | 50 | Claude 3 Haiku | Claude 3 Haiku | 54.13% | $0.20 | |
    | 51 | gpt-oss-20b | Claude Haiku 4.5 | 54.06% | $0.25 | |
    | 52 | gpt-oss-20b | Claude 3 Haiku | 53.08% | $0.08 | |
    | 53 | gpt-oss-20b | Qwen3 Next 80B A3B | 52.87% | $0.05 | |
    | 54 | gpt-oss-20b | gpt-oss-20b | 52.69% | $0.05 | |
    | 55 | Ministral 3 8B | Claude 3 Haiku | 51.65% | $0.16 | |
    | 56 | gpt-oss-20b | Qwen3 32B | 49.60% | $0.06 | |
    | 57 | Qwen3 Next 80B A3B | Claude 3 Haiku | 48.86% | $0.17 | |
    | 58 | Qwen3 32B | Claude 3 Haiku | 48.29% | $0.16 | |
    | 59 | Claude 3 Haiku | Claude Haiku 4.5 | 47.28% | $0.71 | |
    | 60 | Claude Haiku 4.5 | Claude Opus 4.6 | 43.40% | $1.80 | |
    | 61 | Claude Haiku 4.5 | Kimi K2.5 | 41.51% | $0.55 | |
    | 62 | Claude Haiku 4.5 | Ministral 3 8B | 41.21% | $0.45 | |
    | 63 | Claude Haiku 4.5 | gpt-oss-20b | 41.18% | $0.45 | |
    | 64 | Claude Haiku 4.5 | gpt-oss-120b | 40.83% | $0.47 | |
    | 65 | Claude Haiku 4.5 | Qwen3 Next 80B A3B | 40.54% | $0.46 | |
    | 66 | Kimi K2.5 | Claude Haiku 4.5 | 40.37% | $0.87 | |
    | 67 | Claude Haiku 4.5 | Qwen3 32B | 35.64% | $0.46 | |
    | 68 | Claude Haiku 4.5 | Claude 3 Haiku | 34.10% | $0.48 | |
    | 69 | Ministral 3 8B | Claude Haiku 4.5 | 33.95% | $0.74 | |
    | 70 | Claude Opus 4.6 | Claude Opus 4.6 | 32.70% | $2.00 | |
    | 71 | Claude Opus 4.6 | Kimi K2.5 | 32.44% | $2.01 | role2_never_called |
    | 72 | Claude Opus 4.6 | Qwen3 Next 80B A3B | 32.05% | $2.01 | role2_never_called |
    | 73 | Claude Opus 4.6 | gpt-oss-120b | 32.00% | $2.01 | role2_never_called |
    | 74 | Claude Opus 4.6 | Ministral 3 8B | 31.80% | $2.01 | role2_never_called |
    | 75 | Claude Opus 4.6 | Claude 3 Haiku | 31.80% | $2.01 | role2_never_called |
    | 76 | Claude Opus 4.6 | Qwen3 32B | 31.52% | $2.01 | role2_never_called |
    | 77 | Claude Opus 4.6 | gpt-oss-20b | 31.31% | $2.01 | role2_never_called |
    | 78 | Qwen3 Next 80B A3B | Claude Haiku 4.5 | 30.85% | $0.71 | |
    | 79 | Claude Opus 4.6 | Claude Haiku 4.5 | 30.81% | $2.01 | role2_never_called |
    | 80 | Claude Haiku 4.5 | Claude Haiku 4.5 | 26.57% | $0.79 | |
    | 81 | Qwen3 32B | Claude Haiku 4.5 | 25.11% | $0.72 | |

### Selector Comparison

| Selector | Find Rate | Mean Accuracy | Evaluations | Cost | Savings |
|:---------|:----------|:--------------|:------------|:-----|:--------|
| Brute Force | 100% | 74.78% | 16,108 | $51.48 | -- |
| Arm Elimination | 90% | 74.12% | 4,654 | $18.49 | **64%** |
| Hill Climbing | 44% | 73.38% | 5,031 | $19.21 | 63% |
| Bayesian Opt | 8% | 72.78% | 3,979 | $12.13 | 76% |
| Random Search | 30% | 72.34% | 4,176 | $13.26 | 74% |
| Epsilon LUCB | 14% | 69.96% | 477 | $1.86 | 96% |
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

### Bottom 15 Combos

| Rank | Answer Model | Critic Model | Accuracy | Cost |
|:-----|:-------------|:-------------|:---------|:-----|
| 67 | Kimi K2.5 | Claude Haiku 4.5 | 78.01% | $1.20 |
| 68 | Claude 3 Haiku | Ministral 3 8B | 77.64% | $0.30 |
| 69 | Qwen3 Next 80B A3B | Claude Opus 4.6 | 77.60% | $2.03 |
| 70 | Kimi K2.5 | Claude Opus 4.6 | 77.55% | $2.55 |
| 71 | Claude 3 Haiku | Kimi K2.5 | 77.48% | $0.46 |
| 72 | gpt-oss-120b | gpt-oss-20b | 77.32% | $0.17 |
| 73 | Kimi K2.5 | Claude 3 Haiku | 76.96% | $0.92 |
| 74 | Qwen3 Next 80B A3B | Kimi K2.5 | 76.96% | $0.46 |
| 75 | Qwen3 Next 80B A3B | gpt-oss-120b | 76.84% | $0.32 |
| 76 | gpt-oss-120b | Qwen3 Next 80B A3B | 74.74% | $0.20 |
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
    | 16 | gpt-oss-20b | gpt-oss-120b | 93.96% | $0.12 | |
    | 17 | Claude Haiku 4.5 | Qwen3 32B | 93.50% | $2.72 | |
    | 18 | Claude Haiku 4.5 | gpt-oss-20b | 93.50% | $2.93 | |
    | 19 | gpt-oss-20b | Kimi K2.5 | 93.44% | $0.23 | |
    | 20 | gpt-oss-20b | Claude Haiku 4.5 | 92.97% | $0.36 | |
    | 21 | Claude 3 Haiku | Claude Opus 4.6 | 92.94% | $2.04 | |
    | 22 | Claude Haiku 4.5 | gpt-oss-120b | 92.50% | $2.35 | |
    | 23 | gpt-oss-20b | Qwen3 Next 80B A3B | 92.43% | $0.15 | |
    | 24 | gpt-oss-20b | Claude Opus 4.6 | 92.35% | $0.99 | |
    | 25 | gpt-oss-20b | gpt-oss-20b | 91.94% | $0.09 | |
    | 26 | Claude Haiku 4.5 | Claude 3 Haiku | 91.50% | $2.95 | |
    | 27 | gpt-oss-20b | Qwen3 32B | 91.21% | $0.08 | |
    | 28 | gpt-oss-20b | Claude 3 Haiku | 90.76% | $0.16 | |
    | 29 | Ministral 3 8B | gpt-oss-120b | 90.59% | $0.07 | |
    | 30 | gpt-oss-20b | Ministral 3 8B | 90.43% | $0.13 | |
    | 31 | Ministral 3 8B | Qwen3 Next 80B A3B | 90.20% | $0.03 | |
    | 32 | Ministral 3 8B | Claude Opus 4.6 | 89.53% | $0.87 | |
    | 33 | Ministral 3 8B | Claude Haiku 4.5 | 88.89% | $0.30 | |
    | 34 | Ministral 3 8B | Kimi K2.5 | 88.82% | $0.09 | |
    | 35 | Ministral 3 8B | gpt-oss-20b | 88.76% | $0.04 | |
    | 36 | Qwen3 32B | Qwen3 Next 80B A3B | 88.72% | $0.21 | |
    | 37 | Ministral 3 8B | Ministral 3 8B | 88.19% | $0.03 | |
    | 38 | Claude 3 Haiku | Claude Haiku 4.5 | 87.21% | $0.69 | |
    | 39 | Ministral 3 8B | Qwen3 32B | 86.98% | $0.04 | |
    | 40 | Qwen3 32B | Ministral 3 8B | 86.73% | $0.35 | |
    | 41 | Qwen3 32B | gpt-oss-120b | 86.67% | $0.25 | |
    | 42 | Qwen3 32B | Claude Opus 4.6 | 85.35% | $2.01 | |
    | 43 | Qwen3 32B | gpt-oss-20b | 85.05% | $0.19 | |
    | 44 | Qwen3 32B | Claude Haiku 4.5 | 84.02% | $0.53 | |
    | 45 | Qwen3 32B | Kimi K2.5 | 82.74% | $1.11 | |
    | 46 | Qwen3 32B | Qwen3 32B | 82.56% | $0.17 | |
    | 47 | Qwen3 32B | Claude 3 Haiku | 82.47% | $0.27 | |
    | 48 | Qwen3 Next 80B A3B | Claude 3 Haiku | 82.29% | $0.42 | |
    | 49 | Qwen3 Next 80B A3B | Qwen3 32B | 81.87% | $0.29 | |
    | 50 | Kimi K2.5 | gpt-oss-120b | 81.44% | $0.82 | |
    | 51 | Kimi K2.5 | gpt-oss-20b | 81.35% | $0.85 | |
    | 52 | Kimi K2.5 | Kimi K2.5 | 81.25% | $1.13 | |
    | 53 | gpt-oss-120b | Qwen3 32B | 80.41% | $0.15 | |
    | 54 | Qwen3 Next 80B A3B | Qwen3 Next 80B A3B | 80.32% | $0.31 | |
    | 55 | gpt-oss-120b | Claude Haiku 4.5 | 80.31% | $0.47 | |
    | 56 | gpt-oss-120b | Kimi K2.5 | 80.10% | $0.27 | |
    | 57 | gpt-oss-120b | Ministral 3 8B | 80.00% | $0.18 | |
    | 58 | Qwen3 Next 80B A3B | gpt-oss-20b | 79.79% | $0.32 | |
    | 59 | gpt-oss-120b | Claude 3 Haiku | 79.69% | $0.19 | |
    | 60 | Kimi K2.5 | Ministral 3 8B | 79.49% | $0.86 | |
    | 61 | gpt-oss-120b | Claude Opus 4.6 | 79.49% | $1.17 | |
    | 62 | Kimi K2.5 | Qwen3 Next 80B A3B | 79.06% | $0.82 | |
    | 63 | gpt-oss-120b | gpt-oss-120b | 78.87% | $0.20 | |
    | 64 | Qwen3 Next 80B A3B | Claude Haiku 4.5 | 78.65% | $0.81 | |
    | 65 | Qwen3 Next 80B A3B | Ministral 3 8B | 78.65% | $0.38 | |
    | 66 | Claude 3 Haiku | gpt-oss-120b | 78.62% | $0.36 | |
    | 67 | Kimi K2.5 | Claude Haiku 4.5 | 78.01% | $1.20 | |
    | 68 | Claude 3 Haiku | Ministral 3 8B | 77.64% | $0.30 | |
    | 69 | Qwen3 Next 80B A3B | Claude Opus 4.6 | 77.60% | $2.03 | |
    | 70 | Kimi K2.5 | Claude Opus 4.6 | 77.55% | $2.55 | |
    | 71 | Claude 3 Haiku | Kimi K2.5 | 77.48% | $0.46 | |
    | 72 | gpt-oss-120b | gpt-oss-20b | 77.32% | $0.17 | |
    | 73 | Kimi K2.5 | Claude 3 Haiku | 76.96% | $0.92 | |
    | 74 | Qwen3 Next 80B A3B | Kimi K2.5 | 76.96% | $0.46 | |
    | 75 | Qwen3 Next 80B A3B | gpt-oss-120b | 76.84% | $0.32 | |
    | 76 | gpt-oss-120b | Qwen3 Next 80B A3B | 74.74% | $0.20 | |
    | 77 | Claude 3 Haiku | gpt-oss-20b | 72.96% | $0.31 | |
    | 78 | Kimi K2.5 | Qwen3 32B | 72.77% | $0.67 | |
    | 79 | Claude 3 Haiku | Qwen3 Next 80B A3B | 68.94% | $0.36 | |
    | 80 | Claude 3 Haiku | Qwen3 32B | 63.86% | $0.27 | |
    | 81 | Claude 3 Haiku | Claude 3 Haiku | 59.88% | $0.30 | |

### Selector Comparison

| Selector | Find Rate | Mean Accuracy | Evaluations | Cost | Savings |
|:---------|:----------|:--------------|:------------|:-----|:--------|
| Brute Force | 100% | 98.83% | 14,855 | $113.01 | -- |
| Arm Elimination | 96% | 98.80% | 3,632 | $61.22 | **46%** |
| Random Search | 28% | 98.04% | 3,850 | $28.83 | 74% |
| Hill Climbing | 72% | 97.81% | 4,058 | $45.72 | 60% |
| Epsilon LUCB | 0% | 97.46% | 443 | $5.55 | 95% |
| LM Proposal | 0% | 96.87% | 149 | $5.15 | 95% |
| Bayesian Opt | 4% | 95.39% | 3,608 | $31.05 | 73% |
| Threshold SE | 0% | 77.23% | 369 | $1.95 | 98% |
