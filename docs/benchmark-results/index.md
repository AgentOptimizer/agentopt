# Benchmark Results

We evaluated **9 models** across **4 benchmarks** using LangGraph-based agents, then ran 8 model selection algorithms to measure how efficiently each finds the best model without exhaustive search. All results use 198–200 samples per benchmark with brute force ground truth. Selector comparisons were run with 50 random seeds.

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
| GPQA Diamond | 1-tuple | 198 | 9 | Claude Opus 4.6 | **74.75%** | $4.71 | 24% |
| BFCL Multi-Turn | 1-tuple | 200 | 9 | Kimi K2.5 (tied with Opus, Qwen3 Next) | **70.00%** | $84.80 | 12% |
| HotpotQA | 2-tuple | 200 | 81 | planner=Ministral 3 8B + solver=Claude Opus 4.6 | **74.27%** | $51.90 | 67% |
| MathQA | 2-tuple | 200 | 81 | answer=Claude Opus 4.6 + critic=Claude Haiku 4.5 | **98.84%** | $123.87 | 58% |

---

## GPQA Diamond

**Graduate-level science QA** — 198 multiple-choice questions from the GPQA Diamond dataset. Single-agent architecture: one LLM answers directly.

### Model Results

| Rank | Model | Accuracy | Avg Latency (s) | Cost |
|:-----|:------|:---------|:-----------------|:-----|
| 1 | Claude Opus 4.6 | **74.75%** | 9.16 | $2.48 |
| 2 | Kimi K2.5 | 72.73% | 16.41 | $1.13 |
| 3 | gpt-oss-120b | 68.18% | 6.46 | $0.20 |
| 4 | Claude Haiku 4.5 | 59.60% | 3.70 | $0.51 |
| 5 | Qwen3 Next 80B A3B | 51.01% | 10.33 | $0.14 |
| 6 | gpt-oss-20b | 50.00% | 6.21 | $0.14 |
| 7 | Qwen3 32B | 46.97% | 1.54 | $0.08 |
| 8 | Ministral 3 8B | 36.87% | 0.25 | $0.00 |
| 9 | Claude 3 Haiku | 34.85% | 1.79 | $0.06 |

### Selector Comparison

| Selector | Find Rate | Mean Accuracy | Evaluations | Cost | Savings |
|:---------|:----------|:--------------|:------------|:-----|:--------|
| Brute Force | 100% | 74.75% | 1,782 | $4.71 | -- |
| LM Proposal | 100% | 74.75% | 198 | $2.47 | 48% |
| Arm Elimination | 94% | 74.10% | 666 | $3.57 | **24%** |
| Hill Climbing | 90% | 74.55% | 1,501 | $4.03 | 14% |
| Epsilon LUCB | 72% | 73.14% | 380 | $2.51 | 47% |
| Bayesian Opt | 56% | 72.43% | 990 | $2.59 | 45% |
| Random Search | 36% | 68.57% | 594 | $1.73 | 63% |
| Threshold SE | 16% | 57.48% | 252 | $1.80 | 62% |

### Thinking Effort Ablation

Impact of thinking/reasoning budget on GPQA accuracy for Opus (adaptive effort) and Haiku 4.5 (budget_tokens). Baseline "none" rows use brute force results.

| Model | Effort/Budget Tokens | Accuracy | Cost/Sample | Server Latency/Sample (s) |
|:------|:---------------------|:---------|:------------|:--------------------------|
| Opus | high | 83.90% | $0.113 | 70.4 |
| Opus | medium | 79.30% | $0.0341 | 23.9 |
| Opus | none | 74.75% | $0.0125 | 9.16 |
| Haiku 4.5 | 16K | 71.20% | $0.0192 | 30.6 |
| Haiku 4.5 | 32K | 71.20% | $0.0361 | 57.4 |
| Opus | low | 61.60% | $0.00302 | 3.06 |
| Haiku 4.5 | 5K | 60.10% | $0.00925 | 15.0 |
| Haiku 4.5 | none | 59.60% | $0.0026 | 3.70 |

---

## BFCL Multi-Turn

**Multi-turn function calling** — 200 samples from the Berkeley Function Calling Leaderboard (BFCL v3). Each sample has multiple turns with tool-calling loops. Models that don't support native function calling (Qwen3 32B, Kimi K2.5, Ministral 3 8B) use a text-based prompting fallback.

!!! note "Comparison with official BFCL leaderboard"
    Our evaluation uses a **live LangGraph agent** that executes tool calls against real backend state machines, whereas the official BFCL leaderboard uses static response matching. Our accuracy numbers are not directly comparable to the leaderboard — they reflect end-to-end agent performance including tool execution, state management, and multi-step reasoning.

### Model Results

| Rank | Model | Accuracy | Avg Latency (s) | Cost |
|:-----|:------|:---------|:-----------------|:-----|
| 1 | Kimi K2.5 | **70.00%** | 21.30 | $3.86 |
| 2 | Claude Opus 4.6 | 70.00% | 42.35 | $60.14 |
| 3 | Qwen3 Next 80B A3B | 70.00% | 60.54 | $1.90 |
| 4 | Claude Haiku 4.5 | 65.00% | 20.90 | $11.98 |
| 5 | gpt-oss-120b | 58.50% | 20.01 | $1.16 |
| 6 | Qwen3 32B | 47.00% | 10.78 | $1.00 |
| 7 | Claude 3 Haiku | 43.50% | 17.96 | $3.42 |
| 8 | gpt-oss-20b | 42.00% | 10.03 | $0.42 |
| 9 | Ministral 3 8B | 34.00% | 29.03 | $0.92 |

### Selector Comparison

| Selector | Find Rate | Mean Accuracy | Evaluations | Cost | Savings |
|:---------|:----------|:--------------|:------------|:-----|:--------|
| Brute Force | 100% | 70.00% | 1,800 | $84.80 | -- |
| Hill Climbing | 100% | 70.00% | 1,664 | $72.12 | 15% |
| Arm Elimination | 88% | 69.37% | 912 | $74.39 | **12%** |
| Bayesian Opt | 44% | 69.27% | 1,000 | $50.64 | 40% |
| Random Search | 36% | 67.13% | 600 | $31.39 | 63% |
| Epsilon LUCB | 28% | 69.90% | 399 | $40.03 | 53% |
| Threshold SE | 10% | 58.19% | 186 | $18.82 | 78% |
| LM Proposal | 0% | 44.03% | 200 | $3.39 | 96% |

---

## HotpotQA

**Multi-hop question answering** — 200 samples from the HotpotQA distractor setting. Two-agent architecture: a **planner** proposes search steps, and a **solver** executes them with tool access. 81 model combinations (9 planners x 9 solvers).

### Top 15 Combos

| Rank | Planner | Solver | Accuracy | Cost |
|:-----|:--------|:-------|:---------|:-----|
| 1 | Ministral 3 8B | Claude Opus 4.6 | **74.27%** | $2.64 |
| 2 | Claude 3 Haiku | Claude Opus 4.6 | 73.25% | $2.79 |
| 3 | Qwen3 32B | Claude Opus 4.6 | 73.02% | $2.65 |
| 4 | Qwen3 Next 80B A3B | Claude Opus 4.6 | 72.10% | $2.67 |
| 5 | Qwen3 Next 80B A3B | gpt-oss-120b | 71.83% | $0.13 |
| 6 | Qwen3 32B | gpt-oss-120b | 70.04% | $0.13 |
| 7 | Kimi K2.5 | Claude Opus 4.6 | 69.96% | $2.43 |
| 8 | Claude 3 Haiku | gpt-oss-120b | 69.86% | $0.17 |
| 9 | Ministral 3 8B | gpt-oss-20b | 69.34% | $0.09 |
| 10 | Claude 3 Haiku | Qwen3 Next 80B A3B | 69.27% | $0.16 |
| 11 | Qwen3 Next 80B A3B | gpt-oss-20b | 68.89% | $0.09 |
| 12 | Ministral 3 8B | gpt-oss-120b | 68.70% | $0.12 |
| 13 | Qwen3 Next 80B A3B | Qwen3 Next 80B A3B | 68.15% | $0.11 |
| 14 | Ministral 3 8B | Qwen3 Next 80B A3B | 67.98% | $0.11 |
| 15 | Qwen3 32B | Qwen3 Next 80B A3B | 67.53% | $0.11 |

### Bottom 15 Combos

| Rank | Planner | Solver | Accuracy | Cost |
|:-----|:--------|:-------|:---------|:-----|
| 67 | Kimi K2.5 | Claude Haiku 4.5 | 37.19% | $0.88 |
| 68 | Claude Haiku 4.5 | Qwen3 32B | 36.13% | $0.46 |
| 69 | Claude Haiku 4.5 | Claude 3 Haiku | 34.34% | $0.49 |
| 70 | Ministral 3 8B | Claude Haiku 4.5 | 32.42% | $0.70 |
| 71 | Qwen3 Next 80B A3B | Claude Haiku 4.5 | 32.19% | $0.72 |
| 72 | Claude Opus 4.6 | Kimi K2.5 | 31.96% | $2.02 |
| 73 | Claude Opus 4.6 | Ministral 3 8B | 31.96% | $2.02 |
| 74 | Claude Opus 4.6 | Qwen3 32B | 31.96% | $2.02 |
| 75 | Claude Opus 4.6 | Qwen3 Next 80B A3B | 31.96% | $2.02 |
| 76 | Claude Opus 4.6 | gpt-oss-120b | 31.95% | $2.02 |
| 77 | Claude Opus 4.6 | gpt-oss-20b | 31.88% | $2.03 |
| 78 | Claude Opus 4.6 | Claude 3 Haiku | 31.78% | $2.02 |
| 79 | Claude Opus 4.6 | Claude Haiku 4.5 | 31.77% | $2.03 |
| 80 | Qwen3 32B | Claude Haiku 4.5 | 26.63% | $0.69 |
| 81 | Claude Haiku 4.5 | Claude Haiku 4.5 | 26.49% | $0.79 |

!!! warning "Capability as Liability"
    **Claude Opus 4.6 as planner achieves only ~32% accuracy** regardless of solver — the worst planner in the benchmark. Opus is "too smart" for the planner role: it calls `terminate()` and answers directly instead of delegating to the solver. The solver is never invoked. Meanwhile, the cheapest model (Ministral 3 8B) as planner with Opus as solver achieves the **best accuracy at 74.27%**. This demonstrates that stronger models can underperform in multi-agent architectures when the role requires delegation, not direct answering.

??? note "Full 81 Combo Results"

    | Rank | Planner | Solver | Accuracy | Cost | Note |
    |:-----|:--------|:-------|:---------|:-----|:-----|
    | 1 | Ministral 3 8B | Claude Opus 4.6 | 74.27% | $2.64 | |
    | 2 | Claude 3 Haiku | Claude Opus 4.6 | 73.25% | $2.79 | |
    | 3 | Qwen3 32B | Claude Opus 4.6 | 73.02% | $2.65 | |
    | 4 | Qwen3 Next 80B A3B | Claude Opus 4.6 | 72.10% | $2.67 | |
    | 5 | Qwen3 Next 80B A3B | gpt-oss-120b | 71.83% | $0.13 | |
    | 6 | Qwen3 32B | gpt-oss-120b | 70.04% | $0.13 | |
    | 7 | Kimi K2.5 | Claude Opus 4.6 | 69.96% | $2.43 | |
    | 8 | Claude 3 Haiku | gpt-oss-120b | 69.86% | $0.17 | |
    | 9 | Ministral 3 8B | gpt-oss-20b | 69.34% | $0.09 | |
    | 10 | Claude 3 Haiku | Qwen3 Next 80B A3B | 69.27% | $0.16 | |
    | 11 | Qwen3 Next 80B A3B | gpt-oss-20b | 68.89% | $0.09 | |
    | 12 | Ministral 3 8B | gpt-oss-120b | 68.70% | $0.12 | |
    | 13 | Qwen3 Next 80B A3B | Qwen3 Next 80B A3B | 68.15% | $0.11 | |
    | 14 | Ministral 3 8B | Qwen3 Next 80B A3B | 67.98% | $0.11 | |
    | 15 | Qwen3 32B | Qwen3 Next 80B A3B | 67.53% | $0.11 | |
    | 16 | Qwen3 32B | gpt-oss-20b | 66.95% | $0.09 | |
    | 17 | Claude 3 Haiku | Ministral 3 8B | 65.98% | $0.14 | |
    | 18 | Ministral 3 8B | Kimi K2.5 | 65.24% | $0.26 | |
    | 19 | gpt-oss-120b | Qwen3 Next 80B A3B | 64.93% | $0.10 | |
    | 20 | Ministral 3 8B | Ministral 3 8B | 64.89% | $0.09 | |
    | 21 | Claude 3 Haiku | gpt-oss-20b | 64.79% | $0.13 | |
    | 22 | Kimi K2.5 | gpt-oss-120b | 64.70% | $0.29 | |
    | 23 | gpt-oss-120b | Claude Opus 4.6 | 64.59% | $1.61 | |
    | 24 | gpt-oss-120b | Claude Haiku 4.5 | 64.11% | $0.38 | |
    | 25 | Kimi K2.5 | Qwen3 Next 80B A3B | 63.99% | $0.30 | |
    | 26 | Kimi K2.5 | Ministral 3 8B | 63.95% | $0.28 | |
    | 27 | Claude 3 Haiku | Kimi K2.5 | 63.85% | $0.31 | |
    | 28 | gpt-oss-120b | Ministral 3 8B | 63.70% | $0.09 | |
    | 29 | Qwen3 Next 80B A3B | Kimi K2.5 | 63.69% | $0.27 | |
    | 30 | Kimi K2.5 | gpt-oss-20b | 63.35% | $0.26 | |
    | 31 | Qwen3 32B | Kimi K2.5 | 63.17% | $0.28 | |
    | 32 | gpt-oss-120b | Claude 3 Haiku | 62.72% | $0.13 | |
    | 33 | Kimi K2.5 | Kimi K2.5 | 62.28% | $0.44 | |
    | 34 | gpt-oss-120b | gpt-oss-120b | 62.15% | $0.10 | |
    | 35 | Qwen3 Next 80B A3B | Ministral 3 8B | 62.11% | $0.10 | |
    | 36 | gpt-oss-120b | gpt-oss-20b | 61.51% | $0.08 | |
    | 37 | Qwen3 32B | Ministral 3 8B | 61.17% | $0.09 | |
    | 38 | gpt-oss-120b | Kimi K2.5 | 60.85% | $0.18 | |
    | 39 | gpt-oss-120b | Qwen3 32B | 58.80% | $0.10 | |
    | 40 | Claude 3 Haiku | Qwen3 32B | 56.02% | $0.15 | |
    | 41 | Claude 3 Haiku | Claude 3 Haiku | 55.91% | $0.21 | |
    | 42 | gpt-oss-20b | Claude Opus 4.6 | 55.86% | $1.04 | |
    | 43 | Ministral 3 8B | Qwen3 32B | 55.02% | $0.11 | |
    | 44 | Kimi K2.5 | Claude 3 Haiku | 54.90% | $0.34 | |
    | 45 | Qwen3 32B | Qwen3 32B | 54.82% | $0.11 | |
    | 46 | Kimi K2.5 | Qwen3 32B | 54.73% | $0.30 | |
    | 47 | gpt-oss-20b | Claude Haiku 4.5 | 54.28% | $0.26 | |
    | 48 | gpt-oss-20b | Ministral 3 8B | 54.25% | $0.05 | |
    | 49 | Qwen3 Next 80B A3B | Qwen3 32B | 54.13% | $0.11 | |
    | 50 | gpt-oss-20b | Qwen3 Next 80B A3B | 53.89% | $0.06 | |
    | 51 | gpt-oss-20b | Claude 3 Haiku | 52.66% | $0.08 | |
    | 52 | gpt-oss-20b | gpt-oss-120b | 52.17% | $0.06 | |
    | 53 | Ministral 3 8B | Claude 3 Haiku | 51.33% | $0.16 | |
    | 54 | gpt-oss-20b | Kimi K2.5 | 51.01% | $0.12 | |
    | 55 | gpt-oss-20b | gpt-oss-20b | 50.09% | $0.05 | |
    | 56 | Qwen3 Next 80B A3B | Claude 3 Haiku | 49.98% | $0.17 | |
    | 57 | gpt-oss-20b | Qwen3 32B | 49.16% | $0.06 | |
    | 58 | Qwen3 32B | Claude 3 Haiku | 48.77% | $0.16 | |
    | 59 | Claude 3 Haiku | Claude Haiku 4.5 | 46.50% | $0.71 | |
    | 60 | Claude Haiku 4.5 | Claude Opus 4.6 | 43.54% | $1.80 | |
    | 61 | Claude Haiku 4.5 | gpt-oss-20b | 41.49% | $0.45 | |
    | 62 | Claude Haiku 4.5 | gpt-oss-120b | 41.20% | $0.47 | |
    | 63 | Claude Haiku 4.5 | Qwen3 Next 80B A3B | 41.17% | $0.46 | |
    | 64 | Claude Haiku 4.5 | Ministral 3 8B | 41.09% | $0.45 | |
    | 65 | Claude Haiku 4.5 | Kimi K2.5 | 41.00% | $0.54 | |
    | 66 | Kimi K2.5 | Claude Haiku 4.5 | 37.19% | $0.88 | |
    | 67 | Claude Haiku 4.5 | Qwen3 32B | 36.13% | $0.46 | |
    | 68 | Claude Haiku 4.5 | Claude 3 Haiku | 34.34% | $0.49 | |
    | 69 | Ministral 3 8B | Claude Haiku 4.5 | 32.42% | $0.70 | |
    | 70 | Qwen3 Next 80B A3B | Claude Haiku 4.5 | 32.19% | $0.72 | |
    | 71 | Claude Opus 4.6 | Kimi K2.5 | 31.96% | $2.02 | role2_never_called |
    | 72 | Claude Opus 4.6 | Ministral 3 8B | 31.96% | $2.02 | role2_never_called |
    | 73 | Claude Opus 4.6 | Qwen3 32B | 31.96% | $2.02 | role2_never_called |
    | 74 | Claude Opus 4.6 | Qwen3 Next 80B A3B | 31.96% | $2.02 | role2_never_called |
    | 75 | Claude Opus 4.6 | gpt-oss-120b | 31.95% | $2.02 | role2_never_called |
    | 76 | Claude Opus 4.6 | gpt-oss-20b | 31.88% | $2.03 | role2_never_called |
    | 77 | Claude Opus 4.6 | Claude 3 Haiku | 31.78% | $2.02 | role2_never_called |
    | 78 | Claude Opus 4.6 | Claude Haiku 4.5 | 31.77% | $2.03 | role2_never_called |
    | 79 | Claude Opus 4.6 | Claude Opus 4.6 | 31.71% | $2.02 | |
    | 80 | Qwen3 32B | Claude Haiku 4.5 | 26.63% | $0.69 | |
    | 81 | Claude Haiku 4.5 | Claude Haiku 4.5 | 26.49% | $0.79 | |

### Selector Comparison

| Selector | Find Rate | Mean Accuracy | Evaluations | Cost | Savings |
|:---------|:----------|:--------------|:------------|:-----|:--------|
| Brute Force | 100% | 74.27% | 16,168 | $51.90 | -- |
| Arm Elimination | 86% | 73.19% | 4,283 | $16.92 | **67%** |
| Hill Climbing | 52% | 73.13% | 4,635 | $19.39 | 63% |
| Random Search | 30% | 72.25% | 4,192 | $13.37 | 74% |
| Epsilon LUCB | 10% | 69.71% | 478 | $1.75 | 97% |
| Bayesian Opt | 8% | 73.33% | 3,996 | $12.29 | 76% |
| Threshold SE | 4% | 65.42% | 1,642 | $6.45 | 88% |
| LM Proposal | 0% | 34.13% | 200 | $1.84 | 96% |

---

## MathQA

**Self-reflective math reasoning** — 200 samples from the MathQA dataset. Two-agent architecture: an **answer model** solves problems, and a **critic model** checks the work. If the critic rejects, the answer model retries (up to 3 iterations). 81 model combinations (9 answer models x 9 critics).

### Top 15 Combos

| Rank | Answer Model | Critic Model | Accuracy | Cost |
|:-----|:-------------|:-------------|:---------|:-----|
| 1 | Claude Opus 4.6 | Claude Haiku 4.5 | **98.84%** | $6.19 |
| 2 | Claude Opus 4.6 | Qwen3 Next 80B A3B | 98.82% | $5.77 |
| 3 | Claude Opus 4.6 | Ministral 3 8B | 98.72% | $5.26 |
| 4 | Claude Opus 4.6 | gpt-oss-20b | 98.28% | $5.93 |
| 5 | Claude Opus 4.6 | gpt-oss-120b | 97.77% | $6.30 |
| 6 | Claude Opus 4.6 | Qwen3 32B | 97.28% | $6.68 |
| 7 | Claude Opus 4.6 | Claude Opus 4.6 | 97.24% | $6.97 |
| 8 | Claude Opus 4.6 | Kimi K2.5 | 97.24% | $6.58 |
| 9 | Claude Opus 4.6 | Claude 3 Haiku | 95.95% | $5.37 |
| 10 | gpt-oss-20b | Claude Opus 4.6 | 94.57% | $0.97 |
| 11 | gpt-oss-20b | Kimi K2.5 | 94.57% | $0.26 |
| 12 | gpt-oss-20b | gpt-oss-20b | 94.54% | $0.08 |
| 13 | Claude Haiku 4.5 | Qwen3 32B | 94.50% | $2.51 |
| 14 | gpt-oss-20b | Claude Haiku 4.5 | 94.05% | $0.37 |
| 15 | gpt-oss-20b | gpt-oss-120b | 94.02% | $0.11 |

### Bottom 15 Combos

| Rank | Answer Model | Critic Model | Accuracy | Cost |
|:-----|:-------------|:-------------|:---------|:-----|
| 67 | Qwen3 Next 80B A3B | Kimi K2.5 | 75.50% | $0.79 |
| 68 | Qwen3 Next 80B A3B | gpt-oss-20b | 75.00% | $0.48 |
| 69 | Kimi K2.5 | gpt-oss-120b | 74.49% | $0.95 |
| 70 | Kimi K2.5 | gpt-oss-20b | 74.09% | $0.77 |
| 71 | Kimi K2.5 | Kimi K2.5 | 73.58% | $1.34 |
| 72 | Kimi K2.5 | Claude Opus 4.6 | 73.33% | $2.79 |
| 73 | Kimi K2.5 | Claude Haiku 4.5 | 73.20% | $1.36 |
| 74 | Claude 3 Haiku | gpt-oss-120b | 72.19% | $0.32 |
| 75 | Kimi K2.5 | Qwen3 32B | 72.16% | $0.92 |
| 76 | Claude 3 Haiku | gpt-oss-20b | 71.43% | $0.32 |
| 77 | Claude 3 Haiku | Qwen3 Next 80B A3B | 71.07% | $0.39 |
| 78 | Claude 3 Haiku | Kimi K2.5 | 71.01% | $0.53 |
| 79 | Claude 3 Haiku | Ministral 3 8B | 69.28% | $0.32 |
| 80 | Claude 3 Haiku | Qwen3 32B | 59.30% | $0.29 |
| 81 | Claude 3 Haiku | Claude 3 Haiku | 54.37% | $0.30 |

??? note "Full 81 Combo Results"

    | Rank | Answer Model | Critic Model | Accuracy | Cost | Note |
    |:-----|:-------------|:-------------|:---------|:-----|:-----|
    | 1 | Claude Opus 4.6 | Claude Haiku 4.5 | 98.84% | $6.19 | |
    | 2 | Claude Opus 4.6 | Qwen3 Next 80B A3B | 98.82% | $5.77 | |
    | 3 | Claude Opus 4.6 | Ministral 3 8B | 98.72% | $5.26 | |
    | 4 | Claude Opus 4.6 | gpt-oss-20b | 98.28% | $5.93 | |
    | 5 | Claude Opus 4.6 | gpt-oss-120b | 97.77% | $6.30 | |
    | 6 | Claude Opus 4.6 | Qwen3 32B | 97.28% | $6.68 | |
    | 7 | Claude Opus 4.6 | Claude Opus 4.6 | 97.24% | $6.97 | |
    | 8 | Claude Opus 4.6 | Kimi K2.5 | 97.24% | $6.58 | |
    | 9 | Claude Opus 4.6 | Claude 3 Haiku | 95.95% | $5.37 | |
    | 10 | gpt-oss-20b | Claude Opus 4.6 | 94.57% | $0.97 | |
    | 11 | gpt-oss-20b | Kimi K2.5 | 94.57% | $0.26 | |
    | 12 | gpt-oss-20b | gpt-oss-20b | 94.54% | $0.08 | |
    | 13 | Claude Haiku 4.5 | Qwen3 32B | 94.50% | $2.51 | |
    | 14 | gpt-oss-20b | Claude Haiku 4.5 | 94.05% | $0.37 | |
    | 15 | gpt-oss-20b | gpt-oss-120b | 94.02% | $0.11 | |
    | 16 | gpt-oss-20b | Qwen3 Next 80B A3B | 94.02% | $0.14 | |
    | 17 | Claude Haiku 4.5 | Claude Haiku 4.5 | 94.00% | $2.59 | |
    | 18 | gpt-oss-20b | Ministral 3 8B | 93.99% | $0.10 | |
    | 19 | gpt-oss-120b | Claude Opus 4.6 | 93.81% | $1.25 | |
    | 20 | Claude Haiku 4.5 | gpt-oss-20b | 93.50% | $2.20 | |
    | 21 | Claude Haiku 4.5 | Claude Opus 4.6 | 93.50% | $3.77 | |
    | 22 | Claude Haiku 4.5 | Ministral 3 8B | 93.50% | $2.57 | |
    | 23 | Claude Haiku 4.5 | Kimi K2.5 | 93.50% | $2.60 | |
    | 24 | gpt-oss-20b | Qwen3 32B | 93.48% | $0.09 | |
    | 25 | gpt-oss-20b | Claude 3 Haiku | 93.44% | $0.15 | |
    | 26 | gpt-oss-120b | Ministral 3 8B | 93.26% | $0.19 | |
    | 27 | gpt-oss-120b | Qwen3 32B | 93.26% | $0.16 | |
    | 28 | Claude Haiku 4.5 | gpt-oss-120b | 93.00% | $2.90 | |
    | 29 | Claude Haiku 4.5 | Qwen3 Next 80B A3B | 93.00% | $7.81 | |
    | 30 | gpt-oss-120b | Claude Haiku 4.5 | 92.82% | $0.47 | |
    | 31 | gpt-oss-120b | gpt-oss-20b | 92.78% | $0.18 | |
    | 32 | gpt-oss-120b | gpt-oss-120b | 92.78% | $0.19 | |
    | 33 | gpt-oss-120b | Kimi K2.5 | 92.78% | $0.32 | |
    | 34 | gpt-oss-120b | Qwen3 Next 80B A3B | 92.78% | $0.23 | |
    | 35 | gpt-oss-120b | Claude 3 Haiku | 92.75% | $0.20 | |
    | 36 | Claude Haiku 4.5 | Claude 3 Haiku | 92.50% | $2.46 | |
    | 37 | Claude 3 Haiku | Claude Opus 4.6 | 89.66% | $2.26 | |
    | 38 | Qwen3 32B | Qwen3 Next 80B A3B | 88.83% | $0.24 | |
    | 39 | Ministral 3 8B | Claude 3 Haiku | 88.15% | $0.05 | |
    | 40 | Qwen3 32B | gpt-oss-120b | 87.83% | $0.47 | |
    | 41 | Ministral 3 8B | Qwen3 Next 80B A3B | 87.82% | $0.03 | |
    | 42 | Qwen3 32B | Claude Opus 4.6 | 87.56% | $3.43 | |
    | 43 | Ministral 3 8B | Kimi K2.5 | 87.04% | $0.09 | |
    | 44 | Ministral 3 8B | gpt-oss-120b | 86.63% | $0.07 | |
    | 45 | Claude 3 Haiku | Claude Haiku 4.5 | 86.55% | $0.69 | |
    | 46 | Ministral 3 8B | Ministral 3 8B | 86.52% | $0.03 | |
    | 47 | Ministral 3 8B | Claude Opus 4.6 | 86.47% | $0.93 | |
    | 48 | Qwen3 32B | Claude Haiku 4.5 | 86.46% | $0.90 | |
    | 49 | Ministral 3 8B | Claude Haiku 4.5 | 86.23% | $0.30 | |
    | 50 | Ministral 3 8B | gpt-oss-20b | 86.13% | $0.05 | |
    | 51 | Qwen3 32B | Ministral 3 8B | 86.10% | $0.21 | |
    | 52 | Qwen3 32B | Kimi K2.5 | 85.94% | $0.78 | |
    | 53 | Qwen3 32B | gpt-oss-20b | 85.86% | $0.49 | |
    | 54 | Ministral 3 8B | Qwen3 32B | 85.80% | $0.04 | |
    | 55 | Qwen3 32B | Qwen3 32B | 84.82% | $0.62 | |
    | 56 | Kimi K2.5 | Claude 3 Haiku | 80.41% | $0.98 | |
    | 57 | Qwen3 32B | Claude 3 Haiku | 80.00% | $0.67 | |
    | 58 | Qwen3 Next 80B A3B | Claude 3 Haiku | 80.00% | $0.59 | |
    | 59 | Qwen3 Next 80B A3B | Claude Opus 4.6 | 78.00% | $2.96 | |
    | 60 | Kimi K2.5 | Ministral 3 8B | 77.84% | $0.97 | |
    | 61 | Kimi K2.5 | Qwen3 Next 80B A3B | 77.20% | $1.00 | |
    | 62 | Qwen3 Next 80B A3B | Ministral 3 8B | 77.00% | $0.55 | |
    | 63 | Qwen3 Next 80B A3B | Claude Haiku 4.5 | 76.50% | $1.21 | |
    | 64 | Qwen3 Next 80B A3B | gpt-oss-120b | 76.50% | $0.52 | |
    | 65 | Qwen3 Next 80B A3B | Qwen3 32B | 76.00% | $0.42 | |
    | 66 | Qwen3 Next 80B A3B | Qwen3 Next 80B A3B | 76.00% | $0.54 | |
    | 67 | Qwen3 Next 80B A3B | Kimi K2.5 | 75.50% | $0.79 | |
    | 68 | Qwen3 Next 80B A3B | gpt-oss-20b | 75.00% | $0.48 | |
    | 69 | Kimi K2.5 | gpt-oss-120b | 74.49% | $0.95 | |
    | 70 | Kimi K2.5 | gpt-oss-20b | 74.09% | $0.77 | |
    | 71 | Kimi K2.5 | Kimi K2.5 | 73.58% | $1.34 | |
    | 72 | Kimi K2.5 | Claude Opus 4.6 | 73.33% | $2.79 | |
    | 73 | Kimi K2.5 | Claude Haiku 4.5 | 73.20% | $1.36 | |
    | 74 | Claude 3 Haiku | gpt-oss-120b | 72.19% | $0.32 | |
    | 75 | Kimi K2.5 | Qwen3 32B | 72.16% | $0.92 | |
    | 76 | Claude 3 Haiku | gpt-oss-20b | 71.43% | $0.32 | |
    | 77 | Claude 3 Haiku | Qwen3 Next 80B A3B | 71.07% | $0.39 | |
    | 78 | Claude 3 Haiku | Kimi K2.5 | 71.01% | $0.53 | |
    | 79 | Claude 3 Haiku | Ministral 3 8B | 69.28% | $0.32 | |
    | 80 | Claude 3 Haiku | Qwen3 32B | 59.30% | $0.29 | |
    | 81 | Claude 3 Haiku | Claude 3 Haiku | 54.37% | $0.30 | |

### Selector Comparison

| Selector | Find Rate | Mean Accuracy | Evaluations | Cost | Savings |
|:---------|:----------|:--------------|:------------|:-----|:--------|
| Brute Force | 100% | 98.84% | 14,961 | $123.87 | -- |
| Arm Elimination | 86% | 98.83% | 3,356 | $51.86 | **58%** |
| Hill Climbing | 80% | 98.76% | 3,926 | $54.22 | 56% |
| Random Search | 28% | 98.17% | 3,880 | $31.77 | 74% |
| Epsilon LUCB | 4% | 96.99% | 447 | $6.10 | 95% |
| Bayesian Opt | 4% | 95.41% | 3,666 | $35.56 | 71% |
| LM Proposal | 0% | 95.82% | 158 | $5.61 | 95% |
| Threshold SE | 0% | 74.52% | 1,355 | $6.90 | 94% |
