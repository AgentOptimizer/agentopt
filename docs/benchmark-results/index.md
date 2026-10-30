# Benchmark Results

HotpotQA and MathQA evaluate **10 candidates in each of two roles**, giving **100 model combinations and 20,000 configuration–question outcomes per benchmark**. Their tables report complete 200-question means for quality, deployment cost, and server-reported LLM latency.

GPQA Diamond and BFCL are separate single-role benchmarks with nine candidates. Their model tables report mean server LLM latency and invocation cost per complete sample; their selector comparisons use 50 random seeds. Measurement definitions and fixed data-source links accompany the benchmark sections below.

## Models

The table lists the token rates used for cost accounting in the recorded experiments, in USD per million input or output tokens. These are experiment accounting rates, not a live pricing quote. The **All four** rows appear in both candidate sets; DeepSeek R1 and GPT-5.6 Luna are used by HotpotQA/MathQA, while Claude 3 Haiku is used by GPQA/BFCL.

| Model | Provider | Input $/MTok | Output $/MTok | Benchmarks |
|:------|:---------|-------------:|--------------:|:-----------|
| Claude Opus 4.6 | Anthropic | $5.00 | $25.00 | All four |
| Claude Haiku 4.5 | Anthropic | $1.00 | $5.00 | All four |
| Claude 3 Haiku | Anthropic | $0.25 | $1.25 | GPQA / BFCL |
| gpt-oss-120b | OpenAI | $0.15 | $0.60 | All four |
| gpt-oss-20b | OpenAI | $0.07 | $0.30 | All four |
| Kimi K2.5 | Moonshot AI | $0.60 | $3.00 | All four |
| Qwen3 Next 80B A3B | Qwen | $0.15 | $1.20 | All four |
| Qwen3 32B | Qwen | $0.15 | $0.60 | All four |
| Ministral 3 8B | Mistral | $0.15 | $0.15 | All four |
| DeepSeek R1 | DeepSeek | $1.35 | $5.40 | HotpotQA / MathQA |
| GPT-5.6 Luna | OpenAI | $0.22 | $1.32 | HotpotQA / MathQA |

## Cross-Benchmark Summary

| Benchmark | Roles | Questions | Configurations | Best assignment | Best mean score | Full-matrix evaluation cost (USD) |
|:----------|------:|----------:|---------------:|:----------------|----------------:|-----------------:|
| GPQA Diamond | 1 | 198 | 9 | Claude Opus 4.6 | 74.75% accuracy | $4.71 |
| BFCL Multi-Turn | 1 | 200 | 9 | Kimi K2.5, Claude Opus 4.6, Qwen3 Next 80B A3B (tied) | 70.00% accuracy | $84.80 |
| HotpotQA | 2 | 200 | 100 | Qwen3 32B / DeepSeek R1 | 74.66% F1 | $62.58 |
| MathQA | 2 | 200 | 100 | Claude Opus 4.6 / gpt-oss-20b | 97.50% accuracy | $234.34 |

**Full-matrix evaluation cost** is the sum of the archived per-configuration benchmark totals. For GPQA and BFCL these totals are $4.7103 and $84.8049, respectively, displayed to cents above. The HotpotQA and MathQA assignment names are ordered by their first and second workflow roles.

---

## GPQA and BFCL Data and Measurements

The following **Model Results** tables describe one complete sample: a GPQA question or a BFCL multi-turn task. The [GPQA source CSV](https://github.com/AgentOptimizer/agentopt/blob/b87e652b6a0c90e31daeb47054ecca615f25acee/cached_results/gpqa/brute_force_results.csv) and [BFCL source CSV](https://github.com/AgentOptimizer/agentopt/blob/b87e652b6a0c90e31daeb47054ecca615f25acee/cached_results/bfcl/brute_force_results.csv) preserve the model-level server latency means and benchmark cost totals at a fixed data revision.

- **Mean server LLM latency:** sum the recorded Bedrock `metrics.latencyMs` values across LLM calls within each sample, convert milliseconds to seconds, then average across samples with recorded server metrics. For BFCL, the sum spans recorded LLM calls across the multi-turn task; it is not a per-call or per-turn mean. This measure excludes client processing, network transfer, tool execution, and worker-queue time. Missing server metrics are excluded rather than counted as zero. The retained BFCL records provide server totals for all 200 samples per model. GPQA uses the archived model means; its retained replay lookup does not preserve the per-sample server metrics needed to independently audit their coverage.
- **Mean cost:** divide the archived per-model benchmark total by 198 GPQA questions or 200 BFCL samples. Cost accounts for LLM invocation token usage at the recorded experiment prices; it is not a provider invoice or the cost of running tools and other infrastructure. The source CSV's `Cost ($)` is a benchmark total, not a per-sample amount. Calculations retain the source totals at their recorded precision; per-sample means are rounded to six decimal places only for display. Full-matrix totals are summed directly from the archived totals.
- **Mean search cost:** the selector tables report the total evaluation cost incurred during model selection, averaged over 50 seeds, in USD per search run. Savings are relative to exhaustive evaluation of the same benchmark matrix. This is distinct from the cost of serving one sample with the selected model.

The retained GPQA replay lookup assigns each question its model's mean cost and has no per-question token records; it cannot independently recover historical question-level cost variation. BFCL retains question-level cost and token records. These provenance limits apply to the archived GPQA/BFCL model results and replay data.

---

## GPQA Diamond

**Graduate-level science QA** — 198 multiple-choice questions from the GPQA Diamond dataset. Single-agent architecture: one LLM answers directly.

### Model Results

| Rank | Model | Accuracy | Mean server LLM latency (s/sample) | Mean cost (USD/sample) |
|:-----|:------|:---------|----------------------------------:|-----------------------:|
| 1 | Claude Opus 4.6 | **74.75%** | 9.16 | $0.012471 |
| 2 | Kimi K2.5 | 72.73% | 16.41 | $0.005730 |
| 3 | gpt-oss-120b | 68.18% | 6.46 | $0.000960 |
| 4 | Claude Haiku 4.5 | 59.60% | 3.70 | $0.002611 |
| 5 | Qwen3 Next 80B A3B | 51.01% | 10.33 | $0.000671 |
| 6 | gpt-oss-20b | 50.00% | 6.21 | $0.000658 |
| 7 | Qwen3 32B | 46.97% | 1.54 | $0.000370 |
| 8 | Ministral 3 8B | 36.87% | 0.25 | $0.000036 |
| 9 | Claude 3 Haiku | 34.85% | 1.79 | $0.000283 |

### Selector Comparison

| Selector | Find Rate | Mean Accuracy | Evaluations | Mean search cost (USD/run) | Savings |
|:---------|:----------|:--------------|:------------|---------------------------:|:--------|
| Brute Force | 100% | 74.75% | 1,782 | $4.71 | -- |
| Arm Elimination | 94% | 74.10% | 666 | $3.57 | **24%** |
| Bayesian Opt | 56% | 72.43% | 990 | $2.59 | 45% |

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

| Rank | Model | Accuracy | Mean server LLM latency (s/sample) | Mean cost (USD/sample) |
|:-----|:------|:---------|----------------------------------:|-----------------------:|
| 1 | Kimi K2.5 | **70.00%** | 21.30 | $0.019256 |
| 2 | Claude Opus 4.6 | 70.00% | 42.35 | $0.300663 |
| 3 | Qwen3 Next 80B A3B | 70.00% | 60.54 | $0.009481 |
| 4 | Claude Haiku 4.5 | 65.00% | 20.90 | $0.059920 |
| 5 | gpt-oss-120b | 58.50% | 20.01 | $0.005804 |
| 6 | Qwen3 32B | 47.00% | 10.78 | $0.005028 |
| 7 | Claude 3 Haiku | 43.50% | 17.96 | $0.017096 |
| 8 | gpt-oss-20b | 42.00% | 10.03 | $0.002145 |
| 9 | Ministral 3 8B | 34.00% | 29.03 | $0.004633 |

### Selector Comparison

| Selector | Find Rate | Mean Accuracy | Evaluations | Mean search cost (USD/run) | Savings |
|:---------|:----------|:--------------|:------------|---------------------------:|:--------|
| Brute Force | 100% | 70.00% | 1,800 | $84.80 | -- |
| Arm Elimination | 88% | 69.37% | 912 | $74.39 | **12%** |
| Bayesian Opt | 44% | 69.27% | 1,000 | $50.64 | 40% |

---

## HotpotQA and MathQA Data and Measurements

Each linked CSV contains 20,000 rows keyed by `configuration_id` and `question_id`, with scores, token usage, invocation cost, `latency_seconds`, and `server_latency_ms`. These are per-question workflow summaries, rather than per-call transcripts.

- [HotpotQA complete 10×10 CSV](https://github.com/AgentOptimizer/agentopt/blob/fb21fbbb29d757a38ca314e0df7b25a3c8f335ce/experiments/missing_cells/completed/hotpotqa_lookup.10x10.csv) — 100 configurations × 200 questions.
- [MathQA complete 10×10 CSV](https://github.com/AgentOptimizer/agentopt/blob/fb21fbbb29d757a38ca314e0df7b25a3c8f335ce/experiments/missing_cells/completed/mathqa_lookup.10x10.csv) — 100 configurations × 200 questions.

The links pin a specific data commit so the published tables can be reproduced.

- **Quality:** HotpotQA reports mean token-level F1; MathQA reports the proportion of correct multiple-choice answers. Every configuration uses the same 200 questions, including zero-score outcomes.
- **Deployment cost:** mean recorded invocation cost per question, calculated from model-specific input and output token usage. It is an accounting measure, not an amount read from a provider bill. The full-matrix costs above sum all 20,000 outcomes.
- **Server latency:** each workflow/question's `server_latency_ms` field sums the server-reported `metrics.latencyMs` values of successful LLM calls. The tables divide this field by 1,000 to obtain seconds, then average over the 200 questions. This measure excludes tool execution, client processing, network transfer, and worker queue time. It is distinct from the CSV's `latency_seconds` field and the package's runtime workflow timer.

The tables rank configurations by full-benchmark mean quality, breaking exact ties by lower deployment cost, then lower server latency. All costs in the following configuration tables are **USD per question**.

---

## HotpotQA

**Multi-hop question answering** — 200 questions from the HotpotQA distractor setting. A planner proposes search steps and can delegate to a solver with tool access. Ten candidates per role give **100 model combinations**. Quality is mean token-level F1 over all 200 questions, displayed as a percentage.

### Top 15 Combos

| Rank | Planner | Solver | Mean F1 | Server latency (s/query) | Cost (USD/query) |
|:-----|:------|:-------|:--------|-------------------------:|-----------------:|
| 1 | Qwen3 32B | DeepSeek R1 | 74.66% | 3.00 | $0.004038 |
| 2 | Ministral 3 8B | Claude Opus 4.6 | 74.27% | 4.97 | $0.013209 |
| 3 | Qwen3 32B | Claude Opus 4.6 | 73.02% | 4.26 | $0.013245 |
| 4 | Qwen3 Next 80B A3B | Claude Opus 4.6 | 72.10% | 4.67 | $0.013348 |
| 5 | Qwen3 Next 80B A3B | DeepSeek R1 | 72.00% | 3.11 | $0.003867 |
| 6 | Qwen3 Next 80B A3B | gpt-oss-120b | 71.83% | 3.07 | $0.000639 |
| 7 | Kimi K2.5 | DeepSeek R1 | 71.25% | 3.73 | $0.004061 |
| 8 | Ministral 3 8B | DeepSeek R1 | 70.91% | 3.41 | $0.004304 |
| 9 | Qwen3 32B | gpt-oss-120b | 70.04% | 2.66 | $0.000630 |
| 10 | Kimi K2.5 | Claude Opus 4.6 | 69.96% | 4.49 | $0.012142 |
| 11 | gpt-oss-120b | DeepSeek R1 | 69.60% | 2.40 | $0.002414 |
| 12 | Ministral 3 8B | gpt-oss-20b | 68.99% | 5.64 | $0.000441 |
| 13 | Qwen3 Next 80B A3B | GPT-5.6 Luna | 68.89% | 13.52 | $0.000330 |
| 14 | Ministral 3 8B | gpt-oss-120b | 68.70% | 3.65 | $0.000615 |
| 15 | Qwen3 32B | GPT-5.6 Luna | 68.61% | 27.14 | $0.000311 |

### Bottom 15 Combos

| Rank | Planner | Solver | Mean F1 | Server latency (s/query) | Cost (USD/query) |
|:-----|:------|:-------|:--------|-------------------------:|-----------------:|
| 86 | Claude Haiku 4.5 | Qwen3 32B | 36.13% | 2.89 | $0.002297 |
| 87 | Ministral 3 8B | Claude Haiku 4.5 | 32.42% | 4.14 | $0.003514 |
| 88 | Qwen3 Next 80B A3B | Claude Haiku 4.5 | 32.19% | 3.92 | $0.003614 |
| 89 | Claude Opus 4.6 | Kimi K2.5 | 31.96% | 4.72 | $0.010119 |
| 90 | Claude Opus 4.6 | Ministral 3 8B | 31.96% | 4.72 | $0.010119 |
| 91 | Claude Opus 4.6 | Qwen3 32B | 31.96% | 4.72 | $0.010119 |
| 92 | Claude Opus 4.6 | Qwen3 Next 80B A3B | 31.96% | 4.72 | $0.010119 |
| 93 | Claude Opus 4.6 | gpt-oss-120b | 31.95% | 4.60 | $0.010105 |
| 94 | Claude Opus 4.6 | gpt-oss-20b | 31.88% | 4.57 | $0.010136 |
| 95 | Claude Opus 4.6 | Claude Haiku 4.5 | 31.77% | 4.16 | $0.010151 |
| 96 | Claude Opus 4.6 | Claude Opus 4.6 | 31.71% | 4.19 | $0.010111 |
| 97 | Claude Opus 4.6 | DeepSeek R1 | 31.22% | 2.60 | $0.010142 |
| 98 | Claude Opus 4.6 | GPT-5.6 Luna | 31.22% | 2.60 | $0.010142 |
| 99 | Qwen3 32B | Claude Haiku 4.5 | 26.63% | 3.47 | $0.003471 |
| 100 | Claude Haiku 4.5 | Claude Haiku 4.5 | 26.49% | 3.40 | $0.003954 |

!!! note "Role behavior matters"
    Claude Opus 4.6 planner combinations score 31.22–31.96% mean F1 in this workflow. The planner can terminate with a direct answer instead of delegating to the solver, so model capability alone does not determine the best role assignment. The highest-scoring assignment here uses Qwen3 32B as planner and DeepSeek R1 as solver (74.66% mean F1).

??? note "Full 100 Combo Results"

    | Rank | Planner | Solver | Mean F1 | Server latency (s/query) | Cost (USD/query) |
    |:-----|:------|:-------|:--------|-------------------------:|-----------------:|
    | 1 | Qwen3 32B | DeepSeek R1 | 74.66% | 3.00 | $0.004038 |
    | 2 | Ministral 3 8B | Claude Opus 4.6 | 74.27% | 4.97 | $0.013209 |
    | 3 | Qwen3 32B | Claude Opus 4.6 | 73.02% | 4.26 | $0.013245 |
    | 4 | Qwen3 Next 80B A3B | Claude Opus 4.6 | 72.10% | 4.67 | $0.013348 |
    | 5 | Qwen3 Next 80B A3B | DeepSeek R1 | 72.00% | 3.11 | $0.003867 |
    | 6 | Qwen3 Next 80B A3B | gpt-oss-120b | 71.83% | 3.07 | $0.000639 |
    | 7 | Kimi K2.5 | DeepSeek R1 | 71.25% | 3.73 | $0.004061 |
    | 8 | Ministral 3 8B | DeepSeek R1 | 70.91% | 3.41 | $0.004304 |
    | 9 | Qwen3 32B | gpt-oss-120b | 70.04% | 2.66 | $0.000630 |
    | 10 | Kimi K2.5 | Claude Opus 4.6 | 69.96% | 4.49 | $0.012142 |
    | 11 | gpt-oss-120b | DeepSeek R1 | 69.60% | 2.40 | $0.002414 |
    | 12 | Ministral 3 8B | gpt-oss-20b | 68.99% | 5.64 | $0.000441 |
    | 13 | Qwen3 Next 80B A3B | GPT-5.6 Luna | 68.89% | 13.52 | $0.000330 |
    | 14 | Ministral 3 8B | gpt-oss-120b | 68.70% | 3.65 | $0.000615 |
    | 15 | Qwen3 32B | GPT-5.6 Luna | 68.61% | 27.14 | $0.000311 |
    | 16 | Qwen3 Next 80B A3B | gpt-oss-20b | 68.37% | 2.83 | $0.000477 |
    | 17 | Qwen3 Next 80B A3B | Qwen3 Next 80B A3B | 68.15% | 2.69 | $0.000570 |
    | 18 | Ministral 3 8B | Qwen3 Next 80B A3B | 67.98% | 3.85 | $0.000556 |
    | 19 | Qwen3 32B | Qwen3 Next 80B A3B | 67.53% | 3.51 | $0.000541 |
    | 20 | Ministral 3 8B | GPT-5.6 Luna | 67.48% | 50.17 | $0.000307 |
    | 21 | Kimi K2.5 | GPT-5.6 Luna | 66.79% | 16.59 | $0.001192 |
    | 22 | Qwen3 32B | gpt-oss-20b | 66.48% | 2.47 | $0.000447 |
    | 23 | Ministral 3 8B | Kimi K2.5 | 65.42% | 3.26 | $0.001298 |
    | 24 | gpt-oss-120b | Qwen3 Next 80B A3B | 64.93% | 4.68 | $0.000483 |
    | 25 | Ministral 3 8B | Ministral 3 8B | 64.89% | 3.55 | $0.000453 |
    | 26 | Kimi K2.5 | gpt-oss-120b | 64.70% | 4.16 | $0.001465 |
    | 27 | gpt-oss-120b | Claude Opus 4.6 | 64.59% | 4.57 | $0.008047 |
    | 28 | gpt-oss-120b | Claude Haiku 4.5 | 64.11% | 4.26 | $0.001883 |
    | 29 | Kimi K2.5 | Qwen3 Next 80B A3B | 63.99% | 4.39 | $0.001485 |
    | 30 | Kimi K2.5 | Ministral 3 8B | 63.95% | 6.42 | $0.001412 |
    | 31 | Kimi K2.5 | gpt-oss-20b | 63.75% | 6.62 | $0.001344 |
    | 32 | Qwen3 Next 80B A3B | Kimi K2.5 | 63.69% | 2.89 | $0.001329 |
    | 33 | gpt-oss-120b | GPT-5.6 Luna | 63.68% | 15.68 | $0.000339 |
    | 34 | gpt-oss-120b | Ministral 3 8B | 63.38% | 7.34 | $0.000441 |
    | 35 | Qwen3 32B | Kimi K2.5 | 63.36% | 3.25 | $0.001420 |
    | 36 | Kimi K2.5 | Kimi K2.5 | 62.46% | 4.55 | $0.002196 |
    | 37 | gpt-oss-120b | gpt-oss-20b | 62.18% | 2.66 | $0.000415 |
    | 38 | gpt-oss-120b | gpt-oss-120b | 62.15% | 4.59 | $0.000511 |
    | 39 | Qwen3 Next 80B A3B | Ministral 3 8B | 61.80% | 4.25 | $0.000487 |
    | 40 | Qwen3 32B | Ministral 3 8B | 61.17% | 2.89 | $0.000459 |
    | 41 | gpt-oss-120b | Kimi K2.5 | 60.85% | 4.09 | $0.000891 |
    | 42 | gpt-oss-120b | Qwen3 32B | 58.80% | 4.06 | $0.000483 |
    | 43 | gpt-oss-20b | DeepSeek R1 | 57.10% | 2.22 | $0.001698 |
    | 44 | gpt-oss-20b | Claude Opus 4.6 | 55.86% | 2.84 | $0.005212 |
    | 45 | Ministral 3 8B | Qwen3 32B | 55.02% | 3.63 | $0.000535 |
    | 46 | Qwen3 32B | Qwen3 32B | 54.82% | 2.53 | $0.000541 |
    | 47 | Kimi K2.5 | Qwen3 32B | 54.73% | 4.57 | $0.001478 |
    | 48 | gpt-oss-20b | Claude Haiku 4.5 | 54.28% | 2.19 | $0.001299 |
    | 49 | gpt-oss-20b | Ministral 3 8B | 54.25% | 4.35 | $0.000265 |
    | 50 | Qwen3 Next 80B A3B | Qwen3 32B | 54.13% | 2.83 | $0.000570 |
    | 51 | gpt-oss-20b | Qwen3 Next 80B A3B | 53.89% | 2.11 | $0.000296 |
    | 52 | gpt-oss-20b | GPT-5.6 Luna | 53.81% | 3.92 | $0.000193 |
    | 53 | gpt-oss-20b | gpt-oss-120b | 52.17% | 2.11 | $0.000315 |
    | 54 | gpt-oss-20b | Kimi K2.5 | 51.01% | 1.96 | $0.000586 |
    | 55 | gpt-oss-20b | gpt-oss-20b | 50.09% | 2.12 | $0.000252 |
    | 56 | gpt-oss-20b | Qwen3 32B | 49.16% | 2.05 | $0.000296 |
    | 57 | GPT-5.6 Luna | DeepSeek R1 | 47.19% | 7.89 | $0.000192 |
    | 58 | GPT-5.6 Luna | Qwen3 32B | 47.17% | 7.85 | $0.000120 |
    | 59 | GPT-5.6 Luna | Qwen3 Next 80B A3B | 47.10% | 7.85 | $0.000120 |
    | 60 | Claude Haiku 4.5 | GPT-5.6 Luna | 46.73% | 2.61 | $0.002180 |
    | 61 | GPT-5.6 Luna | gpt-oss-120b | 46.71% | 7.86 | $0.000121 |
    | 62 | GPT-5.6 Luna | Ministral 3 8B | 46.71% | 7.84 | $0.000118 |
    | 63 | GPT-5.6 Luna | gpt-oss-20b | 46.70% | 7.86 | $0.000117 |
    | 64 | GPT-5.6 Luna | Kimi K2.5 | 46.68% | 7.88 | $0.000139 |
    | 65 | GPT-5.6 Luna | Claude Opus 4.6 | 46.63% | 7.87 | $0.000430 |
    | 66 | GPT-5.6 Luna | Claude Haiku 4.5 | 46.58% | 7.86 | $0.000178 |
    | 67 | GPT-5.6 Luna | GPT-5.6 Luna | 46.26% | 7.92 | $0.000114 |
    | 68 | Claude Haiku 4.5 | DeepSeek R1 | 44.49% | 3.54 | $0.004705 |
    | 69 | Claude Haiku 4.5 | Claude Opus 4.6 | 43.54% | 4.06 | $0.008998 |
    | 70 | DeepSeek R1 | Claude Haiku 4.5 | 43.15% | 3.69 | $0.004293 |
    | 71 | DeepSeek R1 | Claude Opus 4.6 | 43.15% | 3.69 | $0.004293 |
    | 72 | DeepSeek R1 | DeepSeek R1 | 43.15% | 3.69 | $0.004293 |
    | 73 | DeepSeek R1 | GPT-5.6 Luna | 43.15% | 3.69 | $0.004293 |
    | 74 | DeepSeek R1 | Kimi K2.5 | 43.15% | 3.69 | $0.004293 |
    | 75 | DeepSeek R1 | Ministral 3 8B | 43.15% | 3.69 | $0.004293 |
    | 76 | DeepSeek R1 | Qwen3 32B | 43.15% | 3.69 | $0.004293 |
    | 77 | DeepSeek R1 | Qwen3 Next 80B A3B | 43.15% | 3.69 | $0.004293 |
    | 78 | DeepSeek R1 | gpt-oss-120b | 43.15% | 3.69 | $0.004293 |
    | 79 | DeepSeek R1 | gpt-oss-20b | 43.15% | 3.69 | $0.004293 |
    | 80 | Claude Haiku 4.5 | gpt-oss-20b | 41.49% | 3.03 | $0.002263 |
    | 81 | Claude Haiku 4.5 | gpt-oss-120b | 41.20% | 3.14 | $0.002352 |
    | 82 | Claude Haiku 4.5 | Qwen3 Next 80B A3B | 41.17% | 2.95 | $0.002306 |
    | 83 | Claude Haiku 4.5 | Ministral 3 8B | 41.09% | 3.75 | $0.002257 |
    | 84 | Claude Haiku 4.5 | Kimi K2.5 | 41.00% | 6.16 | $0.002716 |
    | 85 | Kimi K2.5 | Claude Haiku 4.5 | 37.19% | 4.23 | $0.004381 |
    | 86 | Claude Haiku 4.5 | Qwen3 32B | 36.13% | 2.89 | $0.002297 |
    | 87 | Ministral 3 8B | Claude Haiku 4.5 | 32.42% | 4.14 | $0.003514 |
    | 88 | Qwen3 Next 80B A3B | Claude Haiku 4.5 | 32.19% | 3.92 | $0.003614 |
    | 89 | Claude Opus 4.6 | Kimi K2.5 | 31.96% | 4.72 | $0.010119 |
    | 90 | Claude Opus 4.6 | Ministral 3 8B | 31.96% | 4.72 | $0.010119 |
    | 91 | Claude Opus 4.6 | Qwen3 32B | 31.96% | 4.72 | $0.010119 |
    | 92 | Claude Opus 4.6 | Qwen3 Next 80B A3B | 31.96% | 4.72 | $0.010119 |
    | 93 | Claude Opus 4.6 | gpt-oss-120b | 31.95% | 4.60 | $0.010105 |
    | 94 | Claude Opus 4.6 | gpt-oss-20b | 31.88% | 4.57 | $0.010136 |
    | 95 | Claude Opus 4.6 | Claude Haiku 4.5 | 31.77% | 4.16 | $0.010151 |
    | 96 | Claude Opus 4.6 | Claude Opus 4.6 | 31.71% | 4.19 | $0.010111 |
    | 97 | Claude Opus 4.6 | DeepSeek R1 | 31.22% | 2.60 | $0.010142 |
    | 98 | Claude Opus 4.6 | GPT-5.6 Luna | 31.22% | 2.60 | $0.010142 |
    | 99 | Qwen3 32B | Claude Haiku 4.5 | 26.63% | 3.47 | $0.003471 |
    | 100 | Claude Haiku 4.5 | Claude Haiku 4.5 | 26.49% | 3.40 | $0.003954 |

---

## MathQA

**Self-reflective math reasoning** — 200 multiple-choice questions. An answer model solves a problem and a critic model checks the answer; rejected answers can be revised for up to three reflection iterations. Ten candidates per role give **100 model combinations**. Accuracy is the fraction of correct answers over all 200 questions.

### Top 15 Combos

| Rank | Answer model | Critic model | Accuracy | Server latency (s/query) | Cost (USD/query) |
|:-----|:------|:-------|:--------|-------------------------:|-----------------:|
| 1 | Claude Opus 4.6 | gpt-oss-20b | 97.50% | 17.09 | $0.042219 |
| 2 | Claude Opus 4.6 | Kimi K2.5 | 97.00% | 18.39 | $0.040009 |
| 3 | Claude Opus 4.6 | Claude Opus 4.6 | 97.00% | 16.43 | $0.045252 |
| 4 | Claude Opus 4.6 | Qwen3 Next 80B A3B | 97.00% | 15.82 | $0.047379 |
| 5 | Claude Opus 4.6 | Claude Haiku 4.5 | 97.00% | 17.39 | $0.050782 |
| 6 | Claude Opus 4.6 | Qwen3 32B | 96.50% | 15.25 | $0.040012 |
| 7 | Claude Opus 4.6 | gpt-oss-120b | 96.50% | 16.27 | $0.043768 |
| 8 | Claude Opus 4.6 | Ministral 3 8B | 96.00% | 15.91 | $0.053363 |
| 9 | gpt-oss-20b | Claude Opus 4.6 | 95.00% | 7.20 | $0.005698 |
| 10 | Claude Haiku 4.5 | Qwen3 32B | 94.50% | 12.68 | $0.012551 |
| 11 | DeepSeek R1 | Claude Opus 4.6 | 94.50% | 20.17 | $0.021146 |
| 12 | gpt-oss-20b | gpt-oss-20b | 94.00% | 4.23 | $0.000468 |
| 13 | gpt-oss-20b | Ministral 3 8B | 94.00% | 8.51 | $0.000601 |
| 14 | gpt-oss-20b | Kimi K2.5 | 94.00% | 12.17 | $0.001455 |
| 15 | gpt-oss-120b | Claude Opus 4.6 | 94.00% | 8.94 | $0.006331 |

### Bottom 15 Combos

| Rank | Answer model | Critic model | Accuracy | Server latency (s/query) | Cost (USD/query) |
|:-----|:------|:-------|:--------|-------------------------:|-----------------:|
| 86 | Qwen3 Next 80B A3B | Ministral 3 8B | 77.00% | 38.55 | $0.002727 |
| 87 | Qwen3 Next 80B A3B | GPT-5.6 Luna | 76.50% | 24.09 | $0.002471 |
| 88 | Qwen3 Next 80B A3B | gpt-oss-120b | 76.50% | 34.72 | $0.002606 |
| 89 | Qwen3 Next 80B A3B | Claude Haiku 4.5 | 76.50% | 32.33 | $0.006027 |
| 90 | Qwen3 Next 80B A3B | DeepSeek R1 | 76.50% | 25.45 | $0.009190 |
| 91 | Qwen3 Next 80B A3B | Qwen3 32B | 76.00% | 30.64 | $0.002088 |
| 92 | Qwen3 Next 80B A3B | Qwen3 Next 80B A3B | 76.00% | 36.44 | $0.002721 |
| 93 | Qwen3 Next 80B A3B | Kimi K2.5 | 75.50% | 36.37 | $0.003959 |
| 94 | Qwen3 Next 80B A3B | gpt-oss-20b | 75.00% | 32.70 | $0.002391 |
| 95 | Kimi K2.5 | gpt-oss-20b | 75.00% | 25.00 | $0.003928 |
| 96 | Kimi K2.5 | gpt-oss-120b | 75.00% | 31.74 | $0.004762 |
| 97 | Kimi K2.5 | Kimi K2.5 | 74.50% | 43.05 | $0.006772 |
| 98 | Kimi K2.5 | Claude Haiku 4.5 | 74.00% | 26.39 | $0.006879 |
| 99 | Kimi K2.5 | Claude Opus 4.6 | 74.00% | 28.10 | $0.014059 |
| 100 | Kimi K2.5 | Qwen3 32B | 73.00% | 29.60 | $0.004667 |

??? note "Full 100 Combo Results"

    | Rank | Answer model | Critic model | Accuracy | Server latency (s/query) | Cost (USD/query) |
    |:-----|:------|:-------|:--------|-------------------------:|-----------------:|
    | 1 | Claude Opus 4.6 | gpt-oss-20b | 97.50% | 17.09 | $0.042219 |
    | 2 | Claude Opus 4.6 | Kimi K2.5 | 97.00% | 18.39 | $0.040009 |
    | 3 | Claude Opus 4.6 | Claude Opus 4.6 | 97.00% | 16.43 | $0.045252 |
    | 4 | Claude Opus 4.6 | Qwen3 Next 80B A3B | 97.00% | 15.82 | $0.047379 |
    | 5 | Claude Opus 4.6 | Claude Haiku 4.5 | 97.00% | 17.39 | $0.050782 |
    | 6 | Claude Opus 4.6 | Qwen3 32B | 96.50% | 15.25 | $0.040012 |
    | 7 | Claude Opus 4.6 | gpt-oss-120b | 96.50% | 16.27 | $0.043768 |
    | 8 | Claude Opus 4.6 | Ministral 3 8B | 96.00% | 15.91 | $0.053363 |
    | 9 | gpt-oss-20b | Claude Opus 4.6 | 95.00% | 7.20 | $0.005698 |
    | 10 | Claude Haiku 4.5 | Qwen3 32B | 94.50% | 12.68 | $0.012551 |
    | 11 | DeepSeek R1 | Claude Opus 4.6 | 94.50% | 20.17 | $0.021146 |
    | 12 | gpt-oss-20b | gpt-oss-20b | 94.00% | 4.23 | $0.000468 |
    | 13 | gpt-oss-20b | Ministral 3 8B | 94.00% | 8.51 | $0.000601 |
    | 14 | gpt-oss-20b | Kimi K2.5 | 94.00% | 12.17 | $0.001455 |
    | 15 | gpt-oss-120b | Claude Opus 4.6 | 94.00% | 8.94 | $0.006331 |
    | 16 | Claude Haiku 4.5 | Claude Haiku 4.5 | 94.00% | 14.31 | $0.012960 |
    | 17 | DeepSeek R1 | gpt-oss-120b | 94.00% | 21.68 | $0.018156 |
    | 18 | DeepSeek R1 | gpt-oss-20b | 94.00% | 21.71 | $0.018169 |
    | 19 | DeepSeek R1 | Kimi K2.5 | 94.00% | 22.77 | $0.022307 |
    | 20 | gpt-oss-20b | gpt-oss-120b | 93.50% | 5.08 | $0.000619 |
    | 21 | gpt-oss-20b | Qwen3 Next 80B A3B | 93.50% | 9.05 | $0.000823 |
    | 22 | gpt-oss-120b | Qwen3 32B | 93.50% | 5.42 | $0.000838 |
    | 23 | gpt-oss-120b | Ministral 3 8B | 93.50% | 10.26 | $0.000958 |
    | 24 | gpt-oss-20b | Claude Haiku 4.5 | 93.50% | 6.31 | $0.002013 |
    | 25 | Claude Haiku 4.5 | gpt-oss-20b | 93.50% | 12.51 | $0.011006 |
    | 26 | Claude Haiku 4.5 | Ministral 3 8B | 93.50% | 14.70 | $0.012867 |
    | 27 | Claude Haiku 4.5 | Kimi K2.5 | 93.50% | 17.50 | $0.012988 |
    | 28 | DeepSeek R1 | Qwen3 Next 80B A3B | 93.50% | 22.86 | $0.017363 |
    | 29 | Claude Haiku 4.5 | Claude Opus 4.6 | 93.50% | 15.82 | $0.018838 |
    | 30 | DeepSeek R1 | Ministral 3 8B | 93.50% | 23.57 | $0.020497 |
    | 31 | gpt-oss-20b | Qwen3 32B | 93.00% | 4.75 | $0.000535 |
    | 32 | gpt-oss-120b | gpt-oss-20b | 93.00% | 6.36 | $0.000892 |
    | 33 | GPT-5.6 Luna | gpt-oss-120b | 93.00% | 5.84 | $0.000944 |
    | 34 | gpt-oss-120b | gpt-oss-120b | 93.00% | 6.84 | $0.000976 |
    | 35 | gpt-oss-120b | Qwen3 Next 80B A3B | 93.00% | 10.77 | $0.001146 |
    | 36 | gpt-oss-120b | Kimi K2.5 | 93.00% | 11.84 | $0.001647 |
    | 37 | gpt-oss-120b | Claude Haiku 4.5 | 93.00% | 7.67 | $0.002369 |
    | 38 | GPT-5.6 Luna | Claude Opus 4.6 | 93.00% | 6.28 | $0.006081 |
    | 39 | Claude Haiku 4.5 | gpt-oss-120b | 93.00% | 14.65 | $0.014505 |
    | 40 | DeepSeek R1 | GPT-5.6 Luna | 93.00% | 28.99 | $0.019476 |
    | 41 | DeepSeek R1 | DeepSeek R1 | 93.00% | 27.78 | $0.024207 |
    | 42 | Claude Haiku 4.5 | DeepSeek R1 | 93.00% | 15.57 | $0.035559 |
    | 43 | Claude Haiku 4.5 | Qwen3 Next 80B A3B | 93.00% | 20.98 | $0.039025 |
    | 44 | gpt-oss-120b | GPT-5.6 Luna | 92.50% | 8.67 | $0.001056 |
    | 45 | GPT-5.6 Luna | Qwen3 Next 80B A3B | 92.50% | 7.94 | $0.001132 |
    | 46 | gpt-oss-120b | DeepSeek R1 | 92.50% | 10.78 | $0.006317 |
    | 47 | Kimi K2.5 | DeepSeek R1 | 92.50% | 18.43 | $0.011867 |
    | 48 | Claude Opus 4.6 | GPT-5.6 Luna | 92.50% | 33.37 | $0.079804 |
    | 49 | GPT-5.6 Luna | Kimi K2.5 | 92.00% | 5.98 | $0.001366 |
    | 50 | GPT-5.6 Luna | Claude Haiku 4.5 | 92.00% | 5.34 | $0.002348 |
    | 51 | GPT-5.6 Luna | DeepSeek R1 | 92.00% | 9.33 | $0.006062 |
    | 52 | Claude Haiku 4.5 | GPT-5.6 Luna | 92.00% | 33.00 | $0.014394 |
    | 53 | GPT-5.6 Luna | Ministral 3 8B | 91.50% | 4.34 | $0.000863 |
    | 54 | Kimi K2.5 | GPT-5.6 Luna | 91.50% | 16.17 | $0.005511 |
    | 55 | DeepSeek R1 | Qwen3 32B | 91.50% | 18.81 | $0.016605 |
    | 56 | DeepSeek R1 | Claude Haiku 4.5 | 91.50% | 22.39 | $0.019250 |
    | 57 | GPT-5.6 Luna | GPT-5.6 Luna | 91.00% | 3.09 | $0.000592 |
    | 58 | GPT-5.6 Luna | Qwen3 32B | 91.00% | 3.47 | $0.000710 |
    | 59 | GPT-5.6 Luna | gpt-oss-20b | 91.00% | 4.76 | $0.000733 |
    | 60 | Claude Opus 4.6 | DeepSeek R1 | 91.00% | 21.64 | $0.092879 |
    | 61 | gpt-oss-20b | DeepSeek R1 | 90.00% | 11.42 | $0.007534 |
    | 62 | gpt-oss-20b | GPT-5.6 Luna | 89.50% | 7.70 | $0.000779 |
    | 63 | Qwen3 32B | Qwen3 Next 80B A3B | 88.00% | 7.76 | $0.001264 |
    | 64 | Qwen3 32B | Claude Opus 4.6 | 87.50% | 12.01 | $0.017293 |
    | 65 | Ministral 3 8B | Qwen3 Next 80B A3B | 87.00% | 10.09 | $0.000441 |
    | 66 | Qwen3 32B | gpt-oss-120b | 87.00% | 6.95 | $0.002407 |
    | 67 | Ministral 3 8B | Kimi K2.5 | 86.50% | 15.31 | $0.001048 |
    | 68 | Ministral 3 8B | Claude Opus 4.6 | 86.50% | 12.53 | $0.008501 |
    | 69 | Qwen3 32B | Claude Haiku 4.5 | 86.00% | 7.29 | $0.004552 |
    | 70 | Ministral 3 8B | gpt-oss-120b | 85.50% | 12.06 | $0.000583 |
    | 71 | Ministral 3 8B | GPT-5.6 Luna | 85.50% | 16.43 | $0.000649 |
    | 72 | Qwen3 32B | Ministral 3 8B | 85.50% | 17.81 | $0.001751 |
    | 73 | Ministral 3 8B | Claude Haiku 4.5 | 85.50% | 12.08 | $0.002437 |
    | 74 | Qwen3 32B | gpt-oss-20b | 85.50% | 6.26 | $0.002462 |
    | 75 | Qwen3 32B | Kimi K2.5 | 85.50% | 13.09 | $0.003929 |
    | 76 | Ministral 3 8B | gpt-oss-20b | 85.00% | 13.30 | $0.000387 |
    | 77 | Ministral 3 8B | Qwen3 32B | 85.00% | 11.16 | $0.000410 |
    | 78 | Ministral 3 8B | Ministral 3 8B | 85.00% | 12.69 | $0.000620 |
    | 79 | Ministral 3 8B | DeepSeek R1 | 85.00% | 25.37 | $0.009215 |
    | 80 | Qwen3 32B | Qwen3 32B | 84.50% | 5.79 | $0.003099 |
    | 81 | Qwen3 32B | DeepSeek R1 | 80.50% | 12.07 | $0.008997 |
    | 82 | Kimi K2.5 | Ministral 3 8B | 78.50% | 39.76 | $0.004889 |
    | 83 | Kimi K2.5 | Qwen3 Next 80B A3B | 78.00% | 36.54 | $0.005041 |
    | 84 | Qwen3 Next 80B A3B | Claude Opus 4.6 | 78.00% | 31.01 | $0.014812 |
    | 85 | Qwen3 32B | GPT-5.6 Luna | 77.50% | 10.81 | $0.003074 |
    | 86 | Qwen3 Next 80B A3B | Ministral 3 8B | 77.00% | 38.55 | $0.002727 |
    | 87 | Qwen3 Next 80B A3B | GPT-5.6 Luna | 76.50% | 24.09 | $0.002471 |
    | 88 | Qwen3 Next 80B A3B | gpt-oss-120b | 76.50% | 34.72 | $0.002606 |
    | 89 | Qwen3 Next 80B A3B | Claude Haiku 4.5 | 76.50% | 32.33 | $0.006027 |
    | 90 | Qwen3 Next 80B A3B | DeepSeek R1 | 76.50% | 25.45 | $0.009190 |
    | 91 | Qwen3 Next 80B A3B | Qwen3 32B | 76.00% | 30.64 | $0.002088 |
    | 92 | Qwen3 Next 80B A3B | Qwen3 Next 80B A3B | 76.00% | 36.44 | $0.002721 |
    | 93 | Qwen3 Next 80B A3B | Kimi K2.5 | 75.50% | 36.37 | $0.003959 |
    | 94 | Qwen3 Next 80B A3B | gpt-oss-20b | 75.00% | 32.70 | $0.002391 |
    | 95 | Kimi K2.5 | gpt-oss-20b | 75.00% | 25.00 | $0.003928 |
    | 96 | Kimi K2.5 | gpt-oss-120b | 75.00% | 31.74 | $0.004762 |
    | 97 | Kimi K2.5 | Kimi K2.5 | 74.50% | 43.05 | $0.006772 |
    | 98 | Kimi K2.5 | Claude Haiku 4.5 | 74.00% | 26.39 | $0.006879 |
    | 99 | Kimi K2.5 | Claude Opus 4.6 | 74.00% | 28.10 | $0.014059 |
    | 100 | Kimi K2.5 | Qwen3 32B | 73.00% | 29.60 | $0.004667 |
