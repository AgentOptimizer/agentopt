---
date: 2026-03-22
authors:
  - wenyuehua
  - qianxie
  - sripadkarne
  - armaanagrawal
  - nikospagonas
  - kostiskaffes
  - tianyipeng
categories:
  - Technical
  - Model Selection
---

# Why Your Agent Needs a Model Combo Optimizer, Not Just a Model

*Wenyue Hua\*, Qian Xie, Sripad Karne, Armaan Agrawal, Nikos Pagonas, Kostis Kaffes, Tianyi Peng\**

*\* Equal contribution*

Most teams pick a model, usually the latest frontier release, and run every step of their agent on it. Planner? GPT-5.4. Solver? GPT-5.4. Critic? GPT-5.4. It works, so nobody questions it.

But "it works" is not "it's optimal." What if the same accuracy costs 20x less with a different combination? What if a *weaker* model actually performs *better* at one of those steps? These aren't hypotheticals. We ran the experiments.

<!-- more -->

## Server-Side vs User-Side: Who's Optimizing What?

There's a lot of exciting work happening in LLM inference optimization. Systems like **Autellix**, **ThunderAgent**, and **Continuum** tackle it from the provider's side: batching requests, quantizing weights, routing traffic across GPU clusters, speculatively decoding tokens. This is server-side optimization. It makes the provider more efficient, and some of those savings trickle down to you through lower prices.

But there's a completely different optimization surface that these systems don't touch: **which models you choose to run in the first place.**

Server-side optimization saves the *provider's* money. User-side optimization saves *your* money. And the gap between the best and worst model choice is far larger than anything server-side tricks can close.

## What User-Side Optimization Actually Means

Every LLM-powered agent lives in a three-axis tradeoff space:

- **Quality**: does it get the right answer?
- **Cost**: how much do you pay per query?
- **Latency**: how long does the user wait?

You can't maximize all three. A frontier model like Claude Opus gives you the best quality but costs more and runs slower. A small model like Ministral 3 8B is cheap and fast but less capable. The question is: *where on this tradeoff surface do you want to be?*

No server-side system can answer that for you. Only you know your quality requirements, your budget, and your latency SLA. A startup building a coding assistant might happily trade 5% accuracy for 10x cost savings. A medical AI can't afford to trade accuracy at all.

This is why the user must own the objective function. AgentOpt lets you define exactly what "good" means for your use case through a simple `eval_fn`, and then finds the model combination that optimizes for it.

## Model Selection Is the Foundation of Everything Else

Among all the optimization levers available to you (caching, routing, speculative decoding, request scheduling), model selection is the one that dominates. Not because the others don't matter, but because model selection is *upstream* of all of them.

Think about it: if you don't know which model to run, what do you cache? Which GPU do you route to? What workload profile do you schedule for?

Caching, routing, and scheduling all *assume* you've already decided which model to use. They're optimizations applied *on top of* a model choice. Model selection is the prerequisite they all depend on.

And the impact is enormous. Here's what we found across three benchmarks, comparing the most expensive combination to the cheapest one that achieves similar accuracy:

| Benchmark | Expensive Combo | Acc | Cost | Budget Combo | Acc | Cost | Savings |
|-----------|----------------|-----|------|-------------|-----|------|---------|
| GPQA | Opus | 74.75% | $2.47 | gpt-oss-120b | 68.18% | $0.19 | **13x** |
| HotpotQA | Opus + Opus | ~32% | $2.02 | Qwen3 Next + gpt-oss-120b | 71.8% | $0.13 | **16x** |
| MathQA | Opus + Haiku 4.5 | 98.8% | $6.19 | gpt-oss-20b + Kimi | 94.6% | $0.26 | **24x** |
| BFCL | Opus | 70% | $60.13 | Qwen3 Next | 70% | $1.90 | **32x** |

These are real numbers from real benchmarks. Same accuracy band, 13-32x cost difference. No amount of caching or request batching can close a 32x gap. The model choice *is* the optimization.

## Agent Routing Is Not LLM Routing

If you've seen LLM routing systems (the ones that pick GPT-5.4 for hard questions and GPT-4o for easy ones), you might think: "Can't I just do that for each step of my agent?"

No. And here's why.

In single-request LLM routing, credit assignment is trivial. One call, one output, one score. You can directly measure which model is better for which type of query.

In a multi-step agent, the steps *interact*. The planner's output shapes what the solver sees. A critic's feedback loops back to the generator. There's **no intermediate ground truth**: you can only score the final output. Was the answer wrong because the planner gave bad instructions, or because the solver couldn't follow good ones? You can't tell by looking at each step in isolation.

This makes **credit assignment non-trivial**. You can't decompose "pick the best model per layer" into independent decisions. The layers affect each other.

The principled response is to treat the *combination* as the atomic unit. Don't optimize layers independently. Evaluate full combos end-to-end.

And the results prove why this matters. On HotpotQA (multi-hop question answering with a planner + solver architecture), we found something that no per-layer optimization would ever discover:

**The weakest planner + the strongest solver beats the strongest planner + any solver.**

Ministral 3 8B (the cheapest, smallest model) as planner paired with Claude Opus as solver achieves 74.3% accuracy. Claude Opus as *both* planner and solver? Only ~32%. Why? Because Opus as planner is *too capable*: it answers the question directly, bypassing the solver's search tools entirely. The "worse" planner correctly delegates to the tool-augmented solver, producing better results.

You'd never find this by picking "the best model" for each layer independently. The best combo doesn't contain the best individual models. This is the credit assignment problem in action.

## Reducing the Overhead: Bandits and Bayesian Optimization

There's an obvious objection: if you have to evaluate full combinations, doesn't the search space explode? With M models and N layers, you have M^N combinations. 9 models across 2 layers = 81 combos. Each evaluated on 200 datapoints = 16,200 LLM calls. That's expensive.

This is where search algorithms matter. And a good search algorithm is non-trivial.

### Arm Elimination

Borrow an idea from multi-armed bandits: treat each model combination as a slot machine arm. You want to find the arm with the highest payout (accuracy) without pulling every arm thousands of times.

The insight: most combinations are clearly bad after just a few samples. You don't need to run all 200 datapoints to know that a combo scoring 40% after 20 samples isn't going to beat one scoring 75%.

Arm Elimination works in rounds:

1. **Start small**: evaluate all combos on a small initial batch (e.g., 10 datapoints)
2. **Eliminate**: use confidence intervals to statistically identify dominated combos. If a combo's upper confidence bound is below another's lower bound, it's out.
3. **Grow the batch**: double the datapoints, evaluate only the survivors
4. **Repeat**: until one combo remains or you run out of data

Bad combos get eliminated early and cheaply. Good combos earn more search budget. The search cost is far less than brute force.

### Bayesian Optimization

For expensive evaluations, Bayesian Optimization builds a Gaussian Process surrogate model that predicts accuracy as a function of the model combination. It uses Expected Improvement to pick the most informative next evaluation, spending budget where uncertainty is highest and potential is greatest.

### How Much Do These Save?

Across our four benchmarks, Arm Elimination consistently achieves near-optimal accuracy while using up to 67% less budget than brute force:

| Benchmark | Brute Force Accuracy | Arm Elimination Accuracy | Search cost savings |
|-----------|---------------------|------------------------|-------------|
| HotpotQA | 74.27% | 73.19% | 67% |
| MathQA | 98.84% | 98.83% | 58% |
| GPQA | 74.75% | 74.10% | 24% |
| BFCL | 70.00% | 69.37% | 12% |

Nearly identical accuracy to exhaustive search, at roughly half the search cost. These algorithms don't just save budget. They find the right combo with statistical guarantees.

## Empirical Validation

We validated AgentOpt across four diverse benchmarks using 9 models on Amazon Bedrock:

### The Benchmarks

- **HotpotQA**: multi-hop question answering (2-tuple: planner + solver with search tools)
- **GPQA Diamond**: graduate-level science questions (1-tuple: single model)
- **MathQA**: mathematical reasoning (2-tuple: answer model + critic)
- **BFCL v3**: multi-turn function calling (1-tuple: single model)

### Key Finding: Best Combo ≠ Best Models

| Benchmark | Best Combo | Why It's Surprising |
|-----------|-----------|-------------------|
| HotpotQA | Ministral 3 8B + Opus | Weakest planner wins. Opus as planner bypasses search tools and scores only ~32% |
| MathQA | Opus + Haiku 4.5 | Critic barely matters. Opus solves math correctly on the first try |
| BFCL | Opus / Kimi / Qwen3 Next (tied) | Three models tie at 70%. Qwen3 Next costs 32x less than Opus |
| GPQA | Opus | Kimi is within 2pp at less than half the cost |

### Algorithm Comparison (50 random seeds each)

<table>
<thead>
<tr><th>Algorithm</th><th>GPQA</th><th>BFCL</th><th>HotpotQA</th><th>MathQA</th></tr>
</thead>
<tbody>
<tr><td><strong>Brute Force</strong></td><td>74.75% / 0%</td><td>70.00% / 0%</td><td>74.27% / 0%</td><td>98.84% / 0%</td></tr>
<tr><td><strong>Arm Elimination</strong></td><td>74.10% / 24%</td><td>69.37% / 12%</td><td><span class="ao-efficient">73.19% / 67%</span></td><td><span class="ao-efficient">98.83% / 58%</span></td></tr>
<tr><td><strong>Bayesian Opt</strong></td><td>72.43% / 45%</td><td>69.27% / 40%</td><td><span class="ao-efficient">73.33% / 76%</span></td><td><span class="ao-efficient">95.41% / 71%</span></td></tr>
</tbody>
</table>

*Format: obtained accuracy / search cost savings vs brute force. Averaged over 50 seeds. <span class="ao-efficient">Green</span> = within 5% of brute force accuracy AND >50% savings on that metric.*

Arm Elimination consistently achieves near-optimal accuracy while using significantly less budget than brute force across our four benchmarks. No single selector dominates all benchmarks — Arm Elimination performs best when there is clear separation between the best combo and the rest (HotpotQA, MathQA), while Bayesian Optimization can achieve high savings on large search spaces at the cost of lower find rates.

## Get Started

```bash
pip install agentopt-py
```

```python
from agentopt import ModelSelector

selector = ModelSelector(
    agent=YourAgent,
    models={"planner": ["gpt-4o", "claude-sonnet-4-6"], "solver": ["gpt-4o-mini", "claude-haiku-4-5"]},
    eval_fn=lambda expected, actual: 1.0 if expected in str(actual) else 0.0,
    dataset=your_eval_dataset,
)
results = selector.select_best()
results.print_summary()
```

Check out the [Quick Start guide](../../getting-started/quickstart.md) for a complete walkthrough.
