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

# Why Your Agent Needs a Model Optimizer, Not Just a Model

*Wenyue Hua\*, Qian Xie, Sripad Karne, Armaan Agrawal, Nikos Pagonas, Kostis Kaffes, Tianyi Peng\**

*\* Equal contribution*

Most teams pick a model, usually the latest frontier release, and run every step of their agent on it. Planner? GPT-4o. Solver? GPT-4o. Critic? GPT-4o. It works, so nobody questions it.

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

You can't maximize all three. A frontier model like Claude Opus gives you the best quality but costs more and runs slower. A small model like Ministral 3B is cheap and fast but less capable. The question is: *where on this tradeoff surface do you want to be?*

No server-side system can answer that for you. Only you know your quality requirements, your budget, and your latency SLA. A startup building a coding assistant might happily trade 5% accuracy for 10x cost savings. A medical AI can't afford to trade accuracy at all.

This is why the user must own the objective function. AgentOpt lets you define exactly what "good" means for your use case through a simple `eval_fn`, and then finds the model combination that optimizes for it.

## Model Selection Is the Foundation of Everything Else

Among all the optimization levers available to you (caching, routing, speculative decoding, request scheduling), model selection is the one that dominates. Not because the others don't matter, but because model selection is *upstream* of all of them.

Think about it: if you don't know which model to run, what do you cache? Which GPU do you route to? What workload profile do you schedule for?

Caching, routing, and scheduling all *assume* you've already decided which model to use. They're optimizations applied *on top of* a model choice. Model selection is the prerequisite they all depend on.

And the impact is enormous. Here's what we found across three benchmarks, comparing the most expensive combination to the cheapest one that achieves similar accuracy:

| Benchmark | Expensive Combo | Acc | Cost | Budget Combo | Acc | Cost | Savings |
|-----------|----------------|-----|------|-------------|-----|------|---------|
| HotpotQA | Opus + Opus | ~73% | $2.71 | Qwen3 Next + gpt-oss-120b | 71.3% | $0.13 | **21x** |
| MathQA | Opus + Opus | ~98.5% | $5.89 | Ministral + C3 Haiku | 94.0% | $0.05 | **118x** |
| BFCL | Opus | 72% | $60.78 | Qwen3 Next | 71% | $1.87 | **32x** |

These are real numbers from real benchmarks. Same accuracy band, 20-100x cost difference. No amount of caching or request batching can close a 32x gap. The model choice *is* the optimization.

## Agent Routing Is Not LLM Routing

If you've seen LLM routing systems (the ones that pick GPT-4 for hard questions and GPT-3.5 for easy ones), you might think: "Can't I just do that for each step of my agent?"

No. And here's why.

In single-request LLM routing, credit assignment is trivial. One call, one output, one score. You can directly measure which model is better for which type of query.

In a multi-step agent, the steps *interact*. The planner's output shapes what the solver sees. A critic's feedback loops back to the generator. There's **no intermediate ground truth**: you can only score the final output. Was the answer wrong because the planner gave bad instructions, or because the solver couldn't follow good ones? You can't tell by looking at each step in isolation.

This makes **credit assignment non-trivial**. You can't decompose "pick the best model per layer" into independent decisions. The layers affect each other.

The principled response is to treat the *combination* as the atomic unit. Don't optimize layers independently. Evaluate full combos end-to-end.

And the results prove why this matters. On HotpotQA (multi-hop question answering with a planner + solver architecture), we found something that no per-layer optimization would ever discover:

**The weakest planner + the strongest solver beats the strongest planner + any solver.**

Ministral 3B (the cheapest, smallest model) as planner paired with Claude Opus as solver achieves 74.8% accuracy. Claude Opus as *both* planner and solver? Only ~73%. Why? Because Opus as planner is *too capable*: it answers the question directly, bypassing the solver's search tools entirely. The "worse" planner correctly delegates to the tool-augmented solver, producing better results.

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

Bad combos get eliminated early and cheaply. Good combos earn more evaluation budget. The total cost is far less than brute force.

### Epsilon-LUCB

When you just need to find *the single best* combo, epsilon-LUCB (Lower/Upper Confidence Bound) is extremely sample-efficient. Each round, it compares the current leader's lower confidence bound against the best challenger's upper bound. When the gap closes below a threshold epsilon, you've found your winner with statistical confidence.

### Bayesian Optimization

For expensive evaluations, Bayesian Optimization builds a Gaussian Process surrogate model that predicts accuracy as a function of the model combination. It uses Expected Improvement to pick the most informative next evaluation, spending budget where uncertainty is highest and potential is greatest.

### How Much Do These Save?

Across our four benchmarks, Arm Elimination consistently finds the best combination while using 40-60% less budget than brute force:

| Benchmark | Find Rate | Cost Savings vs Brute Force |
|-----------|-----------|---------------------------|
| HotpotQA | 90% | 64% |
| GPQA | 98% | 49% |
| MathQA | 96% | 46% |
| BFCL | 100% | 11% |

Epsilon-LUCB excels in specific scenarios. On GPQA it achieves a 90% find rate with the fewest total evaluations of any algorithm.

These aren't just "faster." They find the best combo with statistical guarantees while spending roughly half the brute-force budget.

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
| HotpotQA | Ministral 3B + Opus | Weakest planner wins. Opus as planner bypasses search tools |
| MathQA | Opus + Qwen3 Next | Critic barely matters. Opus solves math correctly on the first try |
| BFCL | Opus (single) | Qwen3 Next ties at 32x lower cost. Statistical difference is ~1% |
| GPQA | Opus (single) | Straightforward. Raw capability wins for grad-level science |

### Algorithm Comparison (50 random seeds each)

| Algorithm | HotpotQA | GPQA | MathQA | BFCL |
|-----------|---------|------|--------|------|
| **Arm Elimination** | 90% / 64% saved | 98% / 49% | 96% / 46% | 100% / 11% |
| **Hill Climbing** | 44% / 63% | n/a | 72% / 60% | 94% / 6% |
| **Bayesian Opt** | 8% / 76% | 56% / n/a | n/a | n/a |
| **Epsilon-LUCB** | n/a | 90% / best efficiency | 0% | 60% / 50% |
| **LM Proposal** | 0% | 100% | n/a | 0% |

*Format: find rate / cost savings. "n/a" = not tested on that benchmark.*

Arm Elimination is the consistent winner. LM Proposal (asking GPT-4.1 to predict the best combo) fails completely on benchmarks where the answer is counterintuitive. It can't predict that Ministral outperforms Opus as a planner.

### Budget Alternatives

For every benchmark, there exists a combination within 3-5% of the best accuracy that costs 10-100x less:

| Benchmark | Best | Accuracy | Cost | Budget Pick | Accuracy | Cost | Ratio |
|-----------|------|---------|------|------------|---------|------|-------|
| HotpotQA | Ministral + Opus | 74.8% | $2.71 | Qwen3 Next + gpt-oss-120b | 71.3% | $0.13 | 21x |
| MathQA | Opus + Qwen3 Next | 98.8% | $5.89 | Ministral + C3 Haiku | 94.0% | $0.05 | 118x |
| BFCL | Opus | 72.0% | $60.78 | Qwen3 Next | 71.0% | $1.87 | 32x |

## Get Started

```bash
pip install agentopt
```

```python
from agentopt import ArmEliminationModelSelector

selector = ArmEliminationModelSelector(
    agent_fn=your_agent_factory,
    models={"planner": ["gpt-4o", "claude-sonnet-4-6"], "solver": ["gpt-4o-mini", "claude-haiku-4-5"]},
    eval_fn=lambda expected, actual: 1.0 if expected in str(actual) else 0.0,
    dataset=your_eval_dataset,
)
results = selector.select_best()
results.print_summary()
```

Check out the [Quick Start guide](../../getting-started/quickstart.md) for a complete walkthrough.
