<p align="center">
  <img width="360" alt="AgentOpt" src="docs/assets/logo.png" />
</p>

<p align="center">
  <strong>Find the right LLM models for your AI agents.</strong>
</p>

<p align="center">
  <em>A simple model swap can cut your agent's costs by 10–100x without sacrificing performance.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/agentopt-py/"><img src="https://img.shields.io/pypi/v/agentopt-py?logo=python&logoColor=white&color=3776ab" alt="PyPI"></a>
  <!-- <a href="https://pepy.tech/projects/agentopt-py"><img src="https://static.pepy.tech/badge/agentopt-py" alt="Downloads"></a> -->
  <!-- <a href="https://github.com/AgentOptimizer/agentopt"><img src="https://img.shields.io/github/stars/AgentOptimizer/agentopt?style=flat&logo=github&color=181717" alt="GitHub stars"></a> -->
  <a href="https://github.com/AgentOptimizer/agentopt/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green?style=flat" alt="License"></a>
  <a href="https://agentoptimizer.github.io/agentopt/"><img src="https://img.shields.io/badge/docs-website-blue?style=flat&logo=materialformkdocs&logoColor=white" alt="Docs"></a>
</p>

<p align="center">
  AgentOpt is supported by <a href="https://daplab.cs.columbia.edu/">DAPLab</a> at Columbia University.
</p>

---

## News
[2026/04] 🔥 Version 0.1.0 released!


## Why AgentOpt
Choosing models for your agent is surprisingly hard. Which family? Small or big? Thinking or non-thinking? And different steps may need different models. The combinatorial space explodes fast — 3 steps × 8 models = **512 combinations** to evaluate.

AgentOpt solves this automatically. Give it your agent and a small evaluation dataset, and it will efficiently search the model combination space to present you with the **Pareto curve of performance/cost/latency tradeoffs** — so you can make an informed choice. 

AgentOpt works with **almost any agent implementation** and requires **minimal wrappers** to your existing agents.

## Use Cases

Same accuracy band, 20–100x cost difference — just by picking the right model combination:

| Benchmark | Expensive Combo | Acc | Cost | Budget Combo | Acc | Cost | Savings |
|-----------|----------------|-----|------|-------------|-----|------|---------|
| BFCL | Opus | 72% | $60.78 | Qwen3 Next | 71% | $1.87 | **32x** |
| HotpotQA | Opus + Opus | ~73% | $2.71 | Qwen3 Next + gpt-oss-120b | 71.3% | $0.13 | **21x** |
| MathQA | Opus + Opus | ~98.5% | $5.89 | Ministral + C3 Haiku | 94.0% | $0.05 | **118x** |

Read more in our [blog post](https://agentoptimizer.github.io/agentopt/blog/2026/03/22/why-your-agent-needs-a-model-combo-optimizer-not-just-a-model/).

## Installation

```bash
pip install agentopt-py
```
## Quick Start

Say you have an agent with two LLM steps (a planner and a solver) and you want to find the best model for each:

```python
from agentopt import ModelSelector

selector = ModelSelector(
    agent=MyAgent,
    models={
        "planner": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],  # 3 options
        "solver":  ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],  # 3 options
    },  # → 3 × 3 = 9 combinations to evaluate
    eval_fn=eval_fn,
    dataset=dataset,
    method="brute_force",  # or "auto" for smarter selection algorithms
)

results = selector.select_best(parallel=True, max_concurrent=50)
results.print_summary()
```

Output:
```
    Model Selection Results
    ----------------------------------------------------------------------------
    Rank  Model                                     Accuracy  Latency      Price
    ----------------------------------------------------------------------------
>>>    1  planner=gpt-4.1-nano + solver=gpt-4.1-nano 100.00%    0.85s  $0.000420
       2  planner=gpt-4o-mini + solver=gpt-4o-mini   100.00%    1.20s  $0.002372
       3  planner=gpt-4o + solver=gpt-4o              100.00%    2.70s  $0.014355
    ...
```

Conceptually, this is what happens under the hood:

```python
for combo in all_combinations(models):       # e.g. {"planner": "gpt-4o", "solver": "gpt-4o-mini"}
    agent = MyAgent(combo)                   # build agent with this model combo
    for input_data, expected in dataset:
        actual = agent.run(input_data)       # run on each datapoint
        score = eval_fn(expected, actual)    # score the output
# rank combos by quality score, latency & per-query cost
```

But AgentOpt does this efficiently with **smart algorithms, parallelization, per-query cost & latency tracking, and caching**. With `method="auto"` (the default), it **automatically** homes in on the best combination (wired to `arm_elimination` — strong best-arm identification with far fewer evaluations than `brute_force`), eliminating clearly worse combinations after just a few datapoints.

You just provide four things:

**Agent** — wrap your agent into a class with `__init__(self, models)` and `run(self, input_data)`:

- `__init__(self, models)` — receive a model configuration and do your agent creation. `models` is a dict that maps each step you want to optimize to a specific model, e.g. `{"planner": "gpt-4o-mini", "solver": "gpt-4o"}`.
- `run(self, input_data)` — run your agent on a single datapoint and return the output.

```python
from openai import OpenAI

class MyAgent:
    def __init__(self, models):
        self.client = OpenAI()
        self.planner_model = models["planner"]
        self.solver_model = models["solver"]

    def run(self, input_data):
        plan = self.client.chat.completions.create(
            model=self.planner_model,
            messages=[{"role": "user", "content": f"Plan: {input_data}"}],
        ).choices[0].message.content

        answer = self.client.chat.completions.create(
            model=self.solver_model,
            messages=[
                {"role": "system", "content": f"Follow this plan:\n{plan}"},
                {"role": "user", "content": input_data},
            ],
        ).choices[0].message.content
        return answer
```

**Dataset** — a list of `(input_data, expected_output)` pairs:

```python
dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky?", "blue"),
    # We recommend at least 100 samples for production decisions,
    # but even 10-20 samples can surface clear winners during development.
]
```

**Eval function** — compares the agent output against the expected answer, returns a score:

```python
def eval_fn(expected, actual):
    return 1.0 if expected.lower() in str(actual).lower() else 0.0
```

LLM-as-judge is also supported — just call your judge LLM inside `eval_fn`.

**Models** — a dict mapping each step name to a list of candidate models to try. AgentOpt picks one from each list, constructs the agent, and evaluates it.

## Framework Compatibility

AgentOpt works with any LLM framework that uses `httpx` under the hood. Here we provide examples for a few popular frameworks, but it literally works with any custom implementation:

| Framework | Status | Example |
|-----------|--------|---------|
| OpenAI Agents SDK | Supported | [openai_sdk_example.py](examples/openai_sdk_example.py) |
| LangChain / LangGraph | Supported | [langchain_example.py](examples/langchain_example.py), [langgraph_example.py](examples/langgraph_example.py) |
| CrewAI | Supported | [crewai_example.py](examples/crewai_example.py) |
| LlamaIndex | Supported | [llamaindex_example.py](examples/llamaindex_example.py) |
| AG2 | Supported | [ag2_example.py](examples/ag2_example.py) |
| OpenAI-Compatible API SDK | Supported | [custom_agent_example.py](examples/custom_agent_example.py) |

## Selection Algorithms

AgentOpt includes a rich set of selection algorithms. Advanced users may get significant speedups by choosing the right method for their use case. See the [documentation](https://agentoptimizer.github.io/agentopt/) and [advanced_selection_example.py](examples/advanced_selection_example.py) for details.

If you do not need the strict best model combination and want **lower search cost**, `epsilon_lucb` is often a good choice: it stops once an **ε-optimal** arm is found (tune `epsilon` to trade off how close to optimal you need to be versus how many runs you spend).

| `method=` | Best for | How it works |
|-----------|----------|-------------|
| `"auto"` (default) | General use | Automatically finds the best combination (wired to `arm_elimination` — strong best-arm identification with lower search cost than `brute_force`) |
| `"brute_force"` | Small search spaces | Evaluates all combinations |
| `"random"` | Quick exploration | Samples a random fraction |
| `"hill_climbing"` | Topology-aware search | Greedy search using model quality/speed rankings |
| `"arm_elimination"` | Best-arm identification | Bandit; eliminates statistically dominated combinations |
| `"epsilon_lucb"` | Extra search cost savings when ε-optimal is enough | Bandit; stops when an epsilon-optimal best arm is identified |
| `"threshold"` | Thresholding objectives | Bandit; determines whether each combination is above/below a user-defined `threshold` on the performance metric (e.g., mean accuracy) |
| `"lm_proposal"` | LLM-guided search | Uses a proposer LLM to shortlist promising combinations |
| `"bayesian"` | Expensive evaluations | GP-based Bayesian optimization over categorical model choices; uses correlation between combinations (requires `pip install "agentopt-py[bayesian]"`) |

```python
selector = ModelSelector(
    agent=MyAgent, models=models, eval_fn=eval_fn, dataset=dataset,
    method="epsilon_lucb",
    epsilon=0.01
)
results = selector.select_best(parallel=True)
```

## How It Works

AgentOpt intercepts LLM calls at the `httpx` transport layer — the one chokepoint every LLM SDK shares. No proxy server, no framework adapters required.

```
your_agent(input)
  └── framework internals (LangChain, CrewAI, etc.)
        └── httpx.Client.send()   ← intercepted here
              └── LLM API (OpenAI, Anthropic, etc.)
```

For each model combination, AgentOpt:
1. Instantiates your agent class with the candidate models
2. Calls `run()` on every datapoint in your evaluation set
3. Tracks token usage, latency, and per-query cost automatically
4. Scores the output using your evaluation function
5. Reports the Pareto-optimal combinations

Response caching (in-memory + SQLite on disk) is enabled by default — identical LLM calls are never repeated, making iterative experimentation fast and cheap.

## Results API

```python
results = selector.select_best()

results.print_summary()               # formatted table
best = results.get_best()             # ModelResult with highest accuracy
combo = results.get_best_combo()      # {"planner": "gpt-4o", "solver": "gpt-4o-mini"}
results.to_csv("results.csv")         # export all results
results.export_config("config.yaml")  # export best combo as YAML
```

## Advanced Usage

**Custom model pricing** — define pricing for self-hosted or custom models:

```python
selector = ModelSelector(
    ...,
    model_prices={
        "my-custom-model": {"input_price": 2.50, "output_price": 10.00},
    },
)
```

**Custom cache directory** — LLM response caching is enabled by default (`.agentopt_cache/`). To customize:

```python
from agentopt import LLMTracker

tracker = LLMTracker(cache_dir="./my_cache")
selector = ModelSelector(..., tracker=tracker)
results = selector.select_best()  # cache flushed automatically
```

**Using prebuilt LLM instances** — pass framework-specific LLM objects instead of model name strings:

```python
from langchain_openai import ChatOpenAI

selector = ModelSelector(
    agent=MyAgent,
    models={
        "planner": [ChatOpenAI(model="gpt-4o"), ChatOpenAI(model="gpt-4o-mini")],
        "solver":  [ChatOpenAI(model="gpt-4o"), ChatOpenAI(model="gpt-4o-mini")],
    },
    eval_fn=eval_fn,
    dataset=dataset,
)
```

## Documentation

Full documentation at **[agentoptimizer.github.io/agentopt](https://agentoptimizer.github.io/agentopt/)** — including detailed guides on the [Results API](https://agentoptimizer.github.io/agentopt/api/results/), [response caching](https://agentoptimizer.github.io/agentopt/concepts/caching/), and [custom model pricing](https://agentoptimizer.github.io/agentopt/api/selectors/).

## License

Apache 2.0
