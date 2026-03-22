<p align="center">
  <img src="logo.png" alt="AgentOpt Logo" width="200">
</p>

<h1 align="center">AgentOpt</h1>

<p align="center">
  <strong>Find the right LLM models for your AI agents.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/agentopt/"><img src="https://img.shields.io/pypi/v/agentopt?logo=python&logoColor=white&color=3776ab" alt="PyPI"></a>
  <!-- <a href="https://pepy.tech/projects/agentopt"><img src="https://static.pepy.tech/badge/agentopt" alt="Downloads"></a> -->
  <!-- <a href="https://github.com/AgentOptimizer/agentopt"><img src="https://img.shields.io/github/stars/AgentOptimizer/agentopt?style=flat&logo=github&color=181717" alt="GitHub stars"></a> -->
  <a href="https://github.com/AgentOptimizer/agentopt/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green?style=flat" alt="License"></a>
  <a href="https://agentoptimizer.github.io/agentopt/"><img src="https://img.shields.io/badge/docs-website-blue?style=flat&logo=materialformkdocs&logoColor=white" alt="Docs"></a>
</p>

---

Choosing the right LLM model is hard. Different models have different cost, performance, and latency tradeoffs. Should you use a thinking model? What effort level? What about different models for different steps of your agent pipeline? The combinatorial space explodes quickly — if your agent has 3 steps and you're considering 5 models per step, that's 125 combinations to evaluate.

AgentOpt solves this automatically. Give it your agent and a small evaluation dataset (~100 samples), and it will efficiently search the model combination space to present you with the **Pareto curve of accuracy/cost/latency tradeoffs** — so you can make an informed choice.

## Key Features

- **Non-intrusive**: Define your agent as a class with `__init__` and `run` — we take care of the rest. No framework adapters, no code changes to your agent internals.
- **Framework-agnostic**: Works with OpenAI SDK, LangChain, LangGraph, CrewAI, LlamaIndex, AG2, or any framework that uses `httpx` for LLM calls.
- **Smart search algorithms**: Selection algorithms from brute force to advanced methods like Bayesian optimization, so you don't have to evaluate every combination.
- **Automatic tracking**: Transparently intercepts all LLM calls to measure token usage, latency, and cost — no manual instrumentation.
- **Response caching**: Identical LLM calls are cached (in-memory + SQLite on disk), so re-running experiments is instant and free.

## Installation

```bash
pip install agentopt

# With Bayesian optimization support
pip install "agentopt[bayesian]"
```

## Quick Start

**Step 1**: Define your agent as a class with `__init__(self, models)` and `run(self, input_data)`:

```python
from openai import OpenAI

class MyAgent:
    def __init__(self, models):
        self.client = OpenAI()
        self.planner_model = models["planner"]
        self.solver_model = models["solver"]

    def run(self, input_data):
        # Step 1: Plan
        plan = self.client.chat.completions.create(
            model=self.planner_model,
            messages=[{"role": "user", "content": f"Plan: {input_data}"}],
        ).choices[0].message.content

        # Step 2: Solve
        answer = self.client.chat.completions.create(
            model=self.solver_model,
            messages=[
                {"role": "system", "content": f"Follow this plan:\n{plan}"},
                {"role": "user", "content": input_data},
            ],
        ).choices[0].message.content
        return answer
```

**Step 2**: Define your evaluation dataset and scoring function:

```python
dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky?", "blue"),
    # ... ideally ~100 samples
]

def eval_fn(expected, actual):
    return 1.0 if expected.lower() in str(actual).lower() else 0.0
```

**Step 3**: Run model selection:

```python
from agentopt import BruteForceModelSelector

selector = BruteForceModelSelector(
    agent=MyAgent,
    models={
        "planner": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
        "solver":  ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
    },
    eval_fn=eval_fn,
    dataset=dataset,
)

results = selector.select_best(parallel=True)
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

## Selection Algorithms

AgentOpt provides advanced selection algorithms — you don't always need to evaluate every combination:

| Algorithm | Best for | How it works |
|-----------|----------|-------------|
| `BruteForceModelSelector` | Small search spaces | Evaluates all combinations |
| `RandomSearchModelSelector` | Quick exploration | Samples a random fraction |
| `HillClimbingModelSelector` | Topology-aware search | Greedy search using model quality/speed rankings |
| `ArmEliminationModelSelector` | Early pruning | Eliminates statistically dominated combinations |
| `EpsilonLUCBModelSelector` | Best-arm identification | Stops when LUCB confidence gap is within user `epsilon` |
| `ThresholdBanditSEModelSelector` | Thresholding objectives | Classifies combinations above/below user `threshold` |
| `LMProposalModelSelector` | LLM-guided search | Uses a proposer LLM to shortlist promising combinations |
| `BayesianOptimizationModelSelector` | Expensive evaluations | GP-based optimization (requires `torch`, `botorch`) |

All selectors share the same interface:

```python
results = selector.select_best(parallel=True, max_concurrent=20)
```

## Framework Compatibility

AgentOpt works with any LLM framework that uses `httpx` under the hood — which is virtually all of them:

| Framework | Status | Example |
|-----------|--------|---------|
| OpenAI SDK | Supported | [custom_agent_example.py](examples/custom_agent_example.py) |
| OpenAI Agents SDK | Supported | [openai_sdk_example.py](examples/openai_sdk_example.py) |
| LangChain / LangGraph | Supported | [langchain_example.py](examples/langchain_example.py), [langgraph_example.py](examples/langgraph_example.py) |
| CrewAI | Supported | [crewai_example.py](examples/crewai_example.py) |
| LlamaIndex | Supported | [llamaindex_example.py](examples/llamaindex_example.py) |
| AG2 | Supported | [ag2_example.py](examples/ag2_example.py) |
| Anthropic SDK | Supported | Uses httpx |

## How It Works

AgentOpt intercepts LLM calls at the `httpx` transport layer — the one chokepoint every LLM SDK shares. No proxy server, no framework adapters required.

```
your_agent(input)
  +-- framework internals (LangChain, CrewAI, etc.)
        +-- httpx.Client.send()   <-- intercepted here
              +-- LLM API (OpenAI, Anthropic, etc.)
```

For each model combination, AgentOpt:
1. Instantiates your agent class with the candidate models
2. Calls `run()` on every datapoint in your evaluation set
3. Tracks token usage, latency, and cost automatically
4. Scores the output using your evaluation function
5. Reports the Pareto-optimal combinations

Response caching ensures that identical LLM calls (same model + same prompt) are never repeated — making iterative experimentation fast and cheap.

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

### Custom model pricing

```python
selector = BruteForceModelSelector(
    ...,
    model_prices={
        "my-custom-model": {"input_price": 2.50, "output_price": 10.00},
    },
)
```

### Persistent disk cache

Cache LLM responses to disk so they survive process restarts:

```python
from agentopt.proxy import LLMTracker

tracker = LLMTracker(cache_dir="./llm_cache")
selector = BruteForceModelSelector(..., tracker=tracker)
results = selector.select_best()  # cache flushed automatically
```

### Using prebuilt LLM instances

Pass framework-specific LLM objects instead of model name strings:

```python
from langchain_openai import ChatOpenAI

selector = BruteForceModelSelector(
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

Full documentation is available at **[agentoptimizer.github.io/agentopt](https://agentoptimizer.github.io/agentopt/)**.

## Development

```bash
git clone https://github.com/AgentOptimizer/agentopt.git
cd agentopt
uv sync --extra dev
uv run pytest
```

## License

Apache 2.0
