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

AgentOpt solves this automatically. Give it your agent and a small evaluation dataset (~100 samples), and it will efficiently search the model combination space to present you with the **Pareto curve of performance/cost/latency tradeoffs** — so you can make an informed choice. It works with **almost any agent implementation** and requires **minimal wrappers** to your existing agents.

## Use Cases

*Coming soon — model selection is a first-order problem: in many cases, a simple model swap can cut costs by 10x without sacrificing performance.*

## Installation

```bash
pip install agentopt
```
## Quick Start

**The idea**: To find the right models for your agent, you need four things — an agent, a set of model candidates, a dataset, and an evaluation function. A naive approach would be:

```python
# Say you have an agent with two LLM steps ("planner" and "solver"),
# and you want to find the best model for each step.
models = {
    "planner": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
    "solver":  ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
}   # → 3 × 3 = 9 combinations

for combo in all_combinations(models):       # e.g. {"planner": "gpt-4o", "solver": "gpt-4o-mini"}
    agent = MyAgent(combo)                   # build agent with this model combo
    for input_data, expected in dataset:
        actual = agent.run(input_data)       # run on each datapoint
        score = eval_fn(expected, actual)    # score the output
    # aggregate scores → quality score, track latency & cost
```

AgentOpt automates this with **smart algorithms, parallelization, and caching**. You just provide the four pieces:


**Step 1**: Say you have an agent (implemented in arbitrary way), we simply ask you wrap up your agent into a class with two methods:

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

**Step 2**: Define your evaluation dataset — a list of `(input_data, expected_output)` pairs:

```python
dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky?", "blue"),
    # We recommend at least 100 samples for production decisions,
    # but even 10-20 samples can surface clear winners during development.
]
```

**Step 3**: Define your evaluation function. It compares the output of `agent.run(input_data)` against the `expected_output` from the dataset, and returns a score:

```python
def eval_fn(expected, actual):
    """Score the agent's output (actual) against the expected answer."""
    return 1.0 if expected.lower() in str(actual).lower() else 0.0
```

**Step 4**: Run model selection. The `models` dict maps each step name to a **list of candidate models** to try. AgentOpt picks one from each list, constructs the agent, and evaluates it:

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
With `method="auto"` (the default), AgentOpt uses smart algorithms that eliminate clearly worse combinations after just a few datapoints — finding the best model combination with far fewer API calls. Use `method="brute_force"` to evaluate all combinations exhaustively.

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

| `method=` | Best for | How it works |
|-----------|----------|-------------|
| `"auto"` (default) | General use | Automatically picks the best approach |
| `"brute_force"` | Small search spaces | Evaluates all combinations |
| `"random"` | Quick exploration | Samples a random fraction |
| `"hill_climbing"` | Topology-aware search | Greedy search using model quality/speed rankings |
| `"arm_elimination"` | Early pruning | Eliminates statistically dominated combinations |
| `"epsilon_lucb"` | Best-arm identification | Stops when LUCB confidence gap is within user `epsilon` |
| `"threshold"` | Thresholding objectives | Classifies combinations above/below user `threshold` |
| `"lm_proposal"` | LLM-guided search | Uses a proposer LLM to shortlist promising combinations |
| `"bayesian"` | Expensive evaluations | GP-based optimization (requires `pip install "agentopt[bayesian]"`) |

```python
selector = ModelSelector(
    agent=MyAgent, models=models, eval_fn=eval_fn, dataset=dataset,
    method="epsilon_lucb",
    epsilon="0.5"
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
3. Tracks token usage, latency, and cost automatically
4. Scores the output using your evaluation function
5. Reports the Pareto-optimal combinations

Response caching (in-memory + SQLite on disk) is enabled by default — identical LLM calls are never repeated, making iterative experimentation fast and cheap.

## Documentation

Full documentation is available at **[agentoptimizer.github.io/agentopt](https://agentoptimizer.github.io/agentopt/)**, including:

- [Results API](https://agentoptimizer.github.io/agentopt/api/results/) — export results to CSV/YAML, retrieve the best combination
- [Response caching](https://agentoptimizer.github.io/agentopt/concepts/caching/) — persistent SQLite cache, custom cache directories
- [Custom model pricing](https://agentoptimizer.github.io/agentopt/api/selectors/) — define pricing for self-hosted or custom models

## License

Apache 2.0
