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
[2026/04] Version 0.1.0 released.


## Why AgentOpt

**Framework-agnostic by construction.** AgentOpt intercepts LLM calls at the one place every SDK eventually goes through — the outbound HTTP request — so it works the same with anything that ships an LLM call over the wire. No framework adapters, no plugin per provider, no wrapping your client. In-process Python frameworks (LangChain, LangGraph, CrewAI, LlamaIndex, AG2, OpenAI Agents SDK, plain `openai`/`anthropic`) attach through an `httpx` patch; subprocess and CLI agents (Claude Code, Gemini CLI, OpenHarness, Terminal Bench, OpenClaw) attach through `HTTPS_PROXY` with a local CA. The same code works on both — and on anything custom you write tomorrow.

On top of that one primitive, AgentOpt gives you three capabilities that share the same proxy, the same record schema, and the same cache:

- **Selection** — search a combinatorial model space to find the best fixed combination for an agent.
- **Routing** — swap models *per call* at runtime based on prompt, history, or any policy you write.
- **Tracking** — just record token usage, latency, and per-query cost across an agent run.

The combinatorial search problem is real: 3 steps × 8 models = **512 combinations** to evaluate. AgentOpt's selection algorithms (arm elimination, LUCB, Bayesian) home in on the best combination with a fraction of the brute-force cost, and the routing API lets you keep refining at runtime once you've shipped.

## Use Cases

### Offline model selection — find the best fixed combination

Same accuracy band, 20–100x cost difference — just by picking the right model combination:

| Benchmark | Expensive Combo | Acc | Cost | Budget Combo | Acc | Cost | Savings |
|-----------|----------------|-----|------|-------------|-----|------|---------|
| BFCL | Opus | 72% | $60.78 | Qwen3 Next | 71% | $1.87 | **32x** |
| HotpotQA | Opus + Opus | ~73% | $2.71 | Qwen3 Next + gpt-oss-120b | 71.3% | $0.13 | **21x** |
| MathQA | Opus + Opus | ~98.5% | $5.89 | Ministral + C3 Haiku | 94.0% | $0.05 | **118x** |

Run it once against a small evaluation dataset; ship the winner. Read more in our [blog post](https://agentoptimizer.github.io/agentopt/blog/2026/03/22/why-your-agent-needs-a-model-combo-optimizer-not-just-a-model/).

### Online model routing — pick a different model per call

For workloads where one fixed combination isn't optimal — easy prompts shouldn't pay GPT-4o prices, hard ones shouldn't suffer on Haiku — a `Router` decides at every LLM call which model to use, based on the prompt, prior calls in the session, or any feature you can compute. Common policies:

- **Length/complexity-based** — short prompts → small model, long context or tool-call-heavy → big model.
- **First-call-big** — a strong model for the planning hop, cheap models for the follow-ups.
- **Bandit / learned routing** — feed selection results back into a contextual bandit so routing decisions improve with traffic.
- **Provider failover & A/B** — route a fraction of traffic to a candidate model for live comparison without redeploying.

The routing API runs the same in-process or through the `agentopt serve` daemon, so you can prototype locally and switch a single env var to share the policy across many clients.

## Installation

```bash
pip install agentopt-py
```
## Quick Start

Two axes determine which entry point you reach for:

- **Selection vs routing** — find one fixed model combination *offline* (selection), or pick a model *per LLM call* at runtime (routing).
- **In-process vs subprocess** — does your agent live in the same Python process as your script (LangChain, OpenAI SDK, …), or run as an external CLI / Docker container (Gemini CLI, Terminal Bench, Claude Code, …)?

A third deployment axis — **local proxy vs `agentopt serve` daemon** — is just an env-var flip; the code below is byte-identical between modes. The four canonical setups follow.

### 1. In-process agent + offline model selection

The base case. Your agent uses an LLM SDK directly; you search a `{planner, solver}` combination space to pick the cheapest combo that hits the accuracy band.

```python
from openai import OpenAI
from agentopt import ModelSelector


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
        return self.client.chat.completions.create(
            model=self.solver_model,
            messages=[
                {"role": "system", "content": f"Follow this plan:\n{plan}"},
                {"role": "user", "content": input_data},
            ],
        ).choices[0].message.content


dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky?", "blue"),
    # 100+ samples recommended for production; 10–20 surfaces clear winners.
]

def eval_fn(expected, actual):
    return 1.0 if expected.lower() in str(actual).lower() else 0.0

selector = ModelSelector(
    agent=MyAgent,
    models={
        "planner": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
        "solver":  ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
    },                                  # → 3 × 3 = 9 combinations
    eval_fn=eval_fn,
    dataset=dataset,
    method="auto",                      # arm_elimination — smart + cheap
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

With `method="auto"` AgentOpt eliminates clearly worse combinations after a few datapoints; LLM-as-judge is supported — just call your judge LLM inside `eval_fn`.

### 2. Subprocess agent + offline model selection

When the agent is an external CLI (Gemini CLI, Claude Code, Terminal Bench, OpenHarness, …), `run()` shells out via `subprocess.run`. **You write zero env-var plumbing** — while a selection is in flight, AgentOpt patches `subprocess.Popen` to inject `HTTPS_PROXY` + the CA bundle into every child, so the CLI's LLM calls are intercepted and tracked the same way as in-process ones.

```python
import subprocess
from agentopt import ModelSelector


class GeminiCLIAgent:
    def __init__(self, models):
        self.model = models["agent"]

    def run(self, prompt):
        # No agentopt imports here. The subprocess patch handles routing.
        return subprocess.run(
            ["gemini", "-m", self.model, "-p", prompt],
            capture_output=True, text=True,
        ).stdout


selector = ModelSelector(
    agent=GeminiCLIAgent,
    models={"agent": ["gemini-2.5-flash", "gemini-2.5-pro"]},
    eval_fn=eval_fn,
    dataset=dataset,
    method="brute_force",
)
selector.select_best(parallel=False).print_summary()
```

For agents that ignore `HTTPS_PROXY` and need the proxy URL / CA cert injected into a config file (OpenClaw is the canonical case), `agentopt.get_current_session_proxy()` is the escape hatch — see the [`OpenClawAgent`](examples/shared/openclaw_agent.py) wrapper for the pattern. Working examples per CLI: [examples/selection/local/](examples/selection/local/).

### 3. Local backend + online model routing

A `Router` decides at every LLM call which model to use — no `models=` search space, no eval dataset. The same `MyAgent` from §1 is reused; only the harness around it changes.

```python
from agentopt import LLMTracker, RandomRouter

agent = MyAgent({"planner": "gpt-4o-mini", "solver": "gpt-4o-mini"})

router = RandomRouter(candidates=["gpt-4o-mini", "gpt-4.1-nano"], seed=0)
questions = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "What color is the sky?",
]

with LLMTracker(router=router) as tracker:
    for i, q in enumerate(questions, 1):
        with tracker.track(data_id=f"q{i}"):
            print(agent.run(q))
tracker.print_summary()
```

Output:
```
Paris
4
Blue.
============================================================
Routing summary
============================================================

Model usage by datapoint:
  [q1]  2 call(s), 4.11s
      gpt-4.1-nano                     2.06s
      gpt-4.1-nano                     2.06s
  [q2]  2 call(s), 2.22s
      gpt-4.1-nano                     1.11s
      gpt-4.1-nano                     1.11s
  [q3]  2 call(s), 6.03s
      gpt-4o-mini                      3.01s
      gpt-4o-mini                      3.01s

Tokens per model:
  gpt-4.1-nano   prompt= 19268   completion=     8   total= 19276
  gpt-4o-mini    prompt=  9638   completion=     6   total=  9644

Total latency: 12.37s across 6 call(s)
```

`RandomRouter` is the simplest built-in policy. Write your own by subclassing `Router` and implementing `route(ctx) -> Optional[str]` — return a model name to swap, or `None` to keep the client's choice. The same code works for subprocess agents too (CLIs are routed at the mitmproxy hop). See the [router docs](https://agentoptimizer.github.io/agentopt/api/router/) and [`examples/routing/local/`](examples/routing/local/).

### 4. Daemon backend + online model routing

Same Python code as §3 — but the proxy state (cache, records, mitmproxy masters) lives in a long-lived `agentopt serve` daemon instead of this process. One gateway can serve many clients in any language, and routing policies preloaded on the daemon apply across all of them.

Start the daemon (in its own terminal):

```bash
# Plain daemon — clients bring their own routers.
agentopt serve --port 9000 --cache-dir ./shared_cache

# Or set a daemon-wide default router (per-session overrides still allowed).
agentopt serve --port 9000 \
    --routing-policy random --candidate-models gpt-4o,gpt-4o-mini --seed 42

# Or preload custom Router subclasses for clients to push per-session.
agentopt serve --port 9000 --policy-module ./my_policies.py
```

Then run the §3 script against it — only the env var changes:

```bash
AGENTOPT_GATEWAY_URL=http://127.0.0.1:9000 python my_routing_script.py
```

`LLMTracker` detects `AGENTOPT_GATEWAY_URL` in `__init__` and routes through `RemoteBackend`; in-process httpx calls forward through the daemon's per-session proxy port, and subprocess agents get that same port injected into `HTTPS_PROXY`. See [`examples/routing/daemon/`](examples/routing/daemon/) and [`examples/selection/daemon/`](examples/selection/daemon/).

### What you provide

All four setups share the same agent contract:

- `MyAgent.__init__(self, models)` — receive a dict like `{"planner": "gpt-4o", "solver": "gpt-4o-mini"}` and build your agent. For routing, the dict is the *initial* model assignment; the router overrides per call.
- `MyAgent.run(self, input_data)` — run on a single datapoint and return the output.

Selection additionally needs a `dataset` of `(input, expected)` pairs and an `eval_fn(expected, actual) -> float`; neither is required for routing.

## Framework Compatibility

Working examples for the frameworks and CLI agents named above. Examples are organised into four quadrants under [`examples/`](examples/): `{selection, routing} × {local, daemon}`.

| Framework | Type | Selection | Routing |
|-----------|------|-----------|---------|
| OpenAI Agents SDK | in-process | [openai_sdk.py](examples/selection/local/openai_sdk.py) | [openai_sdk.py](examples/routing/local/openai_sdk.py) |
| LangChain | in-process | [langchain.py](examples/selection/local/langchain.py) | [langchain.py](examples/routing/local/langchain.py) |
| LangGraph | in-process | [langgraph.py](examples/selection/local/langgraph.py) | [langgraph.py](examples/routing/local/langgraph.py) |
| CrewAI | in-process | [crewai.py](examples/selection/local/crewai.py) | [crewai.py](examples/routing/local/crewai.py) |
| LlamaIndex | in-process | [llamaindex.py](examples/selection/local/llamaindex.py) | [llamaindex.py](examples/routing/local/llamaindex.py) |
| AG2 | in-process | [ag2.py](examples/selection/local/ag2.py) | [ag2.py](examples/routing/local/ag2.py) |
| OpenAI-Compatible API | in-process | [custom_agent.py](examples/selection/local/custom_agent.py) | [custom_agent.py](examples/routing/local/custom_agent.py) |
| Gemini CLI | subprocess | [gemini_cli.py](examples/selection/local/gemini_cli.py) | [gemini_cli.py](examples/routing/local/gemini_cli.py) |
| OpenHarness | subprocess | [openharness.py](examples/selection/local/openharness.py) | [openharness.py](examples/routing/local/openharness.py) |
| Terminal Bench | subprocess (Docker) | [terminal_bench.py](examples/selection/local/terminal_bench.py) | [terminal_bench.py](examples/routing/local/terminal_bench.py) |
| OpenClaw | subprocess | [openclaw.py](examples/selection/local/openclaw.py) | [openclaw.py](examples/routing/local/openclaw.py) |

## Selection Algorithms

AgentOpt includes a rich set of selection algorithms. Advanced users may get significant speedups by choosing the right method for their use case. See the [documentation](https://agentoptimizer.github.io/agentopt/) and [advanced_algorithms.py](examples/selection/local/advanced_algorithms.py) for details.

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

Everything in AgentOpt builds on a single primitive: **intercept every outbound LLM HTTP call.** Selection, routing, tracking, and caching all hang off that one seam.

### One primitive, two interception sites

| Agent shape | Where we intercept | What we patch |
|---|---|---|
| In-process Python (LangChain, OpenAI SDK, …) | The HTTP library, before encryption | `httpx.Client.send` |
| Subprocess / CLI / Docker (Gemini CLI, Claude Code, Terminal Bench, …) | The network, via a local mitmproxy on a per-session port | `subprocess.Popen.__init__` (to inject `HTTPS_PROXY` + CA bundle) |

```
in-process:                                subprocess:

  agent.run(input)                           agent.run(input)
   └── SDK (langchain/openai/…)               └── subprocess.run([...])  ← Popen patch
       └── httpx.Client.send()  ← patched          └── child process inherits HTTPS_PROXY
           └── LLM API                                └── mitmproxy on session port
                                                          ├── TLS-terminates with our CA
                                                          └── forwards to LLM API
```

The mitmproxy CA is generated once and merged with `certifi`'s system CAs into a bundle at `~/.mitmproxy/agentopt-bundle.pem`; the subprocess patch sets `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` / `NODE_EXTRA_CA_CERTS` so the child trusts both. **No agent code changes either way** — the patches install when `LLMTracker.start()` (or the `with LLMTracker:` context) is entered, and uninstall on exit (refcounted so concurrent trackers don't interfere).

### Three capabilities on top of the seam

| | What runs at the intercept | What it produces |
|---|---|---|
| **Tracking** | Record provider, model, tokens, latency, cache hit/miss | A `CallRecord` per LLM call |
| **Caching** | Hash request body → look up SQLite/in-memory cache → short-circuit on hit | Replays of cached responses, with original latency preserved |
| **Routing** | Run the active `Router.route(ctx)` to swap `body["model"]` before forwarding | Per-call model overrides |

Selection orchestrates the same primitive: for each combination it instantiates `MyAgent(combo)`, runs `run()` over the dataset inside a tracking session, and ranks by aggregated accuracy / latency / cost. Smart algorithms (`auto` = arm-elimination by default) drop dominated combinations early so the cost scales sublinearly with the search space.

### Two backends, one API

Where does the mitmproxy state (cache, records, masters) live? You choose at runtime by setting one env var.

| Mode | Selected when | Proxy state lives in | Multi-language / multi-process clients |
|---|---|---|---|
| **Local** | `AGENTOPT_GATEWAY_URL` unset | The Python process running `LLMTracker` | Subprocess agents only |
| **Daemon** | `AGENTOPT_GATEWAY_URL=http://host:port` set | The `agentopt serve` process | First-class — any client that respects `HTTPS_PROXY` |

The user-facing API is byte-identical: `ModelSelector(...).select_best()`, `tracker.track()`, `tracker.get_records()`. In daemon mode the in-process httpx patch forwards through the daemon's per-session port instead of recording locally, and subprocess agents get the daemon's port injected into `HTTPS_PROXY` — the daemon does the cache + record on both paths.

### What this buys you

- **Framework-agnostic.** Anything that ships an LLM call over the wire works — no plugin per framework, no adapter per provider.
- **Subprocess agents are first-class.** Claude Code, Gemini CLI, Terminal Bench, Docker-bound agents — all intercepted with no env-var plumbing in the agent's `run()`.
- **Caching saves real money during iteration.** Identical request bodies are deduplicated across runs.
- **State outlives a single experiment** (daemon mode). Cache, providers, and (optionally) a default routing policy survive across runs and clients.
- **Routing and selection compose.** Today selection picks a winning combination; tomorrow routing decides per call. Future versions can feed selection results into a learned router.

For the full architecture — `_active_session_var` ContextVar attribution, per-session masters, CA bundle plumbing, daemon control plane — see [docs/api/proxy.md](docs/api/proxy.md).

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

Full documentation at **[agentoptimizer.github.io/agentopt](https://agentoptimizer.github.io/agentopt/)** — including the [Selectors](https://agentoptimizer.github.io/agentopt/api/selectors/), [Router](https://agentoptimizer.github.io/agentopt/api/router/), [Tracker](https://agentoptimizer.github.io/agentopt/api/tracker/), and [Results](https://agentoptimizer.github.io/agentopt/api/results/) API references, plus guides on [how it works](https://agentoptimizer.github.io/agentopt/concepts/how-it-works/) and [response caching](https://agentoptimizer.github.io/agentopt/concepts/caching/).

## License

Apache 2.0
