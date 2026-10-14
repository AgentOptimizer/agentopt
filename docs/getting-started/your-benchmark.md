# Run your own benchmark

AgentOpt evaluates your agent on examples you supply, shows the current model
recommendation as evaluation progresses, and lets you stop with the observations
collected so far. You provide an agent, candidate models, input/answer pairs, and
a scoring function. There is no benchmark upload step: your Python program
passes the dataset directly to `ModelSelector`.

## Prepare examples and a scorer

The Python API accepts a nonempty sequence of `(input_data, expected_answer)`
pairs. For a text task, a JSONL file is a convenient way to store them:

```json
{"input": "What is 2 + 2? Reply with only the answer.", "expected": "4"}
{"input": "What is the capital of France? Reply with only the answer.", "expected": "Paris"}
```

The terminal runner requires nonempty strings for both `input` and `expected`.
Each line is one JSON object. Load the records and convert them to pairs before
passing them to the selector; `dataset` does not accept a filename or a generator.
For this format, the example runner below supplies a validated loader:

```python
from examples.selection.benchmark_runner import load_dataset

dataset = load_dataset("my_benchmark.jsonl")
```

That import is for a repository checkout. In your own application, the basic
conversion is:

```python
import json
from pathlib import Path

records = [json.loads(line) for line in Path("my_benchmark.jsonl").read_text(
    encoding="utf-8"
).splitlines() if line.strip()]
dataset = [(record["input"], record["expected"]) for record in records]
if not dataset:
    raise ValueError("Provide at least one benchmark example")

def eval_fn(expected, actual):
    return float(str(expected).strip().casefold() == str(actual).strip().casefold())
```

`eval_fn(expected, actual)` is synchronous and returns a boolean or a finite
numeric score; higher is better. This scorer uses exact matching after trimming
whitespace and ignoring case. Replace it with a task-appropriate metric, such as
answer extraction followed by numerical comparison. Gittins and random methods
expect scores in `[0, 1]` by default; set `score_bounds=(low, high)` for a different
range. The result field named `accuracy` contains your metric's mean, even if
that metric is not classification accuracy.

Inputs and reference answers can also be dictionaries or other Python objects.
Your agent's `run(input_data)` must accept the input, and your scorer must handle
the expected answer and agent output. Convert CSV, Hugging Face, or other data
sources into the same sequence of pairs. Use representative examples for model
selection and a separate held-out set when reporting generalization performance.

## Try a complete terminal run

The repository includes
[`benchmark_runner.py`](https://github.com/AgentOptimizer/agentopt/blob/main/examples/selection/benchmark_runner.py)
and a small illustrative dataset. From the repository root, install the checkout
and the OpenAI SDK in a Python 3.12+ environment:

```bash
python -m pip install -e . openai
```

Set `OPENAI_API_KEY` in your environment. The following command makes real API
calls; replace the model names with ones available to your account:

```bash
python examples/selection/benchmark_runner.py \
  --dataset examples/selection/benchmark_sample.jsonl \
  --models gpt-4o-mini gpt-4.1-nano \
  --output results.json
```

The runner uses one model node and the exact-match scorer above. Its sample data
illustrates the file format and controls; it is not a benchmark for comparing
model quality. Replace `--dataset` with your own JSONL file. For a multi-step
agent, use the Python integration below and provide candidates for each node.

As work finishes, the terminal displays the evaluation count, search spending,
and current recommendation. Press **Ctrl+C** to request a stop. The runner waits
for scheduled work to finish and saves the final result to `results.json`,
including when you stop before any recommendation is available. Keep the process
running while it finishes that work; force-quitting bypasses this save step.


If authentication, rate limits, model calls, or scoring fail, the runner prints
an English error and exits with a nonzero status. Correct the credentials,
quota/concurrency, agent, or scorer before retrying. It does not print raw
provider response bodies or reference answers in its error messages.

When a result snapshot is available, it is saved separately as
`results.partial.json` (or `<output-stem>.partial.json` for your output path),
leaving the complete output file untouched. The file includes `is_partial`,
`failure_message`, and `snapshot_scope`. It contains the last available snapshot;
failed or unfinished work and its charges may be missing. If no snapshot is
available, or saving fails, the terminal explicitly says no results were saved.
Partial results are for inspection, not a resumable checkpoint.

Useful variations, appended to the command above:

| What you want | Arguments |
|---|---|
| Accuracy/cost alternatives | `--objective-mode multi` |
| Random shared questions, about 20% of the dataset | `--method random_questions --evaluation-fraction 0.2` |
| Random complete configurations, about half of the candidates | `--method random_configurations --evaluation-fraction 0.5` |
| UCB with an evaluation allowance | `--method ucb --evaluation-fraction 0.5` |
| Gittins with a soft search-dollar limit | `--max-search-cost-usd 1.0` |
| End automatically on single-objective Gittins advice | `--stop-on-recommendation` |

Small datasets can make a fraction too small to fund a warm start or even one
complete random sampling unit. See the allowance rules below. Run `python
examples/selection/benchmark_runner.py --help` for the runner's options; the
Python API exposes further controls.

## Connect your agent and display live results

An agent is a class with `__init__(self, models)` and `run(self, input_data)`;
`models` contains the selected candidate for each node. No AgentOpt base class is
required. The [Quick Start](quickstart.md) defines a two-step `MyAgent` using this
interface. With that class and the dataset/scorer above:

```python
from pathlib import Path
from threading import Event
from agentopt import ModelSelector

stop = Event()
objective_mode = "single"  # use "multi" for accuracy/cost alternatives

selector = ModelSelector(
    agent=MyAgent,
    models={
        "planner": ["gpt-4o-mini", "gpt-4.1-nano"],
        "solver": ["gpt-4o-mini", "gpt-4.1-nano"],
    },
    dataset=dataset,
    eval_fn=eval_fn,
    objective_mode=objective_mode,
)

def show_progress(snapshot):
    spend = snapshot.selection_cost
    spend_text = "unavailable" if spend is None else f"${spend:.6f}"
    print(f"Evaluations: {snapshot.total_evaluations}; search cost: {spend_text}")
    if objective_mode == "multi":
        candidates = snapshot.get_recommendations()
        if not candidates:
            print("Collecting observations; no recommendation yet.")
        for candidate in candidates:
            print(candidate.combo, candidate.conservative_accuracy,
                  candidate.conservative_cost_usd)
    else:
        print("Current recommendation:", snapshot.get_best_combo())
    if snapshot.stop_recommended and not snapshot.is_final:
        print("Gittins recommends stopping; you may stop now or continue evaluating.")
    if snapshot.is_final:
        print("Finished:", snapshot.stop_reason)

results = selector.select_best(
    parallel=True, max_concurrent=4,
    on_update=show_progress, stop_event=stop,
)
Path("results.json").write_text(results.model_dump_json(indent=2), encoding="utf-8")
```

To connect a Stop button, run selection in a worker thread and have the button
call `stop.set()` or `selector.request_stop()`. The callback executes in the
selection thread: send snapshots to your UI using that framework's thread-safe
mechanism. `selector.latest_results` provides a copy of the latest snapshot for
polling. A callback can also request a stop by returning `False`.

The package supplies progress callbacks and result objects. The terminal runner
provides a ready-to-use display; a web dashboard or notebook interface needs to
connect these callbacks to its own controls.

Updates happen after completed evaluations or batches, not while a model streams
tokens. Gittins needs its uniform warm start before it can recommend a
configuration. Random configurations require a completed configuration; random
questions require a completed sweep across every configuration. Stopping earlier
still returns observed records but can leave the recommendation empty. UCB
recommends from observed configurations; Bayesian optimization recommends only
fully evaluated configurations. See [update boundaries](../api/selectors.md#live-recommendations-and-user-stopping)
for each method.

In single mode, `get_best_combo()` returns the current winner. Gittins bases that
choice on a prediction for the complete supplied dataset; its model predictions
are available through `get_recommendations()`. The raw `results` rows hold the
observed scores. In multi mode, there is no unique best: use
`get_recommendations()` to display the alternatives, then express a preference:

```python
# For multi-objective results:
chosen = results.choose(max_cost_usd=0.01)  # deployment cost per question
if chosen is not None:
    print(chosen.combo)
```

These estimates describe the supplied benchmark. A live recommendation can
change and is not a guarantee of future performance.

## Stop and keep the result

All supported methods allow a user stop. Stopping is cooperative: the currently
scheduled batch finishes, including requests queued behind a concurrency limit;
no next batch starts. It does not cancel or refund an in-flight model call. The
final snapshot has `is_final=True`; a user stop is recorded as
`stop_reason="user_stopped"`.

Single-objective Gittins also publishes `stop_recommended`. By default,
`stop_on_recommendation=False`, so this is advice and evaluation continues. Set
`stop_on_recommendation=True` to finish automatically on that advice. The advice
may change after further observations. Use `is_final` and `stop_reason`, rather
than the advice flag, to tell whether the run has ended.

Saving `results.model_dump_json()` preserves the evaluation records and result
fields for inspection. When single mode has a winner, you can also export its
model configuration:

```python
if results.get_best_combo() is not None:
    results.export_config("selected_models.yaml")
```

**Stopping does not create a resumable checkpoint.** There is no pause/resume API
for restoring the selector's search state. A new run starts selection again.
[Disk caching](../concepts/caching.md), enabled locally at `.agentopt_cache` by
default, can reuse saved responses for identical LLM requests. It does not restore
the selection state or guarantee that every request in the new run is cached.

## Choose an algorithm and change defaults

The default is `method="auto"`, `objective_mode="single"`: Gittins recommends
one configuration, with `cost_scaling_factor=1e-4` for search cost and no
deployment cost or latency penalty. A budget is optional. Without a user stop or
an explicit limit, single-objective Gittins and both random methods can evaluate
the full finite grid. Multi-objective Gittins also has an algorithmic termination
rule.

| `method` | Single objective | Multiple objectives |
|---|---|---|
| `auto`, `gittins` | Gittins winner | Gittins accuracy/cost recommendation set |
| `random_configurations` | Random complete configurations | Same sampling, empirical accuracy/cost alternatives |
| `random_questions` | Shared random questions across all configurations | Same sampling, empirical accuracy/cost alternatives |
| `ucb` | UCB adaptive evaluations | Unsupported |
| `bayesian` | Bayesian optimization of complete configurations | Unsupported |

`random` is a single-mode alias for `random_configurations`; `matrix_ucb` is an
alias for `ucb`. Bayesian optimization requires `python -m pip install
"agentopt-py[bayesian]"` (or `python -m pip install -e '.[bayesian]'` in a checkout).
Gittins, random, and UCB are included in the standard installation.

### Set an evaluation allowance or dollar limit

Let K be the number of model configurations and N the number of examples. One
cell is one evaluation of a configuration on a question; the full grid has K × N
cells. For two nodes with two candidates each, K is four.

| Method | Fraction parameter and default | Allowed dollar limit |
|---|---|---|
| Gittins | `observation_budget_fraction=1.0` of K × N cells | `max_search_cost_usd=None` by default |
| Random configurations | `observation_budget_fraction=1.0`, spent on whole N-question configurations | None; actual spending is reported |
| Random questions | `observation_budget_fraction=1.0`, spent on whole K-configuration question sweeps | None; actual spending is reported |
| UCB | `observation_budget_fraction=1.0` of K × N cells | `max_search_cost_usd=None` by default |
| Bayesian optimization | `sample_fraction=0.25` of K configurations, at least two when available | `max_search_cost_usd=None` by default |

A 20% cell allowance does not generally mean 20% of distinct questions. For
`random_questions`, it means approximately 20% of questions shared by every
configuration; for `random_configurations`, approximately 20% of configurations,
each evaluated on every question. Fractions round up to cells, and random methods
only start whole units that fit, so some allowance may remain unused.

Gittins and random methods also accept `max_total_question_evaluations`, combined
with the fractional cap. Gittins must have enough cells for its warm start:
K × `min(warm_start_batch_size, N)`. Random methods require enough cells for at
least one whole configuration or shared-question sweep. A tiny dollar limit can
stop Gittins during warm start, before any recommendation is available.

Dollar limits apply to new non-cached search charges and are **soft**: scheduled
work can overshoot. Inspect `selection_cost` and `search_cost_overshoot_usd`.
Random methods do not accept `max_search_cost_usd`. All methods can use user
stopping with or without an evaluation allowance.

### Tune behavior beyond budgets

Pass selector parameters to `ModelSelector(...)`; pass execution controls to
`select_best(...)`.

| Control | Default and effect |
|---|---|
| `models`, `dataset`, `eval_fn` | You supply the search space, examples, and quality metric. |
| Gittins `cost_scaling_factor` | `1e-4`, single mode only. Scales raw USD search evaluation cost in its index/stopping calculation; it is not a dollar cap. |
| `lambda_cost`, `lambda_latency` | Both `0.0`, single mode only. Penalize normalized deployment cost/latency in the final scalar objective. These are separate from `cost_scaling_factor`. |
| Gittins `stop_on_recommendation` | `False`. Set `True` for automatic stopping on single-objective advice. |
| Gittins `warm_start_batch_size`, `batch_size` | Both `4`: initial questions per configuration, then questions per adaptive visit. A smaller adaptive batch gives more frequent updates and stop checks. |
| `select_best(parallel=False, max_concurrent=20)` | Execution defaults. The runner uses parallel evaluation with a limit of four. UCB uses `max_concurrent` for both its batch size and concurrency and ignores `parallel`. |
| `seed` | Gittins/random default `42`; UCB defaults to `None`. Controls selection sampling, not model-provider randomness. Bayesian optimization has no selector `seed` argument. |
| `score_bounds` | Gittins/random default `(0.0, 1.0)`. Must contain every score returned by the scorer. |
| `model_prices` | Optional mapping of model name to `input_price` and `output_price`, in USD per million tokens. Supply it for unknown models or price overrides. |
| `tracker` | Optional `LLMTracker`, for example with a `cache_dir` for disk caching. |

Gittins and random selectors require tracked token usage and known nonnegative
prices for every evaluated workflow. Changing the scorer or agent must preserve
that tracking. The [selector API reference](../api/selectors.md) also documents
Gittins' prior/noise settings, recommendation conservatism, multi-objective
search parameters, UCB exploration parameter `a`, and Bayesian `batch_size`.

## Contribute a benchmark example

To share an integration with other users, open a pull request to the
[AgentOpt repository](https://github.com/AgentOptimizer/agentopt) with:

1. A runnable example under `examples/selection/`, with the agent, data loading,
   and scoring code. Use `local/` for a framework-specific local integration.
2. A small redistributable sample or instructions to obtain the dataset, its
   source and license, and the split used for selection and final evaluation.
3. A README entry giving the installation steps, credential environment-variable
   names, run command, candidate models, metric, and evaluation limits. Include
   how to see progress, stop, and inspect the saved result.
4. Reproduction details for any reported results: dataset version, model IDs,
   selector parameters and seeds where supported, pricing assumptions, and
   whether cached responses were used.

Keep credentials out of the example and distinguish illustrative output from
measured results. Test data loading and scoring on the small sample; any automated
tests should be able to run without paid model calls. You do not need to contribute
a pull request to run your own private benchmark.
