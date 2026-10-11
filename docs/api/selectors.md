# Selectors

A `ModelSelector` evaluates model **combinations** for an agent against a dataset. Scalar methods return one winner; Gittins also supports an accuracy/cost recommendation set. All selectors accept an agent, candidate models, a dataset, and `eval_fn`, and run through `select_best()`.

```python
from agentopt import ModelSelector

selector = ModelSelector(
    agent=MyAgent,
    models={"planner": ["gpt-4o", "gpt-4o-mini"], "solver": ["gpt-4o-mini"]},
    eval_fn=lambda expected, actual: float(actual == expected),
    dataset=[(inp, expected), ...],
    method="auto",                # arm_elimination — strong + cheap
)
results = selector.select_best(parallel=True, max_concurrent=20)
results.print_summary()
```

## Common parameters

| Parameter | Type | Description |
|:---|:---|:---|
| `agent` | `type` | Agent class with `__init__(self, models)` and `run(self, input_data)`. Duck-typed — no base class required. |
| `models` | `Dict[str, List]` | Maps node names to candidate model lists (e.g. `{"planner": ["gpt-4o", "gpt-4o-mini"]}`). |
| `eval_fn` | `Callable` | `(expected, actual) -> float` score (higher is better). |
| `dataset` | `Sequence[Tuple]` | `[(input_data, expected_answer), ...]`. |
| `model_prices` | `Dict`, optional | Custom pricing overrides: `{"model": {"input_price": x, "output_price": y}}` in $/MTok. Required for cost terms when `lambda_cost > 0`. |
| `lambda_cost` | `float`, optional | Weight on **normalized** per-sample cost in the combined objective. Default `0.0` (disabled). See [Combined objective](#combined-objective-optional-costlatency-weights) below. |
| `lambda_latency` | `float`, optional | Weight on **normalized** per-sample latency in the combined objective. Default `0.0` (disabled). |
| `tracker` | `LLMTracker`, optional | Bring your own. Defaults to a fresh `LLMTracker()` started in the constructor. Pass one in to share a cache across runs, route via a daemon (`AGENTOPT_GATEWAY_URL`), or post-process records after `select_best()` returns. |

The selector calls `tracker.start()` in the constructor and `tracker.stop()` when `select_best()` returns or raises. Record queries on the tracker remain valid after `stop()`, so post-run analysis works:

```python
tracker = LLMTracker(cache_dir="./shared_cache")
selector = ModelSelector(..., tracker=tracker)
selector.select_best()
print(tracker.get_usage())          # tracker.stop() already called; records still here
```

See [tracker.md](tracker.md) for the full tracker surface.

## Combined objective (optional cost/latency weights)

By default, selectors optimize **`eval_fn` score only** (typically accuracy) and break ties with latency, then price. To trade accuracy against cost and latency in one scalar reward, pass optional weights on the constructor (or via `ModelSelector(..., **kwargs)`):

| Parameter | Default | Effect |
|:---|:---|:---|
| `lambda_cost` | `0.0` | Penalizes normalized per-sample **token cost** (USD from the tracker, or `model_prices`). |
| `lambda_latency` | `0.0` | Penalizes normalized per-sample **wall-clock latency** (seconds). |

Omit both parameters (or leave them at `0.0`) for the original accuracy-centric behavior. Set one or both when you want multi-metric selection.

### Formula

For each datapoint, after observations are recorded:

```
combined = score
         - lambda_cost    * norm(cost)
         - lambda_latency * norm(latency)
```

- **`score`** — return value of `eval_fn` (higher is better).
- **`norm(·)`** — min–max scale to `[0, 1]` using running min/max over **all** samples seen during that selector run (updated as more combos are evaluated).
- **Per combination** — mean of per-datapoint combined values → `ModelResult.combined_objective` (see [results.md](results.md)).

This is a **linear scalarization**, not Pareto exploration. Larger `lambda_*` penalize cost/latency more strongly relative to score.

### Example

```python
selector = ModelSelector(
    agent=MyAgent,
    models=models,
    eval_fn=eval_fn,
    dataset=dataset,
    method="matrix_ucb",
    lambda_cost=0.3,      # optional — omit for accuracy-only
    lambda_latency=0.2,
    model_prices={        # recommended when lambda_cost > 0
        "gpt-4o": {"input_price": 2.5, "output_price": 10.0},
        "gpt-4o-mini": {"input_price": 0.15, "output_price": 0.6},
    },
)
results = selector.select_best(parallel=True)
results.print_summary()   # ranks by combined_objective when lambdas are set
```

### How each method uses the weights

| Methods | During search | Final `is_best` |
|:---|:---|:---|
| `matrix_ucb`, `matrix_ucb_lrf` | UCB rewards use per-cell combined objective | `_find_best` on `combined_objective` |
| `arm_elimination` | Elimination stats on combined per-sample objectives | same |
| `bayesian` | Surrogate target uses combined objective | same |
| `brute_force` | Does not steer *which* combos to try | same |

After `select_best()`, a final pass recomputes every result’s `combined_objective` against the **full-run** normalizer so rankings are comparable.

Gittins single mode freezes its cost/latency min/max scales after uniform warm
start, so the scalar posterior and DP refer to one fixed objective throughout
search. Later values outside those scales are not clipped. Multi mode keeps
accuracy and mean USD separate and rejects scalar lambda weights.

## `select_best()`

```python
results = selector.select_best(
    parallel=False,        # If True, evaluate combos concurrently with asyncio
    max_concurrent=20,     # Total concurrent API-call budget across all combos
)
```

Returns a [`SelectionResults`](results.md). `parallel=True` requires `agent.run` to be either async or threadsafe; the selector splits `max_concurrent` between outer (combos) and inner (datapoints) loops based on dataset size.

!!! note "Automatic cleanup"
    `select_best()` calls `tracker.stop()` on return or exception — caches flush to disk, masters tear down. The tracker remains queryable; only `close()` (which `select_best` does not call) drops the remote backend's HTTP client.

## Choosing a method

| `method` | Algorithm | When to use |
|:---|:---|:---|
| `"auto"` (default) | Arm elimination | Strong best-arm identification at lower search cost than brute force. Same impl as `"arm_elimination"`. |
| `"brute_force"` | Evaluate every combo on the full dataset | Small search space; ground-truth comparison. |
| `"arm_elimination"` | Successive elimination | Best-arm identification with PAC-style guarantees. |
| `"matrix_ucb"` / `"matrix_ucb_lrf"` | UCB exploiting cross-combo structure | Large model x datapoint matrices; `lrf` adds low-rank factorization. |
| `"bayesian"` | Bayesian optimization | Optional extra: `pip install "agentopt-py[bayesian]"`. |
| `"gittins"` | Cost-coupled Gittins | Adaptive batches with a scalar winner or an accuracy/cost recommendation set. Optional extra: `pip install "agentopt-py[gittins]"`. |

## Gittins: single and multi objective

Gittins collects a uniform warm batch on every combination, freezes its
Gaussian calibration and expected continuation costs, then chooses which
combination to evaluate next. The user supplies an ordinary dataset and
agent; a complete combination-question matrix is unnecessary. The live
tracker supplies token attribution, pricing, and cache information.

```python
single = ModelSelector(
    agent=MyAgent, models=models, eval_fn=eval_fn, dataset=dataset,
    method="gittins", objective_mode="single", max_search_cost_usd=10.0,
)
single_results = single.select_best()
print(single_results.get_best_combo())

multi = ModelSelector(
    agent=MyAgent, models=models, eval_fn=eval_fn, dataset=dataset,
    method="gittins", objective_mode="multi", max_search_cost_usd=10.0,
)
multi_results = multi.select_best(parallel=True, max_concurrent=4)
for candidate in multi_results.get_recommendations():
    print(candidate.combo, candidate.accuracy, candidate.cost_usd)
chosen = multi_results.choose(max_cost_usd=0.01)
if chosen is not None:
    print(chosen.combo)
```

Each combination has an independent seeded question permutation. Adaptive
batches evaluate one combination at a time; `parallel=True` allows concurrent
questions within that batch, up to `max_concurrent`. Warm questions run one
at a time so spending can be checked after every request. Async agents are
supported in both sequential and parallel modes.

### Parameters

| Parameter | Default | Meaning |
|:---|:---|:---|
| `objective_mode` | `"single"` | `"single"` optimizes score or a weighted scalar objective; `"multi"` keeps score and mean USD separate. |
| `batch_size` | `4` | Adaptive questions per visit; the final dataset tail may be smaller. |
| `warm_start_batch_size` | `4` | Initial questions per combination, clamped to dataset size. |
| `observation_budget_fraction` | `1.0` | Maximum fraction of combination-question cells, including warm start; not a dollar fraction. |
| `max_total_question_evaluations` | `None` | Optional absolute cell-count limit, combined with the fraction limit. |
| `max_search_cost_usd` | `None` | Soft USD search cap on actual non-cached token charges, including warm start. |
| `recommendation_beta` | `1.0` | Standard-deviation penalty used to construct conservative recommendations. |
| `recommendation_min_samples` | `0` | Minimum observed questions for recommendation eligibility; does not alter acquisition. |
| `score_bounds` | `(0.0, 1.0)` | Allowed finite `eval_fn` range; accuracy is affinely normalized. |
| `prior_variance` | `0.04` | Normalized accuracy/scalar prior variance. |
| `score_noise_variance` | `0.25` | Per-question normalized accuracy noise. Weighted single mode scales its noise with the scalar weights. |
| `cost_reference_usd` | `None` | Frozen cost scale; inferred from the largest warm arm mean, or one USD if all warm costs are zero. |
| `search_cost_scale_eta` | `1.0` | Scale for workflow-cost continuation penalties; distinct from deployment preference weights. |
| `eta_initial` / `eta_decay` | `1.0` / `0.5` | Per-direction initial multiplier and decay on a direction stop. |
| `effective_cost_bin_ratio` | `2.0` | Multiplicative continuation-cost bins for DP reuse; `None` disables binning. |
| `effective_cost_bin_anchor` | `1e-4` | USD anchor for cost bins. |
| `grid_size` | `129` | Axis DP interpolation resolution, at least 17. |
| `stop_tolerance` | `1e-9` | Direction stopping comparison and numerical-floor tolerance. |
| `seed` | `42` | Nonnegative seed for question permutations. |

Every evaluation must produce tracked LLM token usage, and all recorded calls
must have known nonnegative prices. Supply `model_prices`
when the built-in table does not cover a model. Evaluation or pricing failures
abort the run and stop tracking, rather than silently dropping outcomes.

### Budget and stopping

The cell cap must fund the mandatory uniform warm start; otherwise the
constructor rejects it before tracking starts. If the dollar cap is reached
during warm start, the result contains the collected evaluation records,
`stop_reason="warm_start_budget"`, and an empty recommendation set.

Search spending excludes cache hits. Deployment cost estimates include their
recorded workflow token usage. Before an adaptive batch, the selector reserves
its predicted **new charge**, using the corresponding arm's frozen warm-start
charge per question. Actual batch costs can differ, so a dollar cap is not a
hard guarantee. The selector checks actual spend after each completed batch
and reports `search_cost_overshoot_usd`. A nonzero unused budget can remain
when the next planned batch does not fit.

At a fixed eta, a direction stops when the best fully evaluated combination's
terminal value reaches the largest unfinished Gittins index. That direction
then reduces its own eta and is reconsidered on a later visit. Multi mode
visits the deployment axis and quality axis in 1:1 round-robin order; both
share all observed outcomes. Each new observation invalidates earlier floor
stop decisions.

The run ends when the cell or dollar guard rejects further evaluation, every
combination is complete, or all directions stop at their numerical eta floors
without an intervening observation. It does not stop based on unchanged
recommendations or knowledge of the true frontier. `stop_reason` identifies
the condition that actually ended the run.

### Recommendations

The result contains the recommendation from the actual stopping state.
For a dataset of N questions, with n observed outcomes of sum S and Gaussian
latent posterior `(m, v)`, its realized mean has predictive moments:

```text
mean = (S + (N − n) × m) / N
variance = ((N − n)² × v + (N − n) × question_noise) / N²
```

Single mode chooses the combination with the highest conservative scalar
value `mean − beta × std` and marks its empirical `ModelResult` as `is_best`.
Its predicted value is exposed separately on the recommendation. Full
evaluation makes predictive uncertainty zero.

Multi mode takes the strict non-dominated set of
`(accuracy_mean − beta × accuracy_std, cost_mean + beta × cost_std)` across
all eligible combinations. Two search axes do not limit the output to two
members. Multi mode marks no single `is_best`; `get_best_combo()` returns
`None`. Use `get_recommendations()` or choose a deployment preference:

```python
chosen = results.choose(max_cost_usd=0.01)  # highest conservative score within cost limit
chosen = results.choose(min_score=0.85)    # lowest conservative cost above score floor
chosen = results.choose(cost_weight=20.0)  # conservative score minus 20 * raw USD
```

Infeasible preferences return `None`. A minimum-sample gate can make the
recommendation set empty. These Gaussian estimates concern the supplied
finite dataset; they provide neither joint confidence coverage nor a
generalization or complete-frontier guarantee.

The two-axis statistical protocol is described in
[AgentOpt Research](https://github.com/AgentOptimizer/agentopt-research).
The replay repository contains frozen matrices, comparison baselines, and
reproduction scripts; live selection uses the observations collected by the
agent and tracker.

---

## Selector Classes

::: agentopt.model_selection.brute_force.BruteForceModelSelector
    options:
      members: false
      show_bases: false

::: agentopt.model_selection.arm_elimination.ArmEliminationModelSelector
    options:
      members: false
      show_bases: false

::: agentopt.model_selection.matrix_ucb.MatrixUCBModelSelector
    options:
      members: false
      show_bases: false

::: agentopt.model_selection.matrix_ucb.MatrixUCBLRFModelSelector
    options:
      members: false
      show_bases: false

::: agentopt.model_selection.bayesian_optimization.BayesianOptimizationModelSelector
    options:
      members: false
      show_bases: false

::: agentopt.model_selection.gittins.GittinsModelSelector
    options:
      members: false
      show_bases: false
