# Selectors

`ModelSelector` evaluates combinations of models for an agent against a supplied dataset. `method="auto"` uses Gittins. `objective_mode="single"` returns one configuration; `objective_mode="multi"` returns an accuracy–cost recommendation set.

```python
from agentopt import ModelSelector

selector = ModelSelector(
    agent=MyAgent, models=models, eval_fn=eval_fn, dataset=dataset,
    method="auto", objective_mode="single", cost_scaling_factor=1e-4,
    max_search_cost_usd=10.0,
)
results = selector.select_best(parallel=True, max_concurrent=4)
print(results.get_best_combo())
```

## Common parameters

| Parameter | Meaning |
|---|---|
| `agent` | Class accepting a model dictionary in `__init__` and implementing `run(input_data)`. |
| `models` | Node names mapped to nonempty candidate lists. |
| `dataset` | Nonempty sequence of `(input_data, expected_answer)` pairs. |
| `eval_fn` | `(expected, actual) -> float`, higher is better. Gittins and random selectors validate against `score_bounds`. |
| `method` | Algorithm name; default `"auto"` uses Gittins. |
| `objective_mode` | `"single"` (default) or `"multi"`. Invalid method/mode combinations fail before tracking starts. |
| `model_prices` | Custom prices in USD per million tokens: `{"model": {"input_price": x, "output_price": y}}`. |
| `tracker` | Optional `LLMTracker`; otherwise one is created. Tracking starts in the constructor and stops when `select_best()` returns or raises. |
| `lambda_cost`, `lambda_latency` | Optional deployment penalties for single mode, both default `0.0`. See below. |

## Methods

| `method` | Single | Multi |
|---|---|---|
| `auto`, `gittins` | Gittins finite-dataset winner | CC-Gittins two-axis recommendation set |
| `random` | Random complete configurations | — |
| `ucb` | UCB-E over configuration–question observations | — |
| `bayesian` | Categorical GP with Log Expected Improvement | — |
| `random_configurations` | Random complete configurations | Random complete configurations |
| `random_questions` | Shared random questions across all configurations | Shared random questions across all configurations |

`random` is the single-mode alias for `random_configurations`. `matrix_ucb` is an alias for `ucb`. Bayesian optimization requires the optional `agentopt-py[bayesian]` dependencies; the other methods are available in the standard installation.

## Live recommendations and user stopping

Every supported method accepts `select_best(on_update=None, stop_event=None)`
in addition to `parallel` and `max_concurrent`. `on_update(snapshot)` receives an
independent result snapshot after completed work and once more with
`is_final=True` when selection finishes. Inspect `total_evaluations`,
`selection_cost`, `selection_wall_time_seconds`, and the current recommendation
to update a display. In multi mode, display `get_recommendations()` instead of a
single winner.

```python
from threading import Event

stop = Event()

def update_display(snapshot):
    print(f"Evaluations: {snapshot.total_evaluations}")
    print(f"Search cost: {snapshot.selection_cost}")
    if snapshot.get_best_combo() is not None:
        print(f"Current recommendation: {snapshot.get_best_combo()}")
    if snapshot.is_final:
        print(f"Stopped: {snapshot.stop_reason}")

results = selector.select_best(
    parallel=True, max_concurrent=4,
    on_update=update_display, stop_event=stop,
)
```

A UI action or another thread can call `stop.set()` or
`selector.request_stop()`. A callback can request a stop by returning `False`;
returning `None` continues selection. `selector.latest_results` returns a copy
of the most recently published snapshot, or `None` before the first update.
The callback runs in the selection thread, so keep it brief and follow your UI
framework's rules for updating widgets from a worker thread.

Stopping is cooperative: the currently scheduled batch finishes, including
work queued behind the concurrency limit. Its observations and costs are
recorded, and no following batch is scheduled. The final reason is
`user_stopped`. A stop request does not cancel an LLM request or refund its cost.
Progress and stopping checks occur at these boundaries:

| Method | Update boundary | Recommendation readiness |
|---|---|---|
| Gittins | Each warm-start evaluation, then each adaptive batch | No recommendation until the uniform warm start finishes; subsequent recommendations use finite-dataset predictions. |
| Random configurations | Each sequential evaluation or concurrent chunk of questions | Only complete configurations are eligible. |
| Random questions | Each configuration evaluation within a shared-question sweep | Only complete sweeps contribute to recommendations. |
| UCB | Each acquisition batch | Ranks observed configurations by their current empirical scalar means; these are partial-data estimates without calibrated uncertainty. |
| Bayesian optimization | Each complete configuration, or completed parallel acquisition batch | Ranks fully evaluated configurations. |

A snapshot can contain observations but no eligible recommendation. Further
evaluations may change a recommendation in either direction; the displayed
score is not a guarantee of improvement or future performance.

Budgets are optional, and every run has a finite dataset. Use a fraction to
choose the evaluation allowance in advance, a user stop to decide while watching
results, or both. Gittins, UCB, and Bayesian optimization also accept an optional
soft dollar cap. Random baselines report spending without using a dollar cap.

## Evaluation fractions and budgets

Let K be the number of configurations and N the number of dataset questions.
A configuration–question **cell** is one agent evaluation; the full grid has
K × N cells. A cell fraction is not a fraction of distinct dataset questions:
adaptive methods can allocate different questions to each configuration.

| Method | Fraction parameter and default | How the allowance is spent | `max_search_cost_usd` |
|---|---|---|---|
| Gittins | `observation_budget_fraction=1.0` of K × N cells | Uniform warm start, then adaptive batches | Optional soft cap, with estimates before adaptive batches |
| Random configurations | `observation_budget_fraction=1.0` of K × N cells | Whole configurations, N cells each | Unsupported; spending is reported |
| Random questions | `observation_budget_fraction=1.0` of K × N cells | Shared-question sweeps, K cells each | Unsupported; spending is reported |
| UCB | `observation_budget_fraction=1.0` of K × N cells | Adaptive cell batches | Optional soft cap, checked between acquisition batches |
| Bayesian optimization | `sample_fraction=0.25` of K configurations | Complete configurations; at least two when available | Optional soft cap, checked between configurations or parallel acquisition batches |

Cell fractions are rounded up to a whole cell. Random baselines start only
complete units that fit, so a remainder can be unused: a `0.5` fraction means
approximately half the configurations for `random_configurations`, or half the
questions shared across all configurations for `random_questions`. Gittins and
random also accept `max_total_question_evaluations` as an absolute cell cap.
Gittins' cap must fund its warm start; random's must fund one complete unit.

Single-objective Gittins additionally publishes `stop_recommended`. Advice does
not end the run by default: `stop_on_recommendation=False` keeps selecting
unfinished configurations until a user stop, a budget, or full completion.
Set it to `True` to end automatically with `stop_reason="gittins_stop"`.
Multi-objective Gittins retains its own direction/eta termination rule.
The advice can change as further observations arrive.
A recommendation to stop, `user_stopped`, and budget exhaustion are separate
signals; inspect `is_final` and `stop_reason` to tell whether a run has ended.

## Combined objective (optional cost/latency weights)

Single mode optimizes the evaluation score by default. To express a scalar deployment preference:

```text
objective = score - lambda_cost × norm(cost) - lambda_latency × norm(latency)
```

Gittins fixes its min/max scales from warm-start observations; subsequent values can lie outside those initial ranges. Random selection uses common min/max scales from completed sampling units. UCB and Bayesian optimization use running min/max scales across observed samples and refresh the stored scalar objectives. A constant or unavailable scale contributes zero; a positive cost weight requires known prices.

These deployment weights are separate from Gittins' **search-cost coefficient** `cost_scaling_factor`. The latter defaults to `1e-4` in single mode and multiplies raw USD evaluation costs inside the DP. It does not change the final deployment score. Multi mode rejects nonzero deployment lambda weights; use `results.choose(...)` instead.

## Gittins: single and multi objective

Each configuration receives a seeded independent permutation of question indices and a uniform warm start. The live tracker supplies token attribution and pricing. Every evaluation must have tracked token usage and all recorded calls must have known nonnegative prices; supply `model_prices` for custom models. Evaluation or pricing failures abort selection and stop tracking.

`parallel=True` runs questions within the selected adaptive batch concurrently, up to `max_concurrent`. Warm-start questions run sequentially so a dollar cap can be checked after each evaluation. Async agents work in both modes.

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `objective_mode` | `"single"` | One scalar winner or a two-objective recommendation set. |
| `batch_size` | `4` | Questions evaluated per adaptive visit. |
| `warm_start_batch_size` | `4` | Questions per configuration before adaptation, clamped to dataset size. |
| `observation_budget_fraction` | `1.0` | Maximum fraction of configuration–question cells, including warm start. |
| `max_total_question_evaluations` | `None` | Absolute cell limit, combined with the fraction cap. |
| `max_search_cost_usd` | `None` | Soft cap on new non-cached token charges, including warm start. |
| `cost_scaling_factor` | `1e-4` | **Single only:** fixed search penalty per USD of expected evaluation cost. |
| `prior_mean` | `0.5` | **Single only:** fixed prior mean in normalized score units (also used for the optional scalar objective). |
| `recommendation_beta` | `None` | Resolves to `0.0` in single mode and `1.0` in multi mode. |
| `recommendation_min_samples` | `0` | Minimum observed questions for recommendation eligibility. |
| `score_bounds` | `(0.0, 1.0)` | Finite permitted score range; normalized affinely for inference. |
| `prior_variance` | `0.04` | Normalized accuracy/scalar prior variance. |
| `score_noise_variance` | `0.25` | Per-question accuracy noise variance. Weighted single mode scales scalar noise with the deployment weights. |
| `cost_reference_usd` | `None` | Multi cost-axis scale, inferred from the largest warm configuration mean; one USD if all warm costs are zero. |
| `search_cost_scale_eta` | `1.0` | **Multi only:** scale applied to workflow-cost continuation penalties. |
| `eta_initial`, `eta_decay` | `1.0`, `0.5` | **Multi only:** initial per-direction eta and decay on a direction stop. |
| `effective_cost_bin_ratio` | `2.0` | **Multi only:** cost binning for DP reuse; `None` disables it. |
| `effective_cost_bin_anchor` | `1e-4` | **Multi only:** USD anchor for cost bins. |
| `grid_size` | `129` | DP interpolation resolution, at least 17. |
| `stop_tolerance` | `1e-9` | Numerical stopping comparison tolerance. |
| `stop_on_recommendation` | `False` | **Single only:** end automatically when Gittins recommends stopping; otherwise publish the advice and continue. |
| `seed` | `42` | Nonnegative seed for question permutations. |

### Single-objective index and stopping

Single mode uses the posterior mean of the complete finite dataset for acquisition, stopping, and recommendation. With N questions, n observed values with sum S, and a Gaussian latent posterior `(m, v)`:

```text
finite mean = (S + (N - n) × m) / N
finite variance = ((N - n)² × v + (N - n) × question_noise) / N²
```

The DP has one transition per question, even when the next selected configuration is evaluated in a batch. The finite-mean transition variance is the latent-mean transition variance multiplied by `a²`, where `a = 1 + question_noise / (N × prior_variance)`. Costs retain their monetary scale:

```text
DP cost per question = cost_scaling_factor × warm mean question cost in USD
```

Costs and lambda remain fixed after warm start. A tiny positive numerical floor (`1e-12`) handles zero or extremely small DP costs. The cost coefficient has units of normalized score per USD; changing cost units requires changing this coefficient. The default `1e-4` does not guarantee a particular evaluation budget.

A completed configuration's index is its exact empirical scalar mean. When the best completed value reaches the largest unfinished index, single mode publishes `stop_recommended=True`. This is a model-based assessment of whether more evaluation is worth its cost. The default `stop_on_recommendation=False` continues with the highest-index unfinished configuration; it does not reduce the search-cost coefficient. Set `stop_on_recommendation=True` to end on the advice with `stop_reason="gittins_stop"`. User stopping, cell/dollar budgets, and full completion can also end a run.

The winner maximizes `finite_mean - recommendation_beta × finite_std`, with lower conservative deployment cost breaking ties. At the default beta of zero this is posterior-mean recommendation. Warm-start calibration of unknown live costs is an adaptation to online evaluation; configurations with known fixed costs can have different trajectories in an offline replay.

### Multi-objective search and recommendations

Multi mode follows the CC-Gittins deployment and quality axes in 1:1 round-robin order. Both directions share all observations. Warm observations determine the Gaussian plug-in calibration and expected continuation costs, which remain fixed. When a direction stops, its eta decreases; the run ends when both directions stop at numerical eta floors without an intervening observation, or when a budget/full completion ends it.

Recommendations are the strict non-dominated set of `(accuracy_mean - beta × accuracy_std, cost_mean + beta × cost_std)` over eligible configurations, using finite-dataset moments. Two search axes do not limit the output to two configurations. Multi mode does not set a unique `is_best`: `get_best()` and `get_best_combo()` return `None`, and `export_config()` cannot export a unique best configuration.

```python
results = ModelSelector(
    agent=MyAgent, models=models, eval_fn=eval_fn, dataset=dataset,
    objective_mode="multi", max_search_cost_usd=10.0,
).select_best()
for candidate in results.get_recommendations():
    print(candidate.combo, candidate.conservative_accuracy, candidate.conservative_cost_usd)
chosen = results.choose(max_cost_usd=0.01)  # highest score within cost limit
# Also supported: choose(min_score=0.85), choose(cost_weight=20.0)
if chosen is not None:
    print(chosen.combo)
```

`choose(cost_weight=w)` maximizes conservative score minus `w × raw_USD`. Infeasible preferences return `None`. The returned recommendation does not change `is_best`. Model-based estimates describe the supplied finite dataset; they do not guarantee joint coverage, performance on future questions, or the complete true frontier. The two-axis protocol and offline reproductions are described in [AgentOpt Research](https://github.com/AgentOptimizer/agentopt-research).

### Budget accounting

The cell cap must fund the uniform warm start. A dollar cap reached during warm start returns observed records, `stop_reason="warm_start_budget"`, and no recommendations. Search spending excludes cache hits; deployment costs include their recorded workflow usage.

Before adaptive evaluation, Gittins checks the planned batch's frozen estimate of new charges. Actual charges can exceed that estimate; the result reports the excess as `search_cost_overshoot_usd`. Unused budget can remain when the next complete adaptive batch does not fit. A user-supplied cell cap is enforced without overshoot.

## Random baselines

`random_configurations` selects configurations without replacement and evaluates each on every question. `random_questions` selects questions without replacement and evaluates each on every configuration. Both support single and multi mode; `random` is the single-mode configuration-sampling alias. Single mode recommends the highest empirical scalar mean among eligible configurations; multi mode returns their empirical accuracy–cost frontier. No all-configuration warm start or Gaussian model is used.

No budget argument is required: by default, both methods evaluate the finite configuration–question grid unless the user stops them. Both accept `observation_budget_fraction=1.0`, `max_total_question_evaluations=None`, `seed=42`, and `score_bounds=(0, 1)`. The fraction refers to K × N cells. The cap must fund a whole configuration (N cells) or shared-question sweep (K cells), and each new unit must fit the remaining cell cap. Thus an unused cell remainder is possible.

Random baselines do not accept `max_search_cost_usd`. They allocate an evaluation allowance independently of measured cost and report actual non-cached search spending in `selection_cost`. Configuration sampling can evaluate its questions concurrently. A user stop can interrupt a unit; its observations remain in evaluation records but are excluded from recommendations. For shared questions, only fully completed sweeps enter single-mode rankings or the multi-mode empirical frontier. This gives every eligible configuration the same observed questions.

`RandomSelectionResults` supports `get_recommendations()` and `choose(...)` and adds `sampling` and `completed_units`. Recommendation score/cost are empirical means on complete units. Conservative fields equal these means; zero standard-deviation fields indicate no uncertainty model, not certainty about unseen data.

## UCB and Bayesian optimization

UCB (`ucb`, alias `matrix_ucb`) uses `mean + sqrt(a/n)`, with `a=1.0`, and samples unseen questions on the selected configuration. `observation_budget_fraction=1.0` caps cells; `sample_fraction` is an alias that takes precedence. `seed` controls question sampling. `max_concurrent` controls both the batch size and concurrency, and `parallel` is ignored. If an entire acquisition batch fails to yield an observation, UCB stops with `stop_reason="evaluation_failed"`.

Bayesian optimization uses a categorical `MixedSingleTaskGP` and Log Expected Improvement. Each selected configuration is evaluated over the complete dataset. If some questions fail, successful observations remain in the results, but that configuration is excluded from GP training and recommendations. `sample_fraction=0.25` caps the number of configurations (at least two if available); `batch_size=1` controls configurations per GP iteration. Both baselines support scalar deployment weights, single-objective recommendations, evaluation fractions, anytime user stopping, and `max_search_cost_usd=None`.

Their dollar cap counts new non-cached token charges, including evaluation and judge calls. Costs must be known when a cap is requested. The cap is checked before scheduling further work: UCB can exceed it by the current acquisition batch; Bayesian optimization can exceed it by the current complete configuration, or by all configurations in the current parallel acquisition batch. Smaller batches reduce the work between spending checks. These are soft spending limits, not guarantees of an exact dollar total.

## Selector classes

::: agentopt.model_selection.gittins.GittinsModelSelector
    options:
      members: false
      show_bases: false

::: agentopt.model_selection.random_search.RandomSearchModelSelector
    options:
      members: false
      show_bases: false

::: agentopt.model_selection.matrix_ucb.MatrixUCBModelSelector
    options:
      members: false
      show_bases: false

::: agentopt.model_selection.bayesian_optimization.BayesianOptimizationModelSelector
    options:
      members: false
      show_bases: false
