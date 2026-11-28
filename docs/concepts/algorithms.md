# Selection Algorithms

Choose the output you need with `objective_mode`, then choose how to allocate evaluations with `method`. `method="auto"` uses Gittins in both modes.

| Method | Objective mode | Evaluation strategy | Recommendation |
|---|---|---|---|
| `auto` / `gittins` | `single` | Cost-aware Gittins index for the finite dataset's mean score | Highest posterior mean score, with an optional uncertainty penalty |
| `random` / `random_configurations` | `single` | Random configurations, each evaluated on all questions | Highest observed mean score among completed configurations |
| `random_questions` | `single` | Shared randomly ordered questions, evaluated on every configuration | Highest observed mean score on completed shared-question sweeps |
| `ucb` | `single` | UCB-E: mean score plus `sqrt(a / n)`; random unseen questions on the selected configuration | Highest observed mean score |
| `bayesian` | `single` | Categorical Gaussian process and Log Expected Improvement; complete configuration evaluations | Highest observed mean score |
| `auto` / `gittins` | `multi` | CC-Gittins alternating deployment and quality axes | Conservative accuracy–cost recommendation set |
| `random_configurations` | `multi` | Random configurations, each evaluated on all questions | Empirical accuracy–cost frontier among completed configurations |
| `random_questions` | `multi` | Shared randomly ordered questions, evaluated on every configuration | Empirical accuracy–cost frontier on the shared observed questions |

All random sampling is without replacement. Multi-objective selection does not designate a unique best configuration. Use `results.get_recommendations()` to inspect the set and `results.choose(...)` to express a deployment preference.

## Gittins

Gittins starts with a small uniform batch to estimate live evaluation costs. Single mode uses a fixed Gaussian prior and the posterior of the **complete finite dataset's average score** for sampling, stopping, and recommendation. Observed scores contribute directly; the posterior predicts unseen scores. A fully evaluated configuration has its exact empirical score and zero prediction uncertainty.

Each single-objective DP step represents one question. The numerical evaluation penalty is:

```text
DP evaluation cost = cost_scaling_factor × expected question cost in USD
```

`cost_scaling_factor` defaults to `1e-4`. A larger value favors an earlier stopping recommendation. It stays fixed during a single-objective run. When the best completed configuration is at least as valuable as every unfinished Gittins index, the result reports `stop_recommended=True`. By default, selection continues with the highest-index unfinished configuration until a user stop, a budget, or full completion. Set `stop_on_recommendation=True` to end automatically on that advice. The default configuration recommendation maximizes posterior mean (`recommendation_beta=0`); a positive beta subtracts a multiple of the predictive standard deviation.

The single-objective finite-dataset convention follows BanditGittinsEval. Live uniform warm-up and estimated monetary costs are package adaptations; they are not identical to experiments with known per-arm costs. The coefficient multiplies the supplied cost units: `1e-4` per USD is not numerically equivalent to `1e-4` per unit evaluation or per cost ratio.

Multi mode preserves score and deployment cost as separate objectives. It alternates a deployment-cost axis and a quality axis, shares every observation between the axes, and reduces each direction's eta when that direction stops. Its default recommendations use score mean minus one standard deviation and cost mean plus one standard deviation. These model-based coordinates concern the supplied dataset, not guaranteed future performance or recovery of the complete true frontier.

## Search cost and deployment preferences

`cost_scaling_factor` prices **search evaluations** in single-objective Gittins. `max_search_cost_usd` is a separate soft cap on new non-cached token charges. Neither parameter subtracts deployment cost from a configuration's score.

For a scalar deployment tradeoff, single-objective selectors accept `lambda_cost` and `lambda_latency` (both default to zero). For several tradeoffs, use multi mode and choose from its recommendations using a deployment cost limit, score floor, or cost weight. See the [selector reference](../api/selectors.md) for the normalization rules and budget semantics.

## Live recommendations

All methods can publish results through `select_best(on_update=...)` while the
search runs. A user can stop through a shared `stop_event`,
`selector.request_stop()`, or a callback that returns `False`. The selector
finishes the currently scheduled batch, including queued evaluations, and
returns its latest result without scheduling a following batch. Both random
methods support this interface in single and multi mode.

The recommendation reflects the evidence available at each update. Random
configuration sampling ranks completed configurations; random question sampling
uses completed shared-question sweeps. UCB ranks empirical estimates from its
partial observations. Bayesian optimization ranks configurations evaluated on
the full dataset. Gittins begins recommending after its uniform warm start and
uses finite-dataset posterior predictions. Thus an early stop can return
observations with no recommendation, and later observations can change a winner.

## Evaluation allowances and spending

Every method supports an evaluation fraction and anytime user stopping. With K
configurations and N questions, `observation_budget_fraction` for Gittins, UCB,
and random means a fraction of K × N configuration–question evaluations. It does
not generally mean that only that fraction of distinct questions will be used.

- Random configurations spend N evaluations per selected configuration, so the fraction approximately specifies a fraction of configurations.
- Random questions spend K evaluations per shared question, so the fraction approximately specifies a fraction of questions.
- Gittins and UCB allocate their cell allowance adaptively. Gittins includes its uniform warm start in that allowance.
- Bayesian optimization's `sample_fraction` counts configurations, each evaluated on the complete dataset, with a minimum of two configurations when available. It requires `pip install "agentopt-py[bayesian]"`.

The cell allowance is rounded up; random methods require whole units to fit and
can leave a remainder. The default evaluation allowance for random, UCB, and
Gittins is the complete grid; Bayesian optimization uses `sample_fraction=0.25`.

Random baselines report spending and offer evaluation limits, without a dollar
budget parameter. Gittins, UCB, and Bayesian optimization additionally accept
`max_search_cost_usd`, a soft cap on new non-cached charges. Gittins checks frozen
cost estimates before adaptive batches. UCB and Bayesian optimization check
actual spending before scheduling their next batch or complete configuration.
Work already scheduled can exceed the cap. A budget, a user's stop request,
and Gittins' model-based stopping advice express different reasons to end or
reconsider a search.

Gittins, UCB, and random baselines are included in the standard installation. See [advanced_algorithms.py](https://github.com/AgentOptimizer/agentopt/blob/main/examples/selection/local/advanced_algorithms.py) for runnable examples.
