# Parallel Evaluation

`select_best(parallel=True, max_concurrent=20)` permits concurrent evaluations where the selected algorithm supports them. The limit controls concurrent agent evaluations; an agent can itself issue several LLM calls within one evaluation.

| Selector | Parallel behavior |
|---|---|
| Gittins | One configuration per adaptive visit, with at most `min(batch_size, max_concurrent)` questions in flight. Warm-start questions are sequential for dollar accounting. |
| Random configurations | One configuration at a time; questions run in chunks of at most `max_concurrent` when `parallel=True`. |
| Random questions | Sequential shared-question sweeps over all configurations. |
| UCB | `max_concurrent` controls both acquisition batch size and evaluation concurrency. The `parallel` flag is ignored. |
| Bayesian optimization | Splits slots between configurations in an acquisition batch and the questions for each configuration. |

```python
results = selector.select_best(parallel=True, max_concurrent=4)
```

Gittins' statistical batch size is configured on the constructor (`batch_size`), independently of the concurrency limit. For single-objective Gittins, the DP still models one question per transition. A final dataset tail can contain fewer questions than the configured batch size.

Gittins, UCB, and Bayesian optimization accept a soft dollar cap. Gittins checks estimated charges before adaptive batches, but actual charges can exceed the estimate. UCB checks spending between acquisition batches; Bayesian optimization checks between complete configurations or parallel acquisition batches. Already scheduled work finishes and can exceed the cap. Inspect `selection_cost` and `search_cost_overshoot_usd`. Smaller batches reduce the work between checks. Random baselines report costs without a dollar cap.

## Progress and cooperative stopping

`on_update` receives a result snapshot after completed work. Returning `False`,
setting the supplied `stop_event`, or calling `selector.request_stop()` prevents
a following batch from being scheduled. The currently scheduled batch finishes,
including evaluations queued behind the concurrency limit; its results and
charges belong to the final snapshot. Its `is_final` field is `True` and its reason is `user_stopped`.

For Gittins, the response time depends on the adaptive batch size, not just the
concurrency limit: a batch can contain more questions than can run at once.
Random configuration sampling checks for stopping between question chunks;
random question sampling checks between configuration evaluations. UCB checks
after an acquisition batch. Bayesian optimization completes the currently
scheduled configuration or parallel acquisition batch. Smaller batches allow
more frequent user decisions, while larger batches permit more concurrent work.

Use [live selection](../api/selectors.md#live-recommendations-and-user-stopping)
for a callback and event example. Parallel execution does not change when a
recommendation is eligible: incomplete random configurations or shared-question
sweeps remain excluded even when some of their evaluations have finished.
