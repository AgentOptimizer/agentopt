"""Cost-coupled Gittins selection over live configuration-question evaluations.

Single-objective selection uses fixed-cost finite-dataset Gittins indices.
Multi-objective selection uses the exact deployment and quality axes and
returns a finite-LCB Pareto set. Both modes share observations across visits,
freeze warm-start costs. Multi mode decays eta independently on direction stops.
"""

from __future__ import annotations

import asyncio
import math
import operator
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

from ..model_price import compute_price
from .axis_gittins_dp import AxisGittinsBoundaryCache
from .base import BaseModelSelector, ModelResult, SelectionResults
from .gittins_statistics import (
    GaussianPosterior, finite_dataset_moments, fit_warm_start, raw_pareto_front_indices,
)


class GittinsRecommendation(BaseModel):
    """A recommended configuration with finite-dataset Gaussian predictions.

    ``accuracy`` and ``cost_usd`` are predictions for the supplied dataset.
    Conservative coordinates use mean minus beta*std for score, and mean
    plus beta*std for cost. They are model estimates, not coverage guarantees.
    """

    model_name: str
    combo: dict[str, str]
    observed_samples: int
    accuracy: float
    accuracy_std: float
    cost_usd: float
    cost_std_usd: float
    conservative_accuracy: float
    conservative_cost_usd: float
    objective: float | None = None
    conservative_objective: float | None = None


class GittinsSelectionResults(SelectionResults):
    """Evaluation records plus the recommendation at the actual stopping time.

    Multi mode marks no arbitrary single ``is_best``. Use ``choose`` to
    express a deployment preference, or inspect ``recommendations`` directly.
    """

    objective_mode: str
    recommendations: list[GittinsRecommendation] = Field(default_factory=list)
    stop_reason: str
    total_evaluations: int
    search_cost_overshoot_usd: float = 0.0
    direction_eta: list[float] = Field(default_factory=list)

    def get_recommendations(self) -> list[GittinsRecommendation]:
        return list(self.recommendations)

    def choose(self, *, max_cost_usd=None, min_score=None, cost_weight=None):
        """Choose a deployment candidate using conservative score/cost.

        With a cost limit, maximize score among feasible candidates. With
        only a score floor, minimize cost. A weight maximizes score minus
        weight * raw USD. Infeasible constraints return ``None``. This does
        not change ``is_best`` or guarantee performance on future questions.
        """
        for name, value in (("max_cost_usd", max_cost_usd), ("min_score", min_score),
                            ("cost_weight", cost_weight)):
            if value is not None and (not math.isfinite(value) or (name != "min_score" and value < 0)):
                raise ValueError(f"{name} must be finite" + (" and nonnegative" if name != "min_score" else ""))
        if all(value is None for value in (max_cost_usd, min_score, cost_weight)):
            if len(self.recommendations) > 1:
                raise ValueError("Specify a cost limit, score floor, or cost weight to choose a combination")
            return self.recommendations[0] if self.recommendations else None
        feasible = [r for r in self.recommendations
                    if (max_cost_usd is None or r.conservative_cost_usd <= max_cost_usd)
                    and (min_score is None or r.conservative_accuracy >= min_score)]
        if not feasible:
            return None
        if cost_weight is not None:
            return max(feasible, key=lambda r: (r.conservative_accuracy - cost_weight * r.conservative_cost_usd,
                                               -r.conservative_cost_usd))
        if max_cost_usd is not None:
            return max(feasible, key=lambda r: (r.conservative_accuracy, -r.conservative_cost_usd))
        return min(feasible, key=lambda r: (r.conservative_cost_usd, -r.conservative_accuracy))


def _integer(value, name, minimum=1):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(result)


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


class GittinsModelSelector(BaseModelSelector):
    """Gittins with scalar or two-axis accuracy/cost objectives.

    Included in the standard installation (NumPy and SciPy). No complete score matrix
    is needed. Each combination receives a seeded independent question
    order and a uniform warm start; no cell is evaluated twice in one run.

    ``single`` optimizes score, optionally penalizing cost and latency with
    fixed warm-start min/max scales. ``multi`` keeps score and mean USD as
    separate objectives; scalar lambda weights are invalid in that mode.

    ``observation_budget_fraction`` counts cells including warm start.
    ``max_search_cost_usd`` counts actual non-cached token charges including
    warm start. Before adaptive batches, their frozen warm-start charge
    estimates are reserved. Actual costs may exceed estimates, so this is
    a soft spend cap; any overshoot is reported. A cap interrupting warm
    start returns observed records and an empty recommendation set.

    Single mode prices each question with ``cost_scaling_factor=1e-4`` times
    its estimated USD cost. It uses a fixed prior (``prior_mean=0.5`` in
    normalized units), finite-dataset means and per-cell DP transitions.
    Its default recommendation_beta is zero; multi mode defaults to one.
    Single mode recommends stopping when a completed empirical value reaches
    every unfinished index, keeping the cost coefficient fixed. By default it
    continues selecting the largest unfinished index until a budget, user stop,
    or full completion. Set ``stop_on_recommendation=True`` to stop automatically.

    In multi mode, a direction stops when a completed arm's value reaches the
    maximum unfinished Gittins index. Its eta then decays; all directions must stop
    at numerical floors without an intervening observation to halt globally.
    Budget exhaustion and full completion can halt earlier. Evaluation or
    pricing failures abort the run, with the base selector stopping tracking.
    """

    def __init__(self, agent=None, models=None, eval_fn=None, dataset=None, *,
                 objective_mode="single", batch_size=4, warm_start_batch_size=4,
                 observation_budget_fraction=1.0, max_total_question_evaluations=None,
                 max_search_cost_usd=None, recommendation_beta=None,
                 recommendation_min_samples=0, score_bounds=(0.0, 1.0),
                 prior_mean=0.5, prior_variance=0.04, score_noise_variance=0.25,
                 cost_scaling_factor=1e-4, stop_on_recommendation=False,
                 cost_reference_usd=None, search_cost_scale_eta=1.0,
                 eta_initial=1.0, eta_decay=0.5, effective_cost_bin_ratio=2.0,
                 effective_cost_bin_anchor=1e-4, grid_size=129, stop_tolerance=1e-9,
                 seed=42, model_prices=None, tracker=None,
                 lambda_cost=0.0, lambda_latency=0.0):
        try:
            import scipy  # noqa: F401
        except ImportError as error:
            raise ImportError('SciPy is required by AgentOpt; install with `pip install agentopt-py`') from error
        if objective_mode not in ("single", "multi"):
            raise ValueError("objective_mode must be 'single' or 'multi'")
        self.objective_mode = objective_mode
        if not isinstance(stop_on_recommendation, bool):
            raise ValueError("stop_on_recommendation must be a boolean")
        if objective_mode != "single" and stop_on_recommendation:
            raise ValueError("stop_on_recommendation requires single mode")
        self.stop_on_recommendation = stop_on_recommendation
        if recommendation_beta is None:
            recommendation_beta = 0.0 if objective_mode == "single" else 1.0
        self.prior_mean = float(prior_mean)
        if not math.isfinite(self.prior_mean):
            raise ValueError("prior_mean must be finite")
        self.cost_scaling_factor = _positive(cost_scaling_factor, "cost_scaling_factor")
        for name, value in (("lambda_cost", lambda_cost), ("lambda_latency", lambda_latency),
                            ("recommendation_beta", recommendation_beta), ("stop_tolerance", stop_tolerance)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if objective_mode == "multi" and (lambda_cost or lambda_latency):
            raise ValueError("multi mode keeps accuracy and cost separate; lambda weights require single mode")
        self.batch_size = _integer(batch_size, "batch_size")
        self.warm_start_batch_size = _integer(warm_start_batch_size, "warm_start_batch_size")
        self.grid_size = _integer(grid_size, "grid_size", 17)
        self.seed = _integer(seed, "seed", 0)
        self.min_samples = _integer(recommendation_min_samples, "recommendation_min_samples", 0)
        self.beta = float(recommendation_beta)
        self.stop_tolerance = float(stop_tolerance)
        low, high = map(float, score_bounds)
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError("score_bounds must be finite with upper > lower")
        self.score_bounds = (low, high)
        self.prior_variance = _positive(prior_variance, "prior_variance")
        self.score_noise_variance = _positive(score_noise_variance, "score_noise_variance")
        self.cost_reference = None if cost_reference_usd is None else _positive(cost_reference_usd, "cost_reference_usd")
        self.cost_scale = _positive(search_cost_scale_eta, "search_cost_scale_eta")
        self.eta_initial = _positive(eta_initial, "eta_initial")
        self.eta_decay = float(eta_decay)
        if not math.isfinite(self.eta_decay) or not 0 < self.eta_decay < 1:
            raise ValueError("eta_decay must lie in (0, 1)")
        self.cost_bin_ratio = effective_cost_bin_ratio
        if effective_cost_bin_ratio is not None and _positive(effective_cost_bin_ratio, "effective_cost_bin_ratio") <= 1:
            raise ValueError("effective_cost_bin_ratio must exceed one or be None")
        self.cost_bin_anchor = _positive(effective_cost_bin_anchor, "effective_cost_bin_anchor")
        fraction = float(observation_budget_fraction)
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("observation_budget_fraction must lie in (0, 1]")
        self.max_spend = None if max_search_cost_usd is None else _positive(max_search_cost_usd, "max_search_cost_usd")
        # Validate all budget dimensions before BaseModelSelector starts tracking.
        if dataset is None or models is None:
            raise TypeError("'models' and 'dataset' are required")
        if not models or any(not candidates for candidates in models.values()):
            raise ValueError("Every model node must have at least one candidate")
        self.warm_size = min(self.warm_start_batch_size, len(dataset))
        n_arms = math.prod(len(candidates) for candidates in models.values())
        self.cell_cap = math.ceil(fraction * n_arms * len(dataset))
        if max_total_question_evaluations is not None:
            self.cell_cap = min(self.cell_cap, _integer(max_total_question_evaluations, "max_total_question_evaluations"))
        if self.cell_cap < n_arms * self.warm_size:
            raise ValueError("The cell budget cannot fund the mandatory uniform warm start")
        super().__init__(agent=agent, models=models, eval_fn=eval_fn, dataset=dataset,
                         model_prices=model_prices, tracker=tracker,
                         lambda_cost=lambda_cost, lambda_latency=lambda_latency)

    def _run_selection(self, parallel=False, max_concurrent=20):
        return asyncio.run(self._select(parallel, _integer(max_concurrent, "max_concurrent")))

    def _selection_cost_so_far(self):
        # Charged calls only, accumulated before tracker archival.
        return self._spent

    def _update_spend(self):
        records = self._tracker.get_records()
        new_records = records[self._record_cursor:]
        self._record_cursor = len(records)
        for record in new_records:
            if record.cached:
                continue
            price = compute_price({record.model: record.prompt_tokens}, {record.model: record.completion_tokens},
                                  custom_prices=self._custom_prices)
            if price is None or not math.isfinite(price) or price < 0:
                raise ValueError(f"Cannot account for Gittins search cost for {record.model!r}; supply model_prices")
            self._spent += price

    async def _pull(self, arm, question_ids, parallel, max_concurrent):
        """Evaluate a batch with stable original dataset indices and unique run IDs."""
        agent = self.agent(self._combos[arm])
        label = f"{self._names[arm]}::gittins_{self._run_id}"
        async def evaluate(question):
            tasks = [self.dataset[question]]
            if parallel or self._is_async_run:
                outcome = await self._evaluate_agent_async(agent, tasks, label, max_concurrent=1, dp_offset=question)
            else:
                outcome = self._evaluate_agent(agent, tasks, label, dp_offset=question)
            return question, outcome
        if parallel:
            semaphore = asyncio.Semaphore(max_concurrent)
            async def limited(question):
                async with semaphore:
                    return await evaluate(question)
            evaluated = await asyncio.gather(*(limited(q) for q in question_ids))
        else:
            evaluated = [await evaluate(q) for q in question_ids]
        before = self._spent
        self._update_spend()
        self._evaluations += len(question_ids)
        for question, (scores, latencies, dp_ids) in evaluated:
            if len(scores) != 1:
                raise RuntimeError(f"Evaluation failed for {self._names[arm]} at dataset index {question}")
            costs = self._per_sample_costs(dp_ids)
            cost = costs[0]
            low, high = self.score_bounds
            if not math.isfinite(scores[0]) or not low <= scores[0] <= high:
                raise ValueError("eval_fn score must be finite and within score_bounds")
            if cost is None or not math.isfinite(cost) or cost < 0:
                raise ValueError("Gittins requires known nonnegative per-evaluation costs; supply model_prices")
            if not math.isfinite(latencies[0]) or latencies[0] < 0:
                raise ValueError("Evaluation latency must be finite and nonnegative")
            dp = self._build_datapoint_results(scores, latencies, dp_ids)[0]
            if not dp.input_tokens and not dp.output_tokens:
                raise ValueError("No LLM token usage was recorded; Gittins requires tracked evaluations")
            dp.datapoint_index = question
            self._datapoints[arm].append(dp)
            self._scores[arm].append(scores[0])
            self._costs[arm].append(cost)
            self._latencies[arm].append(latencies[0])
        self._positions[arm] += len(question_ids)
        return self._spent - before

    def _budget_stop(self, count, predicted_charge=0.0):
        if self._should_stop():
            return "user_stopped"
        if self._evaluations + count > self.cell_cap:
            return "question_budget"
        if self.max_spend is not None and (self._spent >= self.max_spend
                                          or self._spent + predicted_charge > self.max_spend):
            return "search_cost_budget"
        return None

    def _direction_status(self, indices, completed, direction):
        # G0 uses an ordered, tolerance-aware fold for unfinished arms.
        arm, best_index = None, -math.inf
        for candidate in sorted(indices):
            if indices[candidate] > best_index + self.stop_tolerance:
                arm, best_index = candidate, indices[candidate]
        completed_arm, completed_index = None, -math.inf
        if completed:
            terminal = self._completed_states[direction]
            for i in completed:
                if i not in terminal:
                    secondary = (-float(np.mean(self._costs[i]))
                                 if self.objective_mode == "single" or direction == 1
                                 else float(np.mean(self._scores[i])))
                    terminal[i] = (self._active_state(i, direction)[0], secondary)
            values = {i: terminal[i][0] for i in completed}
            maximum = max(values.values())
            tied = [i for i in completed if abs(values[i] - maximum) <= self.stop_tolerance]
            completed_arm = max(tied, key=lambda i: (terminal[i][1], -i))
            completed_index = values[completed_arm]
        return arm, best_index, completed_arm, completed_index

    def _scalar_observations(self, arm):
        low, high = self.score_bounds
        objective = np.asarray(self._scores[arm], dtype=float).copy()
        for values, bounds, weight in ((self._costs[arm], self._fixed_cost_bounds, self.lambda_cost),
                                       (self._latencies[arm], self._fixed_latency_bounds, self.lambda_latency)):
            lo, hi = bounds
            if weight and hi > lo:
                objective -= weight * (np.asarray(values) - lo) / (hi - lo)
        return (objective - low) / (high - low)

    def _active_state(self, arm, direction):
        if self.objective_mode == "single":
            values = self._scalar_observations(arm)
            mean, _ = finite_dataset_moments(
                self._scalar_posteriors[arm], np.array([values.sum()]),
                len(values), len(self.dataset), np.array([self._scalar_noise]),
            )
            # The full-dataset posterior-mean process is an affine transform
            # of the latent mean process. Transform transition variances,
            # retaining the original monetary scale of evaluation costs.
            scale = 1.0 + self._scalar_noise / (len(self.dataset) * self.prior_variance)
            return float(mean[0]), self._initial_scalar_var * scale ** 2, self._scalar_noise * scale ** 2
        if direction == 0:
            scale = self._calibration.cost_reference_usd
            return (1.0 - self._posteriors[arm].mean[1] / scale,
                    self._initial_var[1] / scale ** 2, self._calibration.question_noise[1] / scale ** 2)
        return self._posteriors[arm].mean[0], self._initial_var[0], self._calibration.question_noise[0]

    async def _select(self, parallel, max_concurrent):
        self._combos = self._all_combos()
        self._names = [self._combo_name(combo) for combo in self._combos]
        if len(set(self._names)) != len(self._names):
            raise ValueError("Gittins requires distinct combination labels")
        n_arms, n_questions = len(self._combos), len(self.dataset)
        self._run_id = uuid4().hex
        self._scores = [[] for _ in range(n_arms)]
        self._costs = [[] for _ in range(n_arms)]
        self._latencies = [[] for _ in range(n_arms)]
        self._datapoints = [[] for _ in range(n_arms)]
        self._positions = np.zeros(n_arms, dtype=int)
        children = np.random.SeedSequence(self.seed).spawn(n_arms + 1)
        self._orders = [np.random.default_rng(child).permutation(n_questions).tolist() for child in children[1:]]
        self._record_cursor = len(self._tracker.get_records())
        self._spent, self._evaluations = 0.0, 0
        self._stop_recommended = False
        dimensions = 1 if self.objective_mode == "single" else 2
        # Completed posteriors and observed tie-break statistics never change.
        self._completed_states = [{} for _ in range(dimensions)]
        eta = [1.0] if self.objective_mode == "single" else [self.eta_initial] * dimensions
        warm_charges = np.zeros(n_arms)
        # Warm cells are sequential so actual dollar usage is checked after
        # each request even when their costs cannot yet be predicted.
        for arm in range(n_arms):
            for question in self._orders[arm][:self.warm_size]:
                reason = self._budget_stop(1)
                if reason:
                    return self._results("user_stopped" if reason == "user_stopped" else "warm_start_budget", eta, calibrated=False)
                warm_charges[arm] += await self._pull(arm, [question], False, max_concurrent)
                self._publish_update(self._results("warming_up", eta, calibrated=False))
        self._calibration = fit_warm_start(self._scores, self._costs, score_bounds=self.score_bounds,
                                          prior_variance=self.prior_variance,
                                          score_noise_variance=self.score_noise_variance,
                                          cost_reference_usd=self.cost_reference)
        self._posteriors = [self._calibration.posterior(s, c) for s, c in zip(self._scores, self._costs)]
        if self.objective_mode == "single":
            # A fixed accuracy prior also underlies reported finite-dataset
            # accuracy predictions; deployment weights affect only the scalar.
            for arm, posterior in enumerate(self._posteriors):
                low, high = self.score_bounds
                normalized = (np.mean(self._scores[arm]) - low) / (high - low)
                variance = 1.0 / (1.0 / self.prior_variance + self.warm_size / self.score_noise_variance)
                posterior.mean[0] = variance * (self.prior_mean / self.prior_variance + self.warm_size * normalized / self.score_noise_variance)
                posterior.variance[0] = variance
        self._initial_var = self._posteriors[0].variance.copy()
        self._fixed_cost_bounds = (float(np.min(self._costs)), float(np.max(self._costs)))
        self._fixed_latency_bounds = (float(np.min(self._latencies)), float(np.max(self._latencies)))
        warm_objectives = [self._scalar_observations(arm) for arm in range(n_arms)]
        scalar_prior_mean = self.prior_mean if self.objective_mode == "single" else float(np.mean(warm_objectives))
        score_span = self.score_bounds[1] - self.score_bounds[0]
        self._scalar_noise = self.score_noise_variance * (1 + (self.lambda_cost + self.lambda_latency) / score_span) ** 2
        self._scalar_posteriors = []
        for values in warm_objectives:
            posterior = GaussianPosterior(np.array([scalar_prior_mean]), np.array([self.prior_variance]))
            posterior.update(np.array([np.mean(values)]), np.array([self._scalar_noise]), self.warm_size)
            self._scalar_posteriors.append(posterior)
        self._initial_scalar_var = self._scalar_posteriors[0].variance[0]
        # Continuation penalties use observed workflow cost, as in research;
        # dollar reservations instead estimate new non-cached charges.
        single = self.objective_mode == "single"
        expected_costs = np.mean(self._costs, axis=1) * (1 if single else self.batch_size)
        multiplier = self.cost_scaling_factor if single else self.cost_scale
        base_penalties = np.maximum(multiplier * expected_costs, 1e-12)
        if not single and self.cost_bin_ratio is not None:
            exponents = np.rint(np.log(base_penalties / self.cost_bin_anchor) / math.log(self.cost_bin_ratio))
            base_penalties = self.cost_bin_anchor * np.power(self.cost_bin_ratio, exponents)
        charge_per_question = warm_charges / self.warm_size
        horizon = n_questions - self.warm_size if single else math.ceil((n_questions - self.warm_size) / self.batch_size)
        if single:
            eta = [1.0]
        stages = np.zeros(n_arms, dtype=int)
        cache = AxisGittinsBoundaryCache()
        # An arm's index changes only after observing that arm or changing
        # the current direction's eta. G0 uses the same invalidation rule.
        direction_indices = [{} for _ in range(dimensions)]
        direction, floor_stops = 0, set()
        if not single:
            self._publish_update(self._results("running", eta))
        while True:
            completed = [i for i in range(n_arms) if self._positions[i] == n_questions]
            unfinished = [i for i in range(n_arms) if self._positions[i] < n_questions]
            if not unfinished:
                self._stop_recommended = single
                reason = "all_arms_completed"
                break
            # Single mode recomputes its advice for the current observations,
            # including when a budget ends the run after the preceding batch.
            if not single:
                reason = self._budget_stop(1)
                if reason:
                    break
            indices = {}
            for arm in unfinished:
                if arm not in direction_indices[direction]:
                    mean, initial, noise = self._active_state(arm, direction)
                    table = cache.get(effective_pull_cost=eta[direction] * base_penalties[arm],
                                      initial_var=float(initial), obs_noise_var=float(noise if single else noise / self.batch_size),
                                      horizon=horizon, grid_size=self.grid_size)
                    direction_indices[direction][arm] = table.index(int(stages[arm]), float(mean))
                indices[arm] = direction_indices[direction][arm]
            arm, best_index, completed_arm, best_completed = self._direction_status(indices, completed, direction)
            direction_stop = bool(completed and best_completed >= best_index - self.stop_tolerance)
            if single:
                self._stop_recommended = direction_stop
                self._publish_update(self._results("running", eta))
                reason = self._budget_stop(1)
                if reason:
                    break
                if direction_stop and self.stop_on_recommendation:
                    reason = "gittins_stop"
                    break
            elif direction_stop:
                remaining_penalty = max((horizon - stages[i]) * eta[direction] * base_penalties[i] for i in unfinished)
                threshold = max(self.stop_tolerance, np.finfo(float).eps * max(1.0, abs(best_completed), abs(best_index)))
                next_eta = eta[direction] * self.eta_decay
                if remaining_penalty <= threshold or next_eta <= 0 or np.any(next_eta * base_penalties <= 0):
                    floor_stops.add(direction)
                else:
                    eta[direction] = next_eta
                    direction_indices[direction].clear()
                    floor_stops.discard(direction)
                if len(floor_stops) == dimensions:
                    reason = "direction_eta_numerical_floor"
                    break
                direction = (direction + 1) % dimensions
                continue
            start = int(self._positions[arm])
            questions = self._orders[arm][start:start + self.batch_size]
            reason = self._budget_stop(len(questions), float(charge_per_question[arm] * len(questions)))
            if reason:
                break
            await self._pull(arm, questions, parallel, max_concurrent)
            observations = self._calibration.observations(self._scores[arm][-len(questions):], self._costs[arm][-len(questions):])
            self._posteriors[arm].update(observations.mean(axis=0), self._calibration.question_noise, len(questions))
            values = self._scalar_observations(arm)[-len(questions):]
            self._scalar_posteriors[arm].update(np.array([np.mean(values)]), np.array([self._scalar_noise]), len(questions))
            for local_indices in direction_indices:
                local_indices.pop(arm, None)
            stages[arm] += len(questions) if single else 1
            floor_stops.clear()
            direction = (direction + 1) % dimensions
            if not single:
                self._publish_update(self._results("running", eta))
        return self._results(reason, eta)

    def _results(self, reason, eta, calibrated=True):
        records = []
        for arm, name in enumerate(self._names):
            dps = self._datapoints[arm]
            if not dps:
                continue
            input_tokens, output_tokens = {}, {}
            for dp in dps:
                for target, tokens in ((input_tokens, dp.input_tokens), (output_tokens, dp.output_tokens)):
                    for model, count in tokens.items():
                        target[model] = target.get(model, 0) + count
            objective = None
            if calibrated and self.objective_mode == "single" and self._has_combined_objective:
                low, high = self.score_bounds
                objective = float(np.mean(self._scalar_observations(arm)) * (high - low) + low)
            records.append(self._make_result(model_name=name, accuracy=float(np.mean(self._scores[arm])),
                                            latency_seconds=float(np.mean(self._latencies[arm])),
                                            input_tokens=input_tokens, output_tokens=output_tokens,
                                            attribute="combination", datapoint_results=dps, combined_objective=objective))
        candidates = []
        if calibrated:
            low, high = self.score_bounds
            for arm, name in enumerate(self._names):
                if len(self._scores[arm]) < self.min_samples:
                    continue
                observations = self._calibration.observations(self._scores[arm], self._costs[arm])
                mean, variance = finite_dataset_moments(self._posteriors[arm], observations.sum(axis=0),
                                                        len(observations), len(self.dataset), self._calibration.question_noise)
                std = np.sqrt(variance)
                accuracy, score_std = low + (high - low) * mean[0], (high - low) * std[0]
                if len(observations) == len(self.dataset):
                    accuracy = float(np.mean(self._scores[arm]))
                    mean[1] = float(np.mean(self._costs[arm]))
                objective = conservative_objective = None
                if self.objective_mode == "single":
                    values = self._scalar_observations(arm)
                    scalar_mean, scalar_var = finite_dataset_moments(self._scalar_posteriors[arm], np.array([values.sum()]),
                                                                    len(values), len(self.dataset), np.array([self._scalar_noise]))
                    objective = float(low + (high - low) * scalar_mean[0])
                    conservative_objective = objective - self.beta * (high - low) * math.sqrt(scalar_var[0])
                candidates.append(GittinsRecommendation(
                    model_name=name, combo={node: self._candidate_label(model) for node, model in self._combos[arm].items()},
                    observed_samples=len(observations), accuracy=float(accuracy), accuracy_std=float(score_std),
                    cost_usd=float(mean[1]), cost_std_usd=float(std[1]),
                    conservative_accuracy=float(accuracy - self.beta * score_std),
                    conservative_cost_usd=float(mean[1] + self.beta * std[1]),
                    objective=objective, conservative_objective=conservative_objective,
                ))
        if self.objective_mode == "single" and candidates:
            winner = max(candidates, key=lambda r: (r.conservative_objective, -r.conservative_cost_usd))
            recommendations = [winner]
            for record in records:
                record.is_best = record.model_name == winner.model_name
        elif candidates:
            positions = raw_pareto_front_indices([[r.conservative_accuracy, r.conservative_cost_usd] for r in candidates])
            recommendations = [candidates[i] for i in positions]
        else:
            recommendations = []
        return GittinsSelectionResults(results=records, recommendations=recommendations,
                                       objective_mode=self.objective_mode, stop_reason=reason,
                                       stop_recommended=self._stop_recommended,
                                       total_evaluations=self._evaluations,
                                       search_cost_overshoot_usd=max(0.0, self._spent - self.max_spend) if self.max_spend is not None else 0.0,
                                       direction_eta=list(eta))
