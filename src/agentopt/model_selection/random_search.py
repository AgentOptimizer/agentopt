"""Random configuration and shared-question baselines for live evaluation."""

from __future__ import annotations

import math
from uuid import uuid4

import numpy as np

from .base import BaseModelSelector
from .gittins import GittinsModelSelector, GittinsRecommendation, GittinsSelectionResults, _integer, _positive
from .gittins_statistics import raw_pareto_front_indices


class RandomSelectionResults(GittinsSelectionResults):
    """Empirical recommendations from completed random sampling units.

    Accuracy/cost fields are sample means; their conservative counterparts
    equal these means. Zero std fields indicate no uncertainty model was
    fitted, not certainty about unseen outcomes. Partial units remain in
    evaluation records but do not enter recommendations.
    """

    sampling: str
    completed_units: int


class RandomSearchModelSelector(GittinsModelSelector):
    """Randomly evaluate complete configurations or shared questions.

    Configuration sampling evaluates every question for each chosen random
    configuration. Question sampling evaluates every configuration on each
    chosen shared random question. Both sample without replacement.

    Cell caps fund whole units. A dollar cap is checked after each individual
    evaluation and may overshoot by that evaluation's charge. If interrupted,
    a partial unit is excluded from the empirical recommendations. All
    evaluations require tracked, priced token usage, as with Gittins.
    """

    def __init__(self, agent=None, models=None, eval_fn=None, dataset=None, *,
                 objective_mode="single", sampling="configurations", seed=42,
                 observation_budget_fraction=1.0, max_total_question_evaluations=None,
                 max_search_cost_usd=None, score_bounds=(0.0, 1.0),
                 model_prices=None, tracker=None, lambda_cost=0.0, lambda_latency=0.0):
        if objective_mode not in ("single", "multi"):
            raise ValueError("objective_mode must be 'single' or 'multi'")
        if sampling not in ("configurations", "questions"):
            raise ValueError("sampling must be 'configurations' or 'questions'")
        if models is None or dataset is None:
            raise TypeError("'models' and 'dataset' are required")
        if not models or any(not candidates for candidates in models.values()) or not len(dataset):
            raise ValueError("models and dataset must be nonempty")
        for name, value in (("lambda_cost", lambda_cost), ("lambda_latency", lambda_latency)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if objective_mode == "multi" and (lambda_cost or lambda_latency):
            raise ValueError("multi mode keeps accuracy and cost separate; lambda weights require single mode")
        low, high = map(float, score_bounds)
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError("score_bounds must be finite with upper > lower")
        fraction = float(observation_budget_fraction)
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("observation_budget_fraction must lie in (0, 1]")
        n_arms = math.prod(len(candidates) for candidates in models.values())
        self.cell_cap = math.ceil(fraction * n_arms * len(dataset))
        if max_total_question_evaluations is not None:
            self.cell_cap = min(self.cell_cap, _integer(max_total_question_evaluations, "max_total_question_evaluations"))
        self.unit_size = len(dataset) if sampling == "configurations" else n_arms
        if self.cell_cap < self.unit_size:
            raise ValueError("The cell budget must fund one complete sampling unit")
        self.objective_mode, self.sampling = objective_mode, sampling
        self.seed = _integer(seed, "seed", 0)
        self.score_bounds = (low, high)
        self.max_spend = None if max_search_cost_usd is None else _positive(max_search_cost_usd, "max_search_cost_usd")
        # Reuse the tracked observation path, but not Gittins' warm-start or DP.
        BaseModelSelector.__init__(self, agent=agent, models=models, eval_fn=eval_fn,
                                   dataset=dataset, model_prices=model_prices, tracker=tracker,
                                   lambda_cost=lambda_cost, lambda_latency=lambda_latency)

    async def _select(self, parallel, max_concurrent):
        self._combos = self._all_combos()
        self._names = [self._combo_name(combo) for combo in self._combos]
        if len(set(self._names)) != len(self._names):
            raise ValueError("Random selection requires distinct combination labels")
        n_arms, n_questions = len(self._combos), len(self.dataset)
        self._run_id = uuid4().hex
        self._scores = [[] for _ in range(n_arms)]
        self._costs = [[] for _ in range(n_arms)]
        self._latencies = [[] for _ in range(n_arms)]
        self._datapoints = [[] for _ in range(n_arms)]
        self._positions = np.zeros(n_arms, dtype=int)
        self._record_cursor = len(self._tracker.get_records())
        self._spent, self._evaluations = 0.0, 0
        eligible = np.zeros(n_arms, dtype=int)
        rng = np.random.default_rng(self.seed)
        units = rng.permutation(n_arms if self.sampling == "configurations" else n_questions)
        completed_units = 0
        reason = "all_arms_completed"
        for unit in units:
            reason = self._budget_stop(self.unit_size)
            if reason:
                break
            if self.sampling == "configurations":
                cells = [(int(unit), q) for q in range(n_questions)]
            else:
                cells = [(arm, int(unit)) for arm in range(n_arms)]
            # Bound the next scheduled batch so user stops also take effect
            # within a configuration. Dollar-capped runs check every cell.
            chunk_size = max_concurrent if parallel and self.max_spend is None and self.sampling == "configurations" else 1
            for offset in range(0, len(cells), chunk_size):
                chunk = cells[offset:offset + chunk_size]
                reason = self._budget_stop(len(chunk))
                if reason:
                    break
                arm = chunk[0][0]
                await self._pull(arm, [q for _, q in chunk], parallel, max_concurrent)
                if offset + len(chunk) == len(cells):
                    completed_units += 1
                    eligible = self._positions.copy()
                self._publish_update(self._random_results("running", eligible, completed_units))
            if reason:
                break
        if reason is None:
            reason = "all_arms_completed" if self._evaluations == n_arms * n_questions else "question_budget"
        return self._random_results(reason, eligible, completed_units)

    def _random_results(self, reason, eligible, completed_units):
        # Use common scales across only the comparable, completed units.
        costs = [c for arm, n in enumerate(eligible) for c in self._costs[arm][:n]]
        latencies = [v for arm, n in enumerate(eligible) for v in self._latencies[arm][:n]]
        cost_bounds = (min(costs), max(costs)) if costs else (0.0, 0.0)
        latency_bounds = (min(latencies), max(latencies)) if latencies else (0.0, 0.0)
        def objectives(arm, n):
            values = np.array(self._scores[arm][:n], dtype=float)
            for observations, bounds, weight in ((self._costs[arm][:n], cost_bounds, self.lambda_cost),
                                                   (self._latencies[arm][:n], latency_bounds, self.lambda_latency)):
                lo, hi = bounds
                if weight and hi > lo:
                    values -= weight * (np.asarray(observations) - lo) / (hi - lo)
            return values
        records, candidates = [], []
        for arm, name in enumerate(self._names):
            dps = self._datapoints[arm]
            if not dps:
                continue
            input_tokens, output_tokens = {}, {}
            for dp in dps:
                for target, tokens in ((input_tokens, dp.input_tokens), (output_tokens, dp.output_tokens)):
                    for model, count in tokens.items():
                        target[model] = target.get(model, 0) + count
            records.append(self._make_result(
                model_name=name, accuracy=float(np.mean(self._scores[arm])),
                latency_seconds=float(np.mean(self._latencies[arm])),
                input_tokens=input_tokens, output_tokens=output_tokens,
                attribute="combination", datapoint_results=dps,
                combined_objective=float(np.mean(objectives(arm, len(dps)))) if self._has_combined_objective else None,
            ))
            n = int(eligible[arm])
            if not n:
                continue
            score, cost = float(np.mean(self._scores[arm][:n])), float(np.mean(self._costs[arm][:n]))
            objective = float(np.mean(objectives(arm, n))) if self.objective_mode == "single" else None
            candidates.append(GittinsRecommendation(
                model_name=name, combo={node: self._candidate_label(model) for node, model in self._combos[arm].items()},
                observed_samples=n, accuracy=score, accuracy_std=0.0, cost_usd=cost, cost_std_usd=0.0,
                conservative_accuracy=score, conservative_cost_usd=cost,
                objective=objective, conservative_objective=objective,
            ))
        if self.objective_mode == "single" and candidates:
            recommendations = [max(candidates, key=lambda r: (r.objective, -r.cost_usd))]
            for record in records:
                record.is_best = record.model_name == recommendations[0].model_name
        elif candidates:
            front = raw_pareto_front_indices([[r.accuracy, r.cost_usd] for r in candidates])
            recommendations = [candidates[i] for i in front]
        else:
            recommendations = []
        return RandomSelectionResults(
            results=records, recommendations=recommendations, objective_mode=self.objective_mode,
            stop_reason=reason, sampling=self.sampling, completed_units=completed_units,
            total_evaluations=self._evaluations,
            search_cost_overshoot_usd=max(0.0, self._spent-self.max_spend) if self.max_spend is not None else 0.0,
        )
