"""Cost-aware Pareto allocation over tracked configuration-question cells."""

from __future__ import annotations

import math
from uuid import uuid4

import numpy as np

from .base import BaseModelSelector
from .gittins import GittinsModelSelector, GittinsRecommendation, GittinsSelectionResults, _integer, _positive
from .gittins_statistics import raw_pareto_front_indices


class CostAwareSelectionResults(GittinsSelectionResults):
    """Empirical Pareto recommendations at the last complete in-budget batch.

    Evaluation records and selection_cost include all charged work. The
    recommendation fields exclude a batch that crossed the dollar cap.
    Accuracy/cost standard deviations are zero because these recommendations
    do not fit an uncertainty model; zero does not imply certainty.
    """

    method: str
    recommendation_evaluations: int = 0
    recommendation_search_cost_usd: float = 0.0
    cost_reference_usd: float | None = None


class _CostAwareParetoSelector(GittinsModelSelector):
    """Share the live observation path with the other finite-data selectors."""

    def __init__(
        self, agent=None, models=None, eval_fn=None, dataset=None, *,
        objective_mode="multi", batch_size=4, warm_start_batch_size=4,
        observation_budget_fraction=1.0, max_total_question_evaluations=None,
        max_search_cost_usd=None, score_bounds=(0.0, 1.0),
        cost_reference_usd=None, seed=42, model_prices=None, tracker=None,
        lambda_cost=0.0, lambda_latency=0.0,
    ):
        if objective_mode != "multi":
            raise ValueError("Cost-aware Pareto selectors require objective_mode='multi'")
        if lambda_cost != 0.0 or lambda_latency != 0.0:
            raise ValueError("Multi mode keeps quality and cost separate; use results.choose(...)")
        if models is None or dataset is None:
            raise TypeError("'models' and 'dataset' are required")
        if not models or any(not candidates for candidates in models.values()) or not len(dataset):
            raise ValueError("models and dataset must be nonempty")
        self.batch_size = _integer(batch_size, "batch_size")
        self.warm_size = min(_integer(warm_start_batch_size, "warm_start_batch_size"), len(dataset))
        self.seed = _integer(seed, "seed", 0)
        low, high = map(float, score_bounds)
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError("score_bounds must be finite with upper > lower")
        self.score_bounds = (low, high)
        fraction = float(observation_budget_fraction)
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("observation_budget_fraction must lie in (0, 1]")
        n_arms = math.prod(len(candidates) for candidates in models.values())
        self.cell_cap = math.ceil(fraction * n_arms * len(dataset))
        if max_total_question_evaluations is not None:
            self.cell_cap = min(self.cell_cap, _integer(max_total_question_evaluations, "max_total_question_evaluations"))
        if self.cell_cap < n_arms * self.warm_size:
            raise ValueError("The cell budget cannot fund the mandatory uniform warm start")
        self.max_spend = None if max_search_cost_usd is None else _positive(max_search_cost_usd, "max_search_cost_usd")
        self._specified_reference = None if cost_reference_usd is None else _positive(cost_reference_usd, "cost_reference_usd")
        self.objective_mode = "multi"
        # Gittins supplies tracked observations and cooperative stopping; its
        # calibration, priors and DP are not used by these policies.
        BaseModelSelector.__init__(
            self, agent=agent, models=models, eval_fn=eval_fn, dataset=dataset,
            model_prices=model_prices, tracker=tracker,
            max_search_cost_usd=max_search_cost_usd,
        )

    async def _select(self, parallel, max_concurrent):
        self._combos = self._all_combos()
        self._names = [self._combo_name(combo) for combo in self._combos]
        if len(set(self._names)) != len(self._names):
            raise ValueError("Cost-aware selection requires distinct combination labels")
        n_arms, n_questions = len(self._combos), len(self.dataset)
        self._run_id = uuid4().hex
        self._scores = [[] for _ in range(n_arms)]
        self._costs = [[] for _ in range(n_arms)]
        self._latencies = [[] for _ in range(n_arms)]
        self._datapoints = [[] for _ in range(n_arms)]
        self._positions = np.zeros(n_arms, dtype=int)
        self._eligible = self._positions.copy()
        self._charged_by_arm = np.zeros(n_arms)
        self._spent, self._evaluations = 0.0, 0
        self._recommendation_spend = 0.0
        self._cost_reference = None
        self._sticky_arm = None
        self._record_cursor = len(self._tracker.get_records())
        children = np.random.SeedSequence(self.seed).spawn(n_arms + 1)
        self._orders = [np.random.default_rng(child).permutation(n_questions).tolist() for child in children[1:]]
        # Encode roles directly, so candidate labels need not be parsed.
        levels = {
            node: {label: i for i, label in enumerate(sorted({
                self._candidate_label(combo[node]) for combo in self._combos
            }))}
            for node in self._node_names
        }
        self._features = np.asarray([
            [levels[node][self._candidate_label(combo[node])] for node in self._node_names]
            for combo in self._combos
        ], dtype=float)
        for arm in range(n_arms):
            for question in self._orders[arm][:self.warm_size]:
                reason = self._budget_stop(1)
                if reason:
                    return self._pareto_results("user_stopped" if reason == "user_stopped" else "warm_start_budget")
                self._charged_by_arm[arm] += await self._pull(arm, [question], False, max_concurrent)
                if np.all(self._positions >= self.warm_size) and self._within_budget():
                    means = np.array([np.mean(costs) for costs in self._costs])
                    positive = means[means > 0.0]
                    self._cost_reference = self._specified_reference
                    if self._cost_reference is None:
                        self._cost_reference = float(np.median(positive)) if len(positive) else 1.0
                    self._accept_checkpoint()
                self._publish_update(self._pareto_results("warming_up"))
        if not self._within_budget():
            return self._pareto_results("warm_start_budget")

        step = 0
        while np.any(self._positions < n_questions):
            reason = self._budget_stop(1)
            if reason:
                return self._pareto_results(reason)
            candidates = np.flatnonzero(self._positions < n_questions)
            sizes = np.minimum(
                np.minimum(self.batch_size, n_questions - self._positions[candidates]),
                self.cell_cap - self._evaluations,
            )
            means = self._objective_means()
            # Cached responses still have deployment cost, but no API charge.
            # A floor makes free evaluations well-defined in per-dollar scores.
            prices = np.maximum(self._charged_by_arm[candidates] / self._positions[candidates], 1e-12) * sizes
            arm = self._choose_arm(candidates, sizes, means, prices, step)
            local = int(np.flatnonzero(candidates == arm)[0])
            start = int(self._positions[arm])
            questions = self._orders[arm][start:start + int(sizes[local])]
            self._charged_by_arm[arm] += await self._pull(arm, questions, parallel, max_concurrent)
            if self._within_budget():
                self._accept_checkpoint()
            self._publish_update(self._pareto_results("running"))
            if not self._within_budget():
                return self._pareto_results("search_cost_budget")
            step += 1
        return self._pareto_results("all_arms_completed")

    def _within_budget(self):
        return self.max_spend is None or self._spent <= self.max_spend

    def _accept_checkpoint(self):
        self._eligible = self._positions.copy()
        self._recommendation_spend = self._spent

    def _objective_means(self):
        low, high = self.score_bounds
        quality = (np.array([np.mean(scores) for scores in self._scores]) - low) / (high - low)
        costs = np.array([np.mean(costs) for costs in self._costs])
        return np.column_stack((quality, self._cost_reference / (self._cost_reference + costs)))

    def _pareto_results(self, reason):
        records, candidates = [], []
        for arm, name in enumerate(self._names):
            datapoints = self._datapoints[arm]
            if not datapoints:
                continue
            input_tokens, output_tokens = {}, {}
            for point in datapoints:
                for target, tokens in ((input_tokens, point.input_tokens), (output_tokens, point.output_tokens)):
                    for model, count in tokens.items():
                        target[model] = target.get(model, 0) + count
            records.append(self._make_result(
                model_name=name, accuracy=float(np.mean(self._scores[arm])),
                latency_seconds=float(np.mean(self._latencies[arm])),
                input_tokens=input_tokens, output_tokens=output_tokens,
                attribute="combination", datapoint_results=datapoints,
            ))
            n = int(self._eligible[arm])
            if n:
                score = float(np.mean(self._scores[arm][:n]))
                cost = float(np.mean(self._costs[arm][:n]))
                candidates.append(GittinsRecommendation(
                    model_name=name,
                    combo={node: self._candidate_label(model) for node, model in self._combos[arm].items()},
                    observed_samples=n, accuracy=score, accuracy_std=0.0,
                    cost_usd=cost, cost_std_usd=0.0,
                    conservative_accuracy=score, conservative_cost_usd=cost,
                ))
        front = raw_pareto_front_indices([[r.accuracy, r.cost_usd] for r in candidates]) if candidates else []
        return CostAwareSelectionResults(
            results=records, recommendations=[candidates[i] for i in front],
            objective_mode="multi", method=self.method, stop_reason=reason,
            total_evaluations=self._evaluations,
            recommendation_evaluations=int(self._eligible.sum()),
            recommendation_search_cost_usd=self._recommendation_spend,
            cost_reference_usd=self._cost_reference,
            search_cost_overshoot_usd=max(0.0, self._spent - self.max_spend) if self.max_spend is not None else 0.0,
        )


class CostAwareUCBModelSelector(_CostAwareParetoSelector):
    """Cycle eleven quality-cost weights and maximize optimistic score per USD.

    Four uniform warm questions per configuration calibrate deployment cost.
    Later batches use radius 0.5*sqrt(log(1+evaluations)/observations), bounded
    normalized quality and reciprocal mean-cost desirability. Recommendations
    are empirical quality-cost Pareto sets, with no fitted confidence bounds.
    """

    method = "cost_aware_ucb"

    def _choose_arm(self, candidates, sizes, means, prices, step):
        counts = self._positions[candidates]
        radius = 0.5 * np.sqrt(math.log1p(self._evaluations) / counts)
        upper = np.minimum(1.0, means[candidates] + radius[:, None])
        cost_weight = (step % 11) / 10
        scores = (upper @ np.array([1 - cost_weight, cost_weight])) / prices
        return int(candidates[np.lexsort((candidates, counts, -scores))[0]])


class CostAwareQNEHVIModelSelector(_CostAwareParetoSelector):
    """Select by noisy expected hypervolume improvement per estimated USD.

    Requires agentopt-py[bayesian]. Each role is a categorical GP feature.
    All observed configurations train the GP; fully evaluated configurations
    form the incumbent set. Refit/reselect every eight adaptive batches or
    when the selected configuration is exhausted. The default MC count is 64.
    """

    method = "cost_aware_qnehvi"

    def __init__(self, *args, mc_samples=64, refit_every=8, candidate_chunk_size=64, **kwargs):
        self.mc_samples = _integer(mc_samples, "mc_samples")
        self.refit_every = _integer(refit_every, "refit_every")
        self.candidate_chunk_size = _integer(candidate_chunk_size, "candidate_chunk_size")
        from ._qnehvi import require_botorch
        require_botorch()
        super().__init__(*args, **kwargs)

    def _choose_arm(self, candidates, sizes, means, prices, step):
        if len(candidates) == 1:
            return int(candidates[0])
        if self._sticky_arm not in candidates or step % self.refit_every == 0:
            from ._qnehvi import select_qnehvi_index
            completed = self._positions == len(self.dataset)
            local = select_qnehvi_index(
                self._features, means, self._features[candidates],
                categorical_dims=list(range(self._features.shape[1])),
                baseline_features=self._features[completed], candidate_costs=prices,
                mc_samples=self.mc_samples, seed=self.seed + self._evaluations,
                candidate_chunk_size=self.candidate_chunk_size,
            )
            self._sticky_arm = int(candidates[local])
        return self._sticky_arm
