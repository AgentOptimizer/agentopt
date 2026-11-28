"""Two-objective cost-aware hypervolume UCB on live quality and deployment cost."""
import math

import numpy as np

from .checkpoint import AUTO_CHECKPOINT, CheckpointError
from .gittins import GittinsModelSelector, GittinsRecommendation, GittinsSelectionResults


class CoHVUCBSelectionResults(GittinsSelectionResults):
    """Pareto recommendations use the observed per-configuration means."""
    algorithm: str = "cohv_ucb"


class CoHVUCBModelSelector(GittinsModelSelector):
    """Uniform warm-up followed by the research CoHV-UCB acquisition rule."""

    algorithm_name = "CoHV-UCB"
    _policy_id = "cohv_ucb"
    _multi_directions = 0
    _result_type = CoHVUCBSelectionResults

    def __init__(self, agent=None, models=None, eval_fn=None, dataset=None, *,
                 batch_size=None, warm_start_batch_size=None, observation_budget_fraction=1.0,
                 max_total_question_evaluations=None, max_search_cost_usd=None,
                 recommendation_min_samples=0, score_bounds=(0.0, 1.0), seed=42,
                 model_prices=None, tracker=None, checkpoint_path=AUTO_CHECKPOINT, checkpoint_id=None):
        super().__init__(
            agent=agent, models=models, eval_fn=eval_fn, dataset=dataset, objective_mode="multi",
            batch_size=batch_size, warm_start_batch_size=warm_start_batch_size,
            observation_budget_fraction=observation_budget_fraction,
            max_total_question_evaluations=max_total_question_evaluations,
            max_search_cost_usd=max_search_cost_usd, recommendation_min_samples=recommendation_min_samples,
            score_bounds=score_bounds, seed=seed, model_prices=model_prices, tracker=tracker,
            checkpoint_path=checkpoint_path, checkpoint_id=checkpoint_id,
        )

    def _choose_arm(self, candidates, sizes, log_horizon):
        counts = self._positions[candidates]
        quality = np.array([np.mean(self._scores[i]) for i in candidates])
        low, high = self.score_bounds
        quality = (quality - low) / (high - low)
        cost = np.array([np.mean(self._costs[i]) for i in candidates])
        affordability = self._reference / (self._reference + cost)
        radius = 0.01 * np.sqrt(2 * log_horizon / counts)
        optimistic_volume = np.minimum(1, quality + radius) * np.minimum(1, affordability + radius)
        lower_cost = np.maximum(self._price_floor / 2, cost - radius * self._price_scale)
        indices = optimistic_volume / (lower_cost * sizes)
        # The research tie-break favors fewer observations, then original arm order.
        return int(candidates[np.lexsort((candidates, counts, -indices))[0]])

    async def _search_after_warmup(self, parallel, max_concurrent, warm_charges, eta):
        self._active_eta, self._calibrated = [], True
        costs = np.mean(self._costs, axis=1)
        positive = costs[costs > 0]
        self._reference = float(np.median(positive)) if positive.size else 1.0
        prices = np.maximum(costs, 1e-12)
        self._price_floor, self._price_scale = float(prices.min()), float(prices.max())
        log_horizon = math.log1p(self.cell_cap)
        if self.max_spend is not None:
            horizon = self.max_spend / self._price_floor
            log_horizon = (math.log1p(math.ceil(horizon)) if math.isfinite(horizon)
                           else math.log(self.max_spend) - math.log(self._price_floor))
        charges = [charge / self.warm_size for charge in warm_charges]
        n_questions = len(self._run_dataset)
        arm_by_name = {name: arm for arm, name in enumerate(self._names)}
        while True:
            self._publish_update(self._results("running", []))
            candidates = np.flatnonzero(self._positions < n_questions)
            if not len(candidates):
                return self._results("all_arms_completed", [])
            reason = self._budget_stop(1)
            if reason:
                return self._results(reason, [])
            saved = self._journal.next_saved_arm() if self._journal else None
            if saved is None:
                sizes = np.minimum(self.batch_size, n_questions - self._positions[candidates])
                arm = self._choose_arm(candidates, sizes, log_horizon)
            else:
                # A budget extension can change the exploration radius. Honor the
                # recorded decision; begin_batch still validates its seeded questions.
                arm = arm_by_name[saved]
                if self._positions[arm] >= n_questions:
                    raise CheckpointError("Checkpoint scheduled an already completed configuration")
            start = int(self._positions[arm])
            questions = self._orders[arm][start:start + self.batch_size]
            reason = self._budget_stop(len(questions), charges[arm] * len(questions))
            if reason:
                return self._results(reason, [])
            await self._pull(arm, questions, parallel, max_concurrent, charges[arm])

    def _recommendation_candidates(self):
        candidates = []
        for i, name in enumerate(self._names):
            count = len(self._scores[i])
            if count < self.min_samples:
                continue
            score, cost = float(np.mean(self._scores[i])), float(np.mean(self._costs[i]))
            # Match the empirical frontier representation used by the random
            # baselines. Optimism belongs only to the next-batch acquisition.
            candidates.append(GittinsRecommendation(
                model_name=name, combo={k: self._candidate_label(v) for k, v in self._combos[i].items()},
                observed_samples=count, accuracy=score, cost_usd=cost, accuracy_std=0, cost_std_usd=0,
                conservative_accuracy=score, conservative_cost_usd=cost,
            ))
        return candidates
