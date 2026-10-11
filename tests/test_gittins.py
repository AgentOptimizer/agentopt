"""Live-selector contract, cost budgets, and two-axis recommendation tests."""

import contextvars
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from agentopt import ModelSelector
from agentopt.model_selection.gittins import GittinsModelSelector
from agentopt.model_selection.gittins_statistics import (
    GaussianPosterior, finite_dataset_moments, fit_warm_start, raw_pareto_front_indices,
)


class RecordingTracker:
    """Records actual token attribution without network calls or matrix lookups."""

    def __init__(self, cached=False):
        self.records = []
        self.active = contextvars.ContextVar("gittins_test_session")
        self.stopped = False
        self.started = False
        self.cached = cached

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    @contextmanager
    def track(self, data_id, combo_id):
        token = self.active.set((data_id, combo_id))
        try:
            yield SimpleNamespace()
        finally:
            self.active.reset(token)

    def record(self, model, input_tokens):
        data_id, combo_id = self.active.get()
        self.records.append(SimpleNamespace(model=model, prompt_tokens=input_tokens,
                                            completion_tokens=0, data_id=data_id,
                                            combo_id=combo_id, cached=self.cached))

    def get_records(self):
        return list(self.records)

    def get_usage(self, data_id=None, combo_id=None):
        tokens = {}
        for r in self.records:
            if (data_id is None or r.data_id == data_id) and (combo_id is None or r.combo_id == combo_id):
                tokens[r.model] = tokens.get(r.model, 0) + r.prompt_tokens
        return {model: (count, 0) for model, count in tokens.items()}

    def get_cached_latency(self, data_id):
        return 0.0


def make_selector(*, mode="multi", count=8, tracker=None, outcomes=None, tokens=None,
                  async_agent=False, **kwargs):
    tracker = tracker or RecordingTracker()
    outcomes = outcomes or {"cheap": 0.6, "middle": 0.8, "expensive": 0.9, "dominated": 0.5}
    tokens = tokens or {"cheap": 1000, "middle": 3000, "expensive": 10000, "dominated": 4000}

    class Agent:
        def __init__(self, models):
            self.model = models["solver"]

        def run(self, question):
            tracker.record(self.model, tokens[self.model])
            value = outcomes[self.model]
            return value(question) if callable(value) else value

    class AsyncAgent(Agent):
        async def run(self, question):
            return super().run(question)

    selector = ModelSelector(agent=AsyncAgent if async_agent else Agent,
                             models={"solver": list(outcomes)},
                             eval_fn=lambda expected, actual: actual,
                             dataset=[(i, None) for i in range(count)],
                             method="gittins", objective_mode=mode,
                             tracker=tracker, model_prices={m: {"input_price": 1, "output_price": 0} for m in outcomes},
                             batch_size=2, warm_start_batch_size=2, grid_size=33,
                             **kwargs)
    return selector, tracker


def test_complete_multi_frontier_and_explicit_preference():
    selector, tracker = make_selector(count=2)
    result = selector.select_best()
    assert result.stop_reason == "all_arms_completed"
    assert result.total_evaluations == 8
    assert tracker.stopped
    assert result.get_best() is None
    assert result.get_best_combo() is None
    assert [r.combo["solver"] for r in result.recommendations] == ["cheap", "middle", "expensive"]
    assert all(r.accuracy_std == 0 and r.cost_std_usd == 0 for r in result.recommendations)
    assert result.choose(max_cost_usd=0.004).combo == {"solver": "middle"}
    assert result.choose(min_score=0.85).combo == {"solver": "expensive"}
    assert result.choose(cost_weight=100).combo == {"solver": "cheap"}
    assert result.choose(max_cost_usd=0.0001) is None
    with pytest.raises(ValueError, match="Specify"):
        result.choose()
    assert all(not r.is_best for r in result)
    assert result.selection_cost == pytest.approx(0.036)


def test_two_node_workflow_cost_and_combination_recommendations():
    tracker = RecordingTracker()
    tokens = {"p-cheap": 1000, "p-strong": 2000, "s-cheap": 1000, "s-strong": 4000}

    class Agent:
        def __init__(self, models):
            self.models = models

        def run(self, question):
            for model in self.models.values():
                tracker.record(model, tokens[model])
            return (0.6 + 0.2 * (self.models["planner"] == "p-strong")
                    + 0.15 * (self.models["solver"] == "s-strong"))

    selector = ModelSelector(agent=Agent, method="gittins", objective_mode="multi",
                             models={"planner": ["p-cheap", "p-strong"], "solver": ["s-cheap", "s-strong"]},
                             dataset=[(i, None) for i in range(2)], eval_fn=lambda expected,actual:actual,
                             tracker=tracker, batch_size=2, warm_start_batch_size=2,
                             model_prices={m: {"input_price": 1, "output_price": 0} for m in tokens})
    result = selector.select_best()
    assert result.total_evaluations == 8
    assert result.selection_cost == pytest.approx(0.032)
    assert len(result.recommendations) == 3
    assert result.choose(max_cost_usd=0.004).combo == {"planner": "p-strong", "solver": "s-cheap"}
    assert all(len(record.input_tokens) == 2 for record in result)


@pytest.mark.parametrize("async_agent", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_single_dispatch_and_async_evaluation(async_agent, parallel):
    selector, tracker = make_selector(mode="single", count=5, async_agent=async_agent)
    assert isinstance(selector, GittinsModelSelector)
    result = selector.select_best(parallel=parallel, max_concurrent=2)
    assert len(result.recommendations) == 1
    assert sum(r.is_best for r in result) == 1
    assert result.get_best_combo() == {"solver": "expensive"}
    assert result.get_best().model_name == result.recommendations[0].model_name
    assert tracker.stopped
    # Original dataset IDs survive arbitrary per-arm permutations and short tails.
    for record in result:
        ids = [dp.datapoint_index for dp in record.datapoint_results]
        assert len(ids) == len(set(ids))
        assert all(0 <= q < 5 for q in ids)


def test_single_weighted_objective_uses_frozen_warm_scales():
    selector, _ = make_selector(mode="single", count=2, lambda_cost=1.0)
    result = selector.select_best()
    assert result.get_best_combo() == {"solver": "cheap"}
    assert result.get_best().combined_objective == pytest.approx(0.6)
    expensive = next(r for r in result if "expensive" in r.model_name)
    assert expensive.combined_objective == pytest.approx(-0.1)


def test_cell_budget_returns_terminal_recommendation_from_observed_data_only():
    outcomes = {m: (lambda q: 1.0 if q == 7 else 0.0) for m in ("cheap", "middle")}
    selector, tracker = make_selector(count=8, outcomes=outcomes, observation_budget_fraction=0.25)
    result = selector.select_best()
    assert result.stop_reason == "question_budget"
    assert result.total_evaluations == 4  # all spent on the mandatory warm start
    assert len(tracker.records) == 4
    assert len(result.results) == 2
    assert all(len(r.datapoint_results) == 2 for r in result)
    # Recompute finite prediction independently for the observed warm data.
    calibration = fit_warm_start([selector._scores[i] for i in range(2)], selector._costs)
    for recommendation in result.recommendations:
        arm = selector._names.index(recommendation.model_name)
        observation = calibration.observations(selector._scores[arm], selector._costs[arm])
        posterior = calibration.posterior(selector._scores[arm], selector._costs[arm])
        expected = (observation.sum(axis=0) + 6 * posterior.mean) / 8
        np.testing.assert_allclose([recommendation.accuracy, recommendation.cost_usd], expected)


def test_dollar_reservation_stops_without_spending_next_batch():
    selector, tracker = make_selector(count=8, max_search_cost_usd=0.037)
    result = selector.select_best()
    assert result.stop_reason == "search_cost_budget"
    assert result.total_evaluations == 8
    assert len(tracker.records) == 8
    assert result.selection_cost == pytest.approx(0.036)
    assert result.search_cost_overshoot_usd == 0
    assert result.recommendations


def test_warm_budget_exhaustion_returns_records_without_inventing_recommendations():
    selector, tracker = make_selector(max_search_cost_usd=0.0015)
    result = selector.select_best()
    assert result.stop_reason == "warm_start_budget"
    assert result.total_evaluations == 2
    assert result.recommendations == []
    assert result.get_best() is None
    assert result.selection_cost == pytest.approx(0.002)
    assert result.search_cost_overshoot_usd == pytest.approx(0.0005)
    assert tracker.stopped


def test_actual_cost_overshoot_is_reported_and_no_more_batches_are_launched():
    tracker = RecordingTracker()
    selector, _ = make_selector(outcomes={"cheap": 0.8}, count=8, tracker=tracker, max_search_cost_usd=0.005)
    original = tracker.record
    def costly_after_warm(model, input_tokens):
        original(model, input_tokens if len(tracker.records) < 2 else 10000)
    tracker.record = costly_after_warm
    result = selector.select_best()
    assert result.stop_reason == "search_cost_budget"
    assert result.total_evaluations == 4
    assert result.selection_cost == pytest.approx(0.022)
    assert result.search_cost_overshoot_usd == pytest.approx(0.017)


def test_cached_workflow_cost_is_separate_from_new_search_charges():
    tracker = RecordingTracker(cached=True)
    selector, _ = make_selector(count=2, tracker=tracker, max_search_cost_usd=0.0001)
    result = selector.select_best()
    assert result.selection_cost == 0
    assert result.stop_reason == "all_arms_completed"
    assert result.choose(max_cost_usd=0.004).combo == {"solver": "middle"}
    assert result.choose(max_cost_usd=0.004).cost_usd == pytest.approx(0.003)


def test_eta_stop_is_reached_without_evaluating_all_arms():
    selector, _ = make_selector(mode="single", count=6,
                               outcomes={"cheap": 1.0, "middle": 0.0},
                               score_noise_variance=1e-8, prior_variance=1e-8)
    result = selector.select_best()
    assert result.stop_reason == "direction_eta_numerical_floor"
    assert result.total_evaluations < 12
    assert result.direction_eta[0] < 1
    assert result.get_best_combo() == {"solver": "cheap"}


def test_min_samples_gate_can_produce_empty_set():
    selector, _ = make_selector(count=2, recommendation_min_samples=3)
    result = selector.select_best()
    assert result.recommendations == []
    assert result.get_best() is None


def test_invalid_configuration_does_not_start_tracker():
    for kwargs in ({"objective_mode": "wrong"}, {"batch_size": 0}, {"eta_decay": 1},
                   {"max_search_cost_usd": float("nan")}, {"observation_budget_fraction": 0.01},
                   {"objective_mode": "multi", "lambda_cost": 0.1}):
        tracker = RecordingTracker()
        with pytest.raises(ValueError):
            GittinsModelSelector(agent=object, models={"solver": ["cheap"]},
                                 eval_fn=lambda a, b: 1, dataset=[(i, 1) for i in range(8)], tracker=tracker, **kwargs)
        assert not tracker.started


def test_missing_scipy_only_blocks_gittins(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "scipy", None)
    tracker = RecordingTracker()
    with pytest.raises(ImportError, match="agentopt-py\\[gittins\\]"):
        GittinsModelSelector(agent=object, models={"solver": ["cheap"]},
                             eval_fn=lambda a, b: 1, dataset=[(1, 1)], tracker=tracker)
    assert not tracker.started
    selector = ModelSelector(agent=object, models={"solver": ["cheap"]},
                             eval_fn=lambda a, b: 1, dataset=[(1, 1)], tracker=tracker,
                             method="auto")
    assert type(selector).__name__ == "ArmEliminationModelSelector"


def test_multi_adaptive_search_is_reproducible_and_observations_are_shared():
    outputs = []
    for _ in range(2):
        selector, tracker = make_selector(count=7, observation_budget_fraction=0.65)
        result = selector.select_best(parallel=True, max_concurrent=2)
        assert result.total_evaluations <= 19
        assert result.total_evaluations == len(tracker.records)
        for record in result:
            questions = [dp.datapoint_index for dp in record.datapoint_results]
            assert len(questions) == len(set(questions))
        outputs.append(([[dp.datapoint_index for dp in record.datapoint_results] for record in result],
                        [r.model_name for r in result.recommendations], result.stop_reason))
    assert outputs[0] == outputs[1]


def test_prior_tracker_records_are_not_charged_to_new_selection():
    tracker = RecordingTracker()
    tracker.records.append(SimpleNamespace(model="unknown-old-model", prompt_tokens=1000000,
                                           completion_tokens=0, cached=False,
                                           data_id="old", combo_id="old"))
    selector, _ = make_selector(count=2, tracker=tracker)
    result = selector.select_best()
    assert result.selection_cost == pytest.approx(0.036)


def test_zero_cost_workflows_and_nondefault_score_bounds():
    selector, _ = make_selector(mode="single", count=3, outcomes={"cheap": 7.0, "middle": 9.0},
                               tokens={"cheap": 0, "middle": 0}, score_bounds=(0.0, 10.0))
    result = selector.select_best()
    assert result.get_best_combo() == {"solver": "middle"}
    assert result.selection_cost == 0
    assert result.recommendations[0].accuracy > 7


def test_live_token_metrics_do_not_accumulate_previous_runs():
    tracker = RecordingTracker()
    runs = []
    for _ in range(2):
        selector, _ = make_selector(count=2, tracker=tracker)
        result = selector.select_best()
        runs.append([(r.model_name, r.total_input_tokens, r.price) for r in result])
        assert result.selection_cost == pytest.approx(0.036)
    assert runs[0] == runs[1]


def test_failed_evaluation_stops_tracker():
    selector, tracker = make_selector(outcomes={"cheap": lambda q: (_ for _ in ()).throw(ValueError("failed"))})
    with pytest.raises(RuntimeError, match="Evaluation failed"):
        selector.select_best()
    assert tracker.stopped


def test_untracked_llm_usage_is_rejected_instead_of_assuming_zero_spend():
    selector, tracker = make_selector()
    tracker.record = lambda model, count: None
    with pytest.raises(ValueError, match="No LLM token usage"):
        selector.select_best()
    assert tracker.stopped


def test_unpriced_usage_stops_tracker():
    selector, tracker = make_selector()
    original = tracker.record
    tracker.record = lambda model, count: original("unknown-unpriced-model", count)
    with pytest.raises(ValueError, match="supply model_prices"):
        selector.select_best()
    assert tracker.stopped


def test_prediction_moments_have_correct_partial_and_complete_limits():
    posterior = GaussianPosterior(np.array([0.8, 0.003]), np.array([0.04, 1e-6]))
    mean, variance = finite_dataset_moments(posterior, np.array([1.0, 0.002]), 2, 4, np.array([0.25, 2e-6]))
    np.testing.assert_allclose(mean, [0.65, 0.002])
    np.testing.assert_allclose(variance, [0.04125, 5e-7])
    complete_mean, complete_var = finite_dataset_moments(posterior, np.array([3.0, 0.012]), 4, 4, np.array([0.25, 2e-6]))
    np.testing.assert_allclose(complete_mean, [0.75, 0.003])
    np.testing.assert_array_equal(complete_var, [0, 0])


def test_pareto_sweep_preserves_ties_and_matches_pairwise_dominance():
    points = np.array([[0.8, 0.002], [0.8, 0.002], [0.9, 0.004], [0.7, 0.002], [0.8, 0.005]])
    assert raw_pareto_front_indices(points) == (0, 1, 2)
    points = np.random.default_rng(42).random((100, 2))
    expected = tuple(i for i, p in enumerate(points) if not any(
        q[0] >= p[0] and q[1] <= p[1] and (q[0] > p[0] or q[1] < p[1]) for q in points))
    assert raw_pareto_front_indices(points) == expected


def test_warm_calibration_matches_raw_mean_protocol():
    scores = [[0, 1], [1, 1]]
    costs = [[0.001, 0.003], [0.002, 0.006]]
    calibration = fit_warm_start(scores, costs)
    np.testing.assert_allclose(calibration.prior_mean, [0.75, 0.003])
    np.testing.assert_allclose(calibration.prior_variance, [0.04, 2e-6])
    np.testing.assert_allclose(calibration.question_noise, [0.25, 5e-6])
    posterior = calibration.posterior(scores[0], costs[0])
    expected_variance = 1 / (1 / calibration.prior_variance + 2 / calibration.question_noise)
    expected_mean = expected_variance * (calibration.prior_mean / calibration.prior_variance
                                         + 2 * np.array([0.5, 0.002]) / calibration.question_noise)
    np.testing.assert_allclose(posterior.mean, expected_mean)
    np.testing.assert_allclose(posterior.variance, expected_variance)


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_real_tracker_http_calls_produce_costs_and_recommendations(mock_upstream, mode):
    import httpx
    from agentopt import LLMTracker

    class Agent:
        def __init__(self, models):
            self.model = models["solver"]

        def run(self, question):
            with httpx.Client(base_url=mock_upstream.base_url, trust_env=False) as client:
                response = client.post("/v1/chat/completions", json={
                    "model": self.model, "messages": [{"role": "user", "content": str(question)}],
                })
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

    tracker = LLMTracker(cache=False, cache_dir=None)
    selector = ModelSelector(agent=Agent, models={"solver": ["gpt-4o-mini"]},
                             eval_fn=lambda expected, actual: float(actual == expected),
                             dataset=[("capital", "Paris"), ("another", "Paris")],
                             method="gittins", objective_mode=mode, tracker=tracker,
                             warm_start_batch_size=1, batch_size=1, grid_size=17,
                             model_prices={"gpt-4o-mini": {"input_price": 1, "output_price": 2}})
    result = selector.select_best(parallel=True, max_concurrent=2)
    assert result.total_evaluations == 2
    assert result.selection_cost == pytest.approx(0.00004)
    assert result.recommendations[0].cost_usd == pytest.approx(0.00002)
    assert result.recommendations[0].accuracy == 1
    assert len({r.data_id for r in tracker.get_records()}) == 2
    assert mock_upstream.request_count == 2
