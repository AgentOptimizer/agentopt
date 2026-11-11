"""Public cost-aware allocation, tracked charges and in-budget frontiers."""

import importlib.util
from unittest.mock import patch

import numpy as np
import pytest

from agentopt import (
    ModelSelector, CostAwareUCBModelSelector, CostAwareQNEHVIModelSelector,
    CostAwareSelectionResults,
)
from tests.test_gittins import RecordingTracker

HAS_BOTORCH = importlib.util.find_spec("botorch") is not None
METHODS = ["cost_aware_ucb", pytest.param("cost_aware_qnehvi", marks=pytest.mark.skipif(
    not HAS_BOTORCH, reason="requires agentopt-py[bayesian]",
))]


@pytest.fixture(autouse=True)
def limit_torch_threads():
    if not HAS_BOTORCH:
        yield
        return
    import torch
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def make_selector(method="cost_aware_ucb", count=6, tracker=None, async_agent=False, **kwargs):
    tracker = tracker or RecordingTracker()
    observed = []
    scores = {"cheap": .6, "middle": .8, "expensive": .9, "dominated": .5}
    tokens = {"cheap": 1000, "middle": 3000, "expensive": 10000, "dominated": 4000}

    class Agent:
        def __init__(self, models):
            self.model = models["solver"]

        def run(self, question):
            observed.append((self.model, question))
            tracker.record(self.model, tokens[self.model])
            return scores[self.model]

    class AsyncAgent(Agent):
        async def run(self, question):
            return super().run(question)

    options = dict(warm_start_batch_size=2, batch_size=2)
    options.update(kwargs)
    if method == "cost_aware_qnehvi":
        options.setdefault("mc_samples", 8)
    selector = ModelSelector(
        agent=AsyncAgent if async_agent else Agent, models={"solver": list(scores)},
        dataset=[(i, None) for i in range(count)], eval_fn=lambda expected, actual: actual,
        method=method, objective_mode="multi", tracker=tracker,
        model_prices={m: {"input_price": 1, "output_price": 0} for m in scores},
        **options,
    )
    return selector, tracker, observed


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("parallel,async_agent", [(False, False), (True, True)])
def test_full_frontier_through_public_api(method, parallel, async_agent):
    selector, tracker, observed = make_selector(method, count=4, async_agent=async_agent)
    result = selector.select_best(parallel=parallel, max_concurrent=2)
    assert isinstance(result, CostAwareSelectionResults)
    assert result.stop_reason == "all_arms_completed"
    assert result.total_evaluations == result.recommendation_evaluations == 16
    assert len(set(observed)) == len(observed) == 16
    assert {r.combo["solver"] for r in result.recommendations} == {"cheap", "middle", "expensive"}
    assert result.selection_cost == pytest.approx(.072)
    assert result.recommendation_search_cost_usd == pytest.approx(.072)
    assert result.choose(max_cost_usd=.004).combo == {"solver": "middle"}
    assert result.choose(min_score=.85).combo == {"solver": "expensive"}
    assert result.get_best_combo() is None
    assert all(r.price > 0 for r in result.results)
    assert tracker.stopped


def test_scalarization_alias_and_direct_exports():
    selector, tracker, _ = make_selector("cost_aware_scalarization_ucb", count=2)
    assert type(selector) is CostAwareUCBModelSelector
    assert CostAwareQNEHVIModelSelector.method == "cost_aware_qnehvi"
    tracker.stop()


@pytest.mark.parametrize("method", ["cost_aware_ucb", "cost_aware_scalarization_ucb", "cost_aware_qnehvi"])
def test_single_mode_fails_before_tracking(method):
    tracker = RecordingTracker()
    with pytest.raises(ValueError, match="objective_mode"):
        ModelSelector(agent=object, models={"solver": ["cheap"]}, dataset=[(0, None)],
                      eval_fn=lambda x, y: 1, method=method, tracker=tracker)
    assert not tracker.started


@pytest.mark.parametrize("options", [
    {"batch_size": 0}, {"seed": True}, {"warm_start_batch_size": -1},
    {"score_bounds": (1, 1)}, {"observation_budget_fraction": float("nan")},
    {"max_total_question_evaluations": 7}, {"max_search_cost_usd": 0},
    {"cost_reference_usd": float("inf")}, {"lambda_cost": 1},
])
def test_invalid_parameters_do_not_start_tracking(options):
    tracker = RecordingTracker()
    with pytest.raises(ValueError):
        make_selector(tracker=tracker, **options)
    assert not tracker.started


def test_ucb_prefers_more_quality_per_dollar_and_truncates_cell_batch():
    selector, _, observed = make_selector(max_total_question_evaluations=11)
    result = selector.select_best()
    assert result.stop_reason == "question_budget"
    assert result.total_evaluations == result.recommendation_evaluations == 11
    # Equal optimistic warm scores are ranked by actual predicted batch charge.
    assert all(model == "cheap" for model, question in observed[8:10])
    assert len(set(observed)) == 11


def test_crossing_batch_is_charged_but_not_recommended():
    selector, _, _ = make_selector(max_search_cost_usd=.0365)
    warm = []

    def update(snapshot):
        if snapshot.recommendation_evaluations == 8 and snapshot.total_evaluations == 8:
            warm.append(snapshot)

    result = selector.select_best(on_update=update)
    assert result.stop_reason == "search_cost_budget"
    assert result.total_evaluations == 10
    assert result.recommendation_evaluations == 8
    assert result.selection_cost == pytest.approx(.038)
    assert result.recommendation_search_cost_usd == pytest.approx(.036)
    assert result.search_cost_overshoot_usd == pytest.approx(.0015)
    assert result.recommendations == warm[-1].recommendations
    assert sum(len(r.datapoint_results) for r in result) == 10


def test_dollar_cap_during_warm_start_has_no_frontier():
    selector, tracker, observed = make_selector(max_search_cost_usd=.0005)
    result = selector.select_best()
    assert result.stop_reason == "warm_start_budget"
    assert result.total_evaluations == len(observed) == 1
    assert result.recommendation_evaluations == 0
    assert result.recommendations == []
    assert result.selection_cost == pytest.approx(.001)
    assert tracker.stopped


def test_user_stop_and_independent_live_snapshots():
    selector, tracker, observed = make_selector()
    snapshots = []

    def update(snapshot):
        snapshots.append(snapshot)
        if snapshot.total_evaluations >= 10:
            return False

    result = selector.select_best(on_update=update)
    assert result.stop_reason == "user_stopped"
    assert len(observed) == result.total_evaluations == 10
    assert result.recommendation_evaluations == 10
    assert snapshots[-1].is_final
    snapshots[-1].recommendations[0].accuracy = -100
    assert selector.latest_results.recommendations[0].accuracy >= 0
    assert tracker.stopped


def test_cached_responses_keep_deployment_cost_without_search_charge():
    selector, _, observed = make_selector(tracker=RecordingTracker(cached=True), max_search_cost_usd=.000001)
    result = selector.select_best()
    assert len(observed) == 24
    assert result.selection_cost == result.recommendation_search_cost_usd == 0
    assert all(r.cost_usd > 0 for r in result.recommendations)


def test_missing_bayesian_extra_fails_before_tracking():
    tracker = RecordingTracker()
    with patch("agentopt.model_selection._qnehvi.require_botorch", side_effect=ImportError("install bayesian")):
        with pytest.raises(ImportError, match="bayesian"):
            make_selector("cost_aware_qnehvi", tracker=tracker)
    assert not tracker.started


@pytest.mark.skipif(not HAS_BOTORCH, reason="requires agentopt-py[bayesian]")
def test_qnehvi_role_features_incumbents_and_refit_schedule():
    tracker = RecordingTracker()
    calls = []

    class Agent:
        def __init__(self, models):
            self.models = models

        def run(self, question):
            for model in self.models.values():
                tracker.record(model, 1000)
            return .5

    def acquisition(train, means, candidates, **kwargs):
        calls.append((train.copy(), candidates.copy(), kwargs))
        return 0

    with patch("agentopt.model_selection._qnehvi.select_qnehvi_index", side_effect=acquisition):
        selector = ModelSelector(
            agent=Agent, models={"planner": ["p1", "p2"], "solver": ["s1", "s2"]},
            dataset=[(i, None) for i in range(5)], eval_fn=lambda expected, actual: actual,
            method="cost_aware_qnehvi", objective_mode="multi", tracker=tracker,
            model_prices={m: {"input_price": 1, "output_price": 0} for m in ["p1", "p2", "s1", "s2"]},
            warm_start_batch_size=1, batch_size=1, refit_every=2, mc_samples=8,
        )
        result = selector.select_best()
    assert result.total_evaluations == 20
    assert len(calls) == 6  # two acquisitions each for the first three configurations
    assert calls[0][0].tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert calls[0][2]["categorical_dims"] == [0, 1]
    assert calls[0][2]["baseline_features"].shape == (0, 2)
    assert calls[2][2]["baseline_features"].tolist() == [[0, 0]]
    assert all(np.all(call[2]["candidate_costs"] > 0) for call in calls)


@pytest.mark.skipif(not HAS_BOTORCH, reason="requires agentopt-py[bayesian]")
@pytest.mark.parametrize("baseline_rows", [0, 1])
def test_real_qnehvi_empty_and_nonempty_incumbents_preserve_rng(baseline_rows):
    import torch
    from agentopt.model_selection._qnehvi import select_qnehvi_index
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[.3, .9], [.5, .7], [.7, .5], [.9, .3]])
    state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    chosen = select_qnehvi_index(
        x, y, x[1:], categorical_dims=[0, 1], baseline_features=x[:baseline_rows],
        candidate_costs=[.01, .02, .03], mc_samples=8, seed=42, candidate_chunk_size=2,
    )
    assert 0 <= chosen < 3
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.get_num_threads() == threads


@pytest.mark.skipif(not HAS_BOTORCH, reason="requires agentopt-py[bayesian]")
def test_qnehvi_acquisition_is_divided_by_candidate_cost():
    import torch
    from agentopt.model_selection._qnehvi import select_qnehvi_index
    x = np.array([[0], [1], [2]], dtype=float)
    y = np.array([[.3, .9], [.5, .7], [.7, .5]])
    with patch("botorch.fit.fit_gpytorch_mll"), patch(
        "botorch.acquisition.multi_objective.monte_carlo.qNoisyExpectedHypervolumeImprovement"
    ) as acquisition:
        acquisition.return_value.return_value = torch.tensor([.4, .9])
        chosen = select_qnehvi_index(
            x, y, x[:2], categorical_dims=[0], baseline_features=x[:0],
            candidate_costs=[1, 3], mc_samples=8,
        )
    assert chosen == 0
