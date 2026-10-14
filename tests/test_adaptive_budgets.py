"""Search spending caps at adaptive-policy evaluation boundaries."""

import pytest

from agentopt import ModelSelector
from agentopt.model_selection import bayesian_optimization as bo
from tests.test_anytime import make_basic
from tests.test_gittins import RecordingTracker


def make_budget_bo(monkeypatch, *, initial=2, batch_size=2, count=2, **kwargs):
    """Keep real BO evaluation orchestration; replace only dependency/GP work."""
    monkeypatch.setattr(bo, "_require_botorch", lambda: None)
    monkeypatch.setattr(bo.random, "shuffle", lambda values: None)
    selector, tracker, observed = make_basic(
        "bayesian", count=count, batch_size=batch_size, **kwargs,
    )
    choices = selector._models["solver"]
    combos = [(i,) for i in range(len(choices))]
    monkeypatch.setattr(selector, "_bo_setup", lambda: (
        None, None, None, None, None, ["solver"], [choices], 1,
        combos, len(combos), initial, 3, len(combos),
    ))
    monkeypatch.setattr(selector, "_bo_fit_and_acquire", lambda *args: (
        [combo for combo in combos if combo not in args[-1]][:batch_size] or None
    ))
    return selector, tracker, observed


@pytest.mark.parametrize("method", ["ucb", "bayesian"])
@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_invalid_adaptive_cap_fails_before_tracker_starts(monkeypatch, method, budget):
    monkeypatch.setattr(bo, "_require_botorch", lambda: None)
    tracker = RecordingTracker()
    with pytest.raises(ValueError, match="max_search_cost_usd"):
        ModelSelector(
            agent=lambda: None, models={"solver": ["first"]},
            eval_fn=lambda expected, actual: actual, dataset=[(0, None)],
            method=method, tracker=tracker, max_search_cost_usd=budget,
        )
    assert not tracker.started


@pytest.mark.parametrize("batch_size,expected", [(1, 2), (3, 3)])
def test_ucb_cap_stops_before_next_batch_and_reports_overshoot(batch_size, expected):
    selector, tracker, observed = make_basic("ucb", count=4, max_search_cost_usd=.0015)
    updates = []
    final = selector.select_best(max_concurrent=batch_size, on_update=updates.append)
    assert len(observed) == final.total_evaluations == expected
    assert final.stop_reason == "search_cost_budget"
    assert final.selection_cost == pytest.approx(expected * .001)
    assert final.search_cost_overshoot_usd == pytest.approx(expected * .001 - .0015)
    assert updates[-1] == final == selector.latest_results
    assert tracker.stopped


@pytest.mark.parametrize("parallel", [False, True])
def test_bo_cap_stops_after_initial_complete_configuration(monkeypatch, parallel):
    selector, tracker, observed = make_budget_bo(monkeypatch, max_search_cost_usd=.0015)
    final = selector.select_best(parallel=parallel, max_concurrent=2)
    assert len(observed) == final.total_evaluations == 2
    assert final.get_best_combo() == {"solver": "first"}
    assert final.stop_reason == "search_cost_budget"
    assert final.selection_cost == pytest.approx(.002)
    assert final.search_cost_overshoot_usd == pytest.approx(.0005)
    assert tracker.stopped


@pytest.mark.parametrize("parallel,expected", [(False, 6), (True, 8)])
def test_bo_cap_stops_after_scheduled_acquisition_batch(monkeypatch, parallel, expected):
    selector, tracker, observed = make_budget_bo(monkeypatch, max_search_cost_usd=.005)
    updates = []
    final = selector.select_best(parallel=parallel, max_concurrent=4, on_update=updates.append)
    assert len(observed) == final.total_evaluations == expected
    assert final.stop_reason == "search_cost_budget"
    assert final.selection_cost == pytest.approx(expected * .001)
    assert final.search_cost_overshoot_usd == pytest.approx(expected * .001 - .005)
    assert updates[-1] == final
    assert tracker.stopped


@pytest.mark.parametrize("parallel", [False, True])
def test_bo_rechecks_budget_after_acquisition_before_scheduling(monkeypatch, parallel):
    selector, tracker, observed = make_budget_bo(monkeypatch, max_search_cost_usd=1)
    acquired = False
    original_check = selector._search_budget_stop
    original_acquire = selector._bo_fit_and_acquire

    def acquire(*args):
        nonlocal acquired
        acquired = True
        return original_acquire(*args)

    monkeypatch.setattr(selector, "_bo_fit_and_acquire", acquire)
    monkeypatch.setattr(selector, "_search_budget_stop", lambda: (
        "search_cost_budget" if acquired else original_check()
    ))
    final = selector.select_best(parallel=parallel, max_concurrent=4)
    assert acquired
    assert len(observed) == final.total_evaluations == 4
    assert final.stop_reason == "search_cost_budget"
    assert tracker.stopped


@pytest.mark.parametrize("method,parallel", [("ucb", False), ("bayesian", False), ("bayesian", True)])
def test_unpriced_explicit_cap_raises_outside_evaluation_error_handling(monkeypatch, method, parallel):
    if method == "bayesian":
        selector, tracker, observed = make_budget_bo(monkeypatch, max_search_cost_usd=.01)
    else:
        selector, tracker, observed = make_basic("ucb", max_search_cost_usd=.01)
    selector._custom_prices = {}
    with pytest.raises(ValueError, match="price|pricing"):
        selector.select_best(parallel=parallel, max_concurrent=1)
    assert observed
    assert tracker.stopped
    assert selector.latest_results is None or not selector.latest_results.is_final


@pytest.mark.parametrize("method,parallel", [("ucb", False), ("bayesian", False), ("bayesian", True)])
def test_cached_evaluations_do_not_consume_search_cap(monkeypatch, method, parallel):
    if method == "bayesian":
        selector, tracker, observed = make_budget_bo(monkeypatch, max_search_cost_usd=.0001)
    else:
        selector, tracker, observed = make_basic("ucb", max_search_cost_usd=.0001)
    tracker.cached = True
    final = selector.select_best(parallel=parallel, max_concurrent=2)
    assert final.stop_reason == "all_arms_completed"
    assert len(observed) == final.total_evaluations == 10
    assert final.selection_cost == 0
    assert final.search_cost_overshoot_usd == 0


def test_ucb_failed_evaluations_still_consume_search_cap():
    selector, tracker, observed = make_basic(
        "ucb", outcomes={"bad": RuntimeError("provider failed after billing")},
        max_search_cost_usd=.0005,
    )
    final = selector.select_best(max_concurrent=1)
    assert final.stop_reason == "search_cost_budget"
    assert len(observed) == 1
    assert final.total_evaluations == 0
    assert final.get_best() is None
    assert final.selection_cost == pytest.approx(.001)
    assert final.search_cost_overshoot_usd == pytest.approx(.0005)
    assert tracker.stopped
