"""Cooperative stopping, independent progress snapshots, and partial results."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from agentopt import ModelSelector
from agentopt.model_selection import bayesian_optimization as bo
from tests.test_gittins import RecordingTracker, make_selector
from tests.test_random_search import make_random


@pytest.mark.parametrize("mode", ["single", "multi"])
@pytest.mark.parametrize("stop_at", [1, 8, 10])
def test_gittins_callback_stops_during_warmup_or_after_calibration(mode, stop_at):
    selector, tracker = make_selector(mode=mode)
    updates = []

    def update(result):
        updates.append(result)
        # At 8 evaluations wait for the calibrated snapshot.
        if result.total_evaluations >= stop_at and (stop_at != 8 or result.recommendations):
            return False

    final = selector.select_best(on_update=update)
    assert final.stop_reason == "user_stopped"
    assert final.total_evaluations == stop_at
    assert bool(final.recommendations) == (stop_at >= 8)
    assert final.is_final and updates[-1].is_final
    assert all(not r.is_final for r in updates[:-1])
    assert updates[-1] == final == selector.latest_results
    assert final.selection_cost == pytest.approx(sum(r.prompt_tokens for r in tracker.records) / 1e6)
    assert tracker.stopped


@pytest.mark.parametrize("mode", ["single", "multi"])
@pytest.mark.parametrize("method,unit_size", [("random_configurations", 5), ("random_questions", 4)])
@pytest.mark.parametrize("completed", [0, 1])
def test_random_stop_preserves_only_complete_comparable_units(mode, method, unit_size, completed):
    selector, _, observed = make_random(method=method, mode=mode)
    threshold = completed * unit_size + 1
    updates = []

    def update(result):
        updates.append(result)
        return result.total_evaluations < threshold

    final = selector.select_best(on_update=update)
    assert len(observed) == final.total_evaluations == threshold
    assert final.completed_units == completed
    assert final.stop_reason == "user_stopped"
    assert final.is_final
    assert bool(final.recommendations) == bool(completed)
    if completed:
        expected_samples = 5 if method == "random_configurations" else 1
        assert all(r.observed_samples == expected_samples for r in final.recommendations)
        comparable = next(r for r in updates if r.total_evaluations == unit_size)
        assert final.recommendations == comparable.recommendations


def test_callback_and_latest_result_copies_cannot_mutate_other_snapshots():
    selector, _, _ = make_random(method="random_questions", mode="single")
    saved = []

    def update(result):
        saved.append(result.model_copy(deep=True))
        original_count = result.total_evaluations
        result.results.clear()
        result.recommendations.clear()
        result.total_evaluations = -1
        return original_count < 6

    final = selector.select_best(on_update=update)
    assert final.total_evaluations == 6
    assert final.recommendations
    assert saved[0].total_evaluations == 1
    assert len(saved[0].results) == 1
    latest = selector.latest_results
    latest.results[0].datapoint_results.clear()
    latest.recommendations.clear()
    assert selector.latest_results == final
    final.results.clear()
    assert selector.latest_results.results
    assert [s.selection_cost for s in saved] == sorted(s.selection_cost for s in saved)
    assert [s.selection_wall_time_seconds for s in saved] == sorted(s.selection_wall_time_seconds for s in saved)


@pytest.mark.parametrize("method", ["gittins", "random_configurations", "random_questions", "ucb"])
def test_preset_stop_event_does_not_evaluate(method):
    if method == "gittins":
        selector, tracker = make_selector()
    elif method == "ucb":
        selector, tracker, _ = make_basic(method)
    else:
        selector, tracker, _ = make_random(method=method)
    stopped = Event()
    stopped.set()
    assert selector.latest_results is None
    updates = []
    final = selector.select_best(stop_event=stopped, on_update=updates.append)
    assert final.stop_reason == "user_stopped"
    assert final.total_evaluations == 0
    assert final.get_best() is None
    assert final.is_final
    assert not tracker.records
    assert tracker.stopped
    assert len(updates) == 1


def test_request_stop_from_another_thread_finishes_only_the_current_cell():
    selector, tracker, observed = make_random(mode="single")
    original = selector.agent.run
    entered, release = Event(), Event()

    def blocking_run(self, question):
        entered.set()
        assert release.wait(5), "test did not release current evaluation"
        return original(self, question)

    selector.agent.run = blocking_run
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(selector.select_best)
        try:
            assert entered.wait(5), "evaluation did not begin"
            selector.request_stop()
        finally:
            release.set()
        final = future.result(timeout=5)
    assert final.stop_reason == "user_stopped"
    assert final.total_evaluations == len(observed) == 1
    assert final.recommendations == []
    assert tracker.stopped


def test_callback_failure_propagates_and_stops_tracker():
    selector, tracker, _ = make_random()

    def broken_callback(result):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        selector.select_best(on_update=broken_callback)
    assert tracker.stopped
    assert selector.latest_results.total_evaluations == 1
    assert not selector.latest_results.is_final


def make_basic(method, *, outcomes=None, count=2, **kwargs):
    tracker = RecordingTracker()
    outcomes = outcomes or {"first": -2.0, "second": -1.0, "third": -0.5, "fourth": -0.2, "fifth": -0.1}
    observed = []

    class Agent:
        def __init__(self, models):
            self.model = models["solver"]

        def run(self, question):
            observed.append((self.model, question))
            tracker.record(self.model, 1000)
            outcome = outcomes[self.model]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome(question) if callable(outcome) else outcome

    selector = ModelSelector(
        agent=Agent, models={"solver": list(outcomes)},
        eval_fn=lambda expected, actual: actual,
        dataset=[(i, None) for i in range(count)], method=method,
        tracker=tracker,
        model_prices={name: {"input_price": 1, "output_price": 0} for name in outcomes},
        **kwargs,
    )
    return selector, tracker, observed


def test_ucb_partial_negative_score_never_recommends_unobserved_combo():
    selector, _, observed = make_basic("ucb")
    final = selector.select_best(max_concurrent=1, on_update=lambda result: False)
    assert len(observed) == 1
    assert final.get_best_combo() == {"solver": observed[0][0]}
    assert final.get_best().accuracy < 0
    assert sum(r.is_best for r in final.results) == 1


def test_ucb_all_failed_batch_terminates_without_recommendation():
    selector, tracker, observed = make_basic("ucb", outcomes={"bad": RuntimeError("provider unavailable")})
    final = selector.select_best(max_concurrent=1)
    assert final.stop_reason == "evaluation_failed"
    assert len(observed) == 1
    assert final.get_best() is None
    assert final.total_evaluations == 0
    assert final.selection_cost == pytest.approx(.001)
    assert tracker.stopped


def make_bo(monkeypatch, *, initial=2, batch_size=2, outcomes=None, count=2):
    """Exercise real evaluation/selection control flow without a GP dependency."""
    monkeypatch.setattr(bo, "_require_botorch", lambda: None)
    monkeypatch.setattr(bo.random, "shuffle", lambda values: None)
    selector, tracker, observed = make_basic("bayesian", outcomes=outcomes, count=count, batch_size=batch_size)
    choices = selector._models["solver"]
    combos = [(i,) for i in range(len(choices))]
    monkeypatch.setattr(selector, "_bo_setup", lambda: (
        None, None, None, None, None, ["solver"], [choices], 1,
        combos, len(combos), initial, 3, len(combos),
    ))

    def acquire(*args):
        evaluated = args[-1]
        return [combo for combo in combos if combo not in evaluated][:batch_size] or None

    monkeypatch.setattr(selector, "_bo_fit_and_acquire", acquire)
    return selector, tracker, observed


@pytest.mark.parametrize("parallel", [False, True])
def test_bo_stops_after_initial_update_without_extra_evaluations(monkeypatch, parallel):
    selector, tracker, observed = make_bo(monkeypatch)
    final = selector.select_best(parallel=parallel, max_concurrent=2, on_update=lambda result: False)
    assert final.total_evaluations == len(observed) == 2
    assert final.get_best_combo() == {"solver": "first"}
    assert final.selection_cost == pytest.approx(.002)
    assert final.stop_reason == "user_stopped"
    assert tracker.stopped


@pytest.mark.parametrize("parallel,expected", [(False, 6), (True, 8)])
def test_bo_stops_after_acquired_update_and_keeps_only_one_winner(monkeypatch, parallel, expected):
    selector, _, observed = make_bo(monkeypatch)
    updates = []

    def update(result):
        updates.append(result)
        return result.total_evaluations < 6

    final = selector.select_best(parallel=parallel, max_concurrent=2, on_update=update)
    assert final.total_evaluations == len(observed) == expected
    assert final.stop_reason == "user_stopped"
    assert all(sum(row.is_best for row in result.results) == 1 for result in updates)
    assert updates[0].get_best_combo() == {"solver": "first"}
    assert final.get_best_combo() == {"solver": "fourth" if parallel else "third"}
    assert final == selector.latest_results


@pytest.mark.parametrize("parallel", [False, True])
def test_bo_failure_cannot_beat_negative_observed_score(monkeypatch, parallel):
    selector, _, _ = make_bo(monkeypatch, initial=2, outcomes={
        "good": -2.0, "bad": RuntimeError("provider unavailable")})
    final = selector.select_best(parallel=parallel)
    assert final.get_best_combo() == {"solver": "good"}
    assert not next(r for r in final.results if r.model_name == "solver=bad").is_best
    assert final.total_evaluations == 2
    assert sum(r.is_best for r in final.results) == 1


@pytest.mark.parametrize("parallel", [False, True])
def test_bo_callback_failure_is_not_swallowed_as_evaluation_error(monkeypatch, parallel):
    selector, tracker, _ = make_bo(monkeypatch)

    def broken_callback(result):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        selector.select_best(parallel=parallel, on_update=broken_callback)
    assert tracker.stopped


@pytest.mark.parametrize("parallel", [False, True])
def test_bo_preset_event_does_not_evaluate(monkeypatch, parallel):
    selector, _, observed = make_bo(monkeypatch)
    stopped = Event()
    stopped.set()
    final = selector.select_best(parallel=parallel, stop_event=stopped)
    assert final.total_evaluations == 0
    assert observed == []
    assert final.stop_reason == "user_stopped"
    assert final.get_best() is None


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("partial_index", [0, 2])
def test_bo_keeps_partial_records_but_excludes_them_from_recommendation_and_gp(monkeypatch, parallel, partial_index):
    def partial_score(question):
        if question == 1:
            raise RuntimeError("second question failed")
        return 1.0

    outcomes = {f"model{i}": -0.5 + .05 * i for i in range(5)}
    outcomes[f"model{partial_index}"] = partial_score
    selector, tracker, observed = make_bo(
        monkeypatch, outcomes=outcomes, batch_size=1,
        initial=3 if partial_index == 0 else 2,
    )
    gp_inputs, updates = [], []
    real_acquire = selector._bo_fit_and_acquire

    def inspect_acquisition(*args):
        gp_inputs.append([tuple(combo) for combo in args[5]])
        return real_acquire(*args)

    monkeypatch.setattr(selector, "_bo_fit_and_acquire", inspect_acquisition)
    final = selector.select_best(parallel=parallel, max_concurrent=2, on_update=updates.append)
    assert gp_inputs
    assert all((partial_index,) not in inputs for inputs in gp_inputs)
    assert all(result.get_best_combo() != {"solver": f"model{partial_index}"} for result in updates)
    partial = next(result for result in final.results if result.model_name == f"solver=model{partial_index}")
    assert len(partial.datapoint_results) == 1
    assert partial.accuracy == 1.0
    assert not partial.is_best
    assert final.total_evaluations == 9
    assert len(observed) == len(tracker.records) == 10
    assert final.selection_cost == pytest.approx(.01)
    assert final.get_best_combo() == {"solver": "model4"}


@pytest.mark.parametrize("method", ["gittins", "random_questions", "ucb", "bayesian"])
def test_final_cost_uses_ledger_before_tracker_archival(monkeypatch, method):
    if method == "gittins":
        selector, tracker = make_selector()
    elif method == "random_questions":
        selector, tracker, _ = make_random(method=method)
    elif method == "bayesian":
        selector, tracker, _ = make_bo(monkeypatch)
    else:
        selector, tracker, _ = make_basic(method)
    original_stop = tracker.stop

    def archive_records():
        # Reproduce a backend exposing both archived and session records.
        tracker.records.extend(list(tracker.records))
        original_stop()

    monkeypatch.setattr(tracker, "stop", archive_records)
    updates = []

    def update(result):
        updates.append(result)
        return False

    final = selector.select_best(max_concurrent=1, on_update=update)
    doubled_cost = sum(record.prompt_tokens for record in tracker.records) / 1e6
    assert doubled_cost > 0
    assert final.selection_cost == pytest.approx(doubled_cost / 2)
    assert updates[-1].selection_cost == final.selection_cost
    assert selector.latest_results.selection_cost == final.selection_cost
    assert tracker.stopped


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_parallel_random_stop_finishes_only_the_scheduled_chunk(mode):
    selector, tracker, observed = make_random(mode=mode)
    final = selector.select_best(parallel=True, max_concurrent=2, on_update=lambda result: False)
    assert len(observed) == final.total_evaluations == 2
    assert final.completed_units == 0
    assert final.recommendations == []
    assert final.stop_reason == "user_stopped"
    assert final.selection_cost == pytest.approx(sum(record.prompt_tokens for record in tracker.records) / 1e6)
