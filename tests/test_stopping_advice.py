"""Gittins advice is independent of user, budget, and optional automatic stops."""
import pytest

from tests.test_gittins import RecordingTracker, make_selector


def advised_selector(**kwargs):
    return make_selector(mode="single", count=6,
                         outcomes={"cheap": 1.0, "middle": 0.0},
                         score_noise_variance=1e-8, prior_variance=1e-8, **kwargs)


def test_default_recommends_stop_but_keeps_evaluating():
    selector, tracker = advised_selector()
    updates = []
    final = selector.select_best(on_update=updates.append)
    advice = [r for r in updates if r.stop_recommended and not r.is_final]
    assert selector.stop_on_recommendation is False
    assert advice and advice[0].total_evaluations < 12
    assert all(r.stop_reason == "running" for r in advice)
    assert final.stop_reason == "all_arms_completed"
    assert final.total_evaluations == 12
    assert final.get_best_combo() == {"solver": "cheap"}
    assert all(r.direction_eta == [1.0] for r in updates)
    assert all(not r.stop_recommended for r in updates if r.stop_reason == "warming_up")
    assert tracker.stopped


def test_user_can_accept_advice_through_callback():
    selector, _ = advised_selector()
    final = selector.select_best(on_update=lambda result: not result.stop_recommended)
    assert final.stop_reason == "user_stopped"
    assert final.stop_recommended
    assert final.total_evaluations < 12
    assert final.is_final


def test_explicit_automatic_stop_has_same_advice_point_as_manual_acceptance():
    automatic, _ = advised_selector(stop_on_recommendation=True)
    automatic_result = automatic.select_best()
    manual, _ = advised_selector()
    manual_result = manual.select_best(on_update=lambda result: not result.stop_recommended)
    assert automatic_result.stop_reason == "gittins_stop"
    assert automatic_result.stop_recommended
    assert automatic_result.total_evaluations == manual_result.total_evaluations
    assert automatic_result.get_best_combo() == manual_result.get_best_combo()


def test_continuing_after_advice_still_obeys_cell_cap():
    automatic, _ = advised_selector(stop_on_recommendation=True)
    first_advice = automatic.select_best().total_evaluations
    limit = first_advice + 2
    assert limit < 12
    selector, _ = advised_selector(max_total_question_evaluations=limit)
    updates = []
    final = selector.select_best(on_update=updates.append)
    assert final.total_evaluations == limit
    assert final.stop_reason == "question_budget"
    assert any(r.stop_recommended and r.total_evaluations < limit for r in updates)
    assert final.stop_recommended


def test_continuing_after_advice_still_obeys_dollar_cap():
    automatic, _ = advised_selector(stop_on_recommendation=True)
    first = automatic.select_best()
    selector, _ = advised_selector(max_search_cost_usd=first.selection_cost + .0061)
    final = selector.select_best()
    assert final.stop_reason == "search_cost_budget"
    assert first.total_evaluations < final.total_evaluations < 12
    assert final.stop_recommended


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_automatic_stop_flag_must_be_boolean_before_tracker_start(value):
    tracker = RecordingTracker()
    with pytest.raises(ValueError, match="stop_on_recommendation must be a boolean"):
        advised_selector(tracker=tracker, stop_on_recommendation=value)
    assert not tracker.started


def test_stopping_advice_is_single_objective_only():
    tracker = RecordingTracker()
    with pytest.raises(ValueError, match="requires single mode"):
        make_selector(mode="multi", tracker=tracker, stop_on_recommendation=True)
    assert not tracker.started
    selector, _ = make_selector(mode="multi", count=2)
    assert selector.select_best().stop_recommended is False
