"""Public dispatch and single-objective finite-dataset contracts."""
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

from agentopt import ModelSelector, GittinsModelSelector, MatrixUCBModelSelector, RandomSearchModelSelector
from tests.test_gittins import RecordingTracker, make_selector


def arguments(tracker):
    return dict(agent=object, models={"solver": ["cheap"]}, eval_fn=lambda a,b:1,
                dataset=[(i, None) for i in range(8)], tracker=tracker)


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_auto_uses_gittins_without_an_extra(mode):
    tracker=RecordingTracker()
    selector=ModelSelector(**arguments(tracker), objective_mode=mode)
    assert type(selector) is GittinsModelSelector
    assert selector.cost_scaling_factor == 1e-4
    assert selector.beta == (0.0 if mode == "single" else 1.0)
    tracker.stop()
    metadata=tomllib.loads((Path(__file__).parents[1]/"pyproject.toml").read_text())
    assert any(requirement.startswith("scipy") for requirement in metadata["project"]["dependencies"])


@pytest.mark.parametrize("method,mode", [("ucb","multi"),("bayesian","multi"),("random","multi"),
                                        ("arm_elimination","single"),
                                        ("brute_force","single"),("matrix_ucb_lrf","single"),
                                        ("auto","invalid")])
def test_invalid_method_mode_rejected_before_tracker_starts(method,mode):
    tracker=RecordingTracker()
    with pytest.raises(ValueError):
        ModelSelector(**arguments(tracker), method=method, objective_mode=mode)
    assert not tracker.started


@pytest.mark.parametrize("method,kind", [("ucb",MatrixUCBModelSelector),("matrix_ucb",MatrixUCBModelSelector),
                                         ("random",RandomSearchModelSelector)])
def test_single_baseline_dispatch(method,kind):
    tracker=RecordingTracker()
    selector=ModelSelector(**arguments(tracker), method=method)
    assert type(selector) is kind
    tracker.stop()


def test_single_finite_mean_recommendations_use_fixed_prior_and_no_default_penalty():
    selector,_=make_selector(mode="single", count=8, observation_budget_fraction=.25)
    result=selector.select_best()
    assert result.total_evaluations == 8
    assert result.direction_eta == [1.0]
    winner=result.recommendations[0]
    assert winner.combo == {"solver":"expensive"}
    posterior_variance=1/(1/.04+2/.25)
    posterior_mean=posterior_variance*(.5/.04+2*.9/.25)
    expected=(2*.9+6*posterior_mean)/8
    assert winner.objective == pytest.approx(expected)
    assert winner.accuracy == pytest.approx(expected)
    assert winner.conservative_objective == winner.objective
    mean,initial,noise=selector._active_state(2,0)
    scale=1+.25/(8*.04)
    assert mean == pytest.approx(expected)
    assert initial == pytest.approx(posterior_variance*scale**2)
    assert noise == pytest.approx(.25*scale**2)
    # Independent next-cell conditional variance: finite mean is affine in Y.
    slope=(1+5*posterior_variance/(posterior_variance+.25))/8
    transition_variance=slope**2*(posterior_variance+.25)
    transformed_next=1/(1/initial+1/noise)
    assert initial-transformed_next == pytest.approx(transition_variance)


def test_completed_single_index_is_exact_empirical_mean():
    selector,_=make_selector(mode="single",count=2)
    result=selector.select_best()
    for i,mean in enumerate((.6,.8,.9,.5)):
        assert selector._active_state(i,0)[0] == pytest.approx(mean)
    assert result.get_best_combo()=={"solver":"expensive"}


def test_single_natural_stopping_keeps_lambda_fixed():
    selector,_=make_selector(mode="single",count=8,cost_scaling_factor=1000.)
    result=selector.select_best()
    assert result.stop_reason=="gittins_stop"
    assert result.total_evaluations<32
    assert result.direction_eta==[1.0]


def test_per_cell_dp_uses_raw_cost_lambda_and_actual_cell_stage(monkeypatch):
    import agentopt.model_selection.gittins as module
    calls=[]
    class Table:
        def index(self,stage,mean):
            calls[-1]["stage"]=stage
            return mean+100
    class Cache:
        def get(self,**kwargs):
            calls.append(kwargs)
            return Table()
    monkeypatch.setattr(module,"AxisGittinsBoundaryCache",Cache)
    selector,_=make_selector(mode="single",count=7,cost_scaling_factor=1e-4,
                            outcomes={"cheap":.6},max_total_question_evaluations=6)
    selector.select_best()
    scale=1+.25/(7*.04)
    assert [call["stage"] for call in calls]==[0,2]
    assert all(call["horizon"]==5 for call in calls)
    assert all(call["effective_pull_cost"]==pytest.approx(1e-7) for call in calls)
    assert all(call["obs_noise_var"]==pytest.approx(.25*scale**2) for call in calls)
