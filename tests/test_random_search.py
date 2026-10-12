"""Random units, finite budgets, and deployment recommendations."""
import numpy as np
import pytest

from agentopt import ModelSelector, RandomSelectionResults
from tests.test_gittins import RecordingTracker


def make_random(method="random_configurations", mode="multi", count=5, tracker=None, **kwargs):
    tracker=tracker or RecordingTracker()
    observed=[]
    scores={"cheap":.6,"middle":.8,"expensive":.9,"dominated":.5}
    tokens={"cheap":1000,"middle":3000,"expensive":10000,"dominated":4000}
    class Agent:
        def __init__(self,models):self.model=models["solver"]
        def run(self,q):
            observed.append((self.model,q))
            tracker.record(self.model,tokens[self.model])
            return scores[self.model]
    selector=ModelSelector(agent=Agent,models={"solver":list(scores)},
                           dataset=[(i,None) for i in range(count)],eval_fn=lambda expected,actual:actual,
                           method=method,objective_mode=mode,tracker=tracker,
                           model_prices={m:{"input_price":1,"output_price":0} for m in scores},**kwargs)
    return selector,tracker,observed


@pytest.mark.parametrize("method",["random_configurations","random_questions"])
def test_random_full_frontier_and_no_repeated_cells(method):
    selector,tracker,observed=make_random(method=method)
    result=selector.select_best(parallel=True,max_concurrent=2)
    assert isinstance(result,RandomSelectionResults)
    assert len(observed)==len(set(observed))==20
    assert result.stop_reason=="all_arms_completed"
    assert result.get_best_combo() is None
    assert {r.combo["solver"] for r in result.recommendations}=={"cheap","middle","expensive"}
    assert result.choose(max_cost_usd=.004).combo=={"solver":"middle"}
    assert result.selection_cost==pytest.approx(.09)
    assert tracker.stopped


def test_random_configurations_evaluate_only_complete_rows():
    selector,_,observed=make_random(max_total_question_evaluations=11)
    result=selector.select_best()
    assert result.total_evaluations==10
    assert result.completed_units==2
    assert result.stop_reason=="question_budget"
    assert len({m for m,q in observed})==2
    for model in {m for m,q in observed}:
        assert {q for m,q in observed if m==model}==set(range(5))


def test_random_questions_use_shared_questions_without_replacement():
    selector,_,observed=make_random(method="random_questions",max_total_question_evaluations=10)
    result=selector.select_best()
    assert result.total_evaluations==8
    assert result.completed_units==2
    question_sets=[{q for m,q in observed if m==model} for model in ("cheap","middle","expensive","dominated")]
    assert all(qs==question_sets[0] for qs in question_sets)
    assert len(question_sets[0])==2


def test_random_single_returns_empirical_winner():
    selector,_,_=make_random(method="random",mode="single")
    result=selector.select_best()
    assert result.get_best_combo()=={"solver":"expensive"}
    assert result.recommendations[0].objective==pytest.approx(.9)


def test_random_single_cost_weight_is_separate_from_search_budget():
    selector,_,_=make_random(method="random",mode="single",lambda_cost=1)
    result=selector.select_best()
    assert result.get_best_combo()=={"solver":"cheap"}


@pytest.mark.parametrize("method",["random_configurations","random_questions"])
def test_seed_reproducibility(method):
    histories=[]
    for _ in range(2):
        selector,_,observed=make_random(method=method,seed=13)
        selector.select_best()
        histories.append(observed)
    assert histories[0]==histories[1]


@pytest.mark.parametrize("method",["random_configurations","random_questions"])
def test_dollar_cap_interrupted_unit_has_no_recommendations(method):
    selector,_,_=make_random(method=method,max_search_cost_usd=.0001)
    result=selector.select_best()
    assert result.total_evaluations==1
    assert result.completed_units==0
    assert result.recommendations==[]
    assert result.stop_reason=="search_cost_budget"
    assert result.search_cost_overshoot_usd==pytest.approx(result.selection_cost-.0001)


def test_partial_question_sweep_does_not_change_previous_recommendations():
    selector,_,_=make_random(method="random_questions",max_search_cost_usd=.0191)
    result=selector.select_best()
    assert result.completed_units==1
    assert result.total_evaluations==6
    assert all(r.observed_samples==1 for r in result.recommendations)
    assert sum(r.num_samples for r in result.results)==6


def test_cached_calls_have_deployment_cost_but_no_search_charge():
    selector,_,_=make_random(tracker=RecordingTracker(cached=True),max_search_cost_usd=.001)
    result=selector.select_best()
    assert result.total_evaluations==20
    assert result.selection_cost==0
    assert all(r.cost_usd>0 for r in result.recommendations)


def test_invalid_random_budget_does_not_start_tracking():
    tracker=RecordingTracker()
    with pytest.raises(ValueError,match="complete sampling unit"):
        make_random(tracker=tracker,max_total_question_evaluations=1)
    assert not tracker.started
