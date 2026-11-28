"""CoHV-UCB dispatch, empirical frontier, and checkpoint continuity."""
from collections import Counter

import pytest

from agentopt import ModelSelector
from agentopt.model_selection.checkpoint import CheckpointError
from tests.test_gittins import RecordingTracker


def run(calls, *, limit=18, method="ucb", checkpoint=None, on_update=None, **options):
    tracker = RecordingTracker()

    class Agent:
        def __init__(self, models):
            self.name = models["solver"]

        def run(self, question):
            calls.append((self.name, question))
            score, tokens = {"cheap": (.5, 1000), "strong": (.9, 3000), "dominated": (.4, 2000)}[self.name]
            tracker.record(self.name, tokens)
            return score

    names = ["cheap", "strong", "dominated"]
    selector = ModelSelector(agent=Agent, models={"solver": names}, eval_fn=lambda _, actual: actual,
                             dataset=[(i, None) for i in range(6)], method=method, objective_mode="multi",
                             batch_size=2, max_total_question_evaluations=limit, tracker=tracker,
                             model_prices={name: {"input_price": 1, "output_price": 0} for name in names},
                             checkpoint_path=checkpoint, checkpoint_id="ucb-test", **options)
    result = selector.select_best(parallel=False, show_ui=False, on_update=on_update)
    return selector, result


def test_cohv_uses_uniform_warmup_and_empirical_frontier():
    calls = []
    snapshots = []
    _, result = run(calls, limit=10, on_update=snapshots.append)
    assert Counter(name for name, _ in calls[:6]) == {"cheap": 2, "strong": 2, "dominated": 2}
    assert all(name == "cheap" for name, _ in calls[6:])
    assert all(not s.recommendations for s in snapshots if s.stop_reason == "warming_up")
    first = next(s for s in snapshots if s.recommendations)
    assert first.total_evaluations == 6
    assert [(r.combo["solver"], r.accuracy, r.cost_usd) for r in result.recommendations] == [
        ("cheap", .5, .001), ("strong", .9, .003)]
    assert result.total_evaluations == len(set(calls)) == 10
    assert result.stop_reason == "question_budget" and result.model_dump()["algorithm"] == "cohv_ucb"


def test_resume_previous_decisions_and_extend_both_budgets(tmp_path):
    db = tmp_path / "ucb.sqlite3"
    calls = []
    def run_with_cap(limit, dollars):
        return run(calls, limit=limit, checkpoint=db, max_search_cost_usd=dollars)
    _, initial = run_with_cap(12, .014)
    assert initial.total_evaluations == 8 and initial.stop_reason == "search_cost_budget"
    _, result = run_with_cap(18, .1)
    assert result.total_evaluations == len(set(calls)) == len(calls) == 18
    assert result.selection_cost == pytest.approx(.036)
    reopened, repeated = run_with_cap(18, .2)
    assert reopened.restored_cells == 18 and len(calls) == 18
    assert repeated.recommendations == result.recommendations
    with pytest.raises(CheckpointError):
        run(calls, method="gittins", checkpoint=db)
    assert len(calls) == 18


def test_partial_ucb_batch_resume_reports_empirical_means(tmp_path, monkeypatch):
    from agentopt.model_selection.checkpoint import EvaluationCheckpoint
    db = tmp_path / "partial.sqlite3"
    calls = []
    original = EvaluationCheckpoint.save_cell_at
    def interrupt(journal, batch, question, data):
        original(journal, batch, question, data)
        if len(calls) == 7:
            raise InterruptedError("after first search cell committed")
    with monkeypatch.context() as patch:
        patch.setattr(EvaluationCheckpoint, "save_cell_at", interrupt)
        with pytest.raises(InterruptedError):
            run(calls, checkpoint=db)
    snapshots = []
    selector, result = run(calls, checkpoint=db, on_update=snapshots.append)
    assert selector.restored_cells == 7
    assert snapshots[0].pending_batch_evaluations == 1
    assert snapshots[0].total_evaluations == 7
    assert snapshots[0].recommendations[0].observed_samples == 3
    assert result.total_evaluations == len(set(calls)) == len(calls) == 18
