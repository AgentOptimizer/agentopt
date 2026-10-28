"""The graphical view consumes real selector snapshots without model APIs."""

import threading
import time

import pytest

matplotlib = pytest.importorskip("matplotlib")

matplotlib.use("Agg", force=True)

from agentopt import ModelSelector
from agentopt.visualization import LiveParetoPlot
from tests.test_gittins import RecordingTracker, make_selector


def test_live_plot_runs_user_supplied_two_role_workflow(tmp_path):
    tracker = RecordingTracker()
    models = {"draft": ["cheap", "strong"], "review": ["cheap", "strong"]}
    dataset = [(f"question-{i}", None) for i in range(4)]

    class Agent:
        def __init__(self, selected):
            self.selected = selected

        def run(self, question):
            tracker.record(self.selected["draft"], 1000)
            tracker.record(self.selected["review"], 1000)
            return 0.4 + 0.2 * (self.selected["draft"] == "strong") + 0.2 * (
                self.selected["review"] == "strong"
            )

    selector = ModelSelector(
        agent=Agent, models=models, dataset=dataset,
        eval_fn=lambda expected, actual: actual,
        method="gittins", objective_mode="multi",
        tracker=tracker, model_prices={
            "cheap": {"input_price": 1, "output_price": 0},
            "strong": {"input_price": 4, "output_price": 0},
        },
        warm_start_batch_size=2, batch_size=2, grid_size=33,
    )
    output = tmp_path / "frontier.png"
    plot = LiveParetoPlot(total_questions=len(dataset), warm_total=8,
                          show_window=False, output=output)
    updates = []
    result = plot.run(selector, max_concurrent=2, on_update=updates.append)

    assert result.is_final and result.total_evaluations == 16
    assert all(set(r.combo) == {"draft", "review"} for r in result.get_recommendations())
    assert updates[-1].is_final and plot.finished
    assert plot._last_evaluations == result.total_evaluations
    assert output.read_bytes().startswith(b"\x89PNG")


def test_pause_continue_and_finish_control_real_selection(tmp_path):
    def measured(score):
        def run(_question):
            time.sleep(0.04)
            return score
        return run

    selector, tracker = make_selector(count=12, outcomes={
        "cheap": measured(0.6), "middle": measured(0.8),
        "expensive": measured(0.9), "dominated": measured(0.5),
    })
    plot = LiveParetoPlot(total_questions=12, warm_total=8,
                          show_window=False, output=tmp_path / "stopped.png")
    plot.toggle_pause(None)
    updates = []

    def inspect(snapshot):
        assert threading.current_thread() is threading.main_thread()
        updates.append(snapshot)
        if len(updates) == 1:
            assert plot._pause_active
            plot.toggle_pause(None)
        elif len(updates) == 2:
            plot.finish_search(None)

    result = plot.run(selector, max_concurrent=2, on_update=inspect)

    assert result.is_final and result.stop_reason == "user_stopped"
    assert result.total_evaluations < 4 * 12
    assert tracker.stopped and updates[-1].is_final
    assert (tmp_path / "stopped.png").read_bytes().startswith(b"\x89PNG")
