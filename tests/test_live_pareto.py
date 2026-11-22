"""The graphical view consumes real selector snapshots without model APIs."""

import threading
import time

import pytest

matplotlib = pytest.importorskip("matplotlib")

matplotlib.use("Agg", force=True)

from agentopt import ModelSelector
from agentopt.visualization.live_pareto import LiveParetoPlot
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


def test_final_export_survives_closed_window_dimensions(tmp_path):
    from types import SimpleNamespace
    from PIL import Image

    plot = LiveParetoPlot(total_questions=4, warm_total=8, show_window=False,
                          output=tmp_path / "closed.png")
    plot.figure.set_size_inches(0, 0, forward=False)
    plot.update(SimpleNamespace(stop_reason="user_stopped", total_evaluations=2,
                                selection_cost=0.01, is_final=True,
                                get_recommendations=lambda: []))
    with Image.open(plot.output) as image:
        assert image.size == (1888, 1152)
    assert not plot.pause_button.active
    assert not plot.finish_button.active


@pytest.mark.parametrize("finish_while_paused", [False, True])
def test_pause_freezes_work_and_spend_then_resume_or_finish(tmp_path, finish_while_paused):
    selector, tracker = make_selector(count=8)
    plot = LiveParetoPlot(total_questions=8, total_combinations=4, warm_total=8,
                          show_window=False, output=tmp_path / "controlled.png")
    original_update = plot.update
    snapshots = []
    pause_requested = False
    paused_at = None
    frozen = None
    continued = False

    def update(snapshot):
        nonlocal pause_requested
        original_update(snapshot)
        snapshots.append(snapshot)
        assert plot.metric_values[0].get_text() == f"{snapshot.total_evaluations / 32 * 100:.2f}%"
        assert plot.metric_values[1].get_text() == f"${snapshot.selection_cost:.4f}"
        assert plot.metric_values[2].get_text() == str(len(snapshot.get_recommendations()))
        if snapshot.get_recommendations():
            expected = sorted(snapshot.get_recommendations(), key=lambda r: r.conservative_cost_usd)
            assert list(plot.axis.lines[0].get_xdata()) == pytest.approx([r.conservative_cost_usd * 1000 for r in expected])
        if not pause_requested and snapshot.get_recommendations() and not snapshot.is_final:
            plot.pause_button._observers.process("clicked", None)
            assert plot.status_text.get_text() == "Pausing…"
            pause_requested = True
        elif continued and not snapshot.is_final:
            plot.finish_button._observers.process("clicked", None)
            assert plot.status_text.get_text() == "Finishing…"

    def pump():
        nonlocal paused_at, frozen, continued
        plot._render_status()
        if plot._pause_active:
            current = (len(tracker.records), plot._last_evaluations, plot._last_spent)
            if paused_at is None:
                paused_at, frozen = time.monotonic(), current
            assert current == frozen
            assert plot.status_text.get_text() == "Paused"
            if time.monotonic() - paused_at >= .2:
                if finish_while_paused:
                    plot.finish_button._observers.process("clicked", None)
                else:
                    plot.pause_button._observers.process("clicked", None)
                    continued = True
                    assert plot.status_text.get_text() == "Searching"

    plot.update = update
    plot.pump_events = pump
    result = plot.run(selector, max_concurrent=2)
    assert paused_at is not None
    assert result.is_final and result.stop_reason == "user_stopped"
    assert len(tracker.records) == result.total_evaluations
    assert result.selection_cost == pytest.approx(sum(r.prompt_tokens / 1_000_000 for r in tracker.records))
    assert plot.status_text.get_text() == "Finished"
    assert not plot.pause_button.active and not plot.finish_button.active
    assert not plot.paused and not plot._pause_active
    assert result.total_evaluations < 32
    assert snapshots[-1].model_dump() == result.model_dump()
    assert all(a.selection_cost <= b.selection_cost for a, b in zip(snapshots, snapshots[1:]))


def test_percentage_and_green_metrics_fit_fixed_canvas():
    from types import SimpleNamespace
    from agentopt.visualization.live_pareto import MINT

    plot = LiveParetoPlot(total_questions=200, total_combinations=100, warm_total=400, show_window=False)
    recommendations = [SimpleNamespace(model_name=str(i), combo={
        "planner": "Qwen3 Next 80B A3B", "solver": "GPT-5.6 Luna", "reviewer": "Claude Haiku 4.5"
    }, conservative_accuracy=.429+i*.02, conservative_cost_usd=.000851+i*.0002,
        observed_samples=8+i*4) for i in range(8)]
    snapshot = SimpleNamespace(stop_reason="", total_evaluations=424, selection_cost=1.279058,
                               is_final=False, get_recommendations=lambda: recommendations)
    plot.update(snapshot)
    assert plot.metric_values[0].get_text() == "2.12%"
    assert plot.evaluation_count.get_text() == "424 / 20,000"
    plot.figure.canvas.draw()
    renderer = plot.figure.canvas.get_renderer()
    panel_bounds = plot.recommendation_axis.get_window_extent(renderer)
    for artist in plot.recommendation_axis.texts:
        bounds = artist.get_window_extent(renderer)
        assert bounds.x0 >= panel_bounds.x0 and bounds.x1 <= panel_bounds.x1
        assert bounds.y0 >= panel_bounds.y0 and bounds.y1 <= panel_bounds.y1
    metrics = [t for t in plot.recommendation_axis.texts if t.get_color() == MINT]
    assert metrics and all("tested" in t.get_text() and "/ 1k" in t.get_text() for t in metrics)
    for _ in range(12):
        plot._on_scroll(SimpleNamespace(inaxes=plot.recommendation_axis, button="down"))
    assert plot._legend_offset == 7
    assert "36/200 tested" in next(t.get_text() for t in plot.recommendation_axis.texts if t.get_color() == MINT)
    snapshot.is_final = True
    snapshot.stop_reason = "user_stopped"
    plot.update(snapshot)
    assert plot.metric_values[0].get_text() == "2.12%"
    assert not plot.progress_axis.get_visible()


def test_complete_run_reaches_full_coverage():
    selector, tracker = make_selector(count=2)
    plot = LiveParetoPlot(total_questions=2, total_combinations=4, warm_total=8, show_window=False)
    result = plot.run(selector)
    assert result.stop_reason == "all_arms_completed"
    assert plot.metric_values[0].get_text() == "100.00%"
    assert plot.evaluation_count.get_text() == "8 / 8"
    assert plot.status_text.get_text() == "Finished"
