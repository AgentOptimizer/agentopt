"""The graphical view consumes real selector snapshots without model APIs."""

import sys
import threading
import time

import pytest

matplotlib = pytest.importorskip("matplotlib")

matplotlib.use("Agg", force=True)

from agentopt import ModelSelector
from agentopt.visualization.live_pareto import LiveParetoPlot, local_desktop_available
from tests.test_gittins import RecordingTracker, make_selector


def test_interactive_shell_with_non_gui_backend_stays_headless(monkeypatch):
    monkeypatch.setattr(sys, "ps1", ">>> ", raising=False)
    for key in ("CI", "SSH_TTY", "SSH_CONNECTION"):
        monkeypatch.delenv(key, raising=False)
    with matplotlib.rc_context({"backend": "Agg"}):
        assert not local_desktop_available()


def test_script_inheriting_terminal_returns_without_a_window(monkeypatch):
    for key in ("CI", "SSH_TTY", "SSH_CONNECTION"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delattr(sys, "ps1", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(matplotlib.cbook, "_get_running_interactive_framework", lambda: None)
    monkeypatch.setattr(matplotlib, "get_backend", lambda: pytest.fail("script probed the GUI backend"))
    selector, tracker = make_selector(count=1)
    result = selector.select_best()
    assert result.is_final and result.total_evaluations == len(tracker.records) == 4


@pytest.mark.parametrize("mode, desktop, options", [
    ("single", True, {}), ("multi", False, {}), ("multi", True, {"show_ui": False}),
])
def test_selection_without_auto_window(monkeypatch, mode, desktop, options):
    monkeypatch.setattr("agentopt.visualization.live_pareto.local_desktop_available", lambda: desktop)
    def unexpected_window(*args, **kwargs):
        pytest.fail("selection opened a window")
    monkeypatch.setattr(LiveParetoPlot, "__init__", unexpected_window)
    selector, tracker = make_selector(mode=mode, count=1)
    result = selector.select_best(**options)
    assert result.total_evaluations == len(tracker.records) == 4


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


def test_callback_stop_also_applies_in_plot_mode():
    selector, tracker = make_selector()
    plot = LiveParetoPlot(total_questions=8, show_window=False)
    try:
        result = plot.run(selector, warm_start_max_concurrent=1, on_update=lambda snapshot: snapshot.total_evaluations < 2)
        assert result.stop_reason == "user_stopped"
        assert result.total_evaluations == len(tracker.records) == 2
    finally:
        plot.plt.close(plot.figure)


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
        bounds = artist.get_tightbbox(renderer)
        if bounds is None:
            continue
        assert bounds.x0 >= panel_bounds.x0 and bounds.x1 <= panel_bounds.x1
        assert bounds.y0 >= panel_bounds.y0 and bounds.y1 <= panel_bounds.y1
    metrics = [t for t in plot.recommendation_axis.texts if t.get_color() == MINT]
    assert metrics and all("tested" in t.get_text() and "/ 1k" in t.get_text() for t in metrics)
    wheel = SimpleNamespace(inaxes=plot.recommendation_axis, button="down")
    first_card = next(t for t in plot.recommendation_axis.texts if t.get_text() == "1")
    original_y = first_card.get_position()[1]
    plot._on_scroll(wheel)
    first_card = next(t for t in plot.recommendation_axis.texts if t.get_text() == "1")
    assert first_card.get_position()[1] == pytest.approx(original_y + .09)
    for _ in range(35):
        plot._on_scroll(wheel)
    assert plot._legend_offset == plot._legend_scroll_limit
    low, high = plot.scroll_thumb.get_ydata()
    assert low == pytest.approx(0) and high < 1
    assert any("36/200 tested" in t.get_text() for t in plot.recommendation_axis.texts)
    recommendations[:] = recommendations[:2]
    plot.update(snapshot)
    plot._on_scroll(wheel)
    assert plot._legend_offset == 0
    assert plot.scroll_axis.get_visible()
    assert list(plot.scroll_thumb.get_ydata()) == [0, 1]
    assert {"1", "2"} <= {t.get_text() for t in plot.recommendation_axis.texts}
    recommendations[:] = [
        SimpleNamespace(model_name=str(i), combo={"planner": "Planner model", "solver": "Solver model"},
                        conservative_accuracy=.4+i*.05, conservative_cost_usd=.001+i*.0001,
                        observed_samples=8)
        for i in range(4)
    ]
    plot.update(snapshot)
    assert {"1", "2", "3", "4"} <= {t.get_text() for t in plot.recommendation_axis.texts}
    assert list(plot.scroll_thumb.get_ydata()) == [0, 1]
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


def test_finish_before_first_evaluation_exports_empty_frontier(tmp_path):
    selector, tracker = make_selector(count=1, outcomes={"only": 0.0}, tokens={"only": 0})
    output = tmp_path / "empty.png"
    plot = LiveParetoPlot(total_questions=1, total_combinations=1, warm_total=1,
                          show_window=False, output=output)
    try:
        plot.finish_button._observers.process("clicked", None)
        result = plot.run(selector)

        assert result.stop_reason == "user_stopped"
        assert result.total_evaluations == len(tracker.records) == 0
        assert result.get_recommendations() == []
        assert plot.evaluation_count.get_text() == "0 / 1"
        assert [metric.get_text() for metric in plot.metric_values] == ["0.00%", "$0.0000", "0"]
        assert plot.empty_title.get_visible() and plot.empty_title.get_text() == "No recommendation available"
        assert not plot.axis.get_visible() and not plot.progress_axis.get_visible()
        assert plot.status_text.get_text() == "Finished"
        assert output.read_bytes().startswith(b"\x89PNG")
    finally:
        plot.plt.close(plot.figure)


def test_single_zero_cost_zero_score_point_is_selectable():
    from matplotlib.backend_bases import MouseEvent

    selector, tracker = make_selector(count=1, outcomes={"only": 0.0}, tokens={"only": 0})
    plot = LiveParetoPlot(total_questions=1, total_combinations=1, warm_total=1, show_window=False)
    try:
        result = plot.run(selector)
        candidate, = result.get_recommendations()
        assert result.total_evaluations == len(tracker.records) == 1
        assert plot.evaluation_count.get_text() == "1 / 1"
        assert [metric.get_text() for metric in plot.metric_values] == ["100.00%", "$0.0000", "1"]
        assert plot.axis.get_visible() and not plot.empty_title.get_visible()
        point, = plot.axis.collections[0].get_offsets()
        assert tuple(point) == pytest.approx((0, 0))

        canvas = plot.figure.canvas
        canvas.draw()
        x, y = plot.axis.transData.transform(point)
        for action in ("button_press_event", "button_release_event"):
            canvas.callbacks.process(action, MouseEvent(action, canvas, x, y, button=1))
        assert plot._selected_combination == candidate.model_name
        assert any("1/1 tested" in text.get_text() for text in plot.recommendation_axis.texts)
    finally:
        plot.plt.close(plot.figure)


def test_chart_number_reveals_matching_card_without_panning(monkeypatch):
    from types import SimpleNamespace
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.backend_tools import Cursors

    plot = LiveParetoPlot(total_questions=200, total_combinations=8, show_window=False)
    recommendations = [SimpleNamespace(
        model_name=f"option {i}", combo={"planner": f"Model {i}", "solver": "Solver"},
        conservative_accuracy=.4 + i*.03, conservative_cost_usd=.001 + i*.0002,
        observed_samples=i+1) for i in range(8)]
    snapshot = SimpleNamespace(stop_reason="running", total_evaluations=80, selection_cost=.5,
                               is_final=False, get_recommendations=lambda: recommendations)
    plot.update(snapshot)
    plot.figure.canvas.draw()
    initial_limits = (plot.axis.get_xlim(), plot.axis.get_ylim())
    _, label, name = plot._chart_labels[-1]
    bounds = label.get_window_extent()
    x, y = (bounds.x0+bounds.x1)/2, (bounds.y0+bounds.y1)/2
    cursors = []
    monkeypatch.setattr(plot.figure.canvas, "set_cursor", cursors.append)
    point_x, point_y = plot.axis.transData.transform(plot._chart_labels[-1][0])
    for mouse_x, mouse_y, expected in ((point_x, point_y, Cursors.HAND),
                                        (0, 0, Cursors.POINTER), (x, y, Cursors.HAND)):
        event = MouseEvent("motion_notify_event", plot.figure.canvas, mouse_x, mouse_y)
        plot.figure.canvas.callbacks.process(event.name, event)
        assert cursors[-1] == expected
    for action in ("button_press_event", "button_release_event"):
        event = MouseEvent(action, plot.figure.canvas, x, y, button=1)
        plot.figure.canvas.callbacks.process(action, event)
    assert plot._selected_combination == name
    assert plot._legend_offset > 0
    assert plot._legend_offset == pytest.approx(plot._legend_scroll_limit)
    assert initial_limits == (plot.axis.get_xlim(), plot.axis.get_ylim())
    renderer = plot.figure.canvas.get_renderer()
    metric = next(t for t in plot.recommendation_axis.texts if "8/200 tested" in t.get_text())
    assert metric.get_tightbbox(renderer).height == metric.get_window_extent(renderer).height

    recommendations.pop()
    plot.update(snapshot)
    assert plot._selected_combination is None


def test_results_actions_follow_successful_export(tmp_path, monkeypatch):
    plot = LiveParetoPlot(total_questions=10, show_window=False)
    try:
        assert not plot.open_results_button.active
        results = tmp_path / "results.json"
        results.write_text("{}", encoding="utf-8")
        opened = []
        monkeypatch.setattr(plot, "_open_path", opened.append)
        plot.show_saved_results(results)
        assert not plot.finish_button.ax.get_visible()
        assert plot.open_results_button.ax.get_visible()
        plot.open_results_button._observers.process("clicked", None)
        plot.open_folder_button._observers.process("clicked", None)
        assert opened == [results, tmp_path]
        assert "Saved: results.json" in plot.detail_text.get_text()
    finally:
        plot.plt.close(plot.figure)


def test_ui_start_failure_leaves_other_tracker_hooks_owned_by_caller(tmp_path, monkeypatch):
    import httpx
    from agentopt.proxy import LLMTracker, interceptor
    monkeypatch.chdir(tmp_path)
    matplotlib.use("Agg", force=True)
    original = httpx.Client.send
    refs = interceptor._install_refcount, interceptor._subprocess_install_refcount
    with LLMTracker(cache=False, cache_dir=None):
        external = httpx.Client.send
        selector, tracker = make_selector(count=2)
        with pytest.raises(RuntimeError, match="interactive Matplotlib"):
            selector.select_best(show_ui=True)
        assert not tracker.started and httpx.Client.send is external
        assert (interceptor._install_refcount, interceptor._subprocess_install_refcount) == (refs[0]+1, refs[1]+1)
        assert selector.select_best().total_evaluations == len(tracker.records)
    assert httpx.Client.send is original
    assert (interceptor._install_refcount, interceptor._subprocess_install_refcount) == refs
