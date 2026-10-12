"""User benchmark loading and a terminal stop through the real selection loop."""

import importlib.util
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import agentopt
from tests.test_gittins import RecordingTracker


SCRIPT = Path(__file__).parents[1] / "examples" / "selection" / "benchmark_runner.py"
spec = importlib.util.spec_from_file_location("benchmark_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("contents,detail", [
    ("\n", "at least one example"),
    ('{"input": "question", "expected": "answer"}\ninvalid\n', ":2: invalid JSON"),
    ('{"input": "question", "expected": 3}\n', ":1: input and expected"),
    ('[]\n', ":1: input and expected"),
    ('{"input": " ", "expected": "answer"}\n', ":1: input and expected"),
])
def test_bad_dataset_reports_location_before_selection(tmp_path, contents, detail):
    source = tmp_path / "bad.jsonl"
    source.write_text(contents)
    with pytest.raises(ValueError, match=detail):
        runner.load_dataset(str(source))


@pytest.mark.parametrize("mode,stop_at,has_recommendation", [
    ("single", 2, False), ("single", 12, True), ("multi", 12, True),
])
def test_terminal_sigint_saves_real_partial_run(monkeypatch, tmp_path, capsys, mode, stop_at, has_recommendation):
    tracker = RecordingTracker()
    source = SCRIPT.with_name("benchmark_sample.jsonl")
    answers = dict(runner.load_dataset(source))
    assert len(answers) > 4  # The sample includes work beyond the default warm start.

    class FakeOpenAI:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, *, model, messages):
            tracker.record(model, 1000)
            answer = answers[messages[-1]["content"]]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    original_factory = agentopt.ModelSelector
    constructed = []

    def tracked_factory(**kwargs):
        selector = original_factory(
            **kwargs, tracker=tracker,
            model_prices={name: {"input_price": 1, "output_price": 0} for name in ("first", "second")},
        )
        constructed.append(selector)
        return selector

    monkeypatch.setattr(agentopt, "ModelSelector", tracked_factory)
    original_display = runner.show_progress
    updates = []

    def display_and_interrupt(snapshot):
        original_display(snapshot)
        updates.append(snapshot)
        if snapshot.total_evaluations >= stop_at and not snapshot.is_final:
            signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(runner, "show_progress", display_and_interrupt)
    previous_handler = signal.getsignal(signal.SIGINT)
    output = tmp_path / "results.json"
    final = runner.main([
        "--dataset", str(source), "--models", "first", "second",
        "--objective-mode", mode, "--output", str(output), "--max-concurrent", "2",
    ])
    saved = json.loads(output.read_text())
    assert final.is_final and saved["is_final"]
    assert final.stop_reason == saved["stop_reason"] == "user_stopped"
    assert stop_at <= final.total_evaluations < len(answers) * 2
    assert final.total_evaluations == len(tracker.records) == saved["total_evaluations"]
    assert final == constructed[0].latest_results == updates[-1]
    assert bool(saved["recommendations"]) == has_recommendation
    assert tracker.stopped
    assert signal.getsignal(signal.SIGINT) is previous_handler
    displayed = capsys.readouterr().out
    assert "No recommendation available yet." in displayed
    stop_message = "Stop requested. Waiting for the current batch to finish…"
    assert displayed.count(stop_message) == 1
    assert displayed.index(stop_message) < displayed.index("Status: Finished")
    assert "Status: Finished" in displayed
    assert "Reason: You requested a stop." in displayed
    if has_recommendation:
        assert "Current recommendation" in displayed
        for candidate in final.get_recommendations():
            assert f"score={candidate.conservative_accuracy:.4f}" in displayed
            assert f"cost=${candidate.conservative_cost_usd:.6f}" in displayed


@pytest.mark.parametrize("flags", [
    ["--method", "random_questions", "--max-search-cost-usd", "1"],
    ["--objective-mode", "multi", "--method", "ucb"],
    ["--method", "ucb", "--stop-on-recommendation"],
    ["--method", "bayesian", "--seed", "42"],
    ["--evaluation-fraction", "nan"],
])
def test_invalid_options_fail_before_loading_sdk_or_constructing_selector(monkeypatch, tmp_path, flags):
    constructed = []
    monkeypatch.setattr(agentopt, "ModelSelector", lambda **kwargs: constructed.append(kwargs))
    with pytest.raises(SystemExit) as error:
        runner.main(["--dataset", str(tmp_path / "missing.jsonl"), "--models", "first", *flags])
    assert error.value.code == 2
    assert constructed == []


def test_stop_handler_restored_if_selection_fails(tmp_path):
    class BrokenSelector:
        def select_best(self, **kwargs):
            raise RuntimeError("provider error")

    previous_handler = signal.getsignal(signal.SIGINT)
    with pytest.raises(RuntimeError, match="provider error"):
        runner.run_selection(BrokenSelector(), max_concurrent=1, output=tmp_path / "result.json")
    assert signal.getsignal(signal.SIGINT) is previous_handler
