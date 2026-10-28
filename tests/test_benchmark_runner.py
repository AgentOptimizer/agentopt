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

    def display_and_interrupt(snapshot, **kwargs):
        original_display(snapshot, **kwargs)
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
            assert f"evaluated={candidate.observed_samples}/{len(answers)}" in displayed


@pytest.mark.parametrize("flags", [
    ["--method", "random_questions", "--max-search-cost-usd", "1"],
    ["--objective-mode", "multi", "--method", "ucb"],
    ["--method", "ucb", "--stop-on-recommendation"],
    ["--method", "bayesian", "--seed", "42"],
    ["--evaluation-fraction", "nan"],
    ["--live-plot"],
    ["--plot-output", "frontier.png"],
])
def test_invalid_options_fail_before_loading_sdk_or_constructing_selector(monkeypatch, tmp_path, flags):
    constructed = []
    monkeypatch.setattr(agentopt, "ModelSelector", lambda **kwargs: constructed.append(kwargs))
    with pytest.raises(SystemExit) as error:
        runner.main(["--dataset", str(tmp_path / "missing.jsonl"), "--models", "first", *flags])
    assert error.value.code == 2
    assert constructed == []


def test_multi_objective_runner_saves_actual_frontier(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg", force=True)

    tracker = RecordingTracker()
    source = SCRIPT.with_name("benchmark_sample.jsonl")
    answers = dict(runner.load_dataset(source))

    class FakeOpenAI:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, *, model, messages):
            tracker.record(model, 1000)
            answer = answers[messages[-1]["content"]]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    factory = agentopt.ModelSelector
    monkeypatch.setattr(agentopt, "ModelSelector", lambda **kwargs: factory(
        **kwargs, tracker=tracker,
        model_prices={name: {"input_price": 1, "output_price": 0} for name in ("first", "second")},
    ))
    output = tmp_path / "results.json"
    figure = tmp_path / "frontier.png"
    result = runner.main([
        "--dataset", str(source), "--models", "first", "second",
        "--objective-mode", "multi", "--plot-output", str(figure),
        "--output", str(output),
    ])

    assert result.is_final
    assert json.loads(output.read_text())["total_evaluations"] == result.total_evaluations
    assert figure.read_bytes().startswith(b"\x89PNG")
    assert len(tracker.records) == result.total_evaluations


def test_stop_handler_restored_if_selection_fails(tmp_path):
    class BrokenSelector:
        def select_best(self, **kwargs):
            raise RuntimeError("provider error")

    previous_handler = signal.getsignal(signal.SIGINT)
    with pytest.raises(runner.BenchmarkRunError, match="benchmark failed"):
        runner.run_selection(BrokenSelector(), max_concurrent=1, output=tmp_path / "result.json")
    assert signal.getsignal(signal.SIGINT) is previous_handler


@pytest.mark.parametrize("status,detail", [
    (0, "No API key is configured"), (401, "Authentication failed"), (403, "Access denied"),
    (429, "rate limit or quota"),
])
def test_provider_failures_are_reported_without_raw_response(monkeypatch, tmp_path, capsys, status, detail):
    import httpx
    import openai

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status, request=request)
    error_class = {0: openai.OpenAIError, 401: openai.AuthenticationError, 403: openai.PermissionDeniedError,
                   429: openai.RateLimitError}[status]
    error = (error_class("The api_key client option must be set") if status == 0 else
             error_class("private-provider-response", response=response, body={}))
    tracker = RecordingTracker()

    class FailingAgent:
        def __init__(self, models):
            if status == 0:
                raise error
            self.model = models["answer"]

        def run(self, question):
            raise error

    monkeypatch.setattr(runner, "AnswerAgent", FailingAgent)
    original = agentopt.ModelSelector
    monkeypatch.setattr(agentopt, "ModelSelector", lambda **kwargs: original(
        **kwargs, tracker=tracker,
        model_prices={"first": {"input_price": 1, "output_price": 0}},
    ))
    output = tmp_path / "results.json"
    output.write_text("previous complete result")
    with pytest.raises(SystemExit) as failure:
        runner.main(["--dataset", str(SCRIPT.with_name("benchmark_sample.jsonl")),
                     "--models", "first", "--output", str(output)])
    assert failure.value.code == 1
    terminal = capsys.readouterr()
    assert "Status: Failed" in terminal.err and detail in terminal.err
    assert "No evaluation results were saved" in terminal.err
    assert "private-provider-response" not in terminal.out + terminal.err
    assert "Traceback" not in terminal.err
    assert output.read_text() == "previous complete result"
    assert not output.with_name("results.partial.json").exists()
    assert tracker.stopped


@pytest.mark.parametrize("mode,fail_at,scoring", [
    ("single", 3, False), ("single", 9, False),
    ("multi", 9, False), ("single", 3, True),
])
def test_failure_saves_last_snapshot_and_preserves_complete_output(monkeypatch, tmp_path, capsys, mode, fail_at, scoring):
    tracker = RecordingTracker()
    calls = 0
    answers = dict(runner.load_dataset(SCRIPT.with_name("benchmark_sample.jsonl")))

    class FailingAgent:
        def __init__(self, models):
            self.model = models["answer"]

        def run(self, question):
            nonlocal calls
            calls += 1
            tracker.record(self.model, 1000)
            if calls >= fail_at and not scoring:
                raise RuntimeError("private-agent-input")
            return answers[question]

    original_score = runner.eval_fn
    def failing_score(expected, actual):
        if calls >= fail_at:
            raise ValueError("private-reference-answer")
        return original_score(expected, actual)

    monkeypatch.setattr(runner, "AnswerAgent", FailingAgent)
    if scoring:
        monkeypatch.setattr(runner, "eval_fn", failing_score)
    original = agentopt.ModelSelector
    selectors = []
    def factory(**kwargs):
        selector = original(**kwargs, tracker=tracker,
                            model_prices={name: {"input_price": 1, "output_price": 0}
                                          for name in ("first", "second")})
        selectors.append(selector)
        return selector
    monkeypatch.setattr(agentopt, "ModelSelector", factory)
    output = tmp_path / "results.json"
    output.write_text("previous complete result")
    with pytest.raises(SystemExit) as failure:
        runner.main(["--dataset", str(SCRIPT.with_name("benchmark_sample.jsonl")),
                     "--models", "first", "second", "--objective-mode", mode,
                     "--max-concurrent", "1", "--output", str(output)])
    assert failure.value.code == 1
    terminal = capsys.readouterr()
    partial = json.loads(output.with_name("results.partial.json").read_text())
    assert partial["is_partial"] and partial["is_final"]
    assert partial["stop_reason"] == "evaluation_failed"
    latest = selectors[0].latest_results
    assert partial["total_evaluations"] == latest.total_evaluations
    assert partial["results"] == latest.model_dump(mode="json")["results"]
    assert bool(partial["recommendations"]) == (fail_at == 9)
    assert "Saved partial results" in terminal.err
    assert "charges may be missing" in terminal.err
    assert ("Scoring failed" if scoring else "benchmark failed") in terminal.err
    assert "Status: Finished" not in terminal.out
    assert "private-agent-input" not in terminal.out + terminal.err
    assert "private-reference-answer" not in terminal.out + terminal.err
    assert output.read_text() == "previous complete result"
    assert tracker.stopped


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 2, "not a score"])
def test_invalid_scores_are_classified_as_scoring_errors(value):
    state = runner.FailureState()
    with pytest.raises(runner.BenchmarkRunError, match="Scoring failed"):
        state.wrap_scorer(lambda expected, actual: value)("reference", "answer")
    assert state.stop_event.is_set()


def test_partial_save_error_does_not_claim_results_were_saved(monkeypatch, tmp_path, capsys):
    from agentopt.model_selection.base import SelectionResults, ModelResult
    snapshot = SelectionResults(results=[ModelResult(model_name="answer=first", accuracy=1,
                                latency_seconds=1, attribute="combination")])
    def failed_save(*args):
        raise OSError("disk full")
    monkeypatch.setattr(runner, "save_json", failed_save)
    runner.report_failure("Scoring failed.", output=tmp_path / "results.json", snapshot=snapshot)
    output = capsys.readouterr().err
    assert "Partial results could not be saved" in output
    assert "Saved partial results" not in output


@pytest.mark.parametrize("method", ["random_configurations", "ucb"])
def test_skipped_scoring_errors_still_fail_terminal_run(monkeypatch, tmp_path, capsys, method):
    tracker = RecordingTracker()
    class Agent:
        def __init__(self, models):
            self.model = models["answer"]
        def run(self, question):
            tracker.record(self.model, 1000)
            return "answer"
    def score(expected, actual):
        raise ValueError("private-reference")
    monkeypatch.setattr(runner, "AnswerAgent", Agent)
    monkeypatch.setattr(runner, "eval_fn", score)
    original = agentopt.ModelSelector
    monkeypatch.setattr(agentopt, "ModelSelector", lambda **kwargs: original(
        **kwargs, tracker=tracker,
        model_prices={"first": {"input_price": 1, "output_price": 0}},
    ))
    output = tmp_path / "results.json"
    with pytest.raises(SystemExit) as failure:
        runner.main(["--dataset", str(SCRIPT.with_name("benchmark_sample.jsonl")),
                     "--models", "first", "--method", method, "--output", str(output)])
    assert failure.value.code == 1
    terminal = capsys.readouterr()
    assert "Scoring failed" in terminal.err
    assert "Status: Finished" not in terminal.out
    assert not output.exists()
    assert tracker.stopped


def test_normal_save_failure_preserves_existing_output_and_reports_partial(monkeypatch, tmp_path, capsys):
    from agentopt.model_selection.base import SelectionResults, ModelResult
    final = SelectionResults(results=[ModelResult(model_name="answer=first", accuracy=1,
                             latency_seconds=1, attribute="combination")], is_final=True)
    class Selector:
        latest_results = final
        def select_best(self, **kwargs):
            return final
    output = tmp_path / "results.json"
    output.write_text("previous complete result")
    real_save = runner.save_json
    def denied_complete(path, payload):
        if Path(path) == output:
            raise PermissionError("write denied")
        real_save(path, payload)
    monkeypatch.setattr(runner, "save_json", denied_complete)
    with pytest.raises(runner.BenchmarkRunError, match="required file"):
        runner.run_selection(Selector(), max_concurrent=1, output=output)
    assert output.read_text() == "previous complete result"
    assert output.with_name("results.partial.json").exists()
    terminal = capsys.readouterr()
    assert "Saved partial results" in terminal.err
    assert f"Saved results to {output}" not in terminal.out


def test_atomic_result_write_keeps_existing_file_if_serialization_fails(tmp_path):
    output = tmp_path / "results.json"
    output.write_text("previous complete result")
    with pytest.raises(ValueError):
        runner.save_json(output, {"invalid": float("nan")})
    assert output.read_text() == "previous complete result"
    assert not list(tmp_path.glob("*.tmp"))
