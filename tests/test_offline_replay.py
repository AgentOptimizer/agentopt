"""Exercise the reviewer CLI with a small score and cost matrix."""

import csv
import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def replay(monkeypatch):
    examples = Path(__file__).resolve().parents[1] / "examples" / "selection"
    monkeypatch.syspath_prepend(str(examples))
    return importlib.import_module("replay_10x10")


@pytest.fixture
def workflow(tmp_path):
    directory = tmp_path / "scope" / "bird_mini_dev"
    directory.mkdir(parents=True)
    scores = {"local": [1, 1, 0, 0, 0, 0], "hybrid": [1, 1, 1, 1, 1, 0],
              "premium": [1] * 6, "legacy": [1, 0, 0, 0, 0, 0]}
    prices = {"local": 0, "hybrid": .002, "premium": .01, "legacy": .02}
    for filename, values in (("accuracy_matrix.csv", scores),
                             ("cost_matrix_usd.csv", {k: [v] * 6 for k, v in prices.items()})):
        with (directory / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model_name", *(f"question_{i}" for i in range(6))])
            for name, row in values.items():
                writer.writerow([name, *row])
    configs = [{"model_name": name, "config": {"schema_linking": "model-a", "classification": "model-b",
                                               "execution": name, "debug": "model-c"}}
               for name in scores]
    (directory / "configs.json").write_text(json.dumps(configs), encoding="utf-8")
    return directory


def run_replay(replay, monkeypatch, tmp_path, *args):
    monkeypatch.setattr(sys, "argv", ["replay_10x10.py", "--benchmark", "bird_mini_dev",
                        "--data-root", str(tmp_path), "--grid-size", "17",
                        "--delay-seconds", "0", "--output", str(tmp_path / "result.json"), *map(str, args)])
    replay.main()


def test_frontier_survives_stop_and_resume(replay, workflow, tmp_path, monkeypatch, capsys):
    run_replay(replay, monkeypatch, tmp_path, "--stop-after", "16")
    partial = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert partial["total_evaluations"] == 16
    run_replay(replay, monkeypatch, tmp_path)
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["total_evaluations"] == 24
    assert result["selection_cost"] == pytest.approx(6 * (.002 + .01 + .02))
    assert "16 restored, 8 newly evaluated" in capsys.readouterr().out
    recommendations = {r["combo"]["execution"]: r for r in result["recommendations"]}
    assert recommendations.keys() == {"local", "hybrid", "premium"}
    for name, score, cost in [("local", 2/6, 0), ("hybrid", 5/6, .002), ("premium", 1, .01)]:
        assert recommendations[name]["accuracy"] == pytest.approx(score)
        assert recommendations[name]["cost_usd"] == pytest.approx(cost)


@pytest.mark.parametrize("destination", ["input", "plot", "checkpoint-lock"])
def test_conflicting_replay_paths_rejected_before_writes(replay, workflow, tmp_path, monkeypatch, destination):
    source = workflow / "accuracy_matrix.csv"
    original = source.read_bytes()
    checkpoint = tmp_path / "run.db"
    options = {
        "input": ["--output", source],
        "plot": ["--plot-output", tmp_path / "report", "--output", tmp_path / "report.png"],
        "checkpoint-lock": ["--output", tmp_path / "run.db.lock"],
    }
    with pytest.raises(SystemExit) as failure:
        run_replay(replay, monkeypatch, tmp_path, "--checkpoint", checkpoint, *options[destination])
    assert failure.value.code == 2
    assert source.read_bytes() == original
    assert not checkpoint.exists()


def test_windows_exported_files(replay, workflow):
    for path in workflow.iterdir():
        content = path.read_text(encoding="utf-8")
        path.write_text(content, encoding="utf-8-sig")
    _, roles, _, configs, scores, costs, count = replay.load_benchmark(workflow.parents[1], "bird_mini_dev")
    assert count == 6 and len(roles) == 4 and scores.keys() == costs.keys() == configs.keys()


def test_bom_jsonl_keeps_unicode_answers(replay, tmp_path):
    from benchmark_runner import load_dataset

    path = tmp_path / "sample.jsonl"
    path.write_text('{"input":"一加二等于几？","expected":"三"}\n', encoding="utf-8-sig")
    assert load_dataset(path) == [("一加二等于几？", "三")]
