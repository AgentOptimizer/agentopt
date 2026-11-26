import csv
import json

import pytest

from agentopt import ModelSelector
from agentopt.model_selection.base import SelectionResults
from tests.test_gittins import RecordingTracker, make_selector


def test_summary_csv_with_datapoints_and_custom_price(tmp_path):
    result = make_selector(mode="single", count=2)[0].select_best()
    output = tmp_path / "summary.csv"
    result.to_csv(output)
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result.results)
    for row, record in zip(rows, result.results):
        assert row["model_name"] == record.model_name
        assert json.loads(row["input_tokens"]) == record.input_tokens
        assert float(row["price"]) == pytest.approx(record.price)
        assert row["is_best"] == str(record.is_best)


def test_winner_preserves_literal_plus_in_model_alias(tmp_path):
    alias = "local + review"
    selector, _ = make_selector(mode="single", count=2, outcomes={alias: .9}, tokens={alias: 10})
    result = selector.select_best()
    assert result.get_best_combo() == {"solver": alias}
    result.export_config(tmp_path / "winner.yaml")
    assert "#   solver: local + review" in (tmp_path / "winner.yaml").read_text(encoding="utf-8")


@pytest.mark.parametrize("method", ["gittins", "random", "ucb"])
def test_role_mapping_survives_json_round_trip(method):
    tracker = RecordingTracker()
    alias = "local + variant=large"
    models = {"planner": [alias], "solver": ["compact"]}

    class Agent:
        def __init__(self, combo):
            self.combo = combo

        def run(self, question):
            for model in self.combo.values():
                tracker.record(model, 10)
            return 1

    selector = ModelSelector(method=method, agent=Agent, models=models, eval_fn=lambda _, score: score,
                             dataset=[("2 + 2", None)], tracker=tracker,
                             model_prices={m: {"input_price": 1, "output_price": 0} for m in (alias, "compact")})
    result = selector.select_best()
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored.get_best_combo() == {"planner": alias, "solver": "compact"}


def test_legacy_result_with_only_display_label():
    result = SelectionResults.model_validate({"results": [{
        "model_name": "draft=local + review + check=mini", "accuracy": 1.0,
        "latency_seconds": 0.1, "attribute": "combination", "is_best": True}]})
    assert result.get_best_combo() == {"draft": "local + review", "check": "mini"}
