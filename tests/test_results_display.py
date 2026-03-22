"""Tests for per-sample pricing and layered result display."""

import pytest

from agentopt.model_selection.base import (
    DatapointResult,
    ModelResult,
    SelectionResults,
)

# Custom prices: $1.00 per 1M input tokens, $2.00 per 1M output tokens.
CUSTOM_PRICES = {"test-model": (1.0, 2.0)}


def _make_result(
    name: str,
    num_samples: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
    accuracy: float = 0.9,
    latency: float = 1.0,
    is_best: bool = False,
    datapoint_results: list | None = None,
) -> ModelResult:
    r = ModelResult(
        model_name=name,
        accuracy=accuracy,
        latency_seconds=latency,
        input_tokens={"test-model": input_tokens},
        output_tokens={"test-model": output_tokens},
        attribute="combination",
        is_best=is_best,
        num_samples=num_samples,
        datapoint_results=datapoint_results or [],
    )
    r._custom_prices = CUSTOM_PRICES
    return r


def _make_dp(index: int, score: float, latency: float, inp: int, out: int) -> DatapointResult:
    return DatapointResult(
        datapoint_index=index,
        score=score,
        latency_seconds=latency,
        input_tokens={"test-model": inp},
        output_tokens={"test-model": out},
    )


# ---------------------------------------------------------------------------
# ModelResult.price normalisation
# ---------------------------------------------------------------------------


class TestModelResultPrice:
    def test_price_normalized_by_num_samples(self):
        # 1M input + 1M output → total = $1 + $2 = $3; per-sample = $3/5 = $0.60
        r = _make_result("a", num_samples=5, input_tokens=1_000_000, output_tokens=1_000_000)
        assert r.price == pytest.approx(0.6)

    def test_price_defaults_to_total_when_num_samples_1(self):
        r = _make_result("a", num_samples=1, input_tokens=1_000_000, output_tokens=1_000_000)
        assert r.price == pytest.approx(3.0)

    def test_price_none_for_unknown_model(self):
        r = ModelResult(
            model_name="a",
            accuracy=0.9,
            latency_seconds=1.0,
            input_tokens={"unknown-model-xyz": 100},
            output_tokens={"unknown-model-xyz": 100},
            attribute="combination",
            num_samples=1,
        )
        assert r.price is None

    def test_price_zero_tokens(self):
        r = _make_result("a", num_samples=1, input_tokens=0, output_tokens=0)
        assert r.price == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SelectionResults display
# ---------------------------------------------------------------------------


class TestSelectionResultsDisplay:
    def test_flat_table_when_same_num_samples(self):
        results = SelectionResults(
            results=[
                _make_result("combo_A", num_samples=5, accuracy=0.9, is_best=True),
                _make_result("combo_B", num_samples=5, accuracy=0.8),
            ]
        )
        output = str(results)
        assert "Model Selection Results" in output
        assert "Round" not in output
        assert "Eliminated" not in output

    def test_layered_table_when_different_num_samples(self):
        dp1 = _make_dp(0, 1.0, 1.0, 100, 50)
        dp2 = _make_dp(1, 0.8, 1.2, 120, 60)
        dp3 = _make_dp(2, 0.6, 0.9, 110, 55)

        r1 = _make_result(
            "combo_A", num_samples=3, accuracy=0.8, is_best=True,
            datapoint_results=[dp1, dp2, dp3],
        )
        r2 = _make_result(
            "combo_B", num_samples=2, accuracy=0.9,
            datapoint_results=[dp1, dp2],
        )
        r3 = _make_result(
            "combo_C", num_samples=1, accuracy=1.0,
            datapoint_results=[dp1],
        )
        results = SelectionResults(results=[r1, r2, r3])
        output = str(results)
        assert "Round 1" in output
        assert "Round 2" in output
        assert "Final" in output
        assert "Eliminated" in output

    def test_layered_recomputes_metrics_from_datapoints(self):
        dp1 = _make_dp(0, 1.0, 2.0, 1_000_000, 500_000)
        dp2 = _make_dp(1, 0.5, 3.0, 1_000_000, 500_000)
        dp3 = _make_dp(2, 0.8, 1.0, 1_000_000, 500_000)

        r_survivor = _make_result(
            "survivor", num_samples=3, accuracy=0.7667, is_best=True,
            datapoint_results=[dp1, dp2, dp3],
        )
        r_eliminated = _make_result(
            "eliminated", num_samples=2, accuracy=0.75,
            datapoint_results=[dp1, dp2],
        )
        results = SelectionResults(results=[r_survivor, r_eliminated])
        output = str(results)

        # Round 1 (2 datapoints): both combos, accuracy = (1.0 + 0.5) / 2 = 75.00%
        assert "75.00%" in output
        # Final (3 datapoints): survivor, accuracy = (1.0+0.5+0.8)/3 ≈ 76.67%
        assert "76.67%" in output

    def test_best_marker_only_in_final_layer(self):
        dp1 = _make_dp(0, 1.0, 1.0, 100, 50)
        dp2 = _make_dp(1, 0.8, 1.2, 120, 60)

        r1 = _make_result(
            "combo_A", num_samples=2, accuracy=0.9, is_best=True,
            datapoint_results=[dp1, dp2],
        )
        r2 = _make_result(
            "combo_B", num_samples=1, accuracy=1.0,
            datapoint_results=[dp1],
        )
        results = SelectionResults(results=[r1, r2])
        output = str(results)

        # Split into round sections.
        round1_section = output.split("Round 1")[1].split("Final")[0]
        final_section = output.split("Final")[1]

        assert ">>>" not in round1_section
        assert ">>>" in final_section
