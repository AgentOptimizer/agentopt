"""Tests for objective_mode (weighted vs pareto) and Pareto helpers."""

import pytest

from agentopt.model_selection import BruteForceModelSelector
from agentopt.model_selection.base import DatapointResult, ModelResult, SelectionResults
from agentopt.model_selection.objectives import (
    pareto_mask_3d,
    score_to_error,
    validate_objective_config,
)


class _NoopAgent:
    def __init__(self, models):
        self.models = models

    def run(self, input_data):
        return "ok"


def _eval_fn(expected, actual):
    return 1.0 if str(expected).lower() in str(actual).lower() else 0.0


class TestValidateObjectiveConfig:
    def test_requires_mode(self):
        with pytest.raises(ValueError, match="objective_mode is required"):
            validate_objective_config(None, 0.0, 0.0)

    def test_weighted_requires_lambda(self):
        with pytest.raises(ValueError, match="lambda"):
            validate_objective_config("weighted", 0.0, 0.0)

    def test_pareto_rejects_lambdas(self):
        with pytest.raises(ValueError, match="does not accept"):
            validate_objective_config("pareto", 0.1, 0.0)

    def test_valid_modes(self):
        assert validate_objective_config("weighted", 0.2, 0.0) == "weighted"
        assert validate_objective_config("pareto", 0.0, 0.0) == "pareto"


class TestScoreToError:
    def test_perfect_score_zero_error(self):
        assert score_to_error(1.0) == 0.0

    def test_zero_score_full_error(self):
        assert score_to_error(0.0) == 1.0


class TestParetoMask3d:
    def test_nondominated_corner(self):
        errors = [0.1, 0.5, 0.5]
        lats = [1.0, 0.5, 2.0]
        costs = [0.01, 0.02, 0.01]
        mask = pareto_mask_3d(errors, lats, costs)
        assert mask[0] is True
        assert mask.count(True) >= 2


class TestModelResultError:
    def test_error_property(self):
        r = ModelResult(
            model_name="a",
            accuracy=0.8,
            latency_seconds=1.0,
            attribute="combination",
        )
        assert r.error == pytest.approx(0.2)


class TestSelectionResultsPareto:
    def _results(self) -> SelectionResults:
        a = ModelResult(
            model_name="fast",
            accuracy=0.9,
            latency_seconds=1.0,
            attribute="combination",
            datapoint_results=[
                DatapointResult(
                    datapoint_index=0, score=0.9, latency_seconds=1.0,
                ),
            ],
        )
        b = ModelResult(
            model_name="slow",
            accuracy=0.95,
            latency_seconds=5.0,
            attribute="combination",
            datapoint_results=[
                DatapointResult(
                    datapoint_index=0, score=0.95, latency_seconds=5.0,
                ),
            ],
        )
        sel = BruteForceModelSelector(
            agent=_NoopAgent,
            models={"n": ["a"]},
            eval_fn=_eval_fn,
            dataset=[("x", "y")],
            objective_mode="pareto",
        )
        sel._mark_pareto_optimal([a, b])
        return SelectionResults(results=[a, b], objective_mode="pareto")

    def test_get_pareto_front(self):
        res = self._results()
        front = res.get_pareto_front()
        assert len(front) >= 1
