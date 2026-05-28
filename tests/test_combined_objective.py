"""Tests for the combined objective (accuracy / latency / cost weighting).

Covers:
- The math helpers on BaseModelSelector (_minmax_norm, _combined_objective,
  adaptive normalizer state, batch + mean helpers, _absorb_observations).
- _find_best falling back to accuracy when combined_objective is unset, and
  preferring combined_objective when it is set on any result.
- _finalize_combined_objectives recomputing per-result objectives against the
  final normalizer state (idempotent under repeated calls).
- End-to-end brute_force run with lambda_latency > 0 picking a faster combo
  over a more accurate slow one.
"""

import time

import pytest

from agentopt.model_selection import BruteForceModelSelector
from agentopt.model_selection.base import (
    DatapointResult,
    ModelResult,
    SelectionResults,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _NoopAgent:
    """Minimal agent stub that satisfies the BaseModelSelector contract.

    The end-to-end test below overrides this with a per-model behavior agent.
    """

    def __init__(self, models):
        self.models = models

    def run(self, input_data):
        return "ok"


def _eval_fn(expected, actual):
    return 1.0 if str(expected).lower() in str(actual).lower() else 0.0


def _selector(
    lambda_cost: float = 0.0,
    lambda_latency: float = 0.1,
    *,
    objective_mode: str = "weighted",
):
    """Build a BruteForceModelSelector with stub agent/eval/dataset."""
    return BruteForceModelSelector(
        agent=_NoopAgent,
        models={"node": ["m"]},
        eval_fn=_eval_fn,
        dataset=[("x", "ok")],
        objective_mode=objective_mode,
        lambda_cost=lambda_cost,
        lambda_latency=lambda_latency,
    )


def _dp(idx: int, score: float, latency: float) -> DatapointResult:
    return DatapointResult(
        datapoint_index=idx,
        score=score,
        latency_seconds=latency,
        input_tokens={},
        output_tokens={},
    )


def _result(name: str, accuracy: float, latency: float, dps=None, *, combined=None):
    return ModelResult(
        model_name=name,
        accuracy=accuracy,
        latency_seconds=latency,
        input_tokens={},
        output_tokens={},
        attribute="combination",
        is_best=False,
        datapoint_results=dps or [],
        combined_objective=combined,
    )


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


class TestMinMaxNorm:
    def test_zero_when_range_degenerate(self):
        assert BruteForceModelSelector._minmax_norm(5.0, 1.0, 1.0) == 0.0

    def test_zero_when_lo_or_hi_infinite(self):
        assert BruteForceModelSelector._minmax_norm(0.5, float("inf"), 1.0) == 0.0
        assert BruteForceModelSelector._minmax_norm(0.5, 0.0, float("inf")) == 0.0

    def test_zero_when_value_none_or_nan(self):
        assert BruteForceModelSelector._minmax_norm(None, 0.0, 1.0) == 0.0
        assert BruteForceModelSelector._minmax_norm(float("nan"), 0.0, 1.0) == 0.0

    def test_linear_scaling(self):
        # 0.25 of the way from 1 to 5 → 0.25
        assert BruteForceModelSelector._minmax_norm(2.0, 1.0, 5.0) == pytest.approx(0.25)


class TestCombinedObjectiveParetoVsWeighted:
    """Pareto mode skips linear scalar; weighted mode uses lambdas."""

    def test_pareto_returns_score_from_combined_helper(self):
        sel = _selector(0.0, 0.0, objective_mode="pareto")
        assert sel._combined_objective(0.42, 100.0, 5.0) == 0.42

    def test_has_combined_objective_by_mode(self):
        assert _selector(0.0, 0.0, objective_mode="pareto")._has_combined_objective is False
        assert _selector(0.0, 0.1, objective_mode="weighted")._has_combined_objective is True
        assert _selector(0.1, 0.0, objective_mode="weighted")._has_combined_objective is True

    def test_pareto_compute_objectives_returns_score_copy(self):
        sel = _selector(0.0, 0.0, objective_mode="pareto")
        scores = [1.0, 0.0]
        out = sel._compute_objectives(scores, [10.0, 1.0], [0.5, 0.01])
        assert out == scores
        assert out is not scores

    def test_pareto_mean_objective_none(self):
        sel = _selector(0.0, 0.0, objective_mode="pareto")
        assert sel._mean_objective([1.0, 0.0], [1.0, 1.0], [0.0, 0.0]) is None


class TestAbsorbObservations:
    def test_updates_running_min_max_for_lat_and_cost(self):
        sel = _selector(lambda_latency=0.1, lambda_cost=0.1)
        sel._absorb_observations([1.0, 5.0, 3.0], [0.001, 0.005, 0.003])
        assert sel._latency_min == 1.0
        assert sel._latency_max == 5.0
        assert sel._cost_min == 0.001
        assert sel._cost_max == 0.005

    def test_noop_in_pareto_absorb_observations_only(self):
        sel = _selector(0.0, 0.0, objective_mode="pareto")
        sel._absorb_observations([1.0, 5.0], [0.001, 0.005])
        assert sel._latency_min == 1.0
        assert sel._latency_max == 5.0

    def test_skips_none_cost(self):
        sel = _selector(lambda_cost=0.1)
        sel._absorb_observations([1.0], [None])
        assert sel._cost_min == float("inf")
        assert sel._cost_max == float("-inf")


class TestCombinedObjectiveMath:
    def test_normalises_against_running_minmax(self):
        sel = _selector(lambda_cost=0.5, lambda_latency=0.5)
        # Two samples: cost ∈ {0.001, 0.005}, latency ∈ {1, 5}.
        sel._absorb_observations([1.0, 5.0], [0.001, 0.005])
        # Mid-point sample (cost halfway, latency halfway, score 1.0):
        #   norm_cost = 0.5, norm_lat = 0.5
        #   obj = 1.0 - 0.5 * 0.5 - 0.5 * 0.5 = 0.5
        assert sel._combined_objective(1.0, 3.0, 0.003) == pytest.approx(0.5)

    def test_min_cell_drops_penalties_to_zero(self):
        sel = _selector(lambda_cost=1.0, lambda_latency=1.0)
        sel._absorb_observations([1.0, 5.0], [0.001, 0.005])
        # At the min, both norm terms are 0 → obj == score.
        assert sel._combined_objective(0.8, 1.0, 0.001) == pytest.approx(0.8)

    def test_max_cell_subtracts_full_lambdas(self):
        sel = _selector(lambda_cost=0.3, lambda_latency=0.2)
        sel._absorb_observations([1.0, 5.0], [0.001, 0.005])
        # norm_cost = norm_lat = 1.0 → obj = score - 0.3 - 0.2
        assert sel._combined_objective(1.0, 5.0, 0.005) == pytest.approx(0.5)

    def test_mean_objective_matches_per_sample_mean(self):
        sel = _selector(lambda_latency=0.5)
        sel._absorb_observations([1.0, 5.0], [None, None])
        # Two samples at the latency extremes: norm_lat = 0 and 1.
        # obj0 = 1.0 - 0.5 * 0 = 1.0; obj1 = 0.5 - 0.5 * 1 = 0.0
        # mean = 0.5
        assert sel._mean_objective(
            [1.0, 0.5], [1.0, 5.0], [None, None]
        ) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _find_best
# ---------------------------------------------------------------------------


class TestFindBest:
    def test_falls_back_to_accuracy_when_none_set(self):
        a = _result("a", 0.8, 1.0)
        b = _result("b", 0.9, 2.0)
        assert BruteForceModelSelector._find_best([a, b]) == ("b", 0.9)

    def test_prefers_combined_objective_when_any_set(self):
        a = _result("a", 0.9, 1.0, combined=0.4)  # high accuracy, low obj (expensive)
        b = _result("b", 0.7, 1.0, combined=0.6)  # lower accuracy, higher obj
        name, value = BruteForceModelSelector._find_best([a, b])
        assert name == "b"
        assert value == pytest.approx(0.6)

    def test_latency_tiebreaks_within_equal_combined(self):
        a = _result("a", 0.5, 5.0, combined=0.5)
        b = _result("b", 0.5, 1.0, combined=0.5)
        assert BruteForceModelSelector._find_best([a, b])[0] == "b"

    def test_returns_none_for_empty_list(self):
        assert BruteForceModelSelector._find_best([]) is None


# ---------------------------------------------------------------------------
# _finalize_combined_objectives
# ---------------------------------------------------------------------------


class TestFinalizeCombinedObjectives:
    def test_noop_when_pareto_mode(self):
        sel = _selector(0.0, 0.0, objective_mode="pareto")
        r = _result("a", 1.0, 1.0, [_dp(0, 1.0, 1.0)])
        sel._finalize_combined_objectives([r])
        assert r.combined_objective is None

    def test_sets_combined_against_final_normalizer(self):
        sel = _selector(lambda_latency=0.5)
        # Two combos with latencies bracketing the range.
        fast = _result("fast", 1.0, 1.0, [_dp(0, 1.0, 1.0)])
        slow = _result("slow", 1.0, 5.0, [_dp(0, 1.0, 5.0)])
        sel._finalize_combined_objectives([fast, slow])
        # fast → norm_lat = 0 → obj = 1.0
        # slow → norm_lat = 1 → obj = 0.5
        assert fast.combined_objective == pytest.approx(1.0)
        assert slow.combined_objective == pytest.approx(0.5)

    def test_idempotent(self):
        sel = _selector(lambda_latency=0.5)
        r1 = _result("a", 1.0, 1.0, [_dp(0, 1.0, 1.0)])
        r2 = _result("b", 0.8, 4.0, [_dp(0, 0.8, 4.0)])
        sel._finalize_combined_objectives([r1, r2])
        first = (r1.combined_objective, r2.combined_objective)
        sel._finalize_combined_objectives([r1, r2])
        second = (r1.combined_objective, r2.combined_objective)
        assert first == second

    def test_handles_empty_datapoint_results(self):
        sel = _selector(lambda_latency=0.5)
        r = _result("a", 0.0, 0.0, [])
        sel._finalize_combined_objectives([r])
        assert r.combined_objective is None


# ---------------------------------------------------------------------------
# End-to-end: brute_force prefers a faster combo when lambda_latency > 0
# ---------------------------------------------------------------------------


class _LatencyTunedAgent:
    """Returns the correct answer for both models, but model 'slow' sleeps."""

    SLEEP_MAP = {"fast": 0.0, "slow": 0.05}

    def __init__(self, models):
        self.model_name = models["node"]

    def run(self, input_data):
        time.sleep(self.SLEEP_MAP[self.model_name])
        return "correct"


class TestBruteForceLatencyWeighting:
    def test_pareto_mode_marks_frontier_not_single_best(self):
        sel = BruteForceModelSelector(
            agent=_LatencyTunedAgent,
            models={"node": ["fast", "slow"]},
            eval_fn=_eval_fn,
            dataset=[("?", "correct"), ("?", "correct")],
            objective_mode="pareto",
        )
        results = sel.select_best(parallel=False)
        assert results.objective_mode == "pareto"
        assert results.get_best() is None
        assert len(results.get_pareto_front()) >= 1
        assert all(r.combined_objective is None for r in results.results)

    def test_lambda_latency_picks_fast_when_accuracy_tied(self):
        sel = BruteForceModelSelector(
            agent=_LatencyTunedAgent,
            models={"node": ["fast", "slow"]},
            eval_fn=_eval_fn,
            dataset=[("?", "correct"), ("?", "correct")],
            objective_mode="weighted",
            lambda_latency=0.5,
        )
        results = sel.select_best(parallel=False)
        best = results.get_best()
        assert best is not None
        assert best.model_name == "node=fast"
        # combined_objective now populated on results.
        assert all(r.combined_objective is not None for r in results.results)
        # fast combo's objective is strictly higher than slow's.
        fast = next(r for r in results.results if r.model_name == "node=fast")
        slow = next(r for r in results.results if r.model_name == "node=slow")
        assert fast.combined_objective > slow.combined_objective
