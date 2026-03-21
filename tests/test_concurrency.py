"""Tests for the two-level concurrency control in parallel model selection."""

import pytest

from agentopt.model_selection.base import BaseModelSelector


class TestComputeConcurrency:
    """Unit tests for BaseModelSelector._compute_concurrency."""

    def test_basic_split(self):
        """max_concurrent=20, batch_size=5 → 4 combos x 5 dp each."""
        n_combo, dp_conc = BaseModelSelector._compute_concurrency(20, 5)
        assert n_combo == 4
        assert dp_conc == 5
        assert n_combo * dp_conc <= 20

    def test_batch_larger_than_max(self):
        """When batch_size exceeds max_concurrent, dp is capped and only 1 combo runs."""
        n_combo, dp_conc = BaseModelSelector._compute_concurrency(20, 100)
        assert n_combo == 1
        assert dp_conc == 20

    def test_batch_size_one(self):
        """batch_size=1 → all slots go to combo parallelism."""
        n_combo, dp_conc = BaseModelSelector._compute_concurrency(20, 1)
        assert n_combo == 20
        assert dp_conc == 1

    def test_batch_equals_max(self):
        """batch_size == max_concurrent → 1 combo, all slots for dp."""
        n_combo, dp_conc = BaseModelSelector._compute_concurrency(10, 10)
        assert n_combo == 1
        assert dp_conc == 10

    def test_indivisible_split(self):
        """When max_concurrent is not evenly divisible by batch_size."""
        n_combo, dp_conc = BaseModelSelector._compute_concurrency(20, 7)
        assert dp_conc == 7
        assert n_combo == 2  # 20 // 7 = 2
        assert n_combo * dp_conc <= 20

    def test_max_concurrent_one(self):
        """Minimum: max_concurrent=1 → 1 combo, 1 dp."""
        n_combo, dp_conc = BaseModelSelector._compute_concurrency(1, 50)
        assert n_combo == 1
        assert dp_conc == 1

    def test_batch_size_zero(self):
        """Edge case: batch_size=0 returns safe defaults."""
        n_combo, dp_conc = BaseModelSelector._compute_concurrency(20, 0)
        assert n_combo == 20
        assert dp_conc == 1

    def test_total_never_exceeds_max(self):
        """Product of n_combo * dp_concurrent never exceeds max_concurrent."""
        for max_c in [1, 5, 10, 20, 50, 100]:
            for batch in [1, 2, 3, 5, 10, 20, 50, 100, 200]:
                n_combo, dp_conc = BaseModelSelector._compute_concurrency(max_c, batch)
                assert n_combo * dp_conc <= max_c, (
                    f"max_concurrent={max_c}, batch_size={batch}: "
                    f"{n_combo} * {dp_conc} = {n_combo * dp_conc} > {max_c}"
                )
                assert n_combo >= 1
                assert dp_conc >= 1

    def test_prioritizes_datapoint_parallelism(self):
        """dp_concurrent should be as large as possible (up to batch_size)."""
        n_combo, dp_conc = BaseModelSelector._compute_concurrency(20, 5)
        assert dp_conc == 5  # full batch fits

        n_combo, dp_conc = BaseModelSelector._compute_concurrency(20, 3)
        assert dp_conc == 3  # full batch fits
        assert n_combo == 6  # 20 // 3 = 6
