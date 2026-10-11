import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

from agentopt.model_selection.axis_gittins_dp import (
    AxisGittinsBoundaryCache,
    build_axis_gittins_boundary_table,
    scalar_posterior_variance_schedule,
)


def build(cost=0.005, horizon=3, grid_size=129, initial_var=0.04, obs_noise_var=0.0625):
    return build_axis_gittins_boundary_table(
        effective_pull_cost=cost,
        initial_var=initial_var,
        obs_noise_var=obs_noise_var,
        horizon=horizon,
        grid_size=grid_size,
    )


def expected_positive_part(mean, std):
    return mean * norm.cdf(mean / std) + std * norm.pdf(mean / std)


class AxisBoundaryTests(unittest.TestCase):
    def test_scalar_schedule_matches_gaussian_precision_update(self):
        expected = np.array([1.0 / (25.0 + n / 0.0625) for n in range(5)])
        np.testing.assert_allclose(scalar_posterior_variance_schedule(0.04, 0.0625, 4), expected)

    def test_one_step_agrees_with_analytic_gaussian_stop_loss_even_on_coarse_grid(self):
        variances = scalar_posterior_variance_schedule(0.04, 0.0625, 1)
        std = math.sqrt(variances[0] - variances[1])
        for cost in (0.1, 0.005, 1e-6, 1e-12):
            with self.subTest(cost=cost):
                expected = brentq(lambda x: expected_positive_part(x, std) - cost, -5.0, 5.0, xtol=1e-14)
                table = build(cost=cost, horizon=1, grid_size=17)
                self.assertAlmostEqual(table.boundary(0), expected, places=11)
                self.assertEqual(table.boundary(1), 0.0)

    def test_two_step_agrees_with_independent_adaptive_quadrature(self):
        cost = 0.01
        variances = scalar_posterior_variance_schedule(0.04, 0.0625, 2)
        std0, std1 = np.sqrt(variances[:-1] - variances[1:])
        last_root = brentq(lambda x: expected_positive_part(x, std1) - cost, -2.0, 2.0)

        def q0(x):
            def integrand(z):
                return (expected_positive_part(x + std0 * z, std1) - cost) * norm.pdf(z)

            return quad(integrand, (last_root - x) / std0, np.inf, epsabs=1e-11)[0] - cost

        expected = brentq(q0, -1.0, 1.0)
        table = build(cost=cost, horizon=2)
        self.assertAlmostEqual(table.boundary(1), last_root, places=11)
        self.assertLess(abs(table.boundary(0) - expected), 5e-5)

    def test_lower_cost_raises_the_index_at_every_unfinished_stage(self):
        costs = (0.02, 0.005, 1e-8)
        tables = [build(cost=cost, horizon=47) for cost in costs]
        for costly, cheaper in zip(tables, tables[1:]):
            self.assertTrue(np.all(costly.boundaries[:-1] > cheaper.boundaries[:-1]))
            self.assertLess(costly.index(0, 0.7), cheaper.index(0, 0.7))
            self.assertEqual(costly.index(47, 0.7), cheaper.index(47, 0.7))

    def test_tiny_cost_long_horizon_converges_without_a_clipped_boundary(self):
        tables = [build(cost=1e-10, horizon=47, grid_size=size, initial_var=1.0 / 41.0) for size in (65, 129, 513)]
        coarse, medium, fine = [table.boundary(0) for table in tables]
        self.assertTrue(all(np.all(np.isfinite(table.boundaries)) for table in tables))
        self.assertLess(abs(medium - fine), abs(coarse - fine))
        self.assertLess(abs(medium - fine), 0.003)
        self.assertLess(fine, -0.8)

    def test_without_information_required_completion_pays_all_remaining_costs(self):
        table = build(cost=0.03, horizon=6, initial_var=1e-20)
        np.testing.assert_allclose(table.boundaries, np.arange(6, -1, -1) * 0.03, atol=1e-12)

    def test_terminal_utility_has_no_posterior_variance_penalty(self):
        for initial, noise in ((0.001, 0.005), (4.0, 8.0)):
            table = build(initial_var=initial, obs_noise_var=noise)
            self.assertEqual(table.boundary(3), 0.0)
            self.assertAlmostEqual(table.index(3, mean=0.9, reference=0.2), 0.7)

        # Equal posterior-mean increment variance, unequal residual terminal
        # variance: expected linear terminal utility produces the same index.
        first = build(horizon=1, initial_var=0.04, obs_noise_var=0.04)
        second = build(horizon=1, initial_var=0.03, obs_noise_var=0.015)
        np.testing.assert_allclose(first.boundaries, second.boundaries, atol=1e-12)

    def test_zero_horizon_and_reference_translation(self):
        terminal = build(horizon=0)
        np.testing.assert_array_equal(terminal.boundaries, [0.0])
        self.assertEqual(terminal.index(0, -0.4), -0.4)
        table = build()
        self.assertAlmostEqual(table.index(1, 0.9, 0.2), table.index(1, 0.7))

    def test_invalid_parameters_are_rejected_and_boundaries_are_immutable(self):
        for cost in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                build(cost=cost)
        for kwargs in ({"initial_var": 0.0}, {"obs_noise_var": -1.0}, {"initial_var": [0.1, 0.2]}, {"horizon": -1}, {"horizon": 1.5}, {"grid_size": 16}, {"grid_size": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build(**kwargs)
        table = build()
        self.assertFalse(table.boundaries.flags.writeable)
        for stage in (-1, 4, 0.5, float("nan")):
            with self.subTest(stage=stage), self.assertRaises(IndexError):
                table.boundary(stage)
        with self.assertRaises(ValueError):
            table.index(0, float("nan"))

    def test_cache_key_includes_all_parameters_and_tracks_memory_hits(self):
        cache = AxisGittinsBoundaryCache()
        kwargs = dict(effective_pull_cost=0.005, initial_var=0.04, obs_noise_var=0.0625, horizon=3, grid_size=65)
        first = cache.get(**kwargs)
        self.assertIs(first, cache.get(**kwargs))
        for change in ({"effective_pull_cost": 0.006}, {"initial_var": 0.03}, {"obs_noise_var": 0.05}, {"horizon": 4}, {"grid_size": 129}):
            self.assertIsNot(cache.get(**(kwargs | change)), first)
        self.assertEqual(len(cache), 6)
        self.assertEqual(cache.stats_snapshot(), {"builds": 6, "memory_hits": 1})


if __name__ == "__main__":
    unittest.main()
