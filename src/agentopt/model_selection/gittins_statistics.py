"""Gaussian calibration and finite-dataset recommendations for CC-Gittins.

The raw-cost warm-start calibration, finite-test prediction moments, and
strict Pareto sweep follow the AgentOpt Research two-axis protocol. Only
observed outcomes enter these calculations; no complete matrix is required.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianPosterior:
    mean: np.ndarray
    variance: np.ndarray

    def update(self, observation: np.ndarray, question_noise: np.ndarray, count: int) -> None:
        """Apply one batch likelihood, using its actual number of questions."""
        batch_noise = question_noise / count
        variance = 1.0 / (1.0 / self.variance + 1.0 / batch_noise)
        self.mean = variance * (self.mean / self.variance + observation / batch_noise)
        self.variance = variance


@dataclass(frozen=True)
class WarmCalibration:
    prior_mean: np.ndarray
    prior_variance: np.ndarray
    question_noise: np.ndarray
    cost_reference_usd: float
    score_bounds: tuple[float, float]

    def observations(self, scores, costs) -> np.ndarray:
        low, high = self.score_bounds
        return np.column_stack(((np.asarray(scores) - low) / (high - low), costs))

    def posterior(self, scores, costs) -> GaussianPosterior:
        posterior = GaussianPosterior(self.prior_mean.copy(), self.prior_variance.copy())
        posterior.update(self.observations(scores, costs).mean(axis=0), self.question_noise, len(scores))
        return posterior


def fit_warm_start(scores, costs, *, score_bounds=(0.0, 1.0), prior_variance=0.04,
                   score_noise_variance=0.25, cost_reference_usd=None) -> WarmCalibration:
    """Freeze an equal-arm plug-in prior; apply each warm likelihood separately.

Accuracy is normalized, cost remains in USD. Cost prior variance is the
sample variance of warm arm means; per-question cost noise is the average
within-arm sample variance. Both have the research floor ``1e-12 * R**2``.
All-zero observed costs use a scale of one USD, without adding actual cost.
"""
    scores, costs = np.asarray(scores, dtype=float), np.asarray(costs, dtype=float)
    if scores.ndim != 2 or scores.shape != costs.shape or not all(scores.shape):
        raise ValueError("warm scores and costs must be matching nonempty matrices")
    low, high = score_bounds
    if (not np.all(np.isfinite(scores)) or not np.all(np.isfinite(costs))
            or np.any(costs < 0) or np.any(scores < low) or np.any(scores > high)):
        raise ValueError("warm outcomes must be finite, scores within bounds, and costs nonnegative")
    raw_means = costs.mean(axis=1)
    reference = float(cost_reference_usd if cost_reference_usd is not None else max(raw_means.max(), 0.0))
    if cost_reference_usd is None and reference == 0:
        reference = 1.0
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("cost_reference_usd must be finite and positive")
    floor = 1e-12 * reference ** 2
    cost_prior = float(np.var(raw_means, ddof=1)) if len(raw_means) > 1 else 0.0
    cost_noise = float(np.var(costs, axis=1, ddof=1).mean()) if costs.shape[1] > 1 else 0.0
    return WarmCalibration(
        np.column_stack((((scores - low) / (high - low)).mean(axis=1), raw_means)).mean(axis=0),
        np.array([prior_variance, max(cost_prior, floor)]),
        np.array([score_noise_variance, max(cost_noise, floor)]), reference, score_bounds,
    )


def finite_dataset_moments(posterior, observed_sum, observed_count, total_count, question_noise):
    """Predict the realized mean of the supplied dataset, not generalization.

Observed outcomes are fixed; unseen outcomes have a common Gaussian latent
mean and independent per-question noise. At completion the variance is zero.
"""
    remaining = total_count - observed_count
    mean = (observed_sum + remaining * posterior.mean) / total_count
    variance = (remaining ** 2 * posterior.variance + remaining * question_noise) / total_count ** 2
    return mean, variance


def raw_pareto_front_indices(points) -> tuple[int, ...]:
    """Maximize score, minimize USD; retain equal pairs in original arm order."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("points must be a finite (K, 2) array")
    if not len(points):
        return ()
    order = np.lexsort((-points[:, 0], points[:, 1]))
    accuracy, cost = points[order, 0], points[order, 1]
    starts = np.r_[True, cost[1:] != cost[:-1]]
    groups = np.cumsum(starts) - 1
    group_best = accuracy[starts]
    previous_best = np.r_[-np.inf, np.maximum.accumulate(group_best)[:-1]]
    keep = (accuracy == group_best[groups]) & (group_best[groups] > previous_best[groups])
    return tuple(sorted(int(i) for i in order[keep]))
