# Copyright 2026 Qian Xie
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Matrix bandit exploration using Gittins indices (Gaussian conjugate / normal–normal)."""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Sequence

import jax
import jax.numpy as jnp
import torch

from .gittins_shrinking_posterior import (
    compute_gittins_shrinking_posterior_walk_batch_mean,
    compute_gittins_shrinking_posterior_walk_per_observation,
    transition_stds_shrinking_gaussian_posterior,
)
from .gittins_lookup import compute_roots_lookup_table


def _cost_vector_per_arm(
    cost_per_transition: float | Sequence[float] | torch.Tensor,
    n_arms: int,
) -> torch.Tensor:
    """Return ``(n_arms,)`` float64 CPU tensor; broadcast scalars, validate vector length."""
    if isinstance(cost_per_transition, torch.Tensor):
        v = cost_per_transition.detach().reshape(-1).cpu().to(torch.float64)
    elif isinstance(cost_per_transition, numbers.Real):
        return torch.full((n_arms,), float(cost_per_transition), dtype=torch.float64)
    else:
        v = torch.tensor(list(cost_per_transition), dtype=torch.float64)
    if v.numel() == 1:
        return v.expand(n_arms).contiguous()
    if v.numel() != n_arms:
        raise ValueError(
            "cost_per_transition must be a scalar, a length-1 tensor, or length n_arms "
            f"(got {v.numel()} values, n_arms={n_arms})"
        )
    return v


def _normal_normal_posterior(
    mu0: float,
    v0: float,
    tau_sq: float,
    obs_sum: float,
    t: int,
) -> tuple[float, float]:
    """Conjugate N(μ0,v0) prior on θ and Y|θ ~ N(θ, τ²); return posterior mean and variance after t sums."""
    if t <= 0:
        return mu0, v0
    prec = 1.0 / v0 + t / tau_sq
    v_t = 1.0 / prec
    mu_t = v_t * (mu0 / v0 + obs_sum / tau_sq)
    return float(mu_t), float(v_t)


def _gittins_posterior_means(
    observed_matrix: torch.Tensor,
    *,
    prior_mean: float,
    prior_variance: float,
    tau_sq_cell: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (mus_posterior, counts, completely_sensed_mask) for the current mask."""
    counts = (~observed_matrix.isnan()).sum(1)
    completely_sensed_mask = counts == observed_matrix.shape[1]
    obs_sum_per_arm = torch.nan_to_num(observed_matrix, nan=0.0).sum(dim=1).to(torch.float64)
    t = counts.to(torch.float64)
    v0 = float(prior_variance)
    prec = (1.0 / v0) + (t / float(tau_sq_cell))
    v_t = 1.0 / prec
    mus_posterior = (
        v_t * (float(prior_mean) / v0 + obs_sum_per_arm / float(tau_sq_cell))
    ).to(torch.float32)
    mus_posterior[counts == 0] = float(prior_mean)
    return mus_posterior, counts, completely_sensed_mask


def _refresh_gittins_scores(
    observed_matrix: torch.Tensor,
    *,
    scores: torch.Tensor,
    mus_posterior: torch.Tensor,
    completely_sensed_mask: torch.Tensor,
    counts: torch.Tensor,
    arm_indices: Iterable[int],
    prior_mean: float,
    prior_variance: float,
    tau_sq_cell: float,
    obs_noise_variance: float,
    arm_costs: torch.Tensor,
    n_gittins_grid_points: int,
    batch_size: int,
    use_batch_mean_gittins_dp: bool,
    force_per_observation_dp: bool,
    roots_lookup_table: torch.Tensor,
    n_examples: int,
) -> None:
    """Update ``scores[k]`` in place for listed arms; complete arms use posterior mean."""
    n_pts = int(n_gittins_grid_points)
    for k in arm_indices:
        if completely_sensed_mask[k]:
            continue
        t = int(counts[k].item())
        row = observed_matrix[k]
        valid = ~torch.isnan(row)
        obs_sum = float(row[valid].sum().item())
        mu_kt, v_kt = _normal_normal_posterior(
            prior_mean, prior_variance, tau_sq_cell, obs_sum, t
        )
        c_k = float(arm_costs[k].item())
        if use_batch_mean_gittins_dp:
            remaining = n_examples - t
            bsz_plan = int(batch_size)
            n_batch = (remaining + bsz_plan - 1) // bsz_plan
            transition_costs_bm = jnp.full((n_batch,), c_k, dtype=jnp.float32)
            g = compute_gittins_shrinking_posterior_walk_batch_mean(
                jnp.float32(mu_kt),
                jnp.float32(v_kt),
                jnp.float32(obs_noise_variance),
                int(batch_size),
                remaining,
                transition_costs_bm,
                jnp.uint32(n_pts),
            )
        elif force_per_observation_dp:
            transition_costs_per_cell = jnp.full((n_examples,), c_k, dtype=jnp.float32)
            g = compute_gittins_shrinking_posterior_walk_per_observation(
                jnp.uint32(t),
                jnp.float32(mu_kt),
                jnp.float32(prior_variance),
                jnp.float32(tau_sq_cell),
                transition_costs_per_cell,
                jnp.uint32(n_pts),
            )
        else:
            k_roots = min(k, roots_lookup_table.shape[0] - 1)
            root_t = float(roots_lookup_table[k_roots, t].item())
            g = jnp.float32(mu_kt) - jnp.float32(root_t)
        scores[k] = float(jax.device_get(g))

    for k in range(observed_matrix.shape[0]):
        if completely_sensed_mask[k]:
            scores[k] = float(mus_posterior[k].item())


def evaluate_gittins_stopping_rules(
    scores: torch.Tensor,
    mus_posterior: torch.Tensor,
    completely_sensed_mask: torch.Tensor,
    *,
    sim_cum_eval: int,
    natural_stop_cum_eval_holder: list[int | None] | None = None,
    recommendation_aware_stop_cum_eval_holder: list[int | None] | None = None,
) -> None:
    """Record nominal stop times using **post-pull** Γ and μ (``sim_cum_eval`` after the batch)."""
    if (
        recommendation_aware_stop_cum_eval_holder is not None
        and len(recommendation_aware_stop_cum_eval_holder) == 1
        and recommendation_aware_stop_cum_eval_holder[0] is None
    ):
        incomplete = ~completely_sensed_mask
        if bool(incomplete.any()):
            max_gittins = float(scores[incomplete].max().item())
            max_mu = float(mus_posterior.max().item())
            if max_gittins < max_mu:
                recommendation_aware_stop_cum_eval_holder[0] = int(sim_cum_eval)

    best_method_index = int(torch.argmax(scores).item())
    if (
        bool(completely_sensed_mask[best_method_index])
        and natural_stop_cum_eval_holder is not None
        and len(natural_stop_cum_eval_holder) == 1
        and natural_stop_cum_eval_holder[0] is None
    ):
        natural_stop_cum_eval_holder[0] = int(sim_cum_eval)


def gittins_post_pull_update(
    observed_matrix: torch.Tensor,
    *,
    cached_scores: torch.Tensor,
    recompute_arms: Iterable[int],
    prior_mean: float = 0.5,
    prior_variance: float = 0.04,
    obs_noise_variance: float = 0.01,
    cost_per_transition: float | Sequence[float] | torch.Tensor = 1.0,
    cost_scaling_factor: float = 1e-4,
    n_gittins_grid_points: int = 2**10 + 1,
    batch_size: int = 32,
    use_batch_mean_gittins_dp: bool = False,
    force_per_observation_dp: bool = False,
    roots_lookup_table: torch.Tensor | None = None,
    batch_observation_model: bool = False,
    sim_cum_eval: int,
    natural_stop_cum_eval_holder: list[int | None] | None = None,
    recommendation_aware_stop_cum_eval_holder: list[int | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """After a batch is revealed: refresh μ, update Γ, and evaluate stopping rules."""
    observed_matrix = observed_matrix.detach()
    if observed_matrix.device.type != "cpu":
        observed_matrix = observed_matrix.cpu()

    m_methods, n_examples = observed_matrix.shape
    tau_sq = float(obs_noise_variance)
    tau_sq_cell = tau_sq * float(batch_size) if batch_observation_model else tau_sq
    mus_posterior, counts, completely_sensed_mask = _gittins_posterior_means(
        observed_matrix,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        tau_sq_cell=tau_sq_cell,
    )

    if cached_scores.shape != (m_methods,) or cached_scores.dtype != torch.float32:
        raise ValueError(
            "cached_scores must have shape (n_arms,) and dtype float32; "
            f"got shape {tuple(cached_scores.shape)}, dtype {cached_scores.dtype}"
        )
    scores = cached_scores
    arm_costs = _cost_vector_per_arm(cost_per_transition, m_methods) * float(cost_scaling_factor)
    n_pts = int(n_gittins_grid_points)

    if roots_lookup_table is None:
        if use_batch_mean_gittins_dp or force_per_observation_dp:
            raise ValueError("roots_lookup_table is required for post-pull Gittins refresh")
        transition_stds = transition_stds_shrinking_gaussian_posterior(
            jnp.float32(prior_variance), jnp.float32(tau_sq_cell), n_examples
        )
        if arm_costs.numel() == 1 or torch.allclose(arm_costs, arm_costs[0].expand_as(arm_costs)):
            costs_per_arm = jnp.float32(float(arm_costs[0].item()))
            roots_all = compute_roots_lookup_table(
                transition_stds=transition_stds,
                costs_per_arm=costs_per_arm,
                n_points=n_pts,
            )
        else:
            arm_costs_jnp = jnp.asarray(arm_costs.numpy(), dtype=jnp.float32)
            roots_all = compute_roots_lookup_table(
                transition_stds=transition_stds,
                costs_per_arm=arm_costs_jnp,
                n_points=n_pts,
            )
        roots_lookup_table = torch.from_numpy(jax.device_get(roots_all)).to(torch.float32)
    if roots_lookup_table.ndim == 1:
        roots_lookup_table = roots_lookup_table.unsqueeze(0)

    _refresh_gittins_scores(
        observed_matrix,
        scores=scores,
        mus_posterior=mus_posterior,
        completely_sensed_mask=completely_sensed_mask,
        counts=counts,
        arm_indices=recompute_arms,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        tau_sq_cell=tau_sq_cell,
        obs_noise_variance=tau_sq,
        arm_costs=arm_costs,
        n_gittins_grid_points=n_gittins_grid_points,
        batch_size=batch_size,
        use_batch_mean_gittins_dp=use_batch_mean_gittins_dp,
        force_per_observation_dp=force_per_observation_dp,
        roots_lookup_table=roots_lookup_table,
        n_examples=n_examples,
    )
    evaluate_gittins_stopping_rules(
        scores,
        mus_posterior,
        completely_sensed_mask,
        sim_cum_eval=int(sim_cum_eval),
        natural_stop_cum_eval_holder=natural_stop_cum_eval_holder,
        recommendation_aware_stop_cum_eval_holder=recommendation_aware_stop_cum_eval_holder,
    )
    return mus_posterior, scores


def gittins_index_exploration(
    observed_matrix: torch.Tensor,
    *,
    prior_mean: float = 0.5,
    prior_variance: float = 0.04,
    obs_noise_variance: float = 0.01,
    cost_per_transition: float | Sequence[float] | torch.Tensor = 1.0,
    cost_scaling_factor: float = 1e-4,
    n_gittins_grid_points: int = 2**10 + 1,
    batch_size: int = 32,
    return_mus: bool = False,
    cached_scores: torch.Tensor | None = None,
    recompute_arms: Iterable[int] | None = None,
    use_batch_mean_gittins_dp: bool = False,
    allow_early_stop: bool = True,
    sim_cum_eval: int | None = None,
    natural_stop_cum_eval_holder: list[int | None] | None = None,
    recommendation_aware_stop_cum_eval_holder: list[int | None] | None = None,
    roots_lookup_table: torch.Tensor | None = None,
    force_per_observation_dp: bool = False,
    batch_observation_model: bool = False,
):
    """
    One step of Gittins-index exploration on a masked observation matrix.

    Rows are arms (methods); columns are i.i.d. example indices. Each revealed entry is modeled as
    Y | θ_k ~ N(θ_k, τ²) with τ² = ``obs_noise_variance``. A common worst-case bound for
    [0, 1]-valued outcomes uses τ² approximately 1/(4B), where **B is** ``batch_size`` (examples per
    arm per step). To
    match that approximation, set ``obs_noise_variance`` to ``1.0 / (4 * batch_size)`` (or your
    chosen B). Default prior on each θ_k is N(0.5, 0.04); override with ``prior_mean`` and
    ``prior_variance``.

    **Stopping / continuation:** Nominal stop times are **not** recorded here. After each batch is
    revealed, call ``gittins_post_pull_update`` (post-pull Γ and μ, then stopping rules). This
    function only **chooses the next pull** from pre-pull Γ. Fully observed arms use posterior mean
    as their score. If the arm with the largest pre-pull score is already fully observed and
    ``allow_early_stop`` is True (default), return ``None`` so the simulator can end the run. If
    ``allow_early_stop`` is False, fall back to the best **incomplete** arm so exploration can
    continue (e.g. to a fixed eval budget).

    **Batch semantics (not a mixed pair minibatch):** compute the Gittins index for every arm,
    choose the single arm k* with the largest index, then evaluate **that method** on
    ``batch_size`` **distinct unevaluated examples** (columns), chosen uniformly at random. So
    ``batch[0]`` is constant (k* repeated); ``batch[1]`` lists example indices. This is
    ``batch_size`` new **example** evaluations for **one** method—not ``batch_size`` arbitrary
    (method, example) pairs that could split across methods.

    The tensor layout matches ``upper_confidence_bound_exploration`` (stacked row/col indices) so
    the same simulator can call either policy.

    Args:
        observed_matrix: (n_arms, n_examples) with NaN for unevaluated cells.
        prior_mean: μ_0 in the Gaussian prior on each θ_k (default 0.5).
        prior_variance: v_0 in the Gaussian prior on each θ_k (default 0.01).
        obs_noise_variance: τ² in Y | θ_k ~ N(θ_k, τ²). With the 1/(4B) bound above, τ² is
            approximately 1/(4B) when B = ``batch_size``.
        cost_per_transition: Cost per **transition** in the Gittins DP for each arm, in **original**
            units—for example ``1.0`` for every arm in a cost-unaware setting, or per-arm values in
            monetary units (e.g. dollars per transition) in a cost-aware setting. A scalar applies to
            every arm; otherwise pass a sequence or
            ``(n_arms,)`` tensor with one value per row of ``observed_matrix``. The DP uses
            ``cost_per_transition * cost_scaling_factor`` so you can keep interpretable costs while
            matching the numerical scale of the reward side (default: ``1.0`` homogeneous cost).
            If ``use_batch_mean_gittins_dp`` is False (default), each transition is one new per-cell
            observation and the horizon has ``n_examples`` stages. If True, each transition is one
            **batch mean** (mean of up to ``batch_size`` new cells), so the horizon has one stage per
            future batch (about ``ceil(remaining / batch_size)`` from the current posterior).
        cost_scaling_factor: Multiplier applied to ``cost_per_transition`` before the DP (default
            ``1e-4``). Set to ``1.0`` if ``cost_per_transition`` is already scaled for the solver.
        n_gittins_grid_points: Grid size for tabular Q (see ``tabular_q_estimate``).
        batch_size: Number of **examples** to evaluate on the chosen arm this step (capped by how
            many columns are still NaN in that row).
        return_mus: If True, return ``(batch, mus)`` with ``mus`` the row-wise nanmean (same as UCB).
        cached_scores: Optional ``(n_arms,)`` float32 buffer updated **in place** when passed. After
            one step, only the pulled arm’s posterior changes, so pass the same buffer and set
            ``recompute_arms`` to that arm’s index to avoid recomputing every arm’s Gittins index.
            First call: pass ``None`` (allocates internally with ``+inf`` placeholders, UCB-E-style)
            or pass a buffer with ``recompute_arms=None`` to fill all arms.
        recompute_arms: Used only when ``cached_scores`` is not ``None``. If ``None``, recompute
            every arm that is not fully observed. Otherwise recompute only the listed arm indices
            (typically the arm evaluated on the previous step).
        use_batch_mean_gittins_dp: If True, each DP step observes the **mean** of the next batch of
            per-cell draws (likelihood variance ``τ² / b`` for batch size ``b``), matching the idea
            that learning advances once per simulator batch instead of once per matrix cell.
        allow_early_stop: If False, never return ``None`` just because the top-scoring arm is
            complete; instead pull the best arm that still has free cells.
        sim_cum_eval: Unused for stopping (kept for API compatibility). Use
            ``gittins_post_pull_update`` with post-pull ``sim_cum_eval`` for stop-time holders.
        natural_stop_cum_eval_holder: Ignored; use ``gittins_post_pull_update`` instead.
        recommendation_aware_stop_cum_eval_holder: Ignored; use ``gittins_post_pull_update`` instead.

    Returns:
        ``batch`` with shape ``(2, b)``, ``b ≤ batch_size``, or ``None`` if every cell is observed.
        Indexing matches ``observed_matrix[row, col]`` (rows = arms/methods, columns = examples):

        * ``batch[0, i]`` — arm index (method); here the same ``k*`` for all ``i`` (only one arm
          chosen per step).
        * ``batch[1, i]`` — example index (column); ``b`` distinct previously unobserved columns for
          that arm.

        Reveal cells by assigning into ``observed_matrix[batch[0], batch[1]]``. Optionally
        ``(batch, mus)``.
    """
    observed_matrix = observed_matrix.detach()
    if observed_matrix.device.type != "cpu":
        observed_matrix = observed_matrix.cpu()

    m_methods, n_examples = observed_matrix.shape
    tau_sq = float(obs_noise_variance)
    tau_sq_cell = tau_sq * float(batch_size) if batch_observation_model else tau_sq
    mus_posterior, counts, completely_sensed_mask = _gittins_posterior_means(
        observed_matrix,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        tau_sq_cell=tau_sq_cell,
    )

    if completely_sensed_mask.sum() == m_methods:
        return (None, mus_posterior) if return_mus else None

    arm_costs = _cost_vector_per_arm(cost_per_transition, m_methods) * float(cost_scaling_factor)
    n_pts = int(n_gittins_grid_points)

    # Per-observation mode: always use a lookup table unless explicitly forced to run the DP.
    if (not use_batch_mean_gittins_dp) and (not force_per_observation_dp):
        if roots_lookup_table is None:
            transition_stds = transition_stds_shrinking_gaussian_posterior(
                jnp.float32(prior_variance), jnp.float32(tau_sq_cell), n_examples
            )
            if arm_costs.numel() == 1 or torch.allclose(
                arm_costs, arm_costs[0].expand_as(arm_costs)
            ):
                costs_per_arm = jnp.float32(float(arm_costs[0].item()))
                roots_all = compute_roots_lookup_table(
                    transition_stds=transition_stds,
                    costs_per_arm=costs_per_arm,
                    n_points=n_pts,
                )
            else:
                arm_costs_jnp = jnp.asarray(arm_costs.numpy(), dtype=jnp.float32)
                roots_all = compute_roots_lookup_table(
                    transition_stds=transition_stds,
                    costs_per_arm=arm_costs_jnp,
                    n_points=n_pts,
                )
            roots_lookup_table = torch.from_numpy(jax.device_get(roots_all)).to(torch.float32)
        # Normalize expected shapes: keep a 2D table (n_roots_arms, T+1).
        if roots_lookup_table.ndim == 1:
            roots_lookup_table = roots_lookup_table.unsqueeze(0)
        if roots_lookup_table.shape[-1] != (n_examples + 1):
            raise ValueError(
                "roots_lookup_table must have last dimension n_examples+1; "
                f"got {tuple(roots_lookup_table.shape)}, n_examples={n_examples}"
            )

    if cached_scores is None:
        scores = torch.full((m_methods,), float("inf"), dtype=torch.float32)
        arm_indices = range(m_methods)
    else:
        if cached_scores.shape != (m_methods,) or cached_scores.dtype != torch.float32:
            raise ValueError(
                "cached_scores must have shape (n_arms,) and dtype float32; "
                f"got shape {tuple(cached_scores.shape)}, dtype {cached_scores.dtype}"
            )
        scores = cached_scores
        if recompute_arms is None:
            arm_indices = range(m_methods)
        else:
            arm_indices = sorted({int(k) for k in recompute_arms if 0 <= int(k) < m_methods})

    _refresh_gittins_scores(
        observed_matrix,
        scores=scores,
        mus_posterior=mus_posterior,
        completely_sensed_mask=completely_sensed_mask,
        counts=counts,
        arm_indices=arm_indices,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        tau_sq_cell=tau_sq_cell,
        obs_noise_variance=tau_sq,
        arm_costs=arm_costs,
        n_gittins_grid_points=n_gittins_grid_points,
        batch_size=batch_size,
        use_batch_mean_gittins_dp=use_batch_mean_gittins_dp,
        force_per_observation_dp=force_per_observation_dp,
        roots_lookup_table=roots_lookup_table,
        n_examples=n_examples,
    )

    best_method_index = int(torch.argmax(scores).item())
    winner_complete = bool(completely_sensed_mask[best_method_index])

    if winner_complete:
        if allow_early_stop:
            return (None, mus_posterior) if return_mus else None
        scores_eff = scores.clone()
        scores_eff[completely_sensed_mask] = float("-inf")
        if not torch.isfinite(scores_eff).any():
            return (None, mus_posterior) if return_mus else None
        best_method_index = int(torch.argmax(scores_eff).item())

    unobserved_column_indices = (
        observed_matrix[best_method_index].isnan().nonzero().flatten()
    )
    n_unobserved = int(unobserved_column_indices.size(0))
    bsz = min(int(batch_size), n_unobserved)
    perm = torch.randperm(n_unobserved)[:bsz]
    batch = torch.stack(
        [
            torch.full((bsz,), best_method_index, dtype=torch.long),
            unobserved_column_indices[perm].long(),
        ]
    )

    if return_mus:
        return batch, mus_posterior
    return batch
