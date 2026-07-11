# Copyright 2026 Qian Xie
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Gittins index helpers for a conjugate Gaussian bandit arm.

Notation: k is the index of a bandit arm. θ_k is that arm's unknown mean. After t noisy
evaluations of arm k, let D_t denote the data from those pulls; then
θ_k | D_t ~ N(μ_{k,t}, v_{k,t}). Each new evaluation Y_t satisfies Y_t | θ_k ~ N(θ_k, τ²)
with observation variance τ². Under the same worst-case bound, τ² is approximately 1/(4B), where B is
the per-step batch size (``batch_size`` in ``gittins_index_exploration``): number of examples evaluated
on the chosen arm in one step.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar, UInt

from gittins_index_computation import compute_gittins_for_random_walk


def transition_stds_batch_mean_sequence(
    current_posterior_variance: Float[Scalar, ""],
    per_example_obs_variance: Float[Scalar, ""],
    plan_batch_size: int,
    n_cells_remaining: int,
) -> Float[Array, " n_batch_steps"]:
    """
    Standard deviations for the posterior-mean process when each DP transition is **one batch
    summary**: the observation is the mean of ``b`` fresh i.i.d. draws Y | θ ~ N(θ, τ²), which is
    equivalent to a single conjugate update with likelihood variance τ² / b.

    Starting from the current posterior variance (after any number of individual observations on
    the arm), this returns one σ per **future** batch until ``n_cells_remaining`` cells are
    exhausted. Batch sizes follow ``plan_batch_size`` except the last batch, which may be smaller
    (noise τ² / b with that batch's ``b``).

    Args:
        current_posterior_variance: v_{k,t} after the data already in the posterior.
        per_example_obs_variance: τ² for each underlying Y | θ.
        plan_batch_size: Target batch size B (same as policy ``batch_size`` when batches are full).
        n_cells_remaining: How many matrix cells for this arm are still unevaluated.
    """
    if n_cells_remaining <= 0:
        return jnp.array([], dtype=jnp.float32)
    v = float(current_posterior_variance)
    tau_sq = float(per_example_obs_variance)
    b_plan = int(plan_batch_size)
    rem = int(n_cells_remaining)
    sigmas: list[float] = []
    while rem > 0:
        b = min(b_plan, rem)
        noise = tau_sq / float(b)
        sigmas.append(float(jnp.sqrt((v * v) / (v + noise))))
        v = 1.0 / (1.0 / v + 1.0 / noise)
        rem -= b
    return jnp.asarray(sigmas, dtype=jnp.float32)


def compute_gittins_shrinking_posterior_walk_batch_mean(
    current_posterior_mean: Float[Scalar, ""],
    current_posterior_variance: Float[Scalar, ""],
    per_example_obs_variance: Float[Scalar, ""],
    plan_batch_size: int,
    n_cells_remaining: int,
    transition_costs: Float[Array, " n_batch_steps"],
    n_points: UInt[Scalar, ""],
) -> Float[Scalar, ""]:
    """
    Gittins index when the DP advances once per **batch mean**: same shrinking Gaussian posterior as
    ``compute_gittins_shrinking_posterior_walk_per_observation``, but each ``GO`` pays one transition
    cost and observes the mean of the next ``b`` cells (``b`` as in
    ``transition_stds_batch_mean_sequence``). The suffix chain starts **now**, so
    ``current_stage_index`` is always 0 inside ``compute_gittins_for_random_walk``.
    """
    transition_stds = transition_stds_batch_mean_sequence(
        current_posterior_variance,
        per_example_obs_variance,
        plan_batch_size,
        n_cells_remaining,
    )
    if transition_stds.shape[0] != transition_costs.shape[0]:
        raise ValueError(
            "transition_costs length must match the number of future batch steps; "
            f"got costs {transition_costs.shape[0]}, sigmas {transition_stds.shape[0]}"
        )
    return compute_gittins_for_random_walk(
        jnp.uint32(0),
        current_posterior_mean,
        transition_stds,
        transition_costs,
        n_points,
    )


def transition_stds_shrinking_gaussian_posterior(
    initial_variance: Float[Scalar, ""],
    obs_noise_variance: Float[Scalar, ""],
    n_transitions: int,
) -> Float[Array, " n_transitions"]:
    """
    Per-step standard deviations for the posterior mean of one arm (fix an arm index k).

    Model: Gaussian prior on θ_k; likelihood Y_t | θ_k ~ N(θ_k, τ²). After t evaluations of
    arm k, θ_k | D_t ~ N(μ_{k,t}, v_{k,t}). In this function the arm is fixed, so μ_t and v_t
    denote μ_{k,t} and v_{k,t} for that k. Given D_t, the next evaluation updates the belief;
    the increment of μ_t is Gaussian with variance σ_t² = v_t² / (v_t + τ²). The variance
    satisfies v_{t+1} = (1/v_t + 1/τ²)⁻¹, i.e. 1/v_t = 1/v_0 + t/τ² for t ≥ 0.

    Returns σ_0, …, σ_{n_transitions-1} where σ_t is the standard deviation of the increment of
    μ_t from stage t to t+1 (using v_t before that observation).

    Args:
        initial_variance: Posterior variance v_0 for the chosen arm at stage 0 (before any further
            evaluations of that arm).
        obs_noise_variance: τ² in Y_t | θ_k ~ N(θ_k, τ²).
        n_transitions: Length of the returned array (one σ per learning step in the leg).
    """
    t = jnp.arange(n_transitions, dtype=initial_variance.dtype)
    v_t = 1.0 / (1.0 / initial_variance + t / obs_noise_variance)
    return jnp.sqrt((v_t * v_t) / (v_t + obs_noise_variance))


def compute_gittins_shrinking_posterior_walk_per_observation(
    current_stage_index: UInt[Scalar, ""],
    current_posterior_mean: Float[Scalar, ""],
    initial_variance: Float[Scalar, ""],
    obs_noise_variance: Float[Scalar, ""],
    transition_costs: Float[Array, " n_transitions"],
    n_points: UInt[Scalar, ""],
) -> Float[Scalar, ""]:
    """
    Gittins index for one bandit arm (fix an arm index k) with **one DP stage per scalar observation** (matrix cell). The unknown mean θ_k has a conjugate Gaussian posterior; μ_{k,t} is
    its posterior mean after t evaluations of arm k, and D_t is the data from those pulls. The
    process μ_{k,t} behaves like a Gaussian random walk with step variance v_t²/(v_t + τ²), where τ²
    is the variance in Y_t | θ_k ~ N(θ_k, τ²). For one DP stage per **batch mean**, see
    ``compute_gittins_shrinking_posterior_walk_batch_mean``.

    Builds per-step σ_t from (v_0, τ²) via ``transition_stds_shrinking_gaussian_posterior``,
    then calls ``compute_gittins_for_random_walk`` with those sigmas and ``transition_costs``.

    Use the same v_0 and full ``transition_costs`` for the whole planning horizon for that arm.
    Set ``current_stage_index`` to the number of evaluations of arm k already folded into the
    posterior (the current t); the Q solver only applies transitions and costs from that stage
    onward.

    Args:
        current_stage_index: Stage index passed through to ``compute_gittins_for_random_walk``
            (current t for arm k).
        current_posterior_mean: Current μ_{k,t} = E[θ_k | D_t].
        initial_variance: v_0 for arm k at the start of the horizon (used to generate all σ_t).
        obs_noise_variance: τ² in Y_t | θ_k ~ N(θ_k, τ²).
        transition_costs: Per-transition costs c_t, same layout as in
            ``compute_gittins_for_random_walk``.
        n_points: Grid resolution for tabular Q inside ``tabular_q_estimate``.

    Returns:
        Scalar Gittins index for that arm at the given μ_{k,t} and stage.
    """
    transition_stds = transition_stds_shrinking_gaussian_posterior(
        initial_variance,
        obs_noise_variance,
        int(transition_costs.shape[0]),
    )
    return compute_gittins_for_random_walk(
        current_stage_index,
        current_posterior_mean,
        transition_stds,
        transition_costs,
        n_points,
    )
