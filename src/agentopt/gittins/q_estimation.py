# Copyright 2025 Qian Xie, Theo Brown, Ziv Scully, Alexander Terenin
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Functions for computing the action-value function (Q-function) of a random walk with actions STOP and GO."""

from typing import Literal

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float, Scalar, UInt


def gaussian_ei(
    x: Float[Array, "..."], loc: float = 0, scale: float = 1
) -> Float[Array, "..."]:
    """
    Expected improvement: E[max(z - ξ, 0)] where ξ ~ N(0, scale²)

    EI(z) = z * Φ(z/σ) + σ * φ(z/σ)
    where Φ is the standard normal CDF and φ is the standard normal PDF
    """
    z = x - loc
    return z * jsp.stats.norm.cdf(z / scale) + scale * jsp.stats.norm.pdf(z / scale)


@jax.jit
def fft_convolve_gaussian_m_diff(
    y: Float[Array, " n"],
    sigma: float,
    left_bound: float,
    right_bound: float,
    dydx_left: float = 0.0,
    dydx_right: float = 1.0,
) -> Float[Array, " n"]:
    """
    FFT convolution with Gaussian kernel using the m-diff approach:
    f(x_0) + m_0 (s - x_0) + sum_{i=0}^{n-1} (m_{i+1} - m_i) * EI_t(s - x_i)

    where m_i = (f(x_{i+1}) - f(x_i)) / δ for i ∈ {0, ..., n-1}
    and EI_t(z) = z * Φ(z/σ_t) + σ_t * φ(z/σ_t)

    Uses FFT for the discrete convolution terms by exploiting s - x_i = (j - i)δ.
    """
    n = y.shape[0]
    # Use explicit boundary parameters
    x = jnp.linspace(left_bound, right_bound, n)
    dx = x[1] - x[0]

    # Compute m_i values (slopes)
    dy_dx = jnp.diff(y) / dx
    m_values = jnp.concatenate([jnp.array([dydx_left]), dy_dx, jnp.array([dydx_right])])

    # Initialize result
    result = jnp.zeros_like(x)

    # Add base term: f(x_0) + m_0 (s - x_0)
    result = result + y[0] + dydx_left * (x - x[0])

    # Compute differences (m_{i+1} - m_i) for i = 0 to n-1
    m_diffs = jnp.diff(m_values)  # length n

    # Create EI kernel for convolution: EI_t((j - i)δ) for j - i = -(n-1), ..., -1, 0, 1, ..., n-1
    # This creates a length-(2n-1) array where index k corresponds to EI_t(k * δ) for k = -(n-1), ..., n-1
    k_values = jnp.arange(2 * n - 1) - (
        n - 1
    )  # [-(n-1), -(n-2), ..., -1, 0, 1, ..., n-1]
    ei_kernel = gaussian_ei(k_values * dx, loc=0, scale=sigma)  # length 2n-1

    # Use FFT to compute the convolution: sum_{i=0}^{n-1} (m_{i+1} - m_i) * EI_t((j - i)δ)
    # This is equivalent to convolving m_diffs (length n) with ei_kernel (length 2n-1) in "valid" mode
    # The result will have length (n + (2n-1) - 1) = 3n-2, but we only need the middle n elements
    convolved = jsp.signal.fftconvolve(m_diffs, ei_kernel, mode="valid")  # length n

    result = result + convolved

    return result


@jax.jit
def fft_convolve_gaussian_ei_diff(
    y: Float[Array, " n"],
    sigma: float,
    left_bound: float,
    right_bound: float,
    dydx_left: float = 0.0,
    dydx_right: float = 1.0,
) -> Float[Array, " n"]:
    """
    FFT convolution with Gaussian kernel using the EI-diff approach:
    f(x_0) + m_0 (s - x_0) + sum_{i=0}^{n-1} m_i (EI_t(s - x_{i-1}) - EI_t(s - x_i))
    - m_0 * EI_t(s - x_{-1}) + m_n * EI_t(s - x_{n-1})

    where m_i = (f(x_{i+1}) - f(x_i)) / δ for i ∈ {0, ..., n-1}
    and EI_t(z) = z * Φ(z/σ_t) + σ_t * φ(z/σ_t)

    Uses FFT for the discrete convolution terms by exploiting s - x_i = (j - i)δ.
    """
    n = y.shape[0]
    # Use explicit boundary parameters
    x = jnp.linspace(left_bound, right_bound, n)
    dx = x[1] - x[0]

    # Compute m_i values (slopes)
    dy_dx = jnp.diff(y) / dx
    m_values = jnp.concatenate([jnp.array([dydx_left]), dy_dx, jnp.array([dydx_right])])

    # Initialize result
    result = jnp.zeros_like(x)

    # Add base term: f(x_0) + m_0 (s - x_0)
    result = result + y[0] + dydx_left * (x - x[0])

    # Create the EI difference kernel for convolution: EI_t((j - i + 1)δ) - EI_t((j - i)δ)
    # This creates a length-(2n-1) array where index k corresponds to EI_t((k + 1) * δ) - EI_t(k * δ) for k = -(n-1), ..., n-1
    k_values = jnp.arange(2 * n - 1) - (
        n - 1
    )  # [-(n-1), -(n-2), ..., -1, 0, 1, ..., n-1]

    # Compute EI_t((k + 1) * δ) for all k
    ei_k_plus_1 = gaussian_ei((k_values + 1) * dx, loc=0, scale=sigma)

    # Compute EI_t(k * δ) for all k
    ei_k = gaussian_ei(k_values * dx, loc=0, scale=sigma)

    # Create the difference kernel (length 2n-1)
    ei_diffs = ei_k_plus_1 - ei_k  # length 2n-1

    # Use FFT to compute the convolution: sum_{i=0}^{n-1} m_i * (EI_t((j-i+1)δ) - EI_t((j-i)δ))
    # This is equivalent to convolving m_values[:n] (length n) with ei_diffs (length 2n-1) in "valid" mode
    # The result will have length (n + (2n-1) - 1) = 3n-2, but we only need the middle n elements
    convolution_term = jsp.signal.fftconvolve(
        m_values[:n], ei_diffs, mode="valid"
    )  # length n

    result = result + convolution_term

    # Add boundary terms: -m_0 * EI_t(s - x_{-1}) + m_n * EI_t(s - x_{n-1})
    m_0 = m_values[0]
    m_n = m_values[-1]

    # Left correction: -m_0 * EI_t(s - x_{-1}) where x_{-1} = x_0 - δ
    ei_x_minus_1 = gaussian_ei(x - (x[0] - dx), loc=0, scale=sigma)
    result = result - m_0 * ei_x_minus_1

    # Right correction: +m_n * EI_t(s - x_{n-1})
    ei_x_n_minus_1 = gaussian_ei(x - x[-1], loc=0, scale=sigma)
    result = result + m_n * ei_x_n_minus_1

    return result


@jax.jit
def fft_convolve_gaussian_phi_diff(
    y: Float[Array, " n"],
    sigma: float,
    left_bound: float,
    right_bound: float,
    dydx_left: float = 0.0,
    dydx_right: float = 1.0,
) -> Float[Array, " n"]:
    """
    FFT convolution with Gaussian kernel using the phi-diff approach:
    f(x_0) + m_0 (s - x_0) + sum_{i=0}^{n-1} (m_{i+1} - m_i) (s-x_i)Phi((s-x_i)/sigma_t)
    + sum_{i=0}^{n-1} m_i sigma_t (phi((s-x_{i-1})/sigma_t) - phi((s-x_i)/sigma_t))
    - m_0 sigma_t phi((s-x_{-1})/sigma_t) + m_n sigma_t phi((s-x_{n-1})/sigma_t)

    where m_i = (f(x_{i+1}) - f(x_i)) / δ for i ∈ {0, ..., n-1}
    and Phi is the standard normal CDF and phi is the standard normal PDF
    """
    n = y.shape[0]
    # Use explicit boundary parameters
    x = jnp.linspace(left_bound, right_bound, n)
    dx = x[1] - x[0]

    # Compute m_i values (slopes)
    dy_dx = jnp.diff(y) / dx
    m_values = jnp.concatenate([jnp.array([dydx_left]), dy_dx, jnp.array([dydx_right])])

    # Initialize result
    result = jnp.zeros_like(x)

    # Add base term: f(x_0) + m_0 (s - x_0)
    result = result + y[0] + dydx_left * (x - x[0])

    # Compute differences (m_{i+1} - m_i) for i = 0 to n-1
    m_diffs = jnp.diff(m_values)  # length n

    # Create kernel for first sum: (s-x_i)Phi((s-x_i)/sigma_t)
    # This creates a length-(2n-1) array where index k corresponds to (k*δ)Phi(k*δ/sigma) for k = -(n-1), ..., n-1
    k_values = jnp.arange(2 * n - 1) - (
        n - 1
    )  # [-(n-1), -(n-2), ..., -1, 0, 1, ..., n-1]
    x_kernel = k_values * dx  # (s-x_i) terms
    phi_kernel = jsp.stats.norm.cdf(x_kernel / sigma)  # Phi((s-x_i)/sigma_t) terms
    first_kernel = x_kernel * phi_kernel  # (s-x_i)Phi((s-x_i)/sigma_t)

    # Use FFT to compute first convolution: sum_{i=0}^{n-1} (m_{i+1} - m_i) * (s-x_i)Phi((s-x_i)/sigma_t)
    first_convolution = jsp.signal.fftconvolve(
        m_diffs, first_kernel, mode="valid"
    )  # length n
    result = result + first_convolution

    # Create kernel for second sum: phi((s-x_{i-1})/sigma_t) - phi((s-x_i)/sigma_t)
    # This creates a length-(2n-1) array where index k corresponds to phi((k+1)*δ/sigma) - phi(k*δ/sigma) for k = -(n-1), ..., n-1
    phi_k_plus_1 = jsp.stats.norm.pdf(
        (k_values + 1) * dx / sigma
    )  # phi((s-x_{i-1})/sigma_t)
    phi_k = jsp.stats.norm.pdf(k_values * dx / sigma)  # phi((s-x_i)/sigma_t)
    second_kernel = (
        phi_k_plus_1 - phi_k
    )  # phi((s-x_{i-1})/sigma_t) - phi((s-x_i)/sigma_t)

    # Use FFT to compute second convolution: sum_{i=0}^{n-1} m_i * sigma_t * (phi((s-x_{i-1})/sigma_t) - phi((s-x_i)/sigma_t))
    second_convolution = sigma * jsp.signal.fftconvolve(
        m_values[:n], second_kernel, mode="valid"
    )  # length n
    result = result + second_convolution

    # Add boundary terms: -m_0 sigma_t phi((s-x_{-1})/sigma_t) + m_n sigma_t phi((s-x_{n-1})/sigma_t)
    m_0 = m_values[0]
    m_n = m_values[-1]

    # Left correction: -m_0 sigma_t phi((s-x_{-1})/sigma_t) where x_{-1} = x_0 - δ
    phi_x_minus_1 = jsp.stats.norm.pdf((x - (x[0] - dx)) / sigma)
    result = result - m_0 * sigma * phi_x_minus_1

    # Right correction: +m_n sigma_t phi((s-x_{n-1})/sigma_t)
    phi_x_n_minus_1 = jsp.stats.norm.pdf((x - x[-1]) / sigma)
    result = result + m_n * sigma * phi_x_n_minus_1

    return result


@jax.jit
def fft_convolve_gaussian_Phi_diff(
    y: Float[Array, " n"],
    sigma: float,
    left_bound: float,
    right_bound: float,
    dydx_left: float = 0.0,
    dydx_right: float = 1.0,
) -> Float[Array, " n"]:
    """
    FFT convolution with Gaussian kernel using the Phi-diff approach:
    f(x_0) + m_0 (s - x_0) + sum_{i=0}^{n-1} (m_{i+1} - m_i) sigma_t phi((s-x_i)/sigma_t)
    + sum_{i=0}^{n-1} m_i ((s-x_{i-1})Phi((s-x_{i-1})/sigma_t) - (s-x_i)Phi((s-x_i)/sigma_t))
    - m_0 (s-x_{-1}) Phi((s-x_{-1})/sigma_t) + m_n (s-x_{n-1}) Phi((s-x_{n-1})/sigma_t)

    where m_i = (f(x_{i+1}) - f(x_i)) / δ for i ∈ {0, ..., n-1}
    and Phi is the standard normal CDF and phi is the standard normal PDF
    """
    n = y.shape[0]
    # Use explicit boundary parameters
    x = jnp.linspace(left_bound, right_bound, n)
    dx = x[1] - x[0]

    # Compute m_i values (slopes)
    dy_dx = jnp.diff(y) / dx
    m_values = jnp.concatenate([jnp.array([dydx_left]), dy_dx, jnp.array([dydx_right])])

    # Initialize result
    result = jnp.zeros_like(x)

    # Add base term: f(x_0) + m_0 (s - x_0)
    result = result + y[0] + dydx_left * (x - x[0])

    # Compute differences (m_{i+1} - m_i) for i = 0 to n-1
    m_diffs = jnp.diff(m_values)  # length n

    # Create kernel for first sum: sigma_t phi((s-x_i)/sigma_t)
    # This creates a length-(2n-1) array where index k corresponds to sigma * phi(k*δ/sigma) for k = -(n-1), ..., n-1
    k_values = jnp.arange(2 * n - 1) - (
        n - 1
    )  # [-(n-1), -(n-2), ..., -1, 0, 1, ..., n-1]
    first_kernel = sigma * jsp.stats.norm.pdf(
        k_values * dx / sigma
    )  # sigma_t phi((s-x_i)/sigma_t)

    # Use FFT to compute first convolution: sum_{i=0}^{n-1} (m_{i+1} - m_i) * sigma_t phi((s-x_i)/sigma_t)
    first_convolution = jsp.signal.fftconvolve(
        m_diffs, first_kernel, mode="valid"
    )  # length n
    result = result + first_convolution

    # Create kernel for second sum: (s-x_{i-1})Phi((s-x_{i-1})/sigma_t) - (s-x_i)Phi((s-x_i)/sigma_t)
    # This creates a length-(2n-1) array where index k corresponds to (k+1)*δ*Phi((k+1)*δ/sigma) - k*δ*Phi(k*δ/sigma) for k = -(n-1), ..., n-1
    x_k_plus_1 = (k_values + 1) * dx  # (s-x_{i-1}) terms
    x_k = k_values * dx  # (s-x_i) terms
    Phi_k_plus_1 = jsp.stats.norm.cdf(
        x_k_plus_1 / sigma
    )  # Phi((s-x_{i-1})/sigma_t) terms
    Phi_k = jsp.stats.norm.cdf(x_k / sigma)  # Phi((s-x_i)/sigma_t) terms
    second_kernel = (
        x_k_plus_1 * Phi_k_plus_1 - x_k * Phi_k
    )  # (s-x_{i-1})Phi((s-x_{i-1})/sigma_t) - (s-x_i)Phi((s-x_i)/sigma_t)

    # Use FFT to compute second convolution: sum_{i=0}^{n-1} m_i * ((s-x_{i-1})Phi((s-x_{i-1})/sigma_t) - (s-x_i)Phi((s-x_i)/sigma_t))
    second_convolution = jsp.signal.fftconvolve(
        m_values[:n], second_kernel, mode="valid"
    )  # length n
    result = result + second_convolution

    # Add boundary terms: -m_0 (s-x_{-1}) Phi((s-x_{-1})/sigma_t) + m_n (s-x_{n-1}) Phi((s-x_{n-1})/sigma_t)
    m_0 = m_values[0]
    m_n = m_values[-1]

    # Left correction: -m_0 (s-x_{-1}) Phi((s-x_{-1})/sigma_t) where x_{-1} = x_0 - δ
    x_minus_1 = x - (x[0] - dx)  # (s-x_{-1}) terms
    Phi_x_minus_1 = jsp.stats.norm.cdf(
        x_minus_1 / sigma
    )  # Phi((s-x_{-1})/sigma_t) terms
    result = result - m_0 * x_minus_1 * Phi_x_minus_1

    # Right correction: +m_n (s-x_{n-1}) Phi((s-x_{n-1})/sigma_t)
    x_n_minus_1 = x - x[-1]  # (s-x_{n-1}) terms
    Phi_x_n_minus_1 = jsp.stats.norm.cdf(
        x_n_minus_1 / sigma
    )  # Phi((s-x_{n-1})/sigma_t) terms
    result = result + m_n * x_n_minus_1 * Phi_x_n_minus_1

    return result


def fft_convolve_gaussian(
    y: Float[Array, " n"],
    sigma: float,
    left_bound: float,
    right_bound: float,
    method: Literal["m_diff", "ei_diff", "phi_diff", "Phi_diff"] = "m_diff",
    dydx_left: float = 0.0,
    dydx_right: float = 1.0,
) -> Float[Array, " n"]:
    """
    FFT convolution with Gaussian kernel using specified method.

    Args:
        y: Input array to convolve
        sigma: Standard deviation of Gaussian kernel
        left_bound: Left boundary of the grid
        right_bound: Right boundary of the grid
        method: "m_diff", "ei_diff", "phi_diff", or "Phi_diff" - which approach to use
        dydx_left: Left boundary derivative
        dydx_right: Right boundary derivative

    Returns:
        Convolved array
    """
    if method == "m_diff":
        return fft_convolve_gaussian_m_diff(
            y, sigma, left_bound, right_bound, dydx_left, dydx_right
        )
    elif method == "ei_diff":
        return fft_convolve_gaussian_ei_diff(
            y, sigma, left_bound, right_bound, dydx_left, dydx_right
        )
    elif method == "phi_diff":
        return fft_convolve_gaussian_phi_diff(
            y, sigma, left_bound, right_bound, dydx_left, dydx_right
        )
    elif method == "Phi_diff":
        return fft_convolve_gaussian_Phi_diff(
            y, sigma, left_bound, right_bound, dydx_left, dydx_right
        )
    else:
        raise ValueError(
            f"Unknown method: {method}. Must be 'm_diff', 'ei_diff', 'phi_diff', or 'Phi_diff'"
        )


def tabular_q_estimate(
    transition_stds: Float[Array, " n_transitions"],
    transition_costs: Float[Array, " n_transitions"],
    current_stage_index: UInt[Scalar, ""],
    n_points: UInt[Scalar, ""] = 2**14 + 1,
    method: Literal["m_diff", "ei_diff", "phi_diff", "Phi_diff"] = "m_diff",
    k: float = 5.0,
    extra_margin: float = 0.01,
) -> tuple[
    Float[Array, " n_points"],
    Float[Array, "n_transitions+1 n_points"],
    Float[Array, " n_points"],
]:
    """
    Compute Q(s, continue) for a range of s values using tabular estimation and FFT convolution.

     Stage 0                          Stage t                                   Stage T
    |-------|                   |--------------------|                   |--------------------|
    |  x₀   | -----> ... -----> | xₜ ~ N(xₜ₋₁, σₜ₋₁) | -----> ... -----> | xᴛ ~ N(xᴛ₋₁, σᴛ₋₁) |
    |-------|   c₀        cₜ₋₁  |--------------------|  cₜ         cᴛ₋₁  |--------------------|
       θ₀                                θₜ                                        θᴛ

    Args:
        transition_stds: Array of standard deviations for Gaussian stage transitions (σ₀, ..., σᴛ₋₁).
        transition_costs: Array of stage transition costs (c₀, ..., cᴛ₋₁).
        current_stage_index: Index of current stage. Note this is not the same as the transition index!
        n_points: Number of points for tabular estimation
        k: Multiplier for the standard deviation range (default: 5.0)
        extra_margin: Fraction of the cost to add on to the adaptive grid, to avoid running out of grid space.

    Returns:
        Q: An array of estimated Q function values over a grid of s values
    """
    # Adaptive grid, based on high-probability bounds
    total_cost = transition_costs.sum()
    total_std = jnp.sqrt((transition_stds**2).sum())
    x_0 = -k * total_std
    x_n = (1 + extra_margin) * total_cost + k * total_std
    x_values = jnp.linspace(x_0, x_n, n_points)

    n_transitions = transition_stds.shape[0]

    def body_fn(Q, i):
        sigma = transition_stds[i]
        cost = transition_costs[i]

        Q_plus = jnp.maximum(Q, 0.0)
        Q_updated = (
            fft_convolve_gaussian(
                Q_plus, sigma, x_0, x_n, method=method, dydx_left=0.0, dydx_right=1.0
            )
            - cost
        )

        return jnp.where(i >= current_stage_index, Q_updated, Q), Q

    Q_final = x_values
    Q_current, Q_future = jax.lax.scan(
        body_fn, Q_final, jnp.arange(n_transitions), reverse=True
    )

    return Q_current, Q_future, x_values