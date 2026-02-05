"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float64


def compute_parameter_uncertainties(
    residual_fn: Callable[[Float64[Array, "n_params"], None], Float64[Array, "n"]],
    params_opt: Float64[Array, "n_params"],
    weights: Float64[Array, "n"] | None = None,
) -> tuple[
    Float64[Array, "n_params"] | None,
    Float64[Array, "n_params n_params"] | None,
    Float64[Array, "n_params n_params"] | None,
]:
    """
    Compute parameter uncertainties from least-squares solution using the Jacobian.

    This function computes standard errors, covariance matrix, and correlation matrix
    for the fitted parameters using the Jacobian of the residual function at the
    optimal point. Handles both weighted and unweighted cases correctly.

    Parameters
    ----------
    residual_fn : Callable
        Residual function with signature:
            fn(params, args) -> residuals
        This should be the SAME function used in optimization.
    params_opt : Float64[Array, "n_params"]
        Optimal parameter values from fitting
    weights : Float64[Array, "n"], optional
        Weights used in the fit (1/relative_uncertainty). If None, assumes
        unweighted fit and estimates uncertainty from residuals. If provided,
        assumes uncertainties are KNOWN and uses weighted covariance formula.

    Returns
    -------
    std_errors : Float64[Array, "n_params"] or None
        Standard errors of parameters. None if computation fails.
    cov_matrix : Float64[Array, "n_params n_params"] or None
        Covariance matrix of parameters. None if computation fails.
    corr_matrix : Float64[Array, "n_params n_params"] or None
        Correlation matrix of parameters. None if computation fails.

    Notes
    -----
    **Covariance Matrix Formulas:**

    Since the residual function returns weighted residuals r = (y - f) * w,
    the Jacobian J already contains the weights: J = dr/dθ = w * df/dθ.
    Therefore, both cases use similar formulas:

    1. **Unweighted case** (weights=None):
        Uncertainties are UNKNOWN, estimate from residuals:

        .. math::
            \\mathrm{Cov}(\\theta) = \\hat{\\sigma}^2 (J^T J)^{-1}

        where :math:`\\hat{\\sigma}^2 = \\sum r_i^2 / (n - p)` is the residual variance.

    2. **Weighted case** (weights provided):
        Uncertainties are KNOWN, use:

        .. math::
            \\mathrm{Cov}(\\theta) = (J^T J)^{-1}

        where J is the Jacobian of the weighted residuals.

    **Goodness of Fit:**

    For weighted fits, computes reduced chi-squared:

    .. math::
        \\chi^2_{\\text{red}} = \\frac{1}{n-p} \\sum_i r_i^2

    where residuals are already weighted. Ideally :math:`\\chi^2_{\\text{red}} \\approx 1`.

    **Numerical Stability:**

    - Checks condition number before inversion
    - Falls back to pseudo-inverse if matrix is singular
    - Returns None if computation fails
    """
    try:
        # Compute Jacobian at optimal parameters using JAX autodiff
        jac_fn = jax.jacobian(residual_fn, argnums=0)
        J = jac_fn(params_opt, None)

        # Ensure J is 2D (n_data, n_params)
        if J.ndim == 1:
            J = J.reshape(-1, 1)

        n_data, n_params = J.shape

        # Get residuals at optimal point
        residuals = residual_fn(params_opt, None)

        # Compute degrees of freedom
        dof = n_data - n_params
        if dof <= 0:
            return None, None, None

        # Compute J^T J
        JTJ = J.T @ J

        # Check condition number
        cond = jnp.linalg.cond(JTJ)
        if cond > 1e12:
            # Use pseudoinverse for badly conditioned matrices
            JTJ_inv = jnp.linalg.pinv(JTJ)
        else:
            JTJ_inv = jnp.linalg.inv(JTJ)

        # Compute covariance matrix
        if weights is None:
            # Unweighted case: estimate variance from residuals
            sigma2 = jnp.sum(residuals**2) / dof
            cov_matrix = sigma2 * JTJ_inv
        else:
            # Weighted case: uncertainties are known
            cov_matrix = JTJ_inv

        # Compute standard errors
        std_errors = jnp.sqrt(jnp.diag(cov_matrix))

        # Compute correlation matrix
        D_inv = jnp.diag(1.0 / std_errors)
        corr_matrix = D_inv @ cov_matrix @ D_inv

        return std_errors, cov_matrix, corr_matrix

    except Exception:
        return None, None, None
