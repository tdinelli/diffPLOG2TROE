"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float64


def estimate_missing_uncertainties(
    residual_fn_unweighted: Callable[[Float64[Array, "n_params"], None], Float64[Array, "n"]],
    params_opt: Float64[Array, "n_params"],
    uncertainties: Float64[Array, "n"],
    has_uncertainty: Bool[Array, "n"],
) -> Float64[Array, "n"]:
    """
    Estimate missing uncertainties from residual distribution.

    When some data points lack uncertainty estimates, this function uses
    the residual distribution at optimized parameters to estimate reasonable
    uncertainty values. This ensures all data points have weights for
    proper weighted least squares fitting.

    Parameters
    ----------
    residual_fn_unweighted : Callable
        Unweighted residual function: fn(params, args) -> residuals
    params_opt : Float64[Array, "n_params"]
        Optimal parameter values (from initial fit)
    uncertainties : Float64[Array, "n"]
        Array of uncertainties (some may be zero/placeholder)
    has_uncertainty : Bool[Array, "n"]
        Boolean mask indicating which points have valid uncertainties

    Returns
    -------
    Float64[Array, "n"]
        Complete uncertainty array with estimated values filled in

    Notes
    -----
    **Estimation Strategy:**

    1. Compute unweighted residuals at optimal parameters
    2. Calculate RMSE from points that have valid uncertainties
    3. Use this RMSE as the uncertainty for points without measurements
    4. If no points have uncertainties, use global RMSE

    This approach assumes that missing uncertainties have similar magnitude
    to the observed residual scatter in the data.

    References
    ----------
    Based on the approach used in Cantera's kinetics fitting utilities.
    """
    # Evaluate residuals at optimal parameters
    residuals = residual_fn_unweighted(params_opt, None)

    # Estimate uncertainty from residuals of points that have it
    if jnp.any(has_uncertainty):
        # Use RMSE from points with known uncertainty
        residuals_with_uncert = residuals[has_uncertainty]
        rmse = jnp.sqrt(jnp.mean(residuals_with_uncert**2))
    else:
        # No points have uncertainty - use global RMSE
        rmse = jnp.sqrt(jnp.mean(residuals**2))

    # Fill in missing uncertainties with estimated value
    uncertainties_estimated = jnp.where(has_uncertainty, uncertainties, rmse)

    return uncertainties_estimated
