"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import jax.numpy as jnp
from jaxtyping import Array, Float64


def compute_statistics(
    predictions: Float64[Array, "n"],
    observations: Float64[Array, "n"],
) -> dict[str, float]:
    """
    Compute fitting statistics.

    Parameters
    ----------
    predictions : Float64[Array, "n"]
        Predicted values from the model
    observations : Float64[Array, "n"]
        Observed values from experimental data

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - 'R2': Coefficient of determination (R²)
        - 'SSE': Sum of squared errors
        - 'RMSE': Root mean squared error
        - 'MAE': Mean absolute error

    Notes
    -----
    Statistics are typically computed in log space for rate constants,
    which gives equal weight to relative errors across different orders
    of magnitude. This is appropriate for rate constants that can span
    many orders of magnitude.
    """
    residuals = observations - predictions
    SSE = jnp.sum(residuals**2)
    SS_tot = jnp.sum((observations - jnp.mean(observations)) ** 2)
    R2 = float(1.0 - (SSE / SS_tot)) if SS_tot != 0 else 0.0
    RMSE = float(jnp.sqrt(SSE / len(residuals)))
    MAE = float(jnp.mean(jnp.abs(residuals)))

    return {
        "R2": R2,
        "SSE": float(SSE),
        "RMSE": RMSE,
        "MAE": MAE,
    }
