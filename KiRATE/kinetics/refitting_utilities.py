from typing import Dict, Optional, Tuple

import jax.numpy as jnp
from jax import jit, lax
from jaxtyping import Float64

from ..utilities.custom_types import Array64f, Matrix64f


@jit
def refit_arrhenius(
    rate_constant: Array64f,
    temperature: Array64f,
    weights: Optional[Array64f] = None,
    three_params: bool = False,
) -> Dict[str, Float64]:
    log_k = jnp.log(rate_constant)
    inv_T = 1.0 / temperature

    # ==============================================================================
    # Build design matrix
    X = lax.cond(
        three_params,
        lambda _: jnp.vstack([jnp.ones_like(inv_T), jnp.log(temperature), -inv_T]).T,
        lambda _: jnp.vstack([jnp.ones_like(inv_T), jnp.zeros_like(inv_T), -inv_T]).T,
        None,
    )

    # ==============================================================================
    # Apply weights if provided
    weighted_fit = weights is not None
    if weighted_fit:
        X_weighted, y_weighted = apply_weights(X, log_k, weights)
    else:
        X_weighted, y_weighted = X, log_k

    # ==============================================================================
    # Perform least squares fit
    beta = jnp.linalg.lstsq(X_weighted, y_weighted, rcond=None)[0]

    # ==============================================================================
    # Calculate residuals and covariance matrix
    y_pred = X @ beta
    residuals = log_k - y_pred

    # ==============================================================================
    # Handle weighted residuals for RSS calculation
    if weighted_fit:
        rss = jnp.sum(weights * residuals**2)
    else:
        rss = jnp.sum(residuals**2)

    n_params = lax.cond(three_params, lambda _: 3, lambda _: 2, None)
    dof = len(rate_constant) - n_params

    # ==============================================================================
    # Covariance matrix calculation
    XTX_inv = jnp.linalg.inv(X_weighted.T @ X_weighted)

    # ==============================================================================
    # Mean squared error
    mse = rss / dof

    # ==============================================================================
    # Parameter covariance matrix
    cov_matrix = mse * XTX_inv

    # ==============================================================================
    # Standard errors (diagonal elements of covariance matrix)
    param_uncertainties = jnp.sqrt(jnp.diag(cov_matrix))

    # ==============================================================================
    # Handle two-parameter case by padding with zeros
    lnA_unc, n_unc, EaR_unc = lax.cond(
        three_params,
        lambda _: (param_uncertainties[0], param_uncertainties[1], param_uncertainties[2]),
        lambda _: (param_uncertainties[0], 0.0, param_uncertainties[1]),
        None,
    )

    return {
        "lnA": beta[0],
        "n": beta[1],
        "EaR": beta[2],
        "lnA_uncertainty": lnA_unc,
        "n_uncertainty": n_unc,
        "EaR_uncertainty": EaR_unc,
        "RSS": rss,
        "DoF": dof,
    }


@jit
def apply_weights(X: Matrix64f, y: Array64f, weights: Array64f) -> Tuple[Matrix64f, Array64f]:
    sqrt_weights = jnp.sqrt(weights)
    X_weighted = X * sqrt_weights[:, None]
    y_weighted = y * sqrt_weights
    return X_weighted, y_weighted


def validate_fitting_data(rates: Array64f, temps: Array64f, weights: Optional[Array64f], three_params: bool) -> None:
    if len(rates) != len(temps):
        raise ValueError("Rate constants and temperatures must have the same length")

    if len(rates) < (3 if three_params else 2):
        raise ValueError(f"Need at least {3 if three_params else 2} data points for fitting")

    if not jnp.all(jnp.isfinite(rates)):
        raise ValueError("All rate constants must be finite")

    if not jnp.all(jnp.isfinite(temps)):
        raise ValueError("All temperatures must be finite")

    if jnp.any(temps <= 0):
        raise ValueError("All temperatures must be positive")

    if jnp.any(rates <= 0):
        raise ValueError("All rate constants must be positive")

    if weights is not None:
        if len(weights) != len(rates):
            raise ValueError("Weights must have the same length as data")
        if not jnp.all(jnp.isfinite(weights)):
            raise ValueError("All weights must be finite")
        if jnp.any(weights <= 0):
            raise ValueError("All weights must be positive")
