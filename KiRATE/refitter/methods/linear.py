"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import jax.numpy as jnp
from jaxtyping import Array, Float64

from KiRATE.utilities import constants


def arrhenius_linear_fit(
    temperature: Float64[Array, "n"],
    rate_constant: Float64[Array, "n"],
    uncertainties: Float64[Array, "n"] | None = None,
    fixed_params: dict[str, float] | None = None,
) -> Float64[Array, "n_params"]:
    """
    Compute initial Arrhenius parameter guess using weighted linear least squares.

    This function solves the linearized Arrhenius equation in closed form to
    obtain an excellent initial guess for nonlinear optimization. The method
    transforms the nonlinear Arrhenius equation into a linear regression problem
    and properly accounts for uncertainty propagation through the log transform.

    Parameters
    ----------
    temperature : Float64[Array, "n"]
        Temperature values [K] at which rate constants are measured
    rate_constant : Float64[Array, "n"]
        Rate constant values. The function will apply log transform
        internally with proper uncertainty propagation.
    uncertainties : Float64[Array, "n"], optional
        Uncertainties (standard errors) in rate_constant values.
        If provided, weighted least squares is performed with proper weights
        accounting for the log transform: w_i = 1 / (σ_i / k_i)^2
    fixed_params : dict[str, float], optional
        Known parameters to fix during estimation. Keys can be 'A', 'n', or 'Ea'.
        If provided, only unknown parameters will be estimated. For example:
        - fixed_params={'n': 0.0} → estimate only A and Ea (standard Arrhenius)
        - fixed_params={'A': 1e14} → estimate only n and Ea

    Returns
    -------
    Float64[Array, "n_params"]
        Initial guess for free parameters. Order depends on which parameters
        are being estimated:
        - If fixed_params is None: [A, n, Ea] (all three parameters)
        - Otherwise: subset of free parameters in canonical order [A, n, Ea]

    Raises
    ------
    ValueError
        If temperature or rate_constant values are non-positive,
        if uncertainties are non-positive,
        or if insufficient data for the number of free parameters.

    Notes
    -----
    **Mathematical Approach:**

    The modified Arrhenius equation:

    .. math::
        k(T) = A \\cdot T^n \\cdot \\exp\\left(-\\frac{E_a}{R T}\\right)

    Taking logarithms linearizes the equation:

    .. math::
        \\ln k = \\ln A + n \\ln T - \\frac{E_a}{R T}

    This is linear in the transformed parameters [ln(A), n, Ea/R], allowing
    closed-form solution via ordinary least squares:

    .. math::
        \\mathbf{y} = \\mathbf{X} \\boldsymbol{\\beta}

    where:
        - **y** = ln(k) (observations)
        - **X** = [**1**, ln(T), -1/T] (design matrix)
        - **β** = [ln(A), n, Ea/R] (coefficients)

    **Uncertainty Propagation Through Log Transform:**

    When transforming k → ln(k), the uncertainty transforms as:

    .. math::
        \\sigma_{\\ln k} = \\frac{\\sigma_k}{k}

    This is derived from error propagation: if y = f(x), then
    :math:`\\sigma_y \\approx |f'(x)| \\sigma_x`. For f(x) = ln(x),
    f'(x) = 1/x, so :math:`\\sigma_{\\ln(k)} = \\sigma_k / k`.

    **Weighted Least Squares:**

    The weighted least squares solution minimizes:

    .. math::
        \\sum_i w_i (y_i - \\mathbf{x}_i^T \\boldsymbol{\\beta})^2

    where weights are: :math:`w_i = 1 / \\sigma^2_{\\ln(k_i)} = (k_i / \\sigma_{k_i})^2`

    This gives more weight to measurements with smaller relative uncertainties.

    **Parameter Masking:**

    When some parameters are known (via `fixed_params`), the method subtracts their
    contribution from ln(k) and estimates only the remaining parameters.
    """
    # Validate inputs
    if jnp.any(temperature <= 0):
        raise ValueError("Temperature must be positive")
    if jnp.any(rate_constant <= 0):
        raise ValueError("Rate constants must be positive (needed for log transform)")

    # Check for sufficient data vs. free parameters
    n_free = 3 if fixed_params is None else (3 - len(fixed_params))
    if len(temperature) < n_free:
        raise ValueError(f"Need at least {n_free} data points to fit {n_free} free parameters")

    # Transform to log space
    log_k = jnp.log(rate_constant)

    # Compute weights if uncertainties provided
    weights = None
    if uncertainties is not None:
        # Check for valid uncertainties
        if jnp.any(uncertainties <= 0):
            raise ValueError("Uncertainties must be positive")

        # Compute weights with proper log transform propagation
        sigma_log_k = uncertainties / rate_constant
        weights = 1.0 / (sigma_log_k**2)

    if fixed_params is None:
        # CASE 1: Fit all three parameters [A, n, Ea]
        # Design matrix: [1, ln(T), -1/T]
        # Corresponds to: ln(k) = ln(A) + n*ln(T) - Ea/(R*T)
        X = jnp.column_stack([jnp.ones_like(temperature), jnp.log(temperature), -1.0 / temperature])

        coeffs = _weighted_linear_least_squares(X, log_k, weights)

        # Return [A, n, Ea]
        # Note: coeffs[0] = ln(A), coeffs[2] = Ea/R
        return jnp.array(
            [
                jnp.exp(coeffs[0]),  # A
                coeffs[1],  # n
                coeffs[2] * constants.R_cal_mol_K,  # Ea
            ]
        )

    else:
        # CASE 2: Fit subset of parameters
        # Start with y = ln(k)
        y = log_k

        # Subtract known parameter contributions from y
        if "A" in fixed_params:
            y = y - jnp.log(fixed_params["A"])
        if "n" in fixed_params:
            y = y - fixed_params["n"] * jnp.log(temperature)
        if "Ea" in fixed_params:
            y = y + fixed_params["Ea"] / (constants.R_cal_mol_K * temperature)

        # Build design matrix for unknown parameters
        X_cols = []
        free_param_names = []

        if "A" not in fixed_params:
            X_cols.append(jnp.ones_like(temperature))
            free_param_names.append("A")
        if "n" not in fixed_params:
            X_cols.append(jnp.log(temperature))
            free_param_names.append("n")
        if "Ea" not in fixed_params:
            X_cols.append(-1.0 / temperature)
            free_param_names.append("Ea")

        if not X_cols:
            raise ValueError("No free parameters to fit (all are fixed)")

        X = jnp.column_stack(X_cols)
        coeffs = _weighted_linear_least_squares(X, y, weights)

        # Transform coefficients back to parameter space
        result = []
        for i, param_name in enumerate(free_param_names):
            if param_name == "A":
                result.append(jnp.exp(coeffs[i]))
            elif param_name == "Ea":
                result.append(coeffs[i] * constants.R_cal_mol_K)
            else:  # n
                result.append(coeffs[i])

        return jnp.array(result)


def _weighted_linear_least_squares(
    X: Float64[Array, "n p"],
    y: Float64[Array, "n"],
    weights: Float64[Array, "n"] | None = None,
) -> Float64[Array, "p"]:
    """
    Solve weighted linear least squares problem (internal helper).

    Solves: min ||W^(1/2) (y - X β)||^2

    Parameters
    ----------
    X : Float64[Array, "n p"]
        Design matrix (n data points, p parameters)
    y : Float64[Array, "n"]
        Observations
    weights : Float64[Array, "n"], optional
        Weights for each observation. If None, unweighted least squares.

    Returns
    -------
    Float64[Array, "p"]
        Optimal parameter vector β

    Notes
    -----
    Uses JAX's lstsq which employs SVD for numerical stability.
    Weights are normalized to mean=1 for numerical stability.
    """
    if weights is not None:
        # Normalize weights for stability
        weights_norm = weights / jnp.mean(weights)
        W_sqrt = jnp.sqrt(weights_norm)
        X_weighted = X * W_sqrt[:, None]
        y_weighted = y * W_sqrt
        coeffs, *_ = jnp.linalg.lstsq(X_weighted, y_weighted, rcond=None)
    else:
        coeffs, *_ = jnp.linalg.lstsq(X, y, rcond=None)

    return coeffs
