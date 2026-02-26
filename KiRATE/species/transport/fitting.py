"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details

Polynomial fitting for gas-phase transport properties.

Replicates the fitting procedure used in CHEMKIN and OpenSMOKE++ to represent
viscosity and thermal conductivity as cubic polynomials in log(T).

References
----------
.. [1] Kee, R. J., Rupley, F. M., and Miller, J. A.
       "CHEMKIN-II: A Fortran Chemical Kinetics Package for the Analysis of
       Gas-Phase Chemical Kinetics." Sandia Report SAND89-8009 (1989).
"""

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float64


def fit_log_property(
    eval_func: Callable,
    T_min: float = 300.0,
    T_max: float = 3600.0,
    n_points: int = 10,
) -> Float64[Array, "4"]:
    """
    Fit a transport property to a cubic polynomial in log(T).

    Samples `eval_func` at `n_points` temperatures uniformly spaced between
    `T_min` and `T_max`, then solves the least-squares system:

    .. math::
        \\ln(\\text{prop}(T_i)) = A + B\\ln(T_i) + C\\ln(T_i)^2 + D\\ln(T_i)^3

    This is identical to the fitting loop in OpenSMOKE++ ``Fitting()`` for
    viscosity, thermal conductivity, and binary diffusivities.

    Parameters
    ----------
    eval_func : Callable[[float], float]
        Function that returns the property value at temperature T [K].
        Must accept a single scalar float and return a positive float.
        Example: ``lambda T: species_viscosity(jnp.array(T), MW, sigma, eps_k, mu)``
    T_min : float, optional
        Minimum temperature for the fit [K]. Default: 300 K.
    T_max : float, optional
        Maximum temperature for the fit [K]. Default: 3600 K.
    n_points : int, optional
        Number of sampling points. Default: 10 (same as OpenSMOKE++).

    Returns
    -------
    Float64[Array, "4"]
        Fitting coefficients [A, B, C, D] such that:
        :math:`\\ln(\\text{prop}) \\approx A + B\\ln(T) + C\\ln(T)^2 + D\\ln(T)^3`

    Notes
    -----
    The normal equations :math:`(X^T X) c = X^T y` are solved via ``jnp.linalg.lstsq``
    for numerical stability. This is a preprocessing step, so NumPy is used for
    temperature grid construction and the result is converted to JAX arrays.

    OpenSMOKE++ uses ``fullPivLu()`` decomposition; ``jnp.linalg.lstsq`` gives
    identical results for well-conditioned systems.
    """
    T_points = jnp.linspace(T_min, T_max, n_points)

    # Build Vandermonde matrix X  (n_points x 4)
    log_T = jnp.log(T_points)
    X = jnp.column_stack([jnp.ones_like(log_T), log_T, jnp.pow(log_T, 2), jnp.pow(log_T, 3)])

    # Sample the property and take log
    y = eval_func(T_points)

    # Solve least squares: (X^T X) coeffs = X^T y
    coeffs, *_ = jnp.linalg.lstsq(X, y)

    return coeffs


def eval_log_poly(
    coeffs: Float64[Array, "4"],
    T: Float64[Array, ""] | Float64[Array, "n"],
) -> Float64[Array, ""] | Float64[Array, "n"]:
    """
    Evaluate a log-cubic polynomial fit at temperature T.

    Reconstructs the property from its fitting coefficients using Horner's method:

    .. math::
        \\text{prop}(T) = \\exp(A + B\\ln(T) + C\\ln(T)^2 + D\\ln(T)^3)

    Parameters
    ----------
    coeffs : Float64[Array, "4"]
        Fitting coefficients [A, B, C, D] from :func:`fit_log_property`
    T : Float64[Array, ""]
        Temperature [K]

    Returns
    -------
    Float64[Array, ""]
        Reconstructed property value (same units as the fitted property)
    """
    log_T = jnp.log(T)
    return jnp.exp(coeffs[0] + log_T * (coeffs[1] + log_T * (coeffs[2] + log_T * coeffs[3])))
