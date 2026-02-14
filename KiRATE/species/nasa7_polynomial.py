"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import jax.numpy as jnp
from jaxtyping import Array, Float64

# Coefficient multipliers for enthalpy and entropy polynomials
_h_multipliers: Float64[Array, "5"] = jnp.array([1.0, 0.5, 1 / 3, 0.25, 0.2], dtype=jnp.float64)
_s_multipliers: Float64[Array, "5"] = jnp.array([0.0, 1.0, 0.5, 1 / 3, 0.25], dtype=jnp.float64)


def build_temperature_powers(
    T: Float64[Array, ""] | Float64[Array, "n"],
) -> Float64[Array, "5"] | Float64[Array, "n 5"]:
    """
    Assemble precomputed temperature powers for NASA polynomial evaluation.

    Parameters
    ----------
    T : Float64[Array, ""] | Float64[Array, "n"]
        Temperature [K], scalar or array

    Returns
    -------
    Float64[Array, "5"] | Float64[Array, "n 5"]
        Array containing [1, T, T^2, T^3, T^4]

    Notes
    -----
    This method efficiently computes temperature powers by reusing intermediate
    results (T^2, T^3, T^4) rather than computing each power independently.
    Powers are stacked along the last axis to support both scalar and array inputs.
    """
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T

    return jnp.stack([jnp.ones_like(T), T, T2, T3, T4], axis=-1)


def eval_cp_over_R(coeffs: Float64[Array, "7"], T_powers: Float64[Array, "5"]) -> Float64[Array, ""]:
    """
    Evaluate NASA Cp/R polynomial using precomputed temperature powers.

    Parameters
    ----------
    coeffs : Float64[Array, "7"]
        NASA 7-coefficient array [a1, a2, a3, a4, a5, a6, a7]
    T_powers : Float64[Array, "5"]
        Precomputed temperature powers [1, T, T^2, T^3, T^4]

    Returns
    -------
    Float64[Array, ""]
        Dimensionless heat capacity Cp/R

    Notes
    -----
    The NASA polynomial for dimensionless heat capacity is:

    .. math::
        \\frac{C_p}{R} = a_1 + a_2 T + a_3 T^2 + a_4 T^3 + a_5 T^4

    This is implemented as a dot product with precomputed temperature powers:

    .. math::
        \\frac{C_p}{R} = \\mathbf{a}_{1:5} \\cdot [1, T, T^2, T^3, T^4]^T

    where :math:`\\mathbf{a}_{1:5} = [a_1, a_2, a_3, a_4, a_5]`.
    """
    return jnp.dot(coeffs[:5], T_powers)


def eval_h_over_RT(coeffs: Float64[Array, "7"], T_powers: Float64[Array, "5"]) -> Float64[Array, ""]:
    """
    Evaluate NASA H/(RT) polynomial using precomputed temperature powers.

    Parameters
    ----------
    coeffs : Float64[Array, "7"]
        NASA 7-coefficient array [a1, a2, a3, a4, a5, a6, a7]
    T_powers : Float64[Array, "5"]
        Precomputed temperature powers [1, T, T^2, T^3, T^4]

    Returns
    -------
    Float64[Array, ""]
        Dimensionless enthalpy H/(RT)

    Notes
    -----
    The NASA polynomial for dimensionless enthalpy is obtained by integrating
    the Cp/R polynomial divided by T:

    .. math::
        \\frac{H}{RT} = \\int \\frac{C_p/R}{T} dT = a_1 + \\frac{a_2}{2} T + \\frac{a_3}{3} T^2
                       + \\frac{a_4}{4} T^3 + \\frac{a_5}{5} T^4 + \\frac{a_6}{T}

    where :math:`a_6` is the integration constant.

    This is implemented as a dot product with scaled coefficients:

    .. math::
        \\frac{H}{RT} = (\\mathbf{a}_{1:5} \\odot \\mathbf{m}_h) \\cdot [1, T, T^2, T^3, T^4]^T + \\frac{a_6}{T}

    where :math:`\\odot` denotes element-wise multiplication and
    :math:`\\mathbf{m}_h = [1, 1/2, 1/3, 1/4, 1/5]` is the _h_multipliers vector.
    """
    return jnp.dot(coeffs[:5] * _h_multipliers, T_powers) + coeffs[5] / T_powers[1]


def eval_s_over_R(coeffs: Float64[Array, "7"], T_powers: Float64[Array, "5"]) -> Float64[Array, ""]:
    """
    Evaluate NASA S/R polynomial using precomputed temperature powers.

    Parameters
    ----------
    coeffs : Float64[Array, "7"]
        NASA 7-coefficient array [a1, a2, a3, a4, a5, a6, a7]
    T_powers : Float64[Array, "5"]
        Precomputed temperature powers [1, T, T^2, T^3, T^4]

    Returns
    -------
    Float64[Array, ""]
        Dimensionless entropy S/R

    Notes
    -----
    The NASA polynomial for dimensionless entropy is obtained by integrating
    the Cp/R polynomial divided by T:

    .. math::
        \\frac{S}{R} = \\int \\frac{C_p/R}{T} dT = a_1 \\ln T + a_2 T + \\frac{a_3}{2} T^2
                      + \\frac{a_4}{3} T^3 + \\frac{a_5}{4} T^4 + a_7

    where :math:`a_7` is the integration constant.

    This is implemented as a logarithmic term plus a dot product with scaled coefficients:

    .. math::
        \\frac{S}{R} = a_1 \\ln T + (\\mathbf{a}_{2:5} \\odot \\mathbf{m}_s) \\cdot [T, T^2, T^3, T^4]^T + a_7

    where :math:`\\odot` denotes element-wise multiplication and
    :math:`\\mathbf{m}_s = [1, 1/2, 1/3, 1/4]` is the _s_multipliers[1:] vector.
    Note that the :math:`a_1 \\ln T` term is handled separately from the polynomial
    due to its different functional form.
    """
    return coeffs[0] * jnp.log(T_powers[1]) + jnp.dot(coeffs[1:5] * _s_multipliers[1:], T_powers[1:]) + coeffs[6]


def cp_over_R_honer(coeffs: Float64[Array, "7"], T: Float64[Array, ""]) -> Float64[Array, ""]:
    """
    Evaluate NASA Cp/R polynomial using Horner's method.

    Computes: a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4

    Parameters
    ----------
    coeffs : Float64[Array, "7"]
        NASA 7-coefficient array [a1, a2, a3, a4, a5, a6, a7]
    T : Float64[Array, ""]
        Temperature [K] (scalar)

    Returns
    -------
    Float64[Array, ""]
        Dimensionless heat capacity Cp/R at temperature T

    Notes
    -----
    Uses Horner's method for efficient polynomial evaluation with
    fewer operations and better numerical stability.
    """
    return coeffs[0] + T * (coeffs[1] + T * (coeffs[2] + T * (coeffs[3] + T * coeffs[4])))


def raw_h_over_RT(coeffs: Float64[Array, "7"], T: Float64[Array, ""]) -> Float64[Array, ""]:
    """
    Evaluate NASA H/(RT) enthalpy polynomial.

    Computes: a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T

    Parameters
    ----------
    coeffs : Float64[Array, "7"]
        NASA 7-coefficient array [a1, a2, a3, a4, a5, a6, a7]
    T : Float64[Array, ""]
        Temperature [K] (scalar)

    Returns
    -------
    Float64[Array, ""]
        Dimensionless enthalpy H/(RT) at temperature T

    Notes
    -----
    This polynomial is the integral of the Cp/R polynomial divided by T,
    plus the integration constant a6/T.
    """
    return (
        coeffs[0]
        + coeffs[1] * T / 2.0
        + coeffs[2] * jnp.pow(T, 2) / 3.0
        + coeffs[3] * jnp.pow(T, 3) / 4.0
        + coeffs[4] * jnp.pow(T, 4) / 5.0
        + coeffs[5] / T
    )


def raw_s_over_RT(coeffs: Float64[Array, "7"], T: Float64[Array, ""]) -> Float64[Array, ""]:
    """
    Evaluate NASA S/R entropy polynomial.

    Computes: a1*ln(T) + a2*T + a3*T^2/2 + a4*T^3/3 + a5*T^4/4 + a7

    Parameters
    ----------
    coeffs : Float64[Array, "7"]
        NASA 7-coefficient array [a1, a2, a3, a4, a5, a6, a7]
    T : Float64[Array, ""]
        Temperature [K] (scalar)

    Returns
    -------
    Float64[Array, ""]
        Dimensionless entropy S/R at temperature T

    Notes
    -----
    This polynomial is the integral of the Cp/R polynomial divided by T,
    plus the integration constant a7.
    """
    return (
        coeffs[0] * jnp.log(T)
        + coeffs[1] * T
        + coeffs[2] * jnp.pow(T, 2) / 2.0
        + coeffs[3] * jnp.pow(T, 3) / 3.0
        + coeffs[4] * jnp.pow(T, 4) / 4.0
        + coeffs[6]
    )
