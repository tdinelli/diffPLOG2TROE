from enum import IntEnum
from typing import Union

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float64


class FittingType(IntEnum):
    lindemann = 0
    troe = 1
    sri = 2


def convert_to_fitting_type(fitting_type: str) -> int:
    """Convert string representation to FittingType enum."""
    try:
        return {
            "lindemann": FittingType.lindemann,
            "troe": FittingType.troe,
            "sri": FittingType.sri,
        }[fitting_type.lower()]
    except KeyError:
        available = ", ".join(f"'{k}'" for k in ["lindemann", "troe", "sri"])
        raise ValueError(f"Unknown fitting type '{fitting_type}'. Available types: {available}")


def lindemann(T: Union[Float64, Array], Pr: Union[Float64, Array], parameters: Array) -> Union[Float64, Array]:
    return jnp.ones_like(T, dtype=jnp.float64)


def troe(T: Union[Float64, Array], Pr: Union[Float64, Array], parameters: Array) -> Union[Float64, Array]:
    alpha, T3, T1, T2, _ = parameters

    logFcent = jnp.log10((1 - alpha) * jnp.exp(-T / T3) + alpha * jnp.exp(-T / T1) + jnp.exp(-T2 / T))
    c = -0.4 - 0.67 * logFcent
    n = 0.75 - 1.27 * logFcent
    d = jnp.log10(Pr) + c
    f1 = (d / (n - 0.14 * d)) ** 2

    result = jnp.where(
        Pr > 1.0e-32,  # Edge case as handled by A.C. in OpenSMOKE++
        10.0 ** (logFcent / (1.0 + f1)),  # normal case
        10.0 ** (logFcent / (1.0 + (1.0 / 0.14) ** 2)),  # OpenSMOKE does this Asymptotic value for F when f --> -Inf
    )

    return result


def sri(T: Union[Float64, Array], Pr: Union[Float64, Array], parameters: Array) -> Union[Float64, Array]:
    a, b, c, d, e = parameters

    logPr = jnp.log10(Pr)
    X = 1.0 / (1.0 + logPr * logPr)

    base = a * jnp.exp(-b / T) + jnp.exp(-T / c)
    return d * (base**X) * (T**e)


def validate_troe_parameters(parameters: Array) -> Array:
    """
    Validates Troe falloff parameters and returns a padded array of length 5.

    Parameters:
        parameters: Array of TROE parameters [A, T3, T1] or [A, T3, T1, T2]

    Returns:
        A padded array [A, T3, T1, T2, 0.0] with T2=0 if not provided

    Raises:
        ValueError: If parameters have incorrect shape or invalid values
    """
    param_count = parameters.shape[0]

    def three_params(p):
        A, T3, T1 = p
        return jnp.array([A, T3, T1, 0.0, 0.0])

    def four_params(p):
        A, T3, T1, T2 = p
        return jnp.array([A, T3, T1, T2, 0.0])

    valid_shape = (param_count == 3) | (param_count == 4)
    if not valid_shape:
        raise ValueError(f"Invalid number of TROE parameters: expected 3 or 4, but received {param_count}.")

    result = lax.cond(param_count == 3, three_params, four_params, parameters)

    A, T3, T1, T2 = result[:4]

    if A <= 0 or A > 1:
        raise ValueError(f"Parameter A (={A}) is out of valid range: must satisfy 0 < A ≤ 1.")

    if T3 <= 0:
        raise ValueError(f"Parameter T3 (={T3}) must be positive: T3 > 0.")

    if T1 <= 0:
        raise ValueError(f"Parameter T1 (={T1}) must be positive: T1 > 0.")

    if T2 < 0:
        raise ValueError(f"Parameter T2 (={T2}) cannot be negative: T2 ≥ 0.")

    return result


def validate_sri_parameters(parameters: Array) -> Array:
    """
    Validates SRI parameters and returns a padded array of length 5.

    Parameters:
        parameters: Array of SRI parameters [a, b, c] or [a, b, c, d, e]

    Returns:
        A padded array [a, b, c, d, e] with d=1 and e=0 if not provided

    Raises:
        ValueError: If parameters have incorrect shape or invalid values
    """
    param_count = parameters.shape[0]

    def three_params(p):
        a, b, c = p
        return jnp.array([a, b, c, 1.0, 0.0])

    def five_params(p):
        a, b, c, d, e = p
        return jnp.array([a, b, c, d, e])

    valid_shape = (param_count == 3) | (param_count == 5)
    if not valid_shape:
        raise ValueError(f"Invalid number of SRI parameters: expected 3 or 5, but received {param_count}.")

    result = lax.cond(param_count == 3, three_params, five_params, parameters)

    a, b, c, d, e = result

    if c == 0:
        raise ValueError(f"Parameter c (={c}) must be different from 0.")

    if d == 0:
        raise ValueError(f"Parameter d (={d}) must be different from 0.")

    return result
