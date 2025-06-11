from enum import IntEnum
from typing import Dict

import jax.numpy as jnp
from jaxtyping import Array, Float64

from ..utilities.custom_types import Array64f, Array64f_3, Array64f_5


def validate_troe_parameters(parameters: Array64f) -> Array64f_5:
    """
    Validates Troe falloff parameters and returns a padded array of length 5.

    Parameters:
        parameters: Array of TROE parameters [A, T3, T1] or [A, T3, T1, T2]

    Returns:
        A padded array [A, T3, T1, T2, 0.0] with T2=0 if not provided

    Raises:
        ValueError: If parameters have incorrect shape or invalid values
    """

    def three_params(p: Float64[Array, "3"]) -> Array64f_5:
        A, T3, T1 = p
        return jnp.array([A, T3, T1, 0.0, 0.0])

    def four_params(p: Float64[Array, "4"]) -> Array64f_5:
        A, T3, T1, T2 = p
        return jnp.array([A, T3, T1, T2, 0.0])

    number_of_parameters = len(parameters)
    valid_shape = (number_of_parameters == 3) | (number_of_parameters == 4)
    if not valid_shape:
        raise ValueError(f"Invalid number of TROE parameters: expected 3 or 4, but received {number_of_parameters}.")

    result = three_params(parameters) if number_of_parameters == 3 else four_params(parameters)

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


def validate_sri_parameters(parameters: Array64f) -> Array64f_5:
    """
    Validates SRI parameters and returns a padded array of length 5.

    Parameters:
        parameters: Array of SRI parameters [a, b, c] or [a, b, c, d, e]

    Returns:
        A padded array [a, b, c, d, e] with d=1 and e=0 if not provided

    Raises:
        ValueError: If parameters have incorrect shape or invalid values
    """

    def three_params(p: Float64[Array, "3"]) -> Array64f_5:
        a, b, c = p
        return jnp.array([a, b, c, 1.0, 0.0])

    def five_params(p: Float64[Array, "5"]) -> Array64f_5:
        a, b, c, d, e = p
        return jnp.array([a, b, c, d, e])

    number_of_parameters = len(parameters)
    valid_shape = (number_of_parameters == 3) | (number_of_parameters == 5)
    if not valid_shape:
        raise ValueError(f"Invalid number of SRI parameters: expected 3 or 5, but received {number_of_parameters}.")

    result = three_params(parameters) if number_of_parameters == 3 else five_params(parameters)

    a, b, c, d, e = result

    if c == 0:
        raise ValueError(f"Parameter c (={c}) must be different from 0.")

    if d == 0:
        raise ValueError(f"Parameter d (={d}) must be different from 0.")

    return result


def validate_tsang_parameters(parameters: Float64[Array, "2"]) -> Array64f_5:
    """
    Validates TSANG parameters and returns a padded array of length 5.

    Parameters:
        parameters: Array of SRI parameters [A, B]

    Returns:
        A padded array [A, B, 0, 0, 0]

    Raises:
        ValueError: If parameters have incorrect shape or invalid values
    """

    number_of_parameters = len(parameters)
    if number_of_parameters != 2:
        raise ValueError(f"Invalid number of TSANG parameters: expected 2, but received {number_of_parameters}.")

    A, B = parameters

    if A == 0:
        raise ValueError(f"Parameter A (={A}) must be different from 0.")

    if B == 0:
        raise ValueError(f"Parameter B (={B}) must be different from 0.")

    return jnp.array([A, B, 0, 0, 0], dtype=jnp.float64)


def validate_efficiencies(efficiencies: Dict[str, Float64]) -> None:
    for species, efficiency in efficiencies.items():
        if efficiency < 0:
            raise ValueError(f"Collision efficiency must be positive. {species} given {efficiency}")


class FittingType(IntEnum):
    lindemann = 0
    troe = 1
    sri = 2
    tsang = 3


def convert_to_fitting_type(fitting_type: str) -> int:
    """Convert string representation to FittingType enum."""
    try:
        return {
            "lindemann": FittingType.lindemann,
            "troe": FittingType.troe,
            "sri": FittingType.sri,
            "tsang": FittingType.tsang,
        }[fitting_type.lower()]
    except KeyError:
        available = ", ".join(f"'{k}'" for k in ["lindemann", "troe", "sri", "tsang"])
        raise ValueError(f"Unknown fitting type '{fitting_type}'. Available types: {available}")


def validate_arrhenius_parameters(parameters: Array64f_3) -> None:
    """
    Validate Arrhenius parameters to ensure consistency in calculations.

    Parameters
    ----------
    parameters : Float64[Array, "3"]
        Array of [A, n, Ea] Arrhenius parameters.

    Raises
    ------
    ValueError
        If parameters are invalid.
    """
    A, n, Ea = parameters
    if A <= 0:
        raise ValueError("Pre-exponential factor must be positive")
    if not jnp.isfinite(A):
        raise ValueError("Pre-exponential factor must be finite")
    if not jnp.isfinite(n):
        raise ValueError("Temperature exponent must be finite")
    if not jnp.isfinite(Ea):
        raise ValueError("Activation energy must be finite")
