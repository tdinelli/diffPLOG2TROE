from typing import Dict, List, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array

from .rate_utils import (
    FittingType,
    create_falloff_array,
    parse_falloff_params,
    parse_fitting_params,
)


def parse_rate_constant(rate_constant: dict) -> Union[Array, Tuple[Array, int]]:
    """Parse and convert chemical reaction rate constants to JAX arrays.

    This function serves as a dispatcher for different types of chemical reaction
    rate constants, routing them to appropriate parsing functions based on their type.

    Args:
        rate_constant (dict): Dictionary containing rate constant information with keys:
            - type (str): Type of reaction ("Arrhenius", "PLOG", "FallOff", "CABR", "3Body")
            - parameters: Rate constant parameters (format depends on reaction type)
            - fitting_type (str, optional): For FallOff/CABR reactions, specifies the
              fitting model ("Lindemann", "TROE", "SRI")

    Returns:
        Union[Array, Tuple[jnp.ndarray, int]]: Either:
            - Array: For Arrhenius, PLOG, and 3Body reactions
            - Tuple[Array, int]: For FallOff and CABR reactions, includes
              the fitting type as an integer

    Raises:
        ValueError: If reaction type is unsupported or not implemented yet

    Examples:
        >>> # Arrhenius reaction
        >>> params = {"type": "Arrhenius", "parameters": [1e13, 0, 50000]}
        >>> arr = parse_rate_constant(params)

        >>> # Falloff reaction
        >>> params = {
        ...     "type": "FallOff",
        ...     "fitting_type": "Lindemann",
        ...     "parameters": {"k0": [1e14, 0, 0], "kInf": [2e13, 0, 400]}
        ... }
        >>> arr, fit_type = parse_rate_constant(params)
    """
    reaction_type = rate_constant["type"]

    parsers = {
        "Arrhenius": lambda x: parse_arrhenius_parameters(x["parameters"]),
        "PLOG": lambda x: parse_plog_parameters(x["parameters"]),
        "FallOff": parse_falloff,
        "CABR": parse_cabr,
        "3Body": parse_threebody,
    }

    unsupported_types = {
        "RPBR": "Reduced Pressure Based",
        "Extended-PLOG": "Extended PLOG",
        "Extended-FallOff": "Extended FallOff",
        "Chebyshev": "Chebyshev",
    }

    if reaction_type in unsupported_types:
        raise ValueError(
            f"{unsupported_types[reaction_type]} reaction type not supported yet!"
        )

    if reaction_type in parsers:
        return parsers[reaction_type](rate_constant)

    raise ValueError(f"Unsupported reaction type: {reaction_type}")


def parse_arrhenius_parameters(parameters: List[float]) -> Array:
    """Convert Arrhenius rate parameters to a JAX array.

    Args:
        parameters (List[float]): List of three Arrhenius parameters:
            - A: Pre-exponential factor
            - n: Temperature exponent
            - E: Activation energy

    Returns:
        Array: JAX array containing [A, n, E]

    Raises:
        ValueError: If parameters list doesn't contain exactly 3 elements
    """
    if len(parameters) != 3:
        raise ValueError(
            f"Arrhenius must have exactly 3 parameters [A, n, E], got {len(parameters)}"
        )
    return jnp.array(parameters, dtype=jnp.float64)


def parse_plog_parameters(parameters: List[List[float]]) -> Array:
    """Convert pressure-dependent logarithmic (PLOG) parameters to a JAX array.

    Args:
        parameters (List[List[float]]): List of parameter sets, each containing:
            - P: Pressure (atm)
            - A: Pre-exponential factor
            - n: Temperature exponent
            - E: Activation energy

    Returns:
        Array: 2D JAX array where each row contains [P, A, n, E]
    """
    return jnp.array(parameters, dtype=jnp.float64)


def parse_falloff(rate_constant: Dict) -> Tuple[Array, int]:
    """Parse falloff reaction rate constants.

    Handles both simple Lindemann falloff and more complex fitting models (TROE, SRI).

    Args:
        rate_constant (Dict): Dictionary containing:
            - parameters: Dict with k0 (low-pressure) and kInf (high-pressure) parameters
            - fitting_type: Fitting model ("Lindemann", "TROE", "SRI")
            - fitting_parameters: Additional parameters for TROE/SRI models

    Returns:
        Tuple[Array, int]: Tuple containing:
            - JAX array of formatted parameters
            - Integer identifying the fitting type
    """
    k0, kInf = parse_falloff_params(rate_constant["parameters"])
    fitting_type = rate_constant["fitting_type"]

    if fitting_type == "Lindemann":
        return create_falloff_array(kInf, k0), FittingType.LINDEMANN.value

    params, type_value = parse_fitting_params(rate_constant, fitting_type)
    return create_falloff_array(kInf, k0, params), type_value


def parse_cabr(rate_constant: Dict) -> Tuple[Array, int]:
    """Parse Chemically Activated Bimolecular Reaction (CABR) rate constants.

    Similar to falloff reactions but with reversed order of k0 and kInf.

    Args:
        rate_constant (Dict): Dictionary containing:
            - parameters: Dict with k0 (low-pressure) and kInf (high-pressure) parameters
            - fitting_type: Fitting model ("Lindemann", "TROE", "SRI")
            - fitting_parameters: Additional parameters for TROE/SRI models

    Returns:
        Tuple[Array, int]: Tuple containing:
            - JAX array of formatted parameters
            - Integer identifying the fitting type
    """
    k0, kInf = parse_falloff_params(rate_constant["parameters"])
    fitting_type = rate_constant["fitting_type"]

    if fitting_type == "Lindemann":
        return (
            create_falloff_array(kInf, k0, swap_order=True),
            FittingType.LINDEMANN.value,
        )

    params, type_value = parse_fitting_params(rate_constant, fitting_type)
    return create_falloff_array(kInf, k0, params, swap_order=True), type_value


def parse_threebody(rate_constant: Dict) -> Array:
    """Convert three-body reaction parameters to a JAX array.

    Args:
        rate_constant (Dict): Dictionary containing:
            - parameters: List of rate parameters [A, n, E]

    Returns:
        Array: JAX array containing the rate parameters
    """
    return jnp.array(rate_constant["parameters"], dtype=jnp.float64)
