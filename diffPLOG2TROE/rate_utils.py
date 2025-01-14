from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
from jaxtyping import Array


class FittingType(IntEnum):
    """Enumeration of supported falloff reaction fitting types.

    Attributes:
        LINDEMANN (int): Simple Lindemann falloff model
        TROE (int): Troe falloff model with temperature-dependent center broadening
        SRI (int): SRI falloff model with additional pressure-dependent terms
    """

    LINDEMANN = 0
    TROE = 1
    SRI = 2


def parse_falloff_params(
    parameters: Dict, required_params: Optional[List[str]] = None
) -> Tuple[List[float], List[float]]:
    """Parse and validate falloff reaction parameters.

    Extracts and validates the low-pressure (k0) and high-pressure (kInf) rate constants
    from the parameters dictionary. Each rate constant must contain exactly three
    Arrhenius parameters [A, n, E].

    Args:
        parameters (Dict): Dictionary containing falloff parameters
        required_params (List[str], optional): List of required parameter keys.
            Defaults to ["k0", "kInf"].

    Returns:
        Tuple[List[float], List[float]]: A tuple containing:
            - k0: Low-pressure limit Arrhenius parameters [A, n, E]
            - kInf: High-pressure limit Arrhenius parameters [A, n, E]

    Raises:
        KeyError: If any required parameters are missing
        ValueError: If k0 or kInf don't contain exactly 3 Arrhenius parameters
    """
    required_params = required_params or ["k0", "kInf"]
    missing_params = set(required_params) - set(parameters.keys())
    if missing_params:
        raise KeyError(f"Missing required parameters: {missing_params}")

    k0 = parameters["k0"]
    kInf = parameters["kInf"]

    for param_name, param_list in [("k0", k0), ("kInf", kInf)]:
        if len(param_list) != 3:
            raise ValueError(
                f"{param_name} must have exactly 3 parameters [A, n, E], got {len(param_list)}"
            )

    return k0, kInf


def parse_fitting_params(
    rate_constant: Dict, fitting_type: str
) -> Tuple[List[float], int]:
    """Parse and validate fitting parameters for TROE or SRI falloff reactions.

    Args:
        rate_constant (Dict): Dictionary containing reaction rate constant information
        fitting_type (str): Type of fitting model ("TROE" or "SRI")

    Returns:
        Tuple[List[float], int]: A tuple containing:
            - List of fitting parameters
            - Integer value corresponding to the fitting type

    Raises:
        KeyError: If fitting_parameters are missing from the rate constant
        ValueError: If the number of parameters doesn't match the requirements:
            - TROE: 3 or 4 parameters
            - SRI: 4 or 5 parameters
    """
    if "fitting_parameters" not in rate_constant:
        raise KeyError(f"{fitting_type} fitting requires 'fitting_parameters'")

    params = rate_constant["fitting_parameters"]
    valid_lengths = {"TROE": {3, 4}, "SRI": {4, 5}}

    if len(params) not in valid_lengths[fitting_type]:
        raise ValueError(
            f"{fitting_type} formalism requires {valid_lengths[fitting_type]} parameters, got {len(params)}"
        )

    return params, FittingType[fitting_type].value


def create_falloff_array(
    kInf: List[float],
    k0: List[float],
    params: Optional[List[float]] = None,
    swap_order: bool = False,
) -> Array:
    """Create a JAX array containing falloff reaction parameters.
    Combines high-pressure limit (kInf), low-pressure limit (k0), and optional
    fitting parameters into a structured array format suitable for rate calculations.

    Args:
        kInf (List[float]): High-pressure limit Arrhenius parameters [A, n, E]
        k0 (List[float]): Low-pressure limit Arrhenius parameters [A, n, E]
        params (Optional[List[float]], optional): Additional fitting parameters for
            TROE or SRI models. Defaults to None.
        swap_order (bool, optional): If True, switches the order of k0 and kInf
            in the output array. Used for CABR reactions. Defaults to False.

    Returns:
        jnp.ndarray: A JAX array containing the formatted parameters:
            - If params is None: 2x3 array of [kInf, k0] or [k0, kInf]
            - If params provided: 3xN array with padded rows for kInf, k0,
              and fitting parameters
    """
    if params is None:
        return jnp.array([kInf, k0] if not swap_order else [k0, kInf], dtype=jnp.float64)

    pad_length = len(params) - 3
    k_rows = [k0, kInf] if swap_order else [kInf, k0]
    return jnp.array(
        [row + [0.0] * pad_length for row in k_rows] + [params], dtype=jnp.float64
    )
