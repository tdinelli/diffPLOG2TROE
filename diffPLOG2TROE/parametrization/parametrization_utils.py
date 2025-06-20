import warnings
from enum import IntEnum
from typing import Dict, Optional, Union

import jax.numpy as jnp
from jaxtyping import Float64


class BroadeningFunctionType(IntEnum):
    lindemann = 0
    troe = 1
    sri = 2
    tsang = 3


def _convert_to_broadening_type(broadening_type: str) -> int:
    """Convert string representation to BroadeningFunctionType enum."""
    try:
        return {
            "lindemann": BroadeningFunctionType.lindemann,
            "troe": BroadeningFunctionType.troe,
            "sri": BroadeningFunctionType.sri,
            "tsang": BroadeningFunctionType.tsang,
        }[broadening_type.lower()]
    except KeyError:
        available = ", ".join(f"'{k}'" for k in ["lindemann", "troe", "sri", "tsang"])
        raise ValueError(f"Unknown broadening function type '{broadening_type}'. Available types are: {available}")


def validate_broadening_parameters(
    broadening_type: str,
    parameters: Optional[Dict[str, Float64]] = None,
) -> Union[None, Dict[str, Float64]]:
    broadening_type_int = _convert_to_broadening_type(broadening_type)

    if broadening_type_int == 0:
        pass

    if broadening_type_int == 1 and parameters is not None:
        _validate_troe_parameters(parameters)

    if broadening_type_int == 2 and parameters is not None:
        _validate_sri_parameters(parameters)

    if broadening_type_int == 3 and parameters is not None:
        _validate_tsang_parameters(parameters)

    return parameters


def _validate_troe_parameters(parameters: Dict[str, Float64]) -> None:
    required_keys = {"A", "T3", "T1", "T2"}
    missing_keys = required_keys - parameters.keys()
    if missing_keys:
        raise ValueError(f"Missing required parameters: {missing_keys}")

    A = parameters["A"]
    T3 = parameters["T3"]
    T1 = parameters["T1"]
    T2 = parameters["T2"]

    if A <= 0 or A > 1:
        raise ValueError(f"Parameter A (={A}) is out of valid range: must satisfy 0 < A ≤ 1.")

    if T3 <= 0:
        raise ValueError(f"Parameter T3 (={T3}) must be positive: T3 > 0.")

    if T1 <= 0:
        raise ValueError(f"Parameter T1 (={T1}) must be positive: T1 > 0.")

    if T2 < 0:
        raise ValueError(f"Parameter T2 (={T2}) cannot be negative: T2 ≥ 0.")


def _validate_sri_parameters(parameters: Dict[str, Float64]) -> None:
    """ """

    required_keys = {"a", "b", "c", "d", "e"}
    missing_keys = required_keys - parameters.keys()
    if missing_keys:
        raise ValueError(f"Missing required parameters: {missing_keys}")

    a = parameters["a"]
    b = parameters["b"]
    c = parameters["c"]
    d = parameters["d"]
    e = parameters["e"]

    if c == 0:
        raise ValueError(f"Parameter c (={c}) must be different from 0.")

    if d == 0:
        raise ValueError(f"Parameter d (={d}) must be different from 0.")


def _validate_tsang_parameters(parameters: Dict[str, Float64]) -> None:
    """ """

    required_keys = {"A", "B"}
    missing_keys = required_keys - parameters.keys()
    if missing_keys:
        raise ValueError(f"Missing required parameters: {missing_keys}")

    A = parameters["A"]
    B = parameters["B"]

    if A == 0:
        raise ValueError(f"Parameter A (={A}) must be different from 0.")

    if B == 0:
        raise ValueError(f"Parameter B (={B}) must be different from 0.")


def validate_arrhenius_parameters(parameters: Dict[str, Float64]) -> None:
    """
    Validate Arrhenius parameters to ensure they are physically meaningful and
    computationally stable.

    This function performs comprehensive validation of the three Arrhenius parameters:
    pre-exponential factor (A), temperature exponent (n), and activation energy (Ea).
    It checks for mathematical validity, physical reasonableness, and computational
    stability.

    Parameters
    ----------
    parameters : Dict[str, Float64]
        Dictionary containing Arrhenius parameters with keys:
        - "A" : Pre-exponential factor, must be positive
        - "n" : Temperature exponent, typically in range [-2, 4]
        - "Ea" : Activation energy in cal/mol

    Raises
    ------
    ValueError
        If any parameter is missing, not finite, or outside reasonable bounds.
        Specific conditions checked:
        - Missing required keys ("A", "n", "Ea")
        - Any parameter is infinite or NaN
    """
    # ==============================================================================
    # Check for required keys
    required_keys = {"A", "n", "Ea"}
    missing_keys = required_keys - parameters.keys()
    if missing_keys:
        raise ValueError(f"Missing required parameters: {missing_keys}")

    # ==============================================================================
    # Extract parameters for validation
    A = parameters["A"]
    n = parameters["n"]
    Ea = parameters["Ea"]

    # ==============================================================================
    # Validate pre-exponential factor
    if A <= 0:
        warnings.warn(f"Pre-exponential factor A is usually positive be careful, got {A}", UserWarning, stacklevel=2)

    if not jnp.isfinite(A):
        raise ValueError(f"Pre-exponential factor A must be finite, got {A}")

    # ==============================================================================
    # Validate temperature exponent
    if not jnp.isfinite(n):
        raise ValueError(f"Temperature exponent n must be finite, got {n}")

    # ==============================================================================
    if abs(n) > 10:
        warnings.warn(
            f"Temperature exponent n = {n} is unusually large. "
            f"Typical values are in range [-2, 4]. Please verify your input.",
            UserWarning,
            stacklevel=2,
        )

    # ==============================================================================
    # Validate activation energy
    if not jnp.isfinite(Ea):
        raise ValueError(f"Activation energy Ea must be finite, got {Ea}")


def validate_efficiencies(efficiencies: Dict[str, Float64]) -> None:
    for species, efficiency in efficiencies.items():
        if efficiency < 0:
            raise ValueError(f"Collision efficiency must be positive. {species} given {efficiency}")
