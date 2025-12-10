import warnings
from typing import Optional

import jax.numpy as jnp


def validate_broadening_parameters(
    broadening_type: str,
    parameters: Optional[dict[str, float]] = None,
) -> None:
    """Static validation function that raises errors for invalid parameters."""
    if broadening_type == "troe" and parameters is not None:
        required_keys = {"A", "T3", "T1", "T2"}
        missing_keys = required_keys - parameters.keys()
        if missing_keys:
            raise ValueError(f"Missing required parameters: {missing_keys}")

        A, T3, T1, T2 = parameters["A"], parameters["T3"], parameters["T1"], parameters["T2"]
        if A <= 0 or A > 1:
            raise ValueError(f"Parameter A (={A}) is out of valid range: must satisfy 0 < A ≤ 1.")
        if T3 <= 0:
            raise ValueError(f"Parameter T3 (={T3}) must be positive: T3 > 0.")
        if T1 <= 0:
            raise ValueError(f"Parameter T1 (={T1}) must be positive: T1 > 0.")
        if T2 < 0:
            raise ValueError(f"Parameter T2 (={T2}) cannot be negative: T2 ≥ 0.")

    elif broadening_type == "sri" and parameters is not None:
        required_keys = {"a", "b", "c", "d", "e"}
        missing_keys = required_keys - parameters.keys()
        if missing_keys:
            raise ValueError(f"Missing required parameters: {missing_keys}")

        c, d = parameters["c"], parameters["d"]
        if c == 0:
            raise ValueError(f"Parameter c (={c}) must be different from 0.")
        if d == 0:
            raise ValueError(f"Parameter d (={d}) must be different from 0.")

    elif broadening_type == "tsang" and parameters is not None:
        required_keys = {"A", "B"}
        missing_keys = required_keys - parameters.keys()
        if missing_keys:
            raise ValueError(f"Missing required parameters: {missing_keys}")

        A, B = parameters["A"], parameters["B"]
        if A == 0:
            raise ValueError(f"Parameter A (={A}) must be different from 0.")
        if B == 0:
            raise ValueError(f"Parameter B (={B}) must be different from 0.")


def validate_arrhenius_parameters(parameters: dict[str, float]) -> None:
    # Check for required keys
    required_keys = {"A", "n", "Ea"}
    missing_keys = required_keys - parameters.keys()
    if missing_keys:
        raise ValueError(f"Missing required parameters: {missing_keys}")

    # Extract parameters for validation
    A = parameters["A"]
    n = parameters["n"]
    Ea = parameters["Ea"]

    # Validate pre-exponential factor
    if A <= 0:
        warnings.warn(
            f"Pre-exponential factor A is usually positive, be careful, got {A}",
            UserWarning,
            stacklevel=2,
        )

    if not jnp.isfinite(A):
        raise ValueError(f"Pre-exponential factor A must be finite, got {A}")

    # Validate temperature exponent
    if not jnp.isfinite(n):
        raise ValueError(f"Temperature exponent n must be finite, got {n}")

    if abs(n) > 5:
        warnings.warn(
            f"Temperature exponent n = {n} is unusually large. "
            f"Typical values are in range [-5, 5].",
            UserWarning,
            stacklevel=2,
        )

    # Validate activation energy
    # if Ea < 0:
    #     warnings.warn(
    #         "Activation energy is usually either 0 or positive. Negative "
    #         "activation energy is the result of a non constrained fitting procedure.",
    #         UserWarning,
    #         stacklevel=2,
    #     )

    if not jnp.isfinite(Ea):
        raise ValueError(f"Activation energy Ea must be finite, got {Ea}")


def validate_efficiencies(efficiencies: dict[str, float]) -> None:
    for species, efficiency in efficiencies.items():
        if efficiency < 0.0:
            raise ValueError(f"Collision efficiency must be positive. {species} given {efficiency}")
