"""
Copyright (c) 2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import warnings
from typing import Optional

import jax.numpy as jnp


def validate_broadening_parameters(
    broadening_type: str,
    parameters: Optional[dict[str, float]] = None,
) -> None:
    """
    Validate broadening factor parameters for fall-off and CABR reactions.

    This function checks that all required parameters are present and satisfy
    physical constraints specific to each broadening factor type (Troe, SRI, Tsang).
    Lindemann formulation requires no parameters and is not validated.

    Parameters
    ----------
    broadening_type : str
        Type of broadening factor: "troe", "sri", "tsang", or "lindemann"
    parameters : dict[str, float], optional
        Dictionary of broadening parameters (type-dependent)

    Raises
    ------
    ValueError
        If required parameters are missing or have invalid values:

        - **Troe**: Requires {"A", "T3", "T1", "T2"}
            - A: must be in range (0, 1]
            - T3, T1: must be positive (> 0)
            - T2: must be non-negative (>= 0)

        - **SRI**: Requires {"a", "b", "c", "d", "e"}
            - c: must be non-zero (appears in denominator)
            - d: must be non-zero (multiplicative factor)

        - **Tsang**: Requires {"A", "B"}
            - A: must be non-zero (appears in F_cent = A + B·T)
            - B: must be non-zero (temperature coefficient)
    """
    if broadening_type == "troe" and parameters is not None:
        # Troe Broadening Factor Validation
        # Check all required parameters are present
        required_keys = {"A", "T3", "T1", "T2"}
        missing_keys = required_keys - parameters.keys()
        if missing_keys:
            raise ValueError(f"Missing required Troe parameters: {missing_keys}")

        # Extract parameters for validation
        A, T3, T1, T2 = parameters["A"], parameters["T3"], parameters["T1"], parameters["T2"]

        # Validate alpha parameter: must be in range (0, 1]
        if A <= 0 or A > 1:
            raise ValueError(f"Troe parameter A (α={A}) is out of valid range: must satisfy 0 < A ≤ 1.")

        # Validate T3: must be positive (temperature scale for first exponential)
        if T3 <= 0:
            raise ValueError(f"Troe parameter T3 (={T3} K) must be positive: T3 > 0.")

        # Validate T1: must be positive (temperature scale for second exponential)
        if T1 <= 0:
            raise ValueError(f"Troe parameter T1 (={T1} K) must be positive: T1 > 0.")

        # Validate T2: can be 0 (3-parameter form) or positive (4-parameter form)
        if T2 < 0:
            raise ValueError(f"Troe parameter T2 (={T2} K) cannot be negative: T2 >= 0.")

    elif broadening_type == "sri" and parameters is not None:
        # SRI Broadening Factor Validation
        # Check all required parameters are present
        required_keys = {"a", "b", "c", "d", "e"}
        missing_keys = required_keys - parameters.keys()
        if missing_keys:
            raise ValueError(f"Missing required SRI parameters: {missing_keys}")

        # Extract critical parameters for validation
        c, d = parameters["c"], parameters["d"]

        # Validate c: must be non-zero (appears in exp(-T/c) term)
        if c == 0:
            raise ValueError(f"SRI parameter c (={c}) must be different from 0 (division by zero).")

        # Validate d: must be non-zero (multiplicative scaling factor)
        if d == 0:
            raise ValueError(f"SRI parameter d (={d}) must be different from 0 (F would be zero).")

    elif broadening_type == "tsang" and parameters is not None:
        # Tsang Broadening Factor Validation
        # Check all required parameters are present
        required_keys = {"A", "B"}
        missing_keys = required_keys - parameters.keys()
        if missing_keys:
            raise ValueError(f"Missing required Tsang parameters: {missing_keys}")

        # Extract parameters for validation
        A, B = parameters["A"], parameters["B"]

        # Validate A: must be non-zero (intercept of F_cent = A + B·T)
        if A == 0:
            raise ValueError(f"Tsang parameter A (={A}) must be different from 0.")

        # Validate B: must be non-zero (temperature coefficient of F_cent)
        if B == 0:
            raise ValueError(f"Tsang parameter B (={B}) must be different from 0.")


def validate_arrhenius_parameters(parameters: dict[str, float]) -> None:
    """
    Validate Arrhenius rate constant parameters (A, n, Ea).

    This function checks that all required Arrhenius parameters are present,
    have finite values, and fall within typical ranges. It raises errors for
    invalid values and issues warnings for unusual but technically valid parameters.

    Parameters
    ----------
    parameters : dict[str, float]
        Dictionary containing Arrhenius parameters:
        {"A": pre-exponential factor, "n": temperature exponent, "Ea": activation energy}

    Raises
    ------
    ValueError
        If required parameters are missing or have invalid (non-finite) values:

        - Missing parameters: Must include "A", "n", and "Ea"
        - A: Must be finite (not NaN or :math:`\\pm \\infty`)
        - n: Must be finite
        - Ea: Must be finite

    Warns
    -----
    UserWarning
        For unusual but valid parameter values:

        - A =< 0: Pre-exponential factors are typically positive
        - |n| > 5: Temperature exponents are typically in range [-5, 5]
    """
    # Completeness Check: Ensure all required parameters are present
    required_keys = {"A", "n", "Ea"}
    missing_keys = required_keys - parameters.keys()
    if missing_keys:
        raise ValueError(f"Missing required Arrhenius parameters: {missing_keys}")

    # Extract parameters for validation
    A = parameters["A"]
    n = parameters["n"]
    Ea = parameters["Ea"]

    # Validate Pre-Exponential Factor (A)
    # Check if A is positive (typical for most reactions)
    # Negative A would result in negative rate constants (unphysical)
    if A <= 0:
        warnings.warn(
            f"Pre-exponential factor A (={A}) is usually positive. "
            "Negative or zero A will result in negative or zero rate constants. "
            "Please verify your input.",
            UserWarning,
            stacklevel=2,
        )

    # Check if A is finite (not NaN or ±infinity)
    if not jnp.isfinite(A):
        raise ValueError(f"Pre-exponential factor A must be finite, got {A}")

    # Validate Temperature Exponent (n)
    # Check if n is finite
    if not jnp.isfinite(n):
        raise ValueError(f"Temperature exponent n must be finite, got {n}")

    # Check if n is within typical range
    # Most reactions have |n| < 5; larger values may indicate fitting issues
    if abs(n) > 5:
        warnings.warn(
            f"Temperature exponent n = {n} is unusually large. "
            "Typical values are in range [-5, 5]. Large |n| may indicate "
            "over-parameterization or fitting to a limited temperature range.",
            UserWarning,
            stacklevel=2,
        )

    # Validate Activation Energy (Ea)
    # Note: Negative Ea validation is intentionally commented out
    # Reason: Negative Ea can occur from fitting procedures and represents
    # reactions where rate decreases with temperature (barrierless capture)
    #
    # if Ea < 0:
    #     warnings.warn(
    #         f"Activation energy Ea (={Ea} cal/mol) is negative. "
    #         "While physically possible for some reactions (e.g., barrierless), "
    #         "negative Ea often results from unconstrained fitting. "
    #         "Please verify this is intentional.",
    #         UserWarning,
    #         stacklevel=2,
    #     )

    # Check if Ea is finite
    if not jnp.isfinite(Ea):
        raise ValueError(f"Activation energy Ea must be finite, got {Ea}")


def validate_efficiencies(efficiencies: dict[str, float]) -> None:
    """
    Validate third-body collision efficiencies for pressure-dependent reactions.

    This function checks that all collision efficiency values are non-negative.
    Collision efficiencies represent the relative effectiveness of different
    species in stabilizing activated complexes in fall-off and CABR reactions.

    Parameters
    ----------
    efficiencies : dict[str, float]
        Dictionary mapping species names to collision efficiency values.
        Keys are species names (e.g., "AR", "H2O", "N2")
        Values are dimensionless efficiency factors (typically 0.1 to 20)

    Raises
    ------
    ValueError
        If any efficiency value is negative (unphysical)
    """
    # Validate Each Efficiency Value
    for species, efficiency in efficiencies.items():
        # Check for non-negative efficiency
        # Negative efficiency is unphysical (would reduce effective [M])
        if efficiency < 0.0:
            raise ValueError(
                f"Collision efficiency must be non-negative (eff >= 0). "
                f"Species '{species}' has invalid efficiency: eff = {efficiency}"
            )


def validate_chebyshev_parameters(
    coefficients: jnp.ndarray,
    T_limits: tuple[float, float],
    P_limits: tuple[float, float],
) -> None:
    """
    Validate Chebyshev polynomial parameters for pressure-dependent rate constants.

    This function checks that coefficient matrix dimensions are valid, all values
    are finite, and temperature/pressure ranges are physically meaningful. Chebyshev
    polynomials provide a compact representation of rate constants over a T-P grid.

    Parameters
    ----------
    coefficients : jnp.ndarray
        2D array of Chebyshev polynomial coefficients with shape (N_T, N_P)
        where N_T is the order in temperature and N_P is the order in pressure
    T_limits : tuple[float, float]
        Temperature range (T_min, T_max) in Kelvin
    P_limits : tuple[float, float]
        Pressure range (P_min, P_max) in bar

    Raises
    ------
    ValueError
        If parameters violate physical or mathematical constraints:

        - **Coefficient array**:
            - Must be 2D
            - Both dimensions must be >= 1 (at least constant term)
            - All values must be finite (no NaN or :math:`\\pm \\infty`)

        - **Temperature limits**:
            - T_min must be positive (> 0 K)
            - T_max must be greater than T_min
            - Both must be finite

        - **Pressure limits**:
            - P_min must be positive (> 0 bar)
            - P_max must be greater than P_min
            - Both must be finite

    Notes
    -----
    **Chebyshev Polynomial Representation:**

    Rate constants are represented as a bivariate Chebyshev expansion:

    .. math::
        \\log_{10} k(T, P) = \\sum_{t=0}^{N_T-1} \\sum_{p=0}^{N_P-1} \\alpha_{tp} \\phi_t(\\tilde{T}) \\phi_p(\\tilde{P})

    where:
        - :math:`\\alpha_{t, p} are the Chebyshev coefficients
        - :math:`\\phi_n (x) = cos(n \\cdot arccos(x))` are Chebyshev polynomials of the first kind
        - :math:`\\tilde{T}` and :math:`\\tilde{P}` are reduced temperature and pressure mapped to [-1, 1]

    **Parameter Requirements:**

    - **Minimum order**: N_T >= 1, N_P >= 1 (at least constant term)
    - **Typical orders**: N_T = 4-8, N_P = 3-5 (balance accuracy vs. cost)
    - **Coefficient magnitude**: No strict bounds, but :math:`|\\alpha| > 100`
        may indicate issues

    **Extrapolation Warning:**

    Chebyshev polynomials are only defined on [-1, 1]. Extrapolation outside
    [T_min, T_max] x [P_min, P_max] is strongly discouraged and may produce
    unphysical results.

    References
    ----------
    .. [1] Cantera Documentation: Chebyshev Reaction Rate Expressions.
           https://cantera.org/stable/reference/kinetics/rate-constants.html
    """
    # =================================================================================
    # Validate Coefficient Matrix
    # Check array is 2D
    if coefficients.ndim != 2:
        raise ValueError(
            f"Chebyshev coefficients must be a 2D array, got shape {coefficients.shape} "
            f"with {coefficients.ndim} dimensions"
        )

    # Check both dimensions are at least 1 (need at least constant term)
    N_T, N_P = coefficients.shape
    if N_T < 1:
        raise ValueError(f"Temperature order N_T must be ≥ 1, got {N_T}")
    if N_P < 1:
        raise ValueError(f"Pressure order N_P must be ≥ 1, got {N_P}")

    # Check all coefficients are finite (no NaN or ±infinity)
    if not jnp.all(jnp.isfinite(coefficients)):
        raise ValueError(
            "All Chebyshev coefficients must be finite (no NaN or ±∞). "
            f"Found {jnp.sum(~jnp.isfinite(coefficients))} non-finite values."
        )

    # Validate Temperature Limits
    T_min, T_max = T_limits

    # Check T_min is positive (absolute temperature must be > 0)
    if T_min <= 0:
        raise ValueError(f"T_min must be positive (> 0 K), got T_min = {T_min} K")

    # Check T_max > T_min (valid range)
    if T_min >= T_max:
        raise ValueError(
            f"Invalid temperature range: T_min ({T_min} K) must be less than "
            f"T_max ({T_max} K). Got range width = {T_max - T_min} K."
        )

    # Check both limits are finite
    if not jnp.isfinite(T_min) or not jnp.isfinite(T_max):
        raise ValueError(
            f"Temperature limits must be finite, got T_min = {T_min} K, T_max = {T_max} K"
        )

    # Validate Pressure Limits
    P_min, P_max = P_limits

    # Check P_min is positive (pressure must be > 0)
    if P_min <= 0:
        raise ValueError(f"P_min must be positive (> 0 bar), got P_min = {P_min} bar")

    # Check P_max > P_min (valid range)
    if P_min >= P_max:
        raise ValueError(
            f"Invalid pressure range: P_min ({P_min} bar) must be less than "
            f"P_max ({P_max} bar). Got range width = {P_max - P_min} bar."
        )

    # Check both limits are finite
    if not jnp.isfinite(P_min) or not jnp.isfinite(P_max):
        raise ValueError(f"Pressure limits must be finite, got P_min = {P_min} bar, P_max = {P_max} bar")
