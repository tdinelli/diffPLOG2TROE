"""
Copyright (c) 2025 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

from functools import partial
from typing import Optional

import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Float64


@partial(jit, static_argnums=(0,))
def compute_broadening_factor(
    broadening_type: str,
    T: Float64[Array, ""] | Float64[Array, "nt"],
    Pr: Float64[Array, ""],
    parameters: Optional[dict[str, Float64[Array, ""]]] = None,
) -> Float64[Array, ""] | Float64[Array, "nt"]:
    """
    Compute the broadening factor F(T, Pr) for fall-off reactions.

    This dispatcher function selects and evaluates the appropriate broadening factor
    formulation based on the specified type. All implementations are fully differentiable
    with respect to temperature and parameters.

    Parameters
    ----------
    broadening_type : str
        Type of broadening factor: "lindemann", "troe", "sri", or "tsang"
        (static argument for JIT compilation)
    T : Float64[Array, ""] | Float64[Array, "nt"]
        Temperature(s) in Kelvin (scalar or 1D array)
    Pr : Float64[Array, ""]
        Reduced pressure :math:`P_r = k_0 \\cdot [M] / k_\\infty` (dimensionless, scalar)
    parameters : dict[str, Float64[Array, ""]], optional
        Broadening-specific parameters as JAX arrays:

        - **Troe**: {"A": alpha, "T3": T3, "T1": T1, "T2": T2}
        - **SRI**: {"a": a, "b": b, "c": c, "d": d, "e": e}
        - **Tsang**: {"A": A, "B": B}
        - **Lindemann**: None (no parameters needed)

    Returns
    -------
    Float64[Array, ""] | Float64[Array, "nt"]
        Broadening factor F(T, Pr) (dimensionless, always less or equal than 1.0)
        Shape matches input temperature T

    Raises
    ------
    ValueError
        If broadening_type is not recognized or if required parameters are missing

    Notes
    -----
    The broadening factor corrects the simple Lindemann fall-off formula to match
    experimental data more accurately. It accounts for:

    - Energy transfer inefficiencies (weak collision effects)
    - Quantum effects at low temperatures
    - Angular momentum conservation
    """
    if broadening_type == "lindemann":
        return lindemann(T)
    elif broadening_type == "troe" and parameters is not None:
        return troe(T, Pr, parameters)
    elif broadening_type == "sri" and parameters is not None:
        return sri(T, Pr, parameters)
    elif broadening_type == "tsang" and parameters is not None:
        return tsang(T, Pr, parameters)
    else:
        raise ValueError(f"Broadening type {broadening_type} is not unknown!")


def lindemann(T: Float64[Array, ""] | Float64[Array, "nt"]) -> Float64[Array, ""] | Float64[Array, "nt"]:
    """
    Lindemann broadening factor (no correction).

    The Lindemann model assumes F(T, Pr) = 1.0 for all conditions, providing no
    correction to the simple fall-off formula. This is the simplest pressure-dependent
    model.

    Parameters
    ----------
    T : Float64[Array, ""] | Float64[Array, "nt"]
        Temperature(s) in Kelvin (scalar or 1D array)

    Returns
    -------
    Float64[Array, ""] | Float64[Array, "nt"]
        F = 1.0 for all temperatures (shape matches input)

    Notes
    -----
    The Lindemann model represents the earliest theoretical treatment of unimolecular
    reactions, treating all collisions as equally effective at energy transfer (strong
    collision assumption). While simple, it typically shows systematic deviations from
    experimental data at intermediate pressures.

    References
    ----------
    .. [1] Lindemann, F. A. "Discussion on 'the radiation theory of chemical action'."
           Trans. Faraday Soc., 17:598, 1922.
    """
    return jnp.ones_like(T, dtype=jnp.float64)


def troe(
    T: Float64[Array, ""] | Float64[Array, "nt"],
    Pr: Float64[Array, ""],
    parameters: dict[str, Float64[Array, ""]],
) -> Float64[Array, ""] | Float64[Array, "nt"]:
    """
    Troe broadening factor for fall-off reactions.

    The Troe formulation is the most widely used broadening factor, providing an
    empirical interpolation that accurately represents the transition between low-
    and high-pressure limits for a wide range of reactions.

    Parameters
    ----------
    T : Float64[Array, ""] | Float64[Array, "nt"]
        Temperature(s) in Kelvin (scalar or 1D array)
    Pr : Float64[Array, ""]
        Reduced pressure (dimensionless, scalar)
    parameters : dict[str, Float64[Array, ""]]
        Troe parameters as JAX arrays:

        - **A**: :math:`\\alpha` - weighting factor, typically :math:`0 < \\alpha < 1`
        - **T3**: T3 [K] - temperature scale for first exponential
        - **T1**: T1 [K] - temperature scale for second exponential
        - **T2**: T2 [K] - temperature scale for third exponential (optional, can be 0)

    Returns
    -------
    Float64[Array, ""] | Float64[Array, "nt"]
        Troe broadening factor F(T, Pr), dimensionless, typically 0.1 < F < 1.0
        Shape matches input temperature T

    Notes
    -----
    **Mathematical Formulation** (Gilbert et al., 1983):

    The Troe broadening factor is computed using:

    .. math::
        \\log_{10} F(T, P_r) = \\frac{\\log_{10} F_{\\text{cent}}(T)}{1 + f_1^2}

    where the centering factor F_cent is:

    .. math::
        F_{\\text{cent}}(T) = (1-\\alpha) \\exp(-T/T_3) + \\alpha \\exp(-T/T_1) + \\exp(-T_2/T)

    and the interpolation parameter :math:`f_1` is:

    .. math::
        f_1 &= \\frac{\\log_{10} P_r + C}{N - 0.14 (\\log_{10} P_r + C)} \\\\
        C &= -0.4 - 0.67 \\log_{10} F_{\\text{cent}} \\\\
        N &= 0.75 - 1.27 \\log_{10} F_{\\text{cent}}

    **Parameter Ranges**:

    - :math:`\\alpha`: typically 0.1 - 0.9 (controls shape asymmetry)
    - :math:`T_3, T_1`: typically 10 - 10000 K (control temperature dependence)
    - :math:`T_2`: typically 0 or 1000 - 10000 K (often set to 0 for 3-parameter form)

    References
    ----------
    .. [1] Gilbert, R. G., Luther, K., and Troe, J. "Theory of thermal unimolecular
           reactions in the fall-off range. II. weak collision rate constants."
           Berichte der Bunsengesellschaft für physikalische Chemie, 87(2):169-175, 1983.
    """
    # Extract Troe parameters from dictionary
    alpha, T3, T1, T2 = parameters["A"], parameters["T3"], parameters["T1"], parameters["T2"]

    # Step 1: Calculate centering factor F_cent(T)
    # First two terms (always present in 3- or 4-parameter form)
    term1 = (1 - alpha) * jnp.exp(-T / T3)  # Low-temperature contribution
    term2 = alpha * jnp.exp(-T / T1)  # High-temperature contribution

    # Third term: optional high-temperature correction (often T2 = 0)
    # Using jnp.where maintains differentiability w.r.t. T2
    # When T2 = 0: term3 = 0 (3-parameter Troe)
    # When T2 > 0: term3 = exp(-T2/T) (4-parameter Troe)
    term3 = jnp.where(T2 != 0.0, jnp.exp(-T2 / T), 0.0)

    # Combine terms and compute log10(F_cent)
    Fcent = term1 + term2 + term3
    logFcent = jnp.log10(Fcent)

    # Step 2: Calculate interpolation parameters C and N
    c = -0.4 - 0.67 * logFcent  # Offset parameter
    n = 0.75 - 1.27 * logFcent  # Width parameter

    # Step 3: Calculate broadening factor using Troe interpolation formula
    d = jnp.log10(Pr) + c  # Shifted log pressure
    f1 = jnp.power((d / (n - 0.14 * d)), 2)  # Interpolation function squared

    # Final broadening factor F(T, Pr)
    return jnp.power(10, (logFcent / (1.0 + f1)))


def sri(
    T: Float64[Array, ""] | Float64[Array, "nt"],
    Pr: Float64[Array, ""],
    parameters: dict[str, Float64[Array, ""]],
) -> Float64[Array, ""] | Float64[Array, "nt"]:
    """
    SRI broadening factor for fall-off reactions.

    The SRI formulation is an alternative to Troe that uses a different functional
    form. It was originally developed for specific reaction types (particularly
    recombination reactions) and uses 5 parameters for flexibility.

    Parameters
    ----------
    T : Float64[Array, ""] | Float64[Array, "nt"]
        Temperature(s) in Kelvin (scalar or 1D array)
    Pr : Float64[Array, ""]
        Reduced pressure (dimensionless, scalar)
    parameters : dict[str, Float64[Array, ""]]
        SRI parameters as JAX arrays:

        - **a**: dimensionless weighting factor
        - **b**: temperature scale [K] for low-temperature exponential
        - **c**: temperature scale [K] for high-temperature exponential
        - **d**: multiplicative scaling factor (default: 1.0)
        - **e**: temperature exponent (default: 0.0)

    Returns
    -------
    Float64[Array, ""] | Float64[Array, "nt"]
        SRI broadening factor F(T, P_r), dimensionless, typically 0.1 < F < 1.0
        Shape matches input temperature T

    Notes
    -----
    **Mathematical Formulation** (Stewart et al., 1989; Kee et al., 1989):

    The SRI broadening factor is computed using:

    .. math::
        F(T, P_r) = d \\cdot \\left[a \\exp(-b/T) + \\exp(-T/c)\\right]^{X} \\cdot T^e

    where the pressure-dependent exponent X is:

    .. math::
        X = \\frac{1}{1 + (\\log_{10} P_r)^2}

    **Physical Interpretation**:

    - At P_r = 1 (log10(Pr) = 0): X = 1, maximum broadening effect
    - At Pr << 1 or Pr >> 1: X -> 0, broadening effect diminishes
    - The base term captures temperature-dependent collision efficiency
    - The T^e term provides additional temperature scaling

    **Parameter Defaults** (Kee et al., 1989):

    - d = 1.0 (no additional scaling)
    - e = 0.0 (no additional temperature dependence)

    References
    ----------
    .. [1] Stewart, P. H., Larson, C. W., and Golden, D. "Pressure and temperature
           dependence of reactions proceeding via a bound complex. 2. application to
           2 CH3 -> C2H5 + H." Combustion and Flame, 75(1):25-40, 1989.
    .. [2] Kee, R. J., Rupley, F. M., and Miller, J. A. "Chemkin-II: A Fortran
           chemical kinetics package for the analysis of gas-phase chemical kinetics."
           Sandia National Labs Report SAND-89-8009, 1989.
    """
    # Extract SRI parameters from dictionary
    a, b, c, d, e = parameters["a"], parameters["b"], parameters["c"], parameters["d"], parameters["e"]

    # Step 1: Calculate pressure-dependent exponent X
    logPr = jnp.log10(Pr)
    X = 1.0 / (1.0 + logPr * logPr)

    # Step 2: Calculate base term (temperature-dependent collision efficiency)
    base = a * jnp.exp(-b / T) + jnp.exp(-T / c)

    # Step 3: Calculate final SRI broadening factor
    return d * (jnp.power(base, X)) * (jnp.power(T, e))


def tsang(
    T: Float64[Array, ""] | Float64[Array, "nt"],
    Pr: Float64[Array, ""],
    parameters: dict[str, Float64[Array, ""]],
) -> Float64[Array, ""] | Float64[Array, "nt"]:
    """
    Tsang approximation for broadening factor in fall-off reactions.

    The Tsang formulation is a simplified version of the Troe model that uses a
    linear temperature-dependent centering factor instead of the exponential form.
    This reduces the number of parameters while maintaining reasonable accuracy
    for many reactions. This is implemented in CANTERA but its not adopted in the
    CHEMKIN format.

    Parameters
    ----------
    T : Float64[Array, ""] | Float64[Array, "nt"]
        Temperature(s) in Kelvin (scalar or 1D array)
    Pr : Float64[Array, ""]
        Reduced pressure (dimensionless, scalar)
    parameters : dict[str, Float64[Array, ""]]
        Tsang parameters as JAX arrays:

        - **A**: intercept for F_cent (dimensionless)
        - **B**: temperature coefficient for F_cent [K^-1]

    Returns
    -------
    Float64[Array, ""] | Float64[Array, "nt"]
        Tsang broadening factor F(T, P_r), dimensionless, typically 0.1 < F < 1.0
        Shape matches input temperature T

    Notes
    -----
    **Mathematical Formulation** (Tsang & Herron, 1991):

    The Tsang broadening factor uses the same interpolation formula as Troe but
    with a simplified centering factor:

    .. math::
        F_{\\text{cent}}(T) = A + B \\cdot T

    The remaining formulas follow the Troe approach:

    .. math::
        \\log_{10} F(T, P_r) &= \\frac{\\log_{10} F_{\\text{cent}}(T)}{1 + f_1^2} \\\\
        f_1 &= \\frac{\\log_{10} P_r + C}{N - 0.14 (\\log_{10} P_r + C)} \\\\
        C &= -0.4 - 0.67 \\log_{10} F_{\\text{cent}} \\\\
        N &= 0.75 - 1.27 \\log_{10} F_{\\text{cent}}

    References
    ----------
    .. [1] Tsang, W. and Herron, J. T. "Chemical kinetic data base for propellant
           combustion I. reactions involving NO, NO2, HNO, HNO2, HCN and N2O."
           Journal of Physical and Chemical Reference Data, 20(3):779-798, 1991.
    """
    # Extract Tsang parameters from dictionary
    A, B = parameters["A"], parameters["B"]

    # Step 1: Calculate simplified centering factor F_cent
    logFcent = jnp.log10(A + B * T)

    # Step 2: Calculate interpolation parameters C and N (same as Troe)
    c = -0.4 - 0.67 * logFcent  # Offset parameter
    n = 0.75 - 1.27 * logFcent  # Width parameter

    # Step 3: Calculate broadening factor using Troe-style interpolation
    d = jnp.log10(Pr) + c  # Shifted log pressure
    f1 = jnp.power((d / (n - 0.14 * d)), 2)  # Interpolation function squared

    # Final broadening factor F(T, Pr)
    return jnp.power(10.0, (logFcent / (1.0 + f1)))
