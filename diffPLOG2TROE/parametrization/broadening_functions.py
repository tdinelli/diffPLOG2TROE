from functools import partial
from typing import Dict, Optional, Union

import jax.numpy as jnp
from jax import jit, lax
from jaxtyping import Array, Float64


@partial(jit, static_argnums=(0,))
def compute_broadening_factor(
    broadening_type: str,
    T: Union[Float64, Float64[Array, "dim"]],
    Pr: Union[Float64, Float64[Array, "dim"]],
    parameters: Optional[Dict[str, Float64]],
) -> Union[Float64, Float64[Array, "dim"]]:
    if broadening_type == "lindemann":
        return lindemann(T)
    elif broadening_type == "troe" and parameters is not None:
        return troe(T, Pr, parameters)
    elif broadening_type == "sri" and parameters is not None:
        return sri(T, Pr, parameters)
    elif broadening_type == "tsang" and parameters is not None:
        return tsang(T, Pr, parameters)


def lindemann(T: Union[Float64, Float64[Array, "dim"]]) -> Union[Float64, Float64[Array, "dim"]]:
    """
    Lindemann falloff function.

    The Lindemann form assumes F = 1.0 (no broadening factor correction).
    This is the simplest falloff model where the rate constant transitions
    between low and high pressure limits without additional correction.

    Parameters
    ----------
    T : Union[Float64, Float64[Array, "dim"]]
        Temperature in Kelvin
    Pr : Union[Float64, Float64[Array, "dim"]]
        Reduced pressure (Pr = k0[M]/k_{\\infty})
    parameters : Array64f_5
        Unused - required for JAX lax.switch compatibility

    Returns
    -------
    Union[Float64, Float64[Array, "dim"]]
        Falloff correction factor F = 1.0

    Notes
    -----
    The effective rate constant is: k = k_{\\infty} * (Pr/(1+Pr)) * F
    For Lindemann, F = 1.0, so: k = k_{\\infty} * (Pr/(1+Pr))
    """
    return jnp.ones_like(T, dtype=jnp.float64)


def troe(
    T: Union[Float64, Float64[Array, "dim"]],
    Pr: Float64,
    parameters: Dict[str, Float64],
) -> Union[Float64, Float64[Array, "dim"]]:
    """
    Troe falloff function with broadening factor.

    The Troe form provides a more accurate representation of the falloff
    behavior using a centering factor Fcent and specific broadening formulation.

    Parameters
    ----------
    T : Union[Float64, Float64[Array, "dim"]]
        Temperature in Kelvin
    Pr : Union[Float64, Float64[Array, "dim"]]
        Reduced pressure (Pr = k0[M]/k_{\\infty})
    parameters : TODO
        [alpha, T***, T*, T**, unused]
        - alpha: Troe parameter (0 < alpha < 1)
        - T***: Temperature parameter (K)
        - T*: Temperature parameter (K)
        - T**: Temperature parameter (K), optional (0 means not used)
        - unused: Required for JAX compatibility

    Returns
    -------
    Union[Float64, Float64[Array, "dim"]]
        Falloff correction factor F

    Notes
    -----
    The centering factor is:
    Fcent = (1-\\alpha)exp(-T/T***) + \\alpha*exp(-T/T*) + exp(-T**/T)

    The broadening factor F is calculated using:
    log F = log Fcent / (1 + f1²)
    where f1 = (log Pr + c) / (n - 0.14*(log Pr + c))
    with c = -0.4 - 0.67*log Fcent and n = 0.75 - 1.27*log Fcent

    References
    ----------
    Gilbert, R.G., Luther, K. and Troe, J. (1983), Theory of Thermal
    Unimolecular Reactions in the Fall-off Range. II. Weak Collision Rate
    Constants. Berichte der Bunsengesellschaft für physikalische Chemie, 87:
    169-177. DOI: 10.1002/bbpc.19830870218
    """
    alpha, T3, T1, T2 = parameters["A"], parameters["T3"], parameters["T1"], parameters["T2"]

    # ==============================================================================
    # Calculate centering factor Fcent
    logFcent = lax.cond(
        T2 != 0.0,
        lambda _: jnp.log10((1 - alpha) * jnp.exp(-T / T3) + alpha * jnp.exp(-T / T1) + jnp.exp(-T2 / T)),
        lambda _: jnp.log10((1 - alpha) * jnp.exp(-T / T3) + alpha * jnp.exp(-T / T1)),
        None,
    )

    # ==============================================================================
    # Broadening parameters
    c = -0.4 - 0.67 * logFcent
    n = 0.75 - 1.27 * logFcent

    # ==============================================================================
    # Calculate broadening factor
    d = jnp.log10(Pr) + c
    f1 = (d / (n - 0.14 * d)) ** 2

    return 10.0 ** (logFcent / (1.0 + f1))


def sri(
    T: Union[Float64, Float64[Array, "dim"]],
    Pr: Float64,
    parameters: Dict[str, Float64],
) -> Union[Float64, Float64[Array, "dim"]]:
    """
    SRI (Stanford Research Institute) falloff function.

    The SRI form provides an alternative falloff representation with different
    temperature and pressure dependencies.

    Parameters
    ----------
    T : Union[Float64, Float64[Array, "dim"]]
        Temperature in Kelvin
    Pr : Union[Float64, Float64[Array, "dim"]]
        Reduced pressure (Pr = k0[M]/k_{\\infty})
    parameters : Array64f_5
        [a, b, c, d, e] - All five SRI parameters
        - a: Pre-exponential factor parameter
        - b: Temperature parameter (K)
        - c: Temperature parameter (K)
        - d: Scaling factor
        - e: Temperature exponent

    Returns
    -------
    Union[Float64, Float64[Array, "dim"]]
        Falloff correction factor F

    Notes
    -----
    The SRI form is:
    F = d * [a*exp(-b/T) + exp(-T/c)]^X * T^e
    where X = 1 / (1 + (log Pr)^2)

    References
    ----------
    [1] Stewart, P. H., Larson, C. W., & Golden, D. M. (1989). Combust. Flame, 75, 25.
        P.H. Stewart, C.W. Larson, D.M. Golden, Pressure and temperature dependence
        of reactions proceeding via a bound complex. 2. Application to 2CH3 → C2H5
        + H, Combustion and Flame, Volume 75, Issue 1, 1989, Pages 25-31, DOI:
        10.1016/0010-2180(89)90084-9.
    [2] Kee, R. J., et al. "Chemkin-II: A Fortran chemical kinetics package for the
        analysis of gas-phase chemical kinetics." , Sep. 1989. DOI: 10.2172/5681118
    """
    a, b, c, d, e = parameters["a"], parameters["b"], parameters["c"], parameters["d"], parameters["e"]

    # ==============================================================================
    # Calculate X factor based on reduced pressure
    logPr = jnp.log10(Pr)
    X = 1.0 / (1.0 + logPr * logPr)

    # ==============================================================================
    # Calculate base term
    base = a * jnp.exp(-b / T) + jnp.exp(-T / c)

    # ==============================================================================
    # Final SRI falloff factor
    return d * (base**X) * (T**e)


def tsang(
    T: Union[Float64, Float64[Array, "dim"]],
    Pr: Float64,
    parameters: Dict[str, Float64],
) -> Union[Float64, Float64[Array, "dim"]]:
    """
    Tsang falloff function (simplified Troe form).

    The Tsang form is a simplified version of the Troe falloff where the
    centering factor is a linear function of temperature.

    Parameters
    ----------
    T : Union[Float64, Float64[Array, "dim"]]
        Temperature in Kelvin
    Pr : Union[Float64, Float64[Array, "dim"]]
        Reduced pressure (Pr = k0[M]/k_{\\infty})
    parameters : Array64f_5
        [A, B, unused, unused, unused]
        - A: Tsang parameter
        - B: Tsang parameter (K^{-1})
        - unused: Required for JAX compatibility

    Returns
    -------
    Union[Float64, Float64[Array, "dim"]]
        Falloff correction factor F

    Notes
    -----
    The centering factor is simplified to: Fcent = A + B*T
    The broadening calculation follows the same form as Troe.

    References
    ----------
    Wing Tsang, John T. Herron; Chemical Kinetic Data Base for Propellant
    Combustion I. Reactions Involving NO, NO2, HNO, HNO2, HCN and N2O. J. Phys.
    Chem. Ref. Data 1 July 1991; 20 (4): 609–663. DOI: 10.1063/1.555890
    """

    A, B = parameters["A"], parameters["B"]

    # ==============================================================================
    # Simplified centering factor
    logFcent = jnp.log10(A + B * T)

    # ==============================================================================
    # Broadening parameters (same as Troe)
    c = -0.4 - 0.67 * logFcent
    n = 0.75 - 1.27 * logFcent

    # ==============================================================================
    # Calculate broadening factor
    d = jnp.log10(Pr) + c
    f1 = (d / (n - 0.14 * d)) ** 2

    return 10.0 ** (logFcent / (1.0 + f1))
