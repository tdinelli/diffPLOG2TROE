import jax.numpy as jnp
from jax import lax, jit

from ..utilities.custom_types import Array64f_5, ScalarOrVector


@jit
def lindemann(T: ScalarOrVector, Pr: ScalarOrVector, parameters: Array64f_5) -> ScalarOrVector:
    """
    Lindemann falloff function.

    The Lindemann form assumes F = 1.0 (no broadening factor correction).
    This is the simplest falloff model where the rate constant transitions
    between low and high pressure limits without additional correction.

    Parameters
    ----------
    T : ScalarOrVector
        Temperature in Kelvin
    Pr : ScalarOrVector
        Reduced pressure (Pr = k0[M]/k_{\\infty})
    parameters : Array64f_5
        Unused - required for JAX lax.switch compatibility

    Returns
    -------
    ScalarOrVector
        Falloff correction factor F = 1.0

    Notes
    -----
    The effective rate constant is: k = k_{\\infty} * (Pr/(1+Pr)) * F
    For Lindemann, F = 1.0, so: k = k_{\\infty} * (Pr/(1+Pr))
    """
    return jnp.ones_like(T, dtype=jnp.float64)


@jit
def troe(T: ScalarOrVector, Pr: ScalarOrVector, parameters: Array64f_5) -> ScalarOrVector:
    """
    Troe falloff function with broadening factor.

    The Troe form provides a more accurate representation of the falloff
    behavior using a centering factor Fcent and specific broadening formulation.

    Parameters
    ----------
    T : ScalarOrVector
        Temperature in Kelvin
    Pr : ScalarOrVector
        Reduced pressure (Pr = k0[M]/k_{\\infty})
    parameters : Array64f_5
        [alpha, T***, T*, T**, unused]
        - alpha: Troe parameter (0 < alpha < 1)
        - T***: Temperature parameter (K)
        - T*: Temperature parameter (K)
        - T**: Temperature parameter (K), optional (0 means not used)
        - unused: Required for JAX compatibility

    Returns
    -------
    ScalarOrVector
        Falloff correction factor F

    Notes
    -----
    The centering factor is:
    Fcent = (1-α)exp(-T/T***) + α*exp(-T/T*) + exp(-T**/T)

    The broadening factor F is calculated using:
    log F = log Fcent / (1 + f1²)
    where f1 = (log Pr + c) / (n - 0.14*(log Pr + c))
    with c = -0.4 - 0.67*log Fcent and n = 0.75 - 1.27*log Fcent

    References
    ----------
    Gilbert, R. G., Luther, K., & Troe, J. (1983). Ber. Bunsenges. Phys. Chem., 87, 169. (TODO: CHECK)
    """
    alpha, T3, T1, T2, _ = parameters

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


@jit
def tsang(T: ScalarOrVector, Pr: ScalarOrVector, parameters: Array64f_5) -> ScalarOrVector:
    """
    Tsang falloff function (simplified Troe form).

    The Tsang form is a simplified version of the Troe falloff where the
    centering factor is a linear function of temperature.

    Parameters
    ----------
    T : ScalarOrVector
        Temperature in Kelvin
    Pr : ScalarOrVector
        Reduced pressure (Pr = k0[M]/k_{\\infty})
    parameters : Array64f_5
        [A, B, unused, unused, unused]
        - A: Tsang parameter
        - B: Tsang parameter (K^{-1})
        - unused: Required for JAX compatibility

    Returns
    -------
    ScalarOrVector
        Falloff correction factor F

    Notes
    -----
    The centering factor is simplified to: Fcent = A + B*T
    The broadening calculation follows the same form as Troe.

    References
    ----------
    Tsang, W. (1991). J. Phys. Chem. Ref. Data, 20, 221. (TODO: CHECK)
    """
    A, B, _, _, _ = parameters

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


@jit
def sri(T: ScalarOrVector, Pr: ScalarOrVector, parameters: Array64f_5) -> ScalarOrVector:
    """
    SRI (Stanford Research Institute) falloff function.

    The SRI form provides an alternative falloff representation with different
    temperature and pressure dependencies.

    Parameters
    ----------
    T : ScalarOrVector
        Temperature in Kelvin
    Pr : ScalarOrVector
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
    ScalarOrVector
        Falloff correction factor F

    Notes
    -----
    The SRI form is:
    F = d * [a*exp(-b/T) + exp(-T/c)]^X * T^e
    where X = 1 / (1 + (log Pr)^2)

    References
    ----------
    Stewart, P. H., Larson, C. W., & Golden, D. M. (1989). Combust. Flame, 75, 25. (TODO: CHECK)
    """
    a, b, c, d, e = parameters

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
