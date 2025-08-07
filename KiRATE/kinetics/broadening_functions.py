from functools import partial

import jax.numpy as jnp
from jax import jit, lax

from ..types.common import Either, ParamsDict, Real


@partial(jit, static_argnums=(0,))
def compute_broadening_factor(broadening_type: str, T: Either, Pr: Real, parameters: ParamsDict) -> Either:
    if broadening_type == "lindemann":
        return lindemann(T)
    elif broadening_type == "troe" and parameters is not None:
        return troe(T, Pr, parameters)
    elif broadening_type == "sri" and parameters is not None:
        return sri(T, Pr, parameters)
    elif broadening_type == "tsang" and parameters is not None:
        return tsang(T, Pr, parameters)


def lindemann(T: Either) -> Either:
    return jnp.ones_like(T, dtype=jnp.float64)


def troe(T: Either, Pr: Real, parameters: ParamsDict) -> Either:
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


def sri(T: Either, Pr: Real, parameters: ParamsDict) -> Either:
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


def tsang(T: Either, Pr: Real, parameters: ParamsDict) -> Either:
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
