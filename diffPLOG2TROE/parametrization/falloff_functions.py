from typing import Union

import jax.numpy as jnp
from jaxtyping import Array, Float64


def lindemann(T: Union[Float64, Array], Pr: Union[Float64, Array], params: Array) -> Union[Float64, Array]:
    return jnp.ones_like(T, dtype=jnp.float64)


def troe(T: Union[Float64, Array], Pr: Union[Float64, Array], params: Array) -> Union[Float64, Array]:
    alpha, T3, T1, T2, _ = params

    logFcent = jnp.log10((1 - alpha) * jnp.exp(-T / T3) + alpha * jnp.exp(-T / T1) + jnp.exp(-T2 / T))
    c = -0.4 - 0.67 * logFcent
    n = 0.75 - 1.27 * logFcent
    d = jnp.log10(Pr) + c
    f1 = (d / (n - 0.14 * d)) ** 2

    result = jnp.where(
        Pr > 1.0e-32,  # Edge case as handled by A.C. in OpenSMOKE++
        10.0 ** (logFcent / (1.0 + f1)),  # normal case
        10.0 ** (logFcent / (1.0 + (1.0 / 0.14) ** 2)),  # OpenSMOKE does this Asymptotic value for F when f --> -Inf
    )

    return result


def sri(T: Union[Float64, Array], Pr: Union[Float64, Array], params: Array) -> Union[Float64, Array]:
    a, b, c, d, e = params

    logPr = jnp.log10(Pr)
    X = 1.0 / (1.0 + logPr * logPr)

    base = a * jnp.exp(-b / T) + jnp.exp(-T / c)
    return d * (base**X) * (T**e)
