from typing import Union

import jax.numpy as jnp
from jaxtyping import Array, Float64


def lindemann(T: Union[Float64, Array]) -> Union[Float64, Array]:
    return jnp.ones_like(T, dtype=jnp.float64)


def troe(T: Union[Float64, Array], Pr: Union[Float64, Array], params: Array) -> Union[Float64, Array]:
    A, T3, T1, T2, _ = params

    Fcent = (1 - A) * jnp.exp(-T / T3) + A * jnp.exp(-T / T1) + jnp.exp(-T2 / T)
    logFcent = jnp.log10(Fcent)
    c = -0.4 - 0.67 * logFcent
    n = 0.75 - 1.27 * logFcent
    logPr = jnp.log10(Pr)

    d = logPr + c
    f1 = (d / (n - 0.14 * d)) ** 2

    return 10.0 ** (logFcent / (1.0 + f1))


def sri(T: Union[Float64, Array], Pr: Union[Float64, Array], params: Array) -> Union[Float64, Array]:
    a, b, c, d, e = params

    logPr = jnp.log10(Pr)
    X = 1.0 / (1.0 + logPr * logPr)

    base = a * jnp.exp(-b / T) + jnp.exp(-T / c)
    return d * (base**X) * (T**e)
