from typing import Union

import jax.numpy as jnp
from jaxtyping import Array, Float64


def lindemann(T: Union[Float64, Array], Pr: Union[Float64, Array], params: Array) -> Union[Float64, Array]:
    return jnp.ones_like(T, dtype=jnp.float64)


def troe(T: Union[Float64, Array], Pr: Union[Float64, Array], params: Array) -> Union[Float64, Array]:
    """
    Calculate Troe falloff function.

    F = (1-α)exp(-T/T***) + αexp(-T/T*) + exp(-T**/T)

    Args:
        T (Union[Float64, Array]): Temperature in K
        Pr (Union[Float64, Array]): Reduced pressure
        params (Array): Troe parameters [α, T***, T*, T**] where T** is optional (use 0 if not needed)

    Returns:
        Union[Float64, Array]: Troe falloff factor
    """
    A, T3, T1, T2, _ = params  # α, T***, T*, T**

    term1 = (1 - A) * jnp.exp(-jnp.minimum(T / T3, jnp.float64(700)))
    term2 = A * jnp.exp(-jnp.minimum(T / T1, jnp.float64(700)))
    term3 = jnp.where(T2 > 0, jnp.exp(-jnp.minimum(T2 / T, jnp.float64(700))), jnp.zeros_like(T, dtype=jnp.float64))

    Fcent = term1 + term2 + term3
    logFcent = jnp.log10(jnp.maximum(Fcent, jnp.float64(1e-300)))
    c = -0.4 - 0.67 * logFcent
    n = 0.75 - 1.27 * logFcent
    logPr = jnp.log10(jnp.maximum(Pr, jnp.float64(1e-300)))

    d = logPr + c
    f1 = (d / (n - 0.14 * d)) ** 2

    return 10.0 ** (logFcent / (1.0 + f1))


def sri(T: Union[Float64, Array], Pr: Union[Float64, Array], params: Array) -> Union[Float64, Array]:
    """
    Calculate SRI falloff function.

    F = d[a*exp(-b/T) + exp(-T/c)]^X * T^e
    where X = 1/[1 + (log10(Pr))^2]

    Args:
        T (Union[Float64, Array]): Temperature in K
        Pr (Union[Float64, Array]): Reduced pressure
        params (Array): SRI parameters [a, b, c, d, e]

    Returns:
        Union[Float64, Array]: SRI falloff factor
    """
    a, b, c, d, e = params

    logPr = jnp.log10(jnp.maximum(Pr, jnp.float64(1e-300)))
    X = 1.0 / (1.0 + logPr * logPr)

    term1 = a * jnp.exp(-jnp.minimum(b / T, jnp.float64(700)))
    term2 = jnp.exp(-jnp.minimum(T / c, jnp.float64(700)))

    base = term1 + term2
    return d * (base**X) * (T**e)
