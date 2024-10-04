from jax import jit, debug
import jax.numpy as jnp


@jit
def lindemann(T: jnp.float64, Pr: jnp.float64, params: jnp.ndarray) -> jnp.float64:
    F = jnp.float64(1)
    return F


@jit
def troe(T: jnp.float64, Pr: jnp.float64, params: jnp.ndarray) -> jnp.float64:
    A = params[2][0]
    T3 = params[2][1]
    T1 = params[2][2]
    T2 = params[2][3] # T2 is optional here we feed 0, when not needed.

    logFcent = jnp.log10((1 - A) * jnp.exp(-T/T3) + A * jnp.exp(-T/T1) + jnp.exp(-T2/T))

    c = -0.4 - 0.67 * logFcent
    n = 0.75 - 1.27 * logFcent
    f1 = ((jnp.log10(Pr) + c) / (n - 0.14 * (jnp.log10(Pr) + c)))**2
    F = 10**(logFcent / (1 + f1))

    return F


@jit
def sri(T: jnp.float64, Pr: jnp.float64, params: jnp.ndarray) -> jnp.float64:
    a = params[2][0]
    b = params[2][1]
    c = params[2][2]
    d = params[2][3]
    e = params[2][4]

    X = 1 / (1 + (jnp.log10(Pr)**2))
    F = d * ((a*jnp.exp(-b/T) + jnp.exp(-T/c))**X) * T**e
    return F
