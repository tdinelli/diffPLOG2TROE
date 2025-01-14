import jax.numpy as jnp
from jax import jit
from jax.scipy.optimize import minimize
from jaxtyping import Array, Float64


@jit
def kinetic_constant_base(params: Array, T: Float64) -> Float64:
    """ """
    R = jnp.float64(1.987)
    A, b, Ea = params
    return A * T**b * jnp.exp(-Ea / R / T)


@jit
def arrhenius_loss(params: Array, T_range: Array, ln_k0: Array) -> Float64:
    """ """
    ln_k0_fit = params[0] + params[1] * jnp.log(T_range) - params[2] / 1.987 / T_range
    return jnp.sum((ln_k0 - ln_k0_fit) ** 2)  # Sum of the squared errors


def arrhenius_fit(k: Array, T_range: Array, first_guess=None):
    """ """

    ln_k0 = jnp.log(k)

    # First guess for the parameters to be used for optimization
    if first_guess is None:
        initial_guess = jnp.array([ln_k0[0], 0.0, 10000.0])
    else:
        initial_guess = first_guess

    # Minimze the objective function
    result = minimize(
        arrhenius_loss, initial_guess, args=(T_range, ln_k0), method="BFGS"
    )

    popt = result.x

    # Extract parameters
    A = jnp.exp(popt[0])
    b = popt[1]
    Ea = popt[2]

    # Commpute the R2
    ln_k_fit = popt[0] + b * jnp.log(T_range) - Ea / 1.987 / T_range
    R2 = 1 - jnp.sum((ln_k0 - ln_k_fit) ** 2) / jnp.sum((ln_k0 - jnp.mean(ln_k0)) ** 2)

    # Compute the adjusted R2
    R2adj = 1 - (1 - R2) * (len(T_range) - 1) / (len(T_range) - 1 - 2)

    return A, b, Ea, R2adj
