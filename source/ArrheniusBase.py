from jax import jit, lax, debug
import jax.numpy as jnp
from jax.scipy.optimize import minimize


@jit
def kinetic_constant_base(params: jnp.ndarray, T: jnp.float64) -> jnp.float64:
    R = jnp.float64(1.987)
    A, b, Ea = params
    return A * T**b * jnp.exp(-Ea/R/T)


def arrhenius_loss(params, T_range, ln_k0):
    ln_k0_fit = params[0] + params[1] * jnp.log(T_range) - params[2] / 1.987 / T_range
    return jnp.sum((ln_k0 - ln_k0_fit) ** 2)  # Sum of the squared errors


def arrhenius_fit(k: jnp.ndarray, T_range: jnp.ndarray):
    """Fitting Arrhenius law using JAX, this not jitted since optimize apparently is not compatible"""
    ln_k0 = jnp.log(k)

    # First guess for the parameters to be used for optimization
    initial_guess = jnp.array([ln_k0[0], 0.0, 10000.0])

    # Minimze the objective function
    result = minimize(arrhenius_loss, initial_guess, args=(T_range, ln_k0), method='BFGS')

    popt = result.x

    # Estrai i parametri
    A = jnp.exp(popt[0])
    b = popt[1]
    Ea = popt[2]

    # Commpute the R2
    ln_k_fit = popt[0] + b * jnp.log(T_range) - Ea / 1.987 / T_range
    R2 = 1 - jnp.sum((ln_k0 - ln_k_fit) ** 2) / jnp.sum((ln_k0 - jnp.mean(ln_k0)) ** 2)

    # Compute the adjusted R2
    R2adj = 1 - (1 - R2) * (len(T_range) - 1) / (len(T_range) - 1 - 2)

    # Control the floating point precision for A (Maybe in this case is not needed (?))
    A = lax.cond(
        jnp.less_equal(A, jnp.finfo(float).eps) or jnp.isclose(A, 0),
        lambda: jnp.inf,
        lambda: A
    )

    return A, b, Ea, R2adj
