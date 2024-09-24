from jax import jit, lax, debug
import jax.numpy as jnp
from jax.scipy.optimize import minimize


@jit
def kinetic_constant_base(params: jnp.ndarray, T: jnp.float64) -> jnp.float64:
    """
    Function that computes the value of the kinetic constant using the modified Arrhenius equation.
    $$
        k\\left(T\\right) = \\boldsymbol{A} \\: T^{\\boldsymbol{b}} \\: e^{-\\frac{\\boldsymbol{Ea}}{RT}}
    $$
    $\\boldsymbol{A}$, $\\boldsymbol{b}$, $\\boldsymbol{Ea}$, are the parameters to be provided as input.

    Args:
        params (jnp.ndarray): Array containing the parameters needed by the modified Arrhenius equation.

                CHEMKIN:
                  H2+O=H+OH 5.0800e+04 2.670 6292.00
                Internal Arrhenius representation:
                  constant = jnp.array([5.0800e+04, 2.670, 6292.00], dtype=jnp.float64)
        T (jnp.float64): Temperature value for which the kinetic constant is computed.

    Returns:
        (jnp.float64): The value of the computed kinetic constant at the given temperature.

    """
    R = jnp.float64(1.987)
    A, b, Ea = params
    return A * T**b * jnp.exp(-Ea/R/T)


def arrhenius_loss(params: jnp.ndarray, T_range: jnp.ndarray, ln_k0: jnp.ndarray) -> jnp.float64:
    """
    Function that compute the sum of the squared errors of a tabulated kinetic constant against a computed one.

    Args:
        params (jnp.ndarray): Array containing the parameters needed by the logarithm of the modified Arrhenius
                              equation. Pay attention cause here the parameters are log(A), b, Ea.
        T_range (jnp.ndarray): Range of temperature where to compute the kinetic constant.
        ln_k0 (jnp.ndarray): Array containing the tabulated values of the logarithm of the kinetic constant.

    Returns:
        (jnp.float64): Sum of the squared errors.
    """
    ln_k0_fit = params[0] + params[1] * jnp.log(T_range) - params[2] / 1.987 / T_range
    return jnp.sum((ln_k0 - ln_k0_fit) ** 2)  # Sum of the squared errors


def arrhenius_fit(k: jnp.ndarray, T_range: jnp.ndarray, first_guess = None):
    """
    Fitting Arrhenius law using JAX, this not jitted since optimize apparently is not compatible (TO BE verified).

    Args:
        k (jnp.ndarray): Tabulated values of the kinetic constant against which the regression has to be performed.
        T_range (jnp.ndarray): Tabulated values of temperature.

    Returns:
        (Union): Returns the three arrhenius parameters and the adjusted R2 value to assess the goodness of the fit.
    """

    ln_k0 = jnp.log(k)

    # First guess for the parameters to be used for optimization
    if first_guess is None:
        initial_guess = jnp.array([ln_k0[0], 0.0, 10000.0])
    else:
        initial_guess = first_guess

    # Minimze the objective function
    result = minimize(arrhenius_loss, initial_guess, args=(T_range, ln_k0), method='BFGS')

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
