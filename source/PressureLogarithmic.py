from jax import jit, lax, vmap
import jax.numpy as jnp
from .ArrheniusBase import kinetic_constant_base


@jit
def kinetic_constant_plog(params: jnp.ndarray, T: jnp.float64, P: jnp.float64) -> jnp.float64:
    """
    Computes the kinetic constant for a PLOG model using logarithmic interpolation between pressure levels. This
    function is optimized with JAX's Just-In-Time (JIT) compilation.

    Args:
        params (jnp.ndarray): A list of tuples where each tuple contains the pressure level and its associated
                              parameters (A, b, Ea).
        T (jnp.float64): Temperature value for which the kinetic constant is computed.
        P (jnp.float64): Pressure value for which the kinetic constant is computed.

    Returns:
        jnp.float64: The value of the interpolated kinetic constant at the given temperature and pressure.
    """

    n = len(params)  # Number of pressure levels
    _P = jnp.array([i[0] for i in params])  # Convert to jax array
    _P = jnp.sort(_P)  # Sort using JAX version

    lnP = jnp.log(_P)

    # Function to find the index of the pressure interval inside the PLOG definition
    def find_index(pIndex, i):
        """Function needed to identify the pressure index of the current pressure level"""
        return lax.cond(P <= _P[i], lambda _: i, lambda _: pIndex,  None)
    pIndex = lax.fori_loop(0, n, find_index, 0)

    # Definition of the common input structure for the subfunctions
    operand = (params, T, P, pIndex, lnP)

    def low_pressure_case(operand):
        """Function that handles the case where the pressure is lower or equal to the lowest pressure level."""
        params, T, _, _,  _ = operand
        params_array = jnp.array([params[0][1], params[0][2], params[0][3]])
        return kinetic_constant_base(params_array, T)

    def high_pressure_case(operand):
        """Function that handles the case where the pressure is higher or equal to the highest pressure level."""
        params, T, _, _, _ = operand
        params_array = jnp.array([params[-1][1], params[-1][2], params[-1][3]])
        return kinetic_constant_base(params_array, T)

    def mid_pressure_case(operand):
        """
        Handles the case where the pressure falls between two pressure levels. It performs logarithmic interpolation
        between the two kinetic constants corresponding to the pressure levels.
        """
        params, T, P, pIndex, lnP = operand
        params1_array = jnp.array([params[pIndex - 1][1], params[pIndex - 1][2], params[pIndex - 1][3]])
        params2_array = jnp.array([params[pIndex][1], params[pIndex][2], params[pIndex][3]])
        log_k1 = jnp.log(kinetic_constant_base(params1_array, T))
        log_k2 = jnp.log(kinetic_constant_base(params2_array, T))

        # Logarithmic Interpolation
        return jnp.exp(log_k1 + (log_k2 - log_k1) * (jnp.log(P) - lnP[pIndex - 1]) / (lnP[pIndex] - lnP[pIndex - 1]))

    return lax.cond(
        P <= _P[0],
        low_pressure_case,
        lambda operand: lax.cond( P >= _P[-1], high_pressure_case, mid_pressure_case, operand),
        operand
    )


@jit
def compute_plog(plog: jnp.ndarray, T_range: jnp.ndarray, P_range: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the kinetic constant of a given reaction written following the PLOG formalism in a vectorized way over a
    range of pressures and temperatures. This is done in order to avoid the use of double nested loop.
    Parameters
    ----------
    plog : jnp.ndarray
        
    T_range : jnp.ndarray
        
    P_range : jnp.ndarray
        

    Returns
    -------
    jnp.ndarray
        

    """
    def compute_single(t, p):
        _kplog = kinetic_constant_plog(plog, t, p)
        return _kplog
    compute_single_t_fixed = vmap(lambda p: vmap(lambda t: compute_single(t, p))(T_range))
    k_plog = compute_single_t_fixed(P_range)
    return k_plog
