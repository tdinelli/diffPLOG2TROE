import jax.numpy as jnp
from jax import jit, lax, vmap
from jaxtyping import Array, Float64

from .arrhenius_base import kinetic_constant_base


@jit
def kinetic_constant_plog(params: Float64, T: Float64, P: Float64) -> Float64:
    """ """

    n = len(params)  # Number of pressure levels
    _P = jnp.array([i[0] for i in params])  # Convert to jax array
    _P = jnp.sort(_P)  # Sort using JAX version

    lnP = jnp.log(_P)

    # Function to find the index of the pressure interval inside the PLOG definition
    def find_index(pIndex, i):
        """Function needed to identify the pressure index of the current pressure level"""
        return lax.cond(P <= _P[i], lambda _: i, lambda _: pIndex, None)

    pIndex = lax.fori_loop(0, n, find_index, 0)

    # Definition of the common input structure for the subfunctions
    operand = (params, T, P, pIndex, lnP)

    def low_pressure_case(operand):
        """Function that handles the case where the pressure is lower or equal to the lowest pressure level."""
        params, T, _, _, _ = operand
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
        params1_array = jnp.array(
            [params[pIndex - 1][1], params[pIndex - 1][2], params[pIndex - 1][3]]
        )
        params2_array = jnp.array(
            [params[pIndex][1], params[pIndex][2], params[pIndex][3]]
        )
        log_k1 = jnp.log(kinetic_constant_base(params1_array, T))
        log_k2 = jnp.log(kinetic_constant_base(params2_array, T))

        # Logarithmic Interpolation
        return jnp.exp(
            log_k1
            + (log_k2 - log_k1)
            * (jnp.log(P) - lnP[pIndex - 1])
            / (lnP[pIndex] - lnP[pIndex - 1])
        )

    return lax.cond(
        P <= _P[0],
        low_pressure_case,
        lambda operand: lax.cond(
            P >= _P[-1], high_pressure_case, mid_pressure_case, operand
        ),
        operand,
    )


@jit
def compute_plog(plog: Array, T_range: Array, P_range: Array) -> Array:
    """ """

    def compute_single(t, p):
        _kplog = kinetic_constant_plog(plog, t, p)
        return _kplog

    compute_single_t_fixed = vmap(
        lambda p: vmap(lambda t: compute_single(t, p))(T_range)
    )
    k_plog = compute_single_t_fixed(P_range)
    return k_plog
