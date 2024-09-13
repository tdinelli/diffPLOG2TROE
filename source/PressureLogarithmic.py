import jax
from jax import jit, lax
import jax.numpy as jnp
from .ArrheniusBase import kinetic_constant_base


def plog(params: jnp.ndarray):
    n = len(params)  # Number of pressure levels
    _P = jnp.array([i[0] for i in params])  # Convert to jax array
    _P = jnp.sort(_P)  # Sort using JAX version

    lnP = jnp.log(_P)
    return n, _P, lnP


@jit
def kinetic_constant_plog(params: jnp.ndarray, T: jnp.float64, P: jnp.float64) -> jnp.float64:
    n, _P, lnP = plog(params)

    # Function to find the index of the pressure interval inside the PLOG definition
    def find_index(pIndex, i):
        return lax.cond(P <= _P[i], lambda _: i, lambda _: pIndex,  None)
    pIndex = lax.fori_loop(0, n, find_index, 0)

    # Definition of the common input structure for the subfunctions
    operand = (params, T, P, pIndex, lnP)

    def low_pressure_case(operand):
        params, T, _, _,  _ = operand
        params_array = jnp.array([params[0][1], params[0][2], params[0][3]])
        return kinetic_constant_base(params_array, T)

    def high_pressure_case(operand):
        params, T, _, _, _ = operand
        params_array = jnp.array([params[-1][1], params[-1][2], params[-1][3]])
        return kinetic_constant_base(params_array, T)

    def mid_pressure_case(operand):
        params, T, P, pIndex, lnP = operand
        params1 = params[pIndex - 1]
        params2 = params[pIndex]
        params1_array = jnp.array([params1[1], params1[2], params1[3]])
        params2_array = jnp.array([params2[1], params2[2], params2[3]])
        log_k1 = jnp.log(kinetic_constant_base(params1_array, T))
        log_k2 = jnp.log(kinetic_constant_base(params2_array, T))

        # Logarithmic Interpolation
        return jnp.exp(log_k1 + (log_k2 - log_k1) * (jnp.log(P) - lnP[pIndex - 1]) / (lnP[pIndex] - lnP[pIndex - 1]))

    outer_pred = P <= _P[0]
    inner_pred = P >= _P[-1]

    return lax.cond(
        outer_pred,
        low_pressure_case,
        lambda operand: lax.cond(inner_pred, high_pressure_case, mid_pressure_case, operand),
        operand
    )

# import jax as jax
# from jax import jit, lax
# from functools import partial
# import jax.numpy as jnp
# from .ArrheniusBase import kinetic_constant_base
#
#
# def plog(params: jnp.ndarray):
#     n = len(params)  # Number of pressure levels
#     _P = jnp.array([i["P"] for i in params])  # Convert to jax array
#     _P = jnp.sort(_P)  # Sort using JAX version
#
#     lnP = jnp.log(_P)
#     return n, _P, lnP
#
#
# @jit
# def kinetic_constant_plog(params: jnp.ndarray, T: jnp.float64, P: jnp.float64) -> jnp.float64:
#     n, _P, lnP = plog(params)
#
#     jax.debug.print("jax.debug.print(P) -> {}", P)
#     def low_pressure_case(_):
#         jax.debug.print("Low pressure case with params: {}", params[0])
#         return 0
#         # AbEa = jnp.array([params[0]])
#         # return kinetic_constant_base(params[0], T)
#
#     def high_pressure_case(_):
#         jax.debug.print("Low pressure case with params: {}", params[-1])
#         # return kinetic_constant_base(params[-1], T)
#         return 0
#
#     # @partial(jit, static_argnums=0)
#     def mid_pressure_case(pIndex):
#         jax.debug.print("Mid pressure case with params: {}", pIndex)
#         # 2. Compute lower and upper pressure level Kinetic Constant
#         # log_k1 = jnp.log(kinetic_constant_base(params[pIndex], T))
#         # log_k2 = jnp.log(kinetic_constant_base(params[pIndex + 1], T))
#
#         # 3. Logarithmic Interpolation
#         # return jnp.exp(log_k1 + (log_k2 - log_k1) * (jnp.log(P) - lnP[pIndex]) / (lnP[pIndex + 1] - lnP[pIndex]))
#         return 0
#
#     # 1. Identify interval in the pressure levels using lax.while_loop
#     def find_index(pIndex):
#         return P >= _P[pIndex + 1]
#
#     pIndex = lax.fori_loop(0, n - 1, lambda i, _: lax.cond(find_index(i), lambda _: i, lambda _: i+1, None), 0)
#
#     # Conditionally select the case (low pressure, high pressure, or mid pressure)
#     # return lax.cond(
#     #     P <= jnp.float64(_P[0]),
#     #     low_pressure_case,
#     #     lax.cond(P >= jnp.float64(_P[-1]), high_pressure_case, mid_pressure_case, pIndex)
#     # )
#     return lax.cond(
#         P <= jnp.float64(_P[0]),
#         low_pressure_case,
#         lambda _: lax.cond(P >= jnp.float64(_P[-1]), high_pressure_case, lambda _: mid_pressure_case(pIndex), None)
#     )
