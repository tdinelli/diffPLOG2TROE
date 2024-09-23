from jax import jit, lax, vmap
import jax.numpy as jnp
from .arrhenius_base import kinetic_constant_base
from .constant_fit_type import lindemann, troe, sri


@jit
def kinetic_constant_falloff(falloff_constant: tuple, T: jnp.float64, P: jnp.float64) -> jnp.float64:
    """
    Function that compute the value of the kientic constant of a FallOff reaction. This function is optimized with JAX's
    Just-In-Time (JIT) compilation. Refer to the official CHEMKIN manual for the actual formalism.

    Args:
        falloff_constant (tuple): A tuple containing an array and an integer. As reported in the following example:

            Internal TROE representation:
                falloff_troe = (
                    jnp.array([
                        [2.0000e+12, 0.9000, 48749.0, .0],  # Here the last .0 is a dummy parameter
                        [2.4900e+24, -2.300, 48749.0, .0],  # Here the last .0 is a dummy parameter
                        [0.4300, 1.000e-30, 1.000e+30, 0.0]
                    ], dtype=jnp.float64),
                    1
                )

            Internal LINDEMANN representation:
                falloff_lindemann = (
                    jnp.array([
                        [2.0000e+12, 0.9000, 48749.0, .0],  # Here the last .0 is a dummy parameter
                        [2.4900e+24, -2.300, 48749.0, .0],  # Here the last .0 is a dummy parameter
                    ], dtype=jnp.float64),
                    0
                )
        T (jnp.float64): Temperature value for which the kinetic constant is computed.
        P (jnp.float64): Pressure value for which the kinetic constant is computed.

    Returns:
        (Union): The value of the computed kinetic constant at the given temperature and pressure, the value of the
                 LPL, the value of the HPL and the value of the total concentration.
    """

    params, fitting_type = falloff_constant

    # is_lindemann = (fitting_type == 0)
    is_troe = (fitting_type == 1)
    is_sri = (fitting_type == 2)

    _k0 = kinetic_constant_base(params[0, 0:3], T)
    _kInf = kinetic_constant_base(params[1, 0:3], T)
    _M = P / 0.08206 / T * (1/1000)  # P [atm], T [K] -> M [mol/cm3/s]
    _Pr = _k0 * _M / _kInf
    operand = (T, _Pr, params)

    F = lax.cond(
        is_troe,
        lambda x: troe(*x),
        lambda x: lax.cond(
            is_sri,
            lambda y: sri(*y),
            lambda y: lindemann(*y),
            x
        ),
        operand
    )

    k_falloff = (_kInf * (_Pr / (1 + _Pr))) * F
    return (k_falloff, _k0, _kInf, _M)

@jit
def compute_falloff(falloff_constant: tuple, T_range: jnp.ndarray, P_range: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the kinetic constant of a given reaction written following the FallOff formalism in a vectorized way over a
    range of pressures and temperatures. This is done in order to avoid the use of double nested loop.

    Examples:
        Intuitively if someone needs to compute the value of the kinetic constant of a FallOff reaction for different
        pressures and temperature the naive and straightforward implementation would be:
        >>> import jax.numpy as jnp
        >>> T_range = jnp.linspace(500, 2500, 30)
        >>> P_range = jnp.logspace(-1, 2, 30)
        >>> k_troe = jnp.empty(shape=(30, 30))
        >>> troe = (jnp.array([[1.135E+36, -5.246, 1704.8], [6.220E+16, -1.174, 635.80], [0.405, 1120., 69.6]]), 1)
        >>> for i, t in enumerate(T_range):
        ...     for j, p in enumerate(P_range):
        ...         k_troe[i, j] = kinetic_constant_falloff(troe, t, p)

        This function address this and perform the operation in a vectorized way thus the code will be:
        >>> import jax.numpy as jnp
        >>> T_range = jnp.linspace(500, 2500, 30)
        >>> P_range = jnp.logspace(-1, 2, 30)
        >>> troe = (jnp.array([[1.135E+36, -5.246, 1704.8], [6.220E+16, -1.174, 635.80], [0.405, 1120., 69.6]]), 1)
        >>> k_troe = compute_falloff(troe, T_range, P_range)

    Args:
        falloff_constant (tuple): Internal representation of the FallOff formalism.
        T_range (jnp.ndarray): Array containing the temperature values where we need to compute the kinetic constant in
                               Kelvin.
        P_range (jnp.ndarray): Array containing the pressure values where we need to compute the kinetic constant.

    Returns:
        jnp.ndarray: Two dimensional array containing the value of the kinetic constant at each feeded value of pressure
                     and temperature. Keep in mind that columns are for the same value of tempreature and rows for the
                     same value of pressure.
    """
    def compute_single(t, p):
        _kfalloff, _, _, _ = kinetic_constant_falloff(falloff_constant, t, p)
        return _kfalloff
    compute_single_t_fixed = vmap(lambda p: vmap(lambda t: compute_single(t, p))(T_range))
    k_falloff = compute_single_t_fixed(P_range)
    return k_falloff
