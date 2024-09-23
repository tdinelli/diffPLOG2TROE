from jax import jit, lax, vmap
import jax.numpy as jnp
from .arrhenius_base import kinetic_constant_base
from .constant_fit_type import lindemann, troe, sri


@jit
def kinetic_constant_falloff(falloff_constant: tuple, T: jnp.float64, P: jnp.float64) -> jnp.float64:
    """
    TODO This is outdated NOW
    Function that compute the value of the kientic constant of a FallOff reaction. At the moment only two of the
    available formalisms are implemented the Lindemann and the TROE, SRI is still missing. This function is optimized
    with JAX's Just-In-Time (JIT) compilation. Refer to the official CHEMKIN manual or to this
    [link](https://cantera.org/science/kinetics.html) for the theoretical formulation and the actual formulas.

    Args:
        params (jnp.ndarry): An array of arrays of precise dimensions (3 x 4) or (2 x 4). The first row contains the
                             value of the Arrhenius parameters for the expression of the High Pressure Limit (HPL), the
                             second row the Arrhenius parameters for the expression of the Low Pressure Limit (LPL) and
                             the third row the coefficients needed by the TROE expression if the rows are only two so
                             only the HPL and LPL we fall back in the case of the Lindemann expression. As reported in
                             the following example:

                TROE
                    CHEMKIN:
                    H2O2(+M)=2OH(+M) 2.0000e+12 0.900 48749.00  ! HPL
                       LOW/  2.49e+24 -2.300    48749.0       / ! LPL
                       TROE/ 0.4300   1.000e-30 1.000e+30 0.0 / ! TROE Coefficients

                    Internal TROE representation:
                    troe = jnp.array([
                       [2.0000e+12, 0.9000, 48749.0, .0],  # Here the last .0 is a dummy parameter
                       [2.4900e+24, -2.300, 48749.0, .0],  # Here the last .0 is a dummy parameter
                       [0.4300, 1.000e-30, 1.000e+30, 0.0]
                       "fitting_type": "troe"
                    ], dtype=jnp.float64)

                LINDEMANN (I know this is not represented by a lindemann formalism its just an example)
                    CHEMKIN:
                    H2O2(+M)=2OH(+M) 2.0000e+12 0.900 48749.00  ! HPL
                       LOW/  2.49e+24 -2.300    48749.0       / ! LPL

                    Internal LINDEMANN representation:
                    falloff = jnp.array([
                       [2.0000e+12, 0.9000, 48749.0, .0],  # Here the last .0 is a dummy parameter
                       [2.4900e+24, -2.300, 48749.0, .0],  # Here the last .0 is a dummy parameter
                       "fitting_type": "lindemann"
                    ], dtype=jnp.float64)
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
        >>> troe = jnp.array([[1.135E+36, -5.246, 1704.8], [6.220E+16, -1.174, 635.80], [0.405, 1120., 69.6]])
        >>> for i, t in enumerate(T_range):
        ...     for j, p in enumerate(P_range):
        ...         k_troe[i, j] = kinetic_constant_falloff(troe, t, p)

        This function address this and perform the operation in a vectorized way thus the code will be:
        >>> import jax.numpy as jnp
        >>> T_range = jnp.linspace(500, 2500, 30)
        >>> P_range = jnp.logspace(-1, 2, 30)
        >>> troe = jnp.array([[1.135E+36, -5.246, 1704.8], [6.220E+16, -1.174, 635.80], [0.405, 1120., 69.6]])
        >>> k_troe = compute_falloff(troe, T_range, P_range)

    Args:
        falloff (jnp.ndarray): Internal representation of the FallOff formalism.
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
