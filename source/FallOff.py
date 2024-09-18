from jax import jit, lax, vmap, debug
import jax.numpy as jnp
from .ArrheniusBase import kinetic_constant_base


@jit
def kinetic_constant_fall_off(params: jnp.ndarray, T: jnp.float64, P: jnp.float64) -> jnp.float64:
    """
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
                                 ], dtype=jnp.float64)

                             LINDEMANN (I know this is not represented by a lindemann formalism its just an example)
                                 CHEMKIN:
                                 H2O2(+M)=2OH(+M) 2.0000e+12 0.900 48749.00  ! HPL
                                    LOW/  2.49e+24 -2.300    48749.0       / ! LPL

                                 Internal LINDEMANN representation:
                                 lindemann = jnp.array([
                                    [2.0000e+12, 0.9000, 48749.0, .0],  # Here the last .0 is a dummy parameter
                                    [2.4900e+24, -2.300, 48749.0, .0],  # Here the last .0 is a dummy parameter
                                 ], dtype=jnp.float64)
        T (jnp.float64): Temperature value for which the kinetic constant is computed.
        P (jnp.float64): Pressure value for which the kinetic constant is computed.

    Returns:
        (Union): The value of the computed kinetic constant at the given temperature and pressure, the value of the
                 LPL, the value of the HPL and the value of the total concentration.
    """
    _k0 = kinetic_constant_base(params[0, 0:3], T)
    _kInf = kinetic_constant_base(params[1, 0:3], T)

    _M = P / 0.08206 / T * (1/1000)  # P [atm], T [K] -> M [mol/cm3/s]
    _Pr = _k0 * _M / _kInf

    operand = (_k0, _kInf, _Pr, _M, params)

    def troe(operand):
        _k0, _kInf, _Pr, _M, params = operand
        A = params[2][0]
        T3 = params[2][1]
        T1 = params[2][2]
        T2 = params[2][3] # T2 is optional in CHEMKIN here we feed 0, when not needed

        logFcent = jnp.log10((1 - A) * jnp.exp(-T/T3) + A * jnp.exp(-T/T1) + jnp.exp(-T2/T))

        c = -0.4 - 0.67 * logFcent
        n = 0.75 - 1.27 * logFcent
        f1 = ((jnp.log10(_Pr) + c) / (n - 0.14 * (jnp.log10(_Pr) + c)))**2
        F = 10**(logFcent / (1 + f1))

        _k_troe = (_kInf * (_Pr / (1 + _Pr))) * F

        return (_k_troe, _k0, _kInf, _M)

    def lindemann(operand):
        _k0, _kInf, _Pr, _M, _ = operand
        F = 1
        _k_lindemann = (_kInf * (_Pr / (1 + _Pr))) * F

        return (_k_lindemann, _k0, _kInf, _M)

    return lax.cond( jnp.shape(params)[0] == 3, troe, lindemann, operand)

@jit
def compute_fall_off(falloff: jnp.ndarray, T_range: jnp.ndarray, P_range: jnp.ndarray) -> jnp.ndarray:
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
        ...         k_troe[i, j] = kinetic_constant_fall_off(troe, t, p)

        This function address this and perform the operation in a vectorized way thus the code will be:
        >>> import jax.numpy as jnp
        >>> T_range = jnp.linspace(500, 2500, 30)
        >>> P_range = jnp.logspace(-1, 2, 30)
        >>> troe = jnp.array([[1.135E+36, -5.246, 1704.8], [6.220E+16, -1.174, 635.80], [0.405, 1120., 69.6]])
        >>> k_troe = compute_fall_off(troe, T_range, P_range)

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
        _kfalloff, _, _, _ = kinetic_constant_fall_off(falloff, t, p)
        return _kfalloff
    compute_single_t_fixed = vmap(lambda p: vmap(lambda t: compute_single(t, p))(T_range))
    k_fall_off = compute_single_t_fixed(P_range)
    return k_fall_off
