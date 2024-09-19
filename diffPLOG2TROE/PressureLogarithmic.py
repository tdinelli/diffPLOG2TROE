from jax import jit, lax, vmap, debug
import jax.numpy as jnp
from .ArrheniusBase import kinetic_constant_base


@jit
def kinetic_constant_plog(params: jnp.ndarray, T: jnp.float64, P: jnp.float64) -> jnp.float64:
    """
    Computes the kinetic constant for a PLOG model using logarithmic interpolation between pressure levels. This
    function is optimized with JAX's Just-In-Time (JIT) compilation.

    Args:
        params (jnp.ndarray): An array of arrays where each entry contains the pressure level and its associated
                              parameters (A, b, Ea). As reported in the following example:

                              CHEMKIN:
                              NH3=H+NH2  3.4970e+30   -5.224    111163.30
                                  PLOG / 1.000000e-01 7.230000e+29 -5.316000e+00 1.108624e+05 /
                                  PLOG / 1.000000e+00 3.497000e+30 -5.224000e+00 1.111633e+05 /
                                  PLOG / 1.000000e+01 1.975000e+31 -5.155000e+00 1.118878e+05 /
                                  PLOG / 1.000000e+02 2.689000e+31 -4.920000e+00 1.127787e+05 /

                              Internal PLOG representation:
                              plog = jnp.array([
                                  [1.000000e-01, 7.230000e+29, -5.316000e+00, 1.108624e+05],
                                  [1.000000e+00, 3.497000e+30, -5.224000e+00, 1.111633e+05],
                                  [1.000000e+01, 1.975000e+31, -5.155000e+00, 1.118878e+05],
                                  [1.000000e+02, 2.689000e+31, -4.920000e+00, 1.127787e+05],
                              ], dtype=jnp.float64)

        T (jnp.float64): Temperature value for which the kinetic constant is computed.
        P (jnp.float64): Pressure value for which the kinetic constant is computed.

    Returns:
        (jnp.float64): The value of the computed kinetic constant at the given temperature and pressure.
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

    Examples:
        Intuitively if someone needs to compute the value of the kinetic constant of a PLOG reaction for different
        pressures and temperature the naive and straightforward implementation would be:
        >>> import jax.numpy as jnp
        >>> T_range = jnp.linspace(500, 2500, 30)
        >>> P_range = jnp.logspace(-1, 2, 30)
        >>> k_plog = jnp.empty(shape=(30, 30))
        >>> plog = jnp.array([[1.00E-01, 7.23E+29, -5.32E+00, 110862.4], [1.00E+00, 3.50E+30, -5.22E+00, 111163.3], [1.00E+01, 1.98E+31, -5.16E+00, 111887.8], [1.00E+02, 2.69E+31, -4.92E+00, 112778.7]])
        >>> for i, t in enumerate(T_range):
        ...     for j, p in enumerate(P_range):
        ...         k_plog[i, j] = kinetic_constant_plog(plog, t, p)

        This function address this and perform the operation in a vectorized way thus the code will be:
        >>> import jax.numpy as jnp
        >>> T_range = jnp.linspace(500, 2500, 30)
        >>> P_range = jnp.logspace(-1, 2, 30)
        >>> plog = jnp.array([[1.00E-01, 7.23E+29, -5.32E+00, 110862.4], [1.00E+00, 3.50E+30, -5.22E+00, 111163.3], [1.00E+01, 1.98E+31, -5.16E+00, 111887.8], [1.00E+02, 2.69E+31, -4.92E+00, 112778.7]])
        >>> k_plog = compute_plog(plog, T_range, P_range)

    Args:
        plog (jnp.ndarray): Internal representation of the PLOG formalism.
        T_range (jnp.ndarray): Array containing the temperature values where we need to compute the kinetic constant in
                               Kelvin.
        P_range (jnp.ndarray): Array containing the pressure values where we need to compute the kinetic constant.

    Returns:
        jnp.ndarray: Two dimensional array containing the value of the kinetic constant at each feeded value of pressure
                     and temperature. Keep in mind that columns are for the same value of tempreature and rows for the
                     same value of pressure.
    """
    def compute_single(t, p):
        _kplog = kinetic_constant_plog(plog, t, p)
        return _kplog
    compute_single_t_fixed = vmap(lambda p: vmap(lambda t: compute_single(t, p))(T_range))
    k_plog = compute_single_t_fixed(P_range)
    return k_plog
