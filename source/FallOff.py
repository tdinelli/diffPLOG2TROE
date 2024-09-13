from jax import jit
import jax.numpy as jnp
from .ArrheniusBase import kinetic_constant


def fall_off(params: dict):

    is_explicitly_enhanced = False
    is_lindemann = False
    is_troe = False
    is_sri = False
    is_four_params = False
    A = 0
    T3 = 0
    T1 = 0
    T2 = 0

    print(params)

    if "efficiencies" in params:
        is_explicitly_enhanced = True

    if is_explicitly_enhanced is True:
        raise Exception(
            "Troe expression with explicit collider is not handled yet!")

    if params["Type"] == "TROE":
        is_troe = True
        A = jnp.float64(params["Coefficients"]["A"])
        T3 = jnp.float64(params["Coefficients"]["T3"])
        T1 = jnp.float64(params["Coefficients"]["T1"])

        if len(params["Coefficients"]) == 4:
            is_four_params = True
            T2 = jnp.float64(params["Coefficients"]["T2"])

    elif params["Type"] == "Lindemann":
        is_lindemann = True
    elif params["Type"] == "SRI":
        raise ValueError("SRI formulation not implemented yet!")
    else:
        raise ValueError(
            "Unknown type. Allowed  are: TROE | Lindemann | SRI")

    return is_lindemann, is_troe, is_four_params, A, T3, T1, T2
    # N.B. The return should be the following one but most of the things there is not used cause I don't have time to
    #      implement them at the moment
    #      return is_explicitly_enhanced, is_lindemann, is_troe, is_sri, is_four_params, A, T3, T1, T2


@jit
def kinetic_constant(params: dict, T: jnp.float64, P: jnp.float64) -> jnp.float64:

    is_lindemann, is_troe, is_four_params, A, T3, T1, T2 = fall_off(params)

    _k0 = kinetic_constant(params["LPL"], T)
    _kInf = kinetic_constant(params["LPL"], T)

    _M = P / 0.08206 / T * (1/1000)  # P [atm], T [K] -> M [mol/cm3/s]
    _Pr = _k0 * _M / _kInf

    if is_troe is True:
        if is_four_params is True:
            logFcent = jnp.log10((1 - A) * jnp.exp(-T/T3) + A * jnp.exp(-T/T1) + jnp.exp(-T2/T))
        else:
            logFcent = jnp.log10((1 - A) * jnp.exp(-T/T3) + A * jnp.exp(-T/T1))

        c = -0.4 - 0.67 * logFcent
        n = 0.75 - 1.27 * logFcent
        f1 = ((jnp.log10(_Pr) + c) / (n - 0.14 * (jnp.log10(_Pr) + c)))**2
        F = 10**(logFcent / (1 + f1))

        return _kInf * (_Pr / (1 + _Pr)) * F

    if is_lindemann is True:
        F = 1
        return _kInf * (_Pr / (1 + _Pr)) * F
