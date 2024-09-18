from jax import jit, lax, vmap
import jax.numpy as jnp
from .ArrheniusBase import kinetic_constant_base


@jit
def kinetic_constant_cabr(params: jnp.ndarray, T: jnp.float64, P: jnp.float64) -> jnp.float64:

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
        T2 = params[2][3]

        logFcent = jnp.log10((1 - A) * jnp.exp(-T/T3) + A * jnp.exp(-T/T1) + jnp.exp(-T2/T)) # Here is T2

        c = -0.4 - 0.67 * logFcent
        n = 0.75 - 1.27 * logFcent
        f1 = ((jnp.log10(_Pr) + c) / (n - 0.14 * (jnp.log10(_Pr) + c)))**2
        F = 10**(logFcent / (1 + f1))

        _k_troe = (_k0 * (1 / (1 + _Pr))) * F

        return (_k_troe, _k0, _kInf, _M)

    def lindemann(operand):
        _k0, _kInf, _Pr, _M, _ = operand
        F = 1
        _k_lindemann = (_k0 * (1 / (1 + _Pr))) * F

        return (_k_lindemann, _k0, _kInf, _M)

    return lax.cond(jnp.shape(params)[0] == 3, troe, lindemann, operand)

@jit
def compute_cabr(cabr: jnp.ndarray, T_range: jnp.ndarray, P_range: jnp.ndarray) -> jnp.ndarray:
    def compute_single(t, p):
        _kcabr, _, _, _ = kinetic_constant_cabr(cabr, t, p)
        return _kcabr
    compute_single_t_fixed = vmap(lambda p: vmap(lambda t: compute_single(t, p))(T_range))
    _k_cabr, _, _, _ = compute_single_t_fixed(P_range)
    return _k_cabr
