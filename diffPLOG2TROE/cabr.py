from jax import jit, lax, vmap
import jax.numpy as jnp
from .arrhenius_base import kinetic_constant_base
from .constant_fit_type import lindemann, troe, sri


@jit
def kinetic_constant_cabr(falloff_constant: tuple, T: jnp.float64, P: jnp.float64) -> jnp.float64:

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
    _k_cabr = (_k0 * (1 / (1 + _Pr))) * F
    return (_k_cabr, _k0, _kInf, _M)


@jit
def compute_cabr(cabr_constant: tuple, T_range: jnp.ndarray, P_range: jnp.ndarray) -> jnp.ndarray:
    def compute_single(t, p):
        _kcabr, _, _, _ = kinetic_constant_cabr(cabr_constant, t, p)
        return _kcabr
    compute_single_t_fixed = vmap(lambda p: vmap(lambda t: compute_single(t, p))(T_range))
    _k_cabr, _, _, _ = compute_single_t_fixed(P_range)
    return _k_cabr
