from jax import jit, vmap
import jax.numpy as jnp
from .arrhenius_base import kinetic_constant_base


@jit
def kinetic_constant_threebody(threebody_constant: tuple, T: jnp.float64, P: jnp.float64) -> jnp.float64:
    """ """

    params = threebody_constant

    _M = P / 0.08206 / T * (1 / 1000)  # P [atm], T [K] -> M [mol/cm3/s]
    _k_threebody = kinetic_constant_base(params, T) * _M

    return (_k_threebody, _M)


@jit
def compute_threebody(threebody_constant: tuple, T_range: jnp.ndarray, P_range: jnp.ndarray) -> jnp.ndarray:
    """ """

    def compute_single(t, p):
        _k_threebody, _, = kinetic_constant_threebody(threebody_constant, t, p)
        return _k_threebody

    compute_single_t_fixed = vmap(lambda p: vmap(lambda t: compute_single(t, p))(T_range))
    k_threebody = compute_single_t_fixed(P_range)
    return k_threebody
