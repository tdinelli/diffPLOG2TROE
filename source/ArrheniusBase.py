from jax import jit
import jax.numpy as jnp
# import jaxlib.xla_extension as xla
# from multipledispatch import dispatch


class Arrhenius:
    def __init__(self, params: dict):
        self.R = jnp.float64(1.987)

        self._A = jnp.float64(params["A"])
        self._b = jnp.float64(params["b"])
        self._Ea = jnp.float64(params["Ea"])

    # @dispatch(xla.ArrayImpl)
    @jit
    def KineticConstant(self, T: jnp.float64) -> jnp.float64:
        return self._A * T**self._b * jnp.exp(-self._Ea/self.R/T)
