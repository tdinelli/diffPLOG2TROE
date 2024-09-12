from jax import jit
import jax.numpy as jnp
# import jaxlib.xla_extension as xla
# from multipledispatch import dispatch
from .ArrheniusBase import Arrhenius


class PressureLogarithmic(Arrhenius):

    def __init__(self, params: jnp.ndarray):
        self.constants = [Arrhenius(i) for i in params]
        self.n = len(self.constants)  # Number of pressure levels
        self._P = [i["P"] for i in params]
        self._P.sort()  # Im not sure if its needed but anyway better safe than sorry

        isADuplicate = len(self._P) != len(set(self._P))
        if isADuplicate is True:
            raise ValueError(
                "The PLOG reaction provided is a duplicate. This is not handled yet! I don't have so much time at the moment.")

        # self._lnP = [jnp.log(i) for i in self._P]
        self._lnP = jnp.log(jnp.array(self._P))

    # @dispatch(xla.ArrayImpl, xla.ArrayImpl)
    @jit
    def KineticConstant(self, T: jnp.float64, P: jnp.float64) -> jnp.float64:
        if P <= jnp.float64(self._P[0]):
            return self.constants[0].KineticConstant(T)
        elif P >= jnp.float64(self._P[-1]):
            return self.constants[-1].KineticConstant(T)
        else:
            # 1. Indentify interval in the pressure levels
            pIndex = 0
            for pIndex in range(self.n - 1):
                if P < self._P[pIndex + 1]:
                    break

            # 2. Compute lower and upper pressure level Kinetic Constant
            log_k1 = jnp.log(self.constants[pIndex].KineticConstant(T))
            log_k2 = jnp.log(self.constants[pIndex+1].KineticConstant(T))

            # 3. Logarithmic Interpolation
            return jnp.exp(log_k1 + (log_k2 - log_k1) * (jnp.log(P) - self._lnP[pIndex]) / (self._lnP[pIndex+1] - self._lnP[pIndex]))

    @property
    def lnP(self):
        return self._lnP

    @property
    def P(self):
        return self._P
