import numpy as np
from multipledispatch import dispatch

class Arrhenius:
    def __init__(self, params: dict):
        self.R = 1.987;

        self._A = params["A"]
        self._b = params["b"]
        self._Ea = params["Ea"]

    @dispatch(float)
    def KineticConstant(self, T: float) -> float:
        return self._A * T**self._b * np.exp(-self._Ea/self.R/T)

    # @property
    # def A(self):
    #     return self._A
    #
    # @property
    # def b(self):
    #     return self._b
    #
    # @property
    # def Ea(self):
    #     return self._Ea
