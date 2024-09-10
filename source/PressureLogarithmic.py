import numpy as np
from multipledispatch import dispatch
from .ArrheniusBase import Arrhenius


class PressureLogarithmic(Arrhenius):

    def __init__(self, params: list) -> None:
        self.constants = [Arrhenius(i) for i in params]
        self.n = len(self.constants)  # Number of pressure levels
        self._P = [i["P"] for i in params]
        self._P.sort()  # Im not sure if its needed but anyway better safe than sorry

        isADuplicate = len(self._P) != len(set(self._P))
        if isADuplicate is True:
            raise ValueError(
                "The PLOG reaction provided is a duplicate. This is not handled yet! I don't have so much time at the moment.")

        self._lnP = [np.log(i) for i in self._P]

    @dispatch(float, float)
    def KineticConstant(self, T: float, P: float) -> float:
        if P <= self._P[0]:
            return self.constants[0].KineticConstant(T)
        elif P >= self._P[-1]:
            return self.constants[-1].KineticConstant(T)
        else:
            # 1. Indentify interval in the pressure levels
            pIndex = 1
            for i, j in enumerate(self._P):
                if P < j:
                    pIndex = i - 1
                    break

            # 2. Compute lower and upper pressure level Kinetic Constant
            log_k1 = np.log(self.constants[pIndex].KineticConstant(T))
            log_k2 = np.log(self.constants[pIndex+1].KineticConstant(T))

            # 3. Logarithmic Interpolation
            return np.exp(log_k1 + (log_k2 - log_k1) * (np.log(P) - self._lnP[pIndex]) / (self._lnP[pIndex+1] - self._lnP[pIndex]))

    @property
    def lnP(self):
        return self._lnP

    @property
    def P(self):
        return self._P
