import numpy as np
from multipledispatch import dispatch
from .ArrheniusBase import Arrhenius

class PressureLogarithmic(Arrhenius):

    def __init__(self, params: list) -> None:
        self.constants = [Arrhenius(i) for i in params]
        self.n = len(self.constants) # Number of pressure levels
        self.p = [i["P"] for i in params]
        self.p.sort() # Im not sure if its needed but anyway better safe than sorry

        isADuplicate = len(self.p) != len(set(self.p))
        if isADuplicate is True:
            raise ValueError("The PLOG reaction provided is a duplicate. This is not handled yet!")

        self.lnP = [np.log(i) for i in self.p]

    @dispatch(float, float)
    def KineticConstant(self, T: float, P: float) -> float:
        if P <= self.p[0]:
            return self.constants[0].KineticConstant(T)
        elif P >= self.p[-1]:
            return self.constants[-1].KineticConstant(T)
        else:
            # 1. Indentify interval in the pressure levels
            pIndex = 1
            for i, j in enumerate(self.p):
                if P < j:
                    pIndex = i - 1
                    break

            # 2. Compute lower and upper pressure level Kinetic Constant
            log_k1 = np.log(self.constants[pIndex].KineticConstant(T))
            log_k2 = np.log(self.constants[pIndex+1].KineticConstant(T))

            # 3. Logarithmic Interpolation
            return np.exp(log_k1 + (log_k2 - log_k1) * (np.log(P) - self.lnP[pIndex]) / (self.lnP[pIndex+1] - self.lnP[pIndex]))
