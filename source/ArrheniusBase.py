import numpy as np
from multipledispatch import dispatch

class Arrhenius:
    def __init__(self, params: dict):
        self.R = 1.987;

        self.A = params["A"]
        self.b = params["b"]
        self.Ea = params["Ea"]

    @dispatch(float)
    def KineticConstant(self, T: float) -> float:
        return self.A * T**self.b * np.exp(-self.Ea/self.R/T)
