import numpy as np
from multipledispatch import dispatch
from .ArrheniusBase import Arrhenius

class Troe(Arrhenius):
    def __init__(self, params: dict):

        self.isExplicitlyEnhanced = False
        self.isFourParameters = False
        self.lpl = Arrhenius(params["LPL"])
        self.hpl = Arrhenius(params["HPL"])

        self.A = params["Troe"]["A"]
        self.T3 = params["Troe"]["T3"]
        self.T1 = params["Troe"]["T1"]
        if len(params["Troe"]) == 4:
            self.isFourParameters = True
            self.T2 = params["Troe"]["T2"]

        if "efficiencies" in params:
            self.isExplicitlyEnhanced = True

        if self.isExplicitlyEnhanced is True:
            raise Exception("Troe expression with explicit collider is not handled yet!")

        self._k0 = 0 # LPL constant
        self._kInf = 0 # HPL constant

        # ==============================
        # Just for testing TOBE REMOVED
        # ==============================
        self._M = 0

    @dispatch(float, float)
    def KineticConstant(self, T: float, P: float) -> float:
        self._k0 = self.lpl.KineticConstant(T)
        self._kInf = self.hpl.KineticConstant(T)
        self._M = P / 0.08206 / T * (1/1000) # P [atm], T [K] -> M [mol/cm3/s]
        Pr = self._k0 * self._M / self._kInf

        if self.isFourParameters is True:
            Fcent = (1 - self.A) * np.exp(-T/self.T3) + self.A * np.exp(-T/self.T1) + np.exp(-self.T2/T)
        else:
            Fcent = (1 - self.A) * np.exp(-T/self.T3) + self.A * np.exp(-T/self.T1)

        c = -0.4 - 0.67 * np.log(Fcent)
        n = 0.75 - 1.27 * np.log(Fcent)

        f1 = ((np.log(Pr) + c) / (n - 0.14 * (np.log(Pr) + c)))**2

        F = np.exp(np.log(Fcent) / (1 + f1))

        return self._kInf * (Pr / (1 + Pr)) * F

    @property
    def k0(self):
        return self._k0

    @property
    def kInf(self):
        return self._kInf

    @property
    def M(self):
        return self._M
