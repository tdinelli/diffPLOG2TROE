import numpy as np
from multipledispatch import dispatch
from .ArrheniusBase import Arrhenius

# -------------------------
# Missing SRI formulation
# -------------------------
class FallOff(Arrhenius):
    def __init__(self, params: dict):

        self.isExplicitlyEnhanced = False

        self.isLindemann = False
        self.isTroe = False
        self.isCabr = False

        self.isFourParameters = False
        self.lpl = Arrhenius(params["LPL"])
        self.hpl = Arrhenius(params["HPL"])

        self.A = params["Coefficients"]["A"]
        self.T3 = params["Coefficients"]["T3"]
        self.T1 = params["Coefficients"]["T1"]
        if len(params["Coefficients"]) == 4:
            self.isFourParameters = True
            self.T2 = params["Coefficients"]["T2"]

        if "efficiencies" in params:
            self.isExplicitlyEnhanced = True

        if self.isExplicitlyEnhanced is True:
            raise Exception("Troe expression with explicit collider is not handled yet!")

        if params["Type"] == "TROE":
            self.isTroe = True
        elif params["Type"] == "CABR":
            self.isCabr = True
        elif params["Type"] == "SRI":
            raise ValueError("SRI formulation not implemented yet!")
        else:
            raise ValueError("Unknown type. Allowed  are: TROE | CABR")

        if params["Lindemann"] == True:
            self.isLindemann = True

        self._k0 = 0 # LPL constant
        self._kInf = 0 # HPL constant

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

        if self.isLindemann is False:
            F = np.exp(np.log(Fcent) / (1 + f1))
        else:
            F = 1

        if self.isTroe is True:
            return self._kInf * (Pr / (1 + Pr)) * F

        if self.isCabr is True:
            return self._k0 * (1 / (1 + Pr)) * F

    @property
    def k0(self):
        return self._k0

    @property
    def kInf(self):
        return self._kInf

    @property
    def M(self):
        return self._M
