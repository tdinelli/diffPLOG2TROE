import numpy as np
from multipledispatch import dispatch
from .ArrheniusBase import Arrhenius


class Cabr(Arrhenius):

    def __init__(self, params: dict):

        self.isExplicitlyEnhanced = False

        self.isLindemann = False
        self.isTroe = False
        self.isSRI = False

        self.isFourParameters = False
        self.lpl = Arrhenius(params["LPL"])
        self.hpl = Arrhenius(params["HPL"])

        if "efficiencies" in params:
            self.isExplicitlyEnhanced = True

        if self.isExplicitlyEnhanced is True:
            raise Exception(
                "Troe expression with explicit collider is not handled yet!")

        if params["Type"] == "TROE":
            self.isTroe = True
            self.A = params["Coefficients"]["A"]
            self.T3 = params["Coefficients"]["T3"]
            self.T1 = params["Coefficients"]["T1"]
            if len(params["Coefficients"]) == 4:
                self.isFourParameters = True
                self.T2 = params["Coefficients"]["T2"]
        elif params["Type"] == "Lindemann":
            self.isLindemann = True
        elif params["Type"] == "SRI":
            raise ValueError("SRI formulation not implemented yet!")
        else:
            raise ValueError(
                "Unknown type. Allowed  are: TROE | Lindemann | SRI")

        self._k0 = 0  # LPL constant
        self._kInf = 0  # HPL constant

        self._M = 0

    @dispatch(float, float)
    def KineticConstant(self, T: float, P: float) -> float:
        self._k0 = self.lpl.KineticConstant(T)
        self._kInf = self.hpl.KineticConstant(T)
        self._M = P / 0.08206 / T * (1/1000)  # P [atm], T [K] -> M [mol/cm3/s]
        Pr = self._k0 * self._M / self._kInf

        if self.isTroe is True:
            if self.isFourParameters is True:
                logFcent = np.log((1 - self.A) * np.exp(-T/self.T3) +
                                  self.A * np.exp(-T/self.T1) + np.exp(-self.T2/T))
            else:
                logFcent = np.log(
                    (1 - self.A) * np.exp(-T/self.T3) + self.A * np.exp(-T/self.T1))

            c = -0.4 - 0.67 * logFcent
            n = 0.75 - 1.27 * logFcent

            f1 = ((np.log(Pr) + c) / (n - 0.14 * (np.log(Pr) + c)))**2

            F = np.exp(logFcent / (1 + f1))
            return self._k0 * (1 / (1 + Pr)) * F

        if self.isLindemann is True:
            F = 1
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
