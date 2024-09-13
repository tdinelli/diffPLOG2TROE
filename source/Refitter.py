import time  # timeit maybe is better
import jax as jax
import jax.numpy as jnp
from scipy.optimize import curve_fit, minimize
from .ArrheniusBase import arrhenius_fit
from .PressureLogarithmic import kinetic_constant_plog
# from .FallOff import kinetic_constant


class Refitter:
    def __init__(self, plog: jnp.ndarray, fit_type: str) -> None:
        self.fg = jnp.array([.0, .0, .0, .0, .0, .0, 1, 1.0, 1.0e+10, 1.0e+10])
        # self.fg = jnp.array([1.380332E+33, -4.32, 1.105952E+05, 1.18231785E+37, 9.63743640E-01, 1.50102675E+05,
        #                      5.79496168E-04, 1.22837204E+02, 1.0e+30, 1.0e+30])
        self.refitted_constant = {
            "LPL": [.0, .0, .0],
            "HPL": [.0, .0, .0],
            "Coefficients": {"A": 1., "T3": 1.0e+3, "T1": 1.0e+3, "T2": 1.0e+3},  # "T2" is optional should I give the option(?)
            "Type": "TROE",
        }

        if fit_type == "FallOff":
            self.isFallOff = True
        elif fit_type == "CABR":
            self.isCabr = True
        else:
            raise ValueError("Unknown fit type! Available are: FallOff | CABR")

        n_range_points = 50
        self.T_range = jnp.linspace(500, 2500, n_range_points)
        # -1, 2 as extreme values in logspace base 10, means that the first pressure value is equal to 0.1 atm and the
        # last one to 100 atm before was -> self.P_range = jnp.linspace(0.1, 100, n_range_points)
        self.P_range = jnp.logspace(-1, 2, n_range_points)

        # Memory allocation
        self.k_plog = jnp.empty(shape=(n_range_points, n_range_points))
        self.k_troe = jnp.empty(shape=(n_range_points, n_range_points))
        self.k0 = jnp.empty(shape=(n_range_points, n_range_points))
        self.kInf = jnp.empty(shape=(n_range_points, n_range_points))

        start_time = time.time()
        for i, p in enumerate(self.P_range):
            for j, t in enumerate(self.T_range):
                self.k_plog = self.k_plog.at[i, j].set(kinetic_constant_plog(plog, t, p))
        end_time = time.time()
        execution_time = end_time - start_time
        self.plog = plog

        print("==============================================")
        print(" PLOG to FallOff/CABR refritter          :)   ")
        print("==============================================")
        print(" Time to compute the PLOG constant: {:.6f} s\n".format(execution_time))

    def ComputePressureLimits(self):
        """ This function compute the First guess Arrhenius parameters for the HPL and LPL. The computation is done in
        such a way that given the pressure and temperature range where we want to refit the PLOG expression into a
        FallOff or a CABR the kinetic constant is computed all along the Temperature interval and at the extreme value
        of the pressure interval
        """
        k0_fg = jnp.array([kinetic_constant_plog(self.plog, i, self.P_range[0]) for i in self.T_range])
        kInf_fg = jnp.array([kinetic_constant_plog(self.plog, i, self.P_range[-1]) for i in self.T_range])

        A0_fg, b0_fg, Ea0_fg, R2adj0 = arrhenius_fit(k0_fg, self.T_range)
        AInf_fg, bInf_fg, EaInf_fg, R2adjInf = arrhenius_fit(kInf_fg, self.T_range)

        self.fg = self.fg.at[0].set(A0_fg)
        self.fg = self.fg.at[1].set(b0_fg - 1)  # A.F. did that
        self.fg = self.fg.at[2].set(Ea0_fg)

        self.plog = self.fg.at[3].set(AInf_fg)
        self.plog = self.fg.at[4].set(bInf_fg - 1)  # A.F. did that
        self.plog = self.fg.at[5].set(EaInf_fg)

        print(" Computing first guesses for the LPL and HPL")
        print("  * Adjusted R2 for the LPL: {:.3}".format(R2adj0))
        print("    - A: {:.3e}, b: {:.3}, Ea: {:.3e}".format(A0_fg, b0_fg + 1, Ea0_fg))
        print("  * Adjusted R2 for the HPL: {:.3}".format(R2adjInf))
        print("    - A: {:.3e}, b: {:.3}, Ea: {:.3e}\n".format(AInf_fg, bInf_fg + 1, EaInf_fg))

    def ObjectiveFunction(self, x) -> float:
        start_time = time.time()
        # Unpacking optimization parameters
        A0, b0, Ea0, AInf, bInf, EaInf, A, T3, T1, T2 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]

        self.refitted_constant["LPL"]["A"] = A0
        self.refitted_constant["LPL"]["b"] = b0
        self.refitted_constant["LPL"]["Ea"] = Ea0

        self.refitted_constant["HPL"]["A"] = AInf
        self.refitted_constant["HPL"]["b"] = bInf
        self.refitted_constant["HPL"]["Ea"] = EaInf

        self.refitted_constant["Coefficients"]["A"] = A
        self.refitted_constant["Coefficients"]["T3"] = T3
        self.refitted_constant["Coefficients"]["T1"] = T1
        self.refitted_constant["Coefficients"]["T2"] = T2

        troe = FallOff(self.refitted_constant)
        for i, p in enumerate(self.P_range):
            for j, t in enumerate(self.T_range):
                self.k_troe = self.k_troe.at[i, j].set(troe.KineticConstant(t, p))
                self.k0 = self.k0.at[i, j].set(troe.k0)
                self.kInf = self.kInf.at[i, j].set(troe.kInf)

        # The following divide will set the values where self.k_plog == 0 equal to 0 in order to handle by 0 division
        # self.ratio = jnp.divide(self.k_troe, self.k_plog, out=jnp.zeros_like(self.k_troe), where=self.k_plog != 0)
        self.ratio = self.k_troe / self.k_plog

        print("k0:    ", self.k0[0, 0])
        print("kInf:  ", self.kInf[0, 0])
        print("T:     ", self.T_range)
        print("P:     ", self.P_range)
        print("TROE:  ", self.k_troe[:, 0])
        print("PLOG:  ", self.k_plog[:, 0])
        print("ratio: ", self.ratio[:, 0])
        exit()

        self.squared_errors = (self.ratio - 1)**2

        obj = jnp.mean(self.squared_errors)
        end_time = time.time()
        execution_time = end_time - start_time
        print("   - {:.6f}\t{:.6f}".format(obj, execution_time))

        return obj

    def fit(self):
        self.ComputePressureLimits()
        print(" * ObjFunction value\tTime")
        # res = minimize(self.ObjectiveFunction, self.fg, method="nelder-mead", tol = 1e-6, options = {"disp": True,
        #                                                                                              "maxiter": 10000,
        #                                                                                              "maxfev": 10000})
