"""
@Authors:
    Timoteo Dinelli [1]
    [1]: CRECK Modeling Lab, Department of Chemistry, Materials, and Chemical Engineering, Politecnico di Milano
@Contacts:
    timoteo.dinelli@polimi.it
@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: timoteo.dinelli@polimi.it
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/Users/tdinelli/Documents/GitHub/PLOG_Converter")
from source.FallOff import FallOff

"""
CH3 + CH3 (+M) = C2H6 (+M)

Example reported within the CHEMKIN manual


"""

constant_troe = {
    "LPL": {"A": 1.135E+36, "b": -5.246, "Ea": 1704.8},
    "HPL": {"A": 6.220E+16, "b": -1.174, "Ea": 635.80},
    "Coefficients": {"A": 0.405, "T3": 1120., "T1": 69.6},  # "T2" is optional
    "Type": "TROE",
}

constant_lindemann = {
    "LPL": {"A": 1.135E+36, "b": -5.246, "Ea": 1704.8},
    "HPL": {"A": 6.220E+16, "b": -1.174, "Ea": 635.80},
    "Type": "Lindemann",
}

troe = FallOff(constant_troe)
lindemann = FallOff(constant_lindemann)

kc = []
k0 = []
kInf = []
M = []
kcl = []

T = np.linspace(1000., 1000., 30000)
P = np.linspace(0.0001, 100., 30000)

for i in P:
    kc.append(troe.KineticConstant(1000., float(i)))
    k0.append(troe.k0 * troe.M)
    kInf.append(troe.kInf)
    M.append(troe.M)

    kcl.append(lindemann.KineticConstant(1000., float(i)))

plt.plot(M, kc, 'b-o', label="TROE form")
plt.plot(M, kcl, 'k-o', label="Lindemann form")
plt.plot(M, kInf, "g--", label="HPL")
plt.plot(M, k0, "r--", label="LPL*[M]")
plt.yscale("log")
plt.xscale("log")
plt.ylim([min(kc), kInf[-1]*1.1])
plt.legend()
plt.show()
