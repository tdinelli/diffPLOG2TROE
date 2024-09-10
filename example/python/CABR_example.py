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
CH3 + CH3(+M) = C2H5 + H(+M)

Example reported within the CHEMKIN manual
"""

constant_cabr = {
    "LPL": {"A": 10**12.698, "b": 0.099, "Ea": 10600},
    "HPL": {"A": 10**-6.42, "b": 4.838, "Ea": 7710},
    "Coefficients": {"A": 1.641, "T3": 4334, "T1": 2725},  # "T2" is optional
    "Type": "CABR",
    "Lindemann": False
}

constant_lindemann = {
    "LPL": {"A": 10**12.698, "b": 0.099, "Ea": 10600},
    "HPL": {"A": 10**-6.42, "b": 4.838, "Ea": 7710},
    "Coefficients": {"A": 1.641, "T3": 4334, "T1": 2725},  # "T2" is optional
    "Type": "CABR",
    "Lindemann": True
}

cabr = FallOff(constant_cabr)
lindemann = FallOff(constant_lindemann)

kc = []
k0 = []
kInf = []
M = []
kcl = []

T = np.linspace(1000., 1000., 30000)
P = np.linspace(0.0001, 100., 30000)

for i in P:
    kc.append(cabr.KineticConstant(1000., float(i)))
    k0.append(cabr.k0)
    kInf.append(cabr.kInf / cabr.M)
    M.append(cabr.M)

    kcl.append(lindemann.KineticConstant(1000., float(i)))

plt.plot(M, kc, 'b-o', label="CABR form")
plt.plot(M, kcl, 'k-o', label="Lindemann form")
plt.plot(M, kInf, "g--", label="HPL / [M]")
plt.plot(M, k0, "r--", label="LPL")
plt.yscale("log")
plt.xscale("log")
plt.ylim([min(kc), k0[-1]*1.1])
plt.legend()
plt.show()
