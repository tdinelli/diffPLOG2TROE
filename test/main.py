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
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.PressureLogarithmic import PressureLogarithmic

"""
Pressure dependent kinetic constant for the reaction NH3=NH2+H as its reported whithin the CRECK chemical kinetic model.
"""

constant = [
    {"P": 1.00E-01, "A": 7.23E+29, "b": -5.32E+00, "Ea": 110862.4},
    {"P": 1.00E+00, "A": 3.50E+30, "b": -5.22E+00, "Ea": 111163.3},
    {"P": 1.00E+01, "A": 1.98E+31, "b": -5.16E+00, "Ea": 111887.8},
    {"P": 1.00E+02, "A": 2.69E+31, "b": -4.92E+00, "Ea": 112778.7},
]

plog = PressureLogarithmic(params=constant)
T = [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500]
P = [0.1, 0.4, 0.7, 1, 2, 5, 10, 20, 30, 40, 50, 70, 100]

for j in P:
    kc = []
    for i in T:
        kc.append(plog.KineticConstant(float(i), float(j)))
    print(kc)
    print(T)
    plt.plot(T, kc)

plt.yscale('log')
plt.show()
