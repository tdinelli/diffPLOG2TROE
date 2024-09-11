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

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/Users/tdinelli/Documents/GitHub/PLOG_Converter")
from source.Refitter import Refitter

"""
TODO: Add description
"""

constant = [
    {"P": 1.00E-01, "A": 7.23E+29, "b": -5.32E+00, "Ea": 110862.4},
    {"P": 1.00E+00, "A": 3.50E+30, "b": -5.22E+00, "Ea": 111163.3},
    {"P": 1.00E+01, "A": 1.98E+31, "b": -5.16E+00, "Ea": 111887.8},
    {"P": 1.00E+02, "A": 2.69E+31, "b": -4.92E+00, "Ea": 112778.7},
]

refitter = Refitter(plog=constant, fit_type="FallOff")
refitter.fit()

n_range_points = 300
T_range = np.linspace(500, 2500, n_range_points)
P_range = np.linspace(0.1, 100, n_range_points)

fig, ax = plt.subplots()

c = ax.imshow(refitter.ratio, cmap='hot', interpolation='nearest')
# ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)

plt.show()
