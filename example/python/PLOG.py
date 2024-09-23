import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/tdinelli/Documents/GitHub/diffPLOG2TROE/")
from diffPLOG2TROE.pressure_logarithmic import kinetic_constant_plog
from diffPLOG2TROE.arrhenius_base import kinetic_constant_base
from diffPLOG2TROE.refitter import refit_plog

"""
Pressure dependent kinetic constant for the reaction NH3=NH2+H as its reported whithin the CRECK chemical kinetic model.
Defined as a jax.numpy.array as follow:
    constant = [
        [1.00E-01, 7.23E+29, -5.32E+00, 110862.4],
        [1.00E+00, 3.50E+30, -5.22E+00, 111163.3],
        [1.00E+01, 1.98E+31, -5.16E+00, 111887.8],
        [1.00E+02, 2.69E+31, -4.92E+00, 112778.7],
    ]
Here each sublist represents a single pressure level in where the number of elements must be four. The first element in
the list represents the pressure the following three are the Arrhenius parameters in this order A, b, Ea.
"""

constant = jnp.array([[1.00E-01, 7.23E+29, -5.32E+00, 110862.4],
                      [1.00E+00, 3.50E+30, -5.22E+00, 111163.3],
                      [1.00E+01, 1.98E+31, -5.16E+00, 111887.8],
                      [1.00E+02, 2.69E+31, -4.92E+00, 112778.7]])

# C2H2+C3H5-A=>C2H4+C3H3   9.4500e+08      1.430           29879.55
#  PLOG /  1.000000e-02    9.450000e+08    1.430000e+00    2.987955e+04     /
#  PLOG /  1.000000e+00    3.010000e+09    1.300000e+00    3.029287e+04     /
#  PLOG /  1.000000e+02    8.630000e+17    -1.010000e+00   3.800488e+04     /
constant = jnp.array([[1.000000e-02, 9.450000e+08, 1.430000e+00 , 2.987955e+04],
                      [1.000000e+00, 3.010000e+09, 1.300000e+00, 3.029287e+04],
                      [1.000000e+02, 8.630000e+17, -1.010000e+00, 3.800488e+04]])


T = jnp.array([500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500])
P = jnp.array([1])

for j in P:
    kc = []
    for i in T:
        kc.append(kinetic_constant_plog(constant, i, j))
    plt.plot(T, kc)

A, b, Ea, R2adj = refit_plog(constant, 1.)
print(A)
print(b)
print(Ea)
kc = [kinetic_constant_base(jnp.array([A, b, Ea]), i) for i in T]
plt.plot(T, kc)


plt.yscale('log')
plt.show()
