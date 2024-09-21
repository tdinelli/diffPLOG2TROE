import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/tdinelli/Documents/GitHub/diffPLOG2TROE/")
from diffPLOG2TROE.PressureLogarithmic import kinetic_constant_plog

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

T = jnp.array([500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500])
P = jnp.array([0.1, 0.4, 0.7, 1, 2, 5, 10, 20, 30, 40, 50, 70, 100])

st = time.time()
kinetic_constant_plog(constant, T[0], P[0])
et = time.time()
print(et -st)

st = time.time()
kinetic_constant_plog(constant, T[0], P[0])
et = time.time()
print(et -st)
exit()
for j in P:
    kc = []
    for i in T:
        kc.append(kinetic_constant_plog(constant, i, j))
    plt.plot(T, kc)

plt.yscale('log')
plt.show()
