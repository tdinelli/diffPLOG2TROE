import matplotlib.pyplot as plt
import jax.numpy as jnp
import sys
sys.path.append("/Users/tdinelli/Documents/GitHub/PLOG_Converter")
from source.Refitter import fit

"""
TODO: Add description
"""

constant = jnp.array([
    [1.00E-01, 7.23E+29, -5.32E+00, 110862.4],
    [1.00E+00, 3.50E+30, -5.22E+00, 111163.3],
    [1.00E+01, 1.98E+31, -5.16E+00, 111887.8],
    [1.00E+02, 2.69E+31, -4.92E+00, 112778.7],
], dtype=jnp.float64)

k_troe_opt, k_plog = fit(plog=constant)

ratio = k_troe_opt / k_plog

T_range = jnp.array([500, 750,1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500])
P_range = jnp.array([0.1, 0.4, 0.7, 1, 2, 5, 10, 20, 30, 40, 50, 70, 100])
P_range = jnp.array([0.1, 0.4, 0.7, 1, 2, 5])

fig, ax = plt.subplots()

c = ax.imshow(ratio, cmap='hot', interpolation='nearest')
# ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)

plt.show()
