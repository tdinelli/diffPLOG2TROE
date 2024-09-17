import jax.numpy as jnp
import sys
sys.path.append("/Users/tdinelli/Documents/GitHub/PLOG_Converter")
from source.FallOff import compute_fall_off, kinetic_constant_fall_off


constant_troe = jnp.array([
    [1.2295E+33, 6.0615E-01, 2.9645E+5, .0],  # LPL
    [1.07077E+37, 5.34428E-01, 3.50434E+5, .0],  # HPL
    [.0, 1.512366, 2.72669994E+10, 5.11885045E+04],  # Coefficients "T2" is optional
], dtype=jnp.float64)

T_range = jnp.array([500, 750,1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500])
P_range = jnp.array([0.1, 0.4, 0.7, 1, 2, 5, 10, 20, 30, 40, 50, 70, 100])

# k_troe = kinetic_constant_fall_off(constant_troe, 500., 0.1)
k_troe = compute_fall_off(constant_troe, T_range, P_range)
print(k_troe[:, 0])
