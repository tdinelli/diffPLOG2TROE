"""
Testing of the computation of the kinetic constant of the following reaction expressed using the PLOG formalism.

NH3=H+NH2               3.4970e+30   -5.224        111163.30
    PLOG / 1.000000e-01 7.230000e+29 -5.316000e+00 1.108624e+05 /
    PLOG / 1.000000e+00 3.497000e+30 -5.224000e+00 1.111633e+05 /
    PLOG / 1.000000e+01 1.975000e+31 -5.160000e+00 1.118878e+05 /
    PLOG / 1.000000e+02 2.689000e+31 -4.920000e+00 1.127787e+05 /
"""


import os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from .test_utils import load_data_matrix
from diffPLOG2TROE.pressure_logarithmic import kinetic_constant_plog, compute_plog

current_file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_file_path, "data", "PLOG.csv")


plog = jnp.array([
    [1.000000e-01, 7.230000e+29, -5.316000e+00, 1.108624e+05],
    [1.000000e+00, 3.497000e+30, -5.224000e+00, 1.111633e+05],
    [1.000000e+01, 1.975000e+31, -5.160000e+00, 1.118878e+05],
    [1.000000e+02, 2.689000e+31, -4.920000e+00, 1.127787e+05],
], dtype=jnp.float64)


def test_plog_computation():
    T_range = jnp.array([500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500], dtype=jnp.float64)
    P_range = jnp.array([0.1, 0.4, 0.7, 1, 2, 5, 10, 20, 30, 40, 50, 70, 100], dtype=jnp.float64)
    data = load_data_matrix(data_path)
    k_plog = jnp.empty_like(data)
    for i, p in enumerate(P_range):
        for j, t in enumerate(T_range):
            k_plog = k_plog.at[i, j].set(kinetic_constant_plog(plog, t, p))

    assert jnp.allclose(data, k_plog, rtol=1e-5)


def test_vectorized_plog_computation():
    T_range = jnp.array([500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500], dtype=jnp.float64)
    P_range = jnp.array([0.1, 0.4, 0.7, 1, 2, 5, 10, 20, 30, 40, 50, 70, 100], dtype=jnp.float64)
    data = load_data_matrix(data_path)
    k_plog = compute_plog(plog, T_range, P_range)

    assert jnp.allclose(data, k_plog, rtol=1e-5)
