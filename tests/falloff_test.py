"""
Testing of the computation of the kinetic constant of the following reaction expressed using the FallOff formalism.

TODO: Do the same stuff for the Lindemann Formulation.
"""


import os
import jax.numpy as jnp
from .tests_utils import load_data_matrix
from diffPLOG2TROE.FallOff import kinetic_constant_falloff, compute_falloff

current_file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_file_path, "data", "FallOff.csv")

troe = jnp.array([
    [1.782E+60, -1.228E+01, 8.398E+04, .0],
    [3.790E+24, -2.230E+00, 8.807E+04, .0],
    [2.352000E-01, 7.240000E+02, 5.000000E+09, 5.000000E+09]
], dtype=jnp.float64)


def test_falloff_troe_computation():
    T_range = jnp.array([800, 830, 860, 890, 920, 950, 980, 1010, 1040, 1070, 1100, 1130, 1160, 1190, 1220, 1250,
                         1280, 1310, 1340, 1370, 1400, 1430, 1460, 1490, 1520], dtype=jnp.float64)
    P_range = jnp.array([1.], dtype=jnp.float64)
    data = load_data_matrix(data_path)
    k0 = jnp.empty_like(data[:, 0])
    kInf = jnp.empty_like(data[:, 1])
    k_troe = jnp.empty_like(data[:, 2])

    for _, p in enumerate(P_range):
        for j, t in enumerate(T_range):
            _k_troe, _k0, _kInf, _ = kinetic_constant_falloff(troe, t, p)
            k_troe = k_troe.at[j].set(_k_troe)
            k0 = k0.at[j].set(_k0)
            kInf = kInf.at[j].set(_kInf)

    assert jnp.allclose(data[:, 0], k0, rtol=1e-5)
    assert jnp.allclose(data[:, 1], kInf, rtol=1e-5)
    assert jnp.allclose(data[:, 2], k_troe, rtol=1e-5)


def test_vectorized_falloff_troe_computation():
    T_range = jnp.array([800, 830, 860, 890, 920, 950, 980, 1010, 1040, 1070, 1100, 1130, 1160, 1190, 1220, 1250,
                         1280, 1310, 1340, 1370, 1400, 1430, 1460, 1490, 1520], dtype=jnp.float64)
    P_range = jnp.array([1.], dtype=jnp.float64)
    data = load_data_matrix(data_path)
    k_troe = jnp.empty_like(data[:, 2])

    k_troe = compute_falloff(troe, T_range, P_range)

    assert jnp.allclose(data[:, 2], k_troe, rtol=1e-5)
