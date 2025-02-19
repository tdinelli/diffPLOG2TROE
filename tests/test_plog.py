import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)

import os


current_file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_file_path, "data", "PLOG.csv")

import sys

from utils import load_data_matrix


sys.path.append("/Users/tdinelli/Documents/GitHub/diffPLOG2TROE")
from diffPLOG2TROE.rate_constants import Plog


plog_constant = {
    "name": "NH3=NH2+H",
    "type": "plog",
    "rate-constant": {
        "coefficients": [
            [1.000000e-01, 7.23e29, -5.316, 110862.4],
            [1.000000e00, 3.497e30, -5.224, 111163.3],
            [1.000000e01, 1.975e31, -5.16, 111887.8],
            [1.000000e02, 2.689e31, -4.92, 112778.7],
        ]
    },
}


def test_plog_computation():
    T_range = jnp.array([500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500], dtype=jnp.float64)
    P_range = jnp.array([0.1, 0.4, 0.7, 1, 2, 5, 10, 20, 30, 40, 50, 70, 100], dtype=jnp.float64)
    data = load_data_matrix(data_path)
    plog = Plog(plog_constant)
    k_plog = plog.kinetic_constant(T_range, P_range)
    for i in range(len(data)):
        assert jnp.allclose(data[i], k_plog[i])
