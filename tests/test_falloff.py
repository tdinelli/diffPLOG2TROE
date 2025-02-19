import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)

import os


current_file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_file_path, "data", "FallOff.csv")

import sys

from utils import load_data_matrix


sys.path.append("/Users/tdinelli/Documents/GitHub/diffPLOG2TROE")
from diffPLOG2TROE.rate_constants import FallOff


falloff_constant = {
    "name": "refitted",
    "type": "falloff",
    "falloff-type": "troe",
    "rate-constant": {
        "hpl-coefficients": [1.8124895700000e+29, -3.909561020, 1.1311387500000e+05],
        "lpl-coefficients": [5.88915776000e+31, -4.00068685000, 1.10139896000e+05],
        "falloff-coefficients": [2.949764370000e-01, 1.606411150000e+02, 8.059921560000e+29, 2.421713850000e+29],
    },
}


def test_falloff_computation():
    T_range = jnp.array([500, 750], dtype=jnp.float64)
    P_range = jnp.array([0.1, 0.4, 0.7, 1, 2, 5, 10, 20, 30, 40, 50, 70, 100], dtype=jnp.float64)
    data = load_data_matrix(data_path)
    constant = FallOff(falloff_constant)
    k_falloff = constant.kinetic_constant(T_range, P_range)

    assert jnp.allclose(data[:, 2], k_falloff[:, 0])
    assert jnp.allclose(data[:, 5], k_falloff[:, 1])

def main():
    test_falloff_computation()


if __name__ == "__main__":
    main()
