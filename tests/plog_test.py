import os
import unittest

import jax.numpy as jnp
import numpy as np

from diffPLOG2TROE.parametrization import Plog


class TestPlog(unittest.TestCase):
    def setUp(self):
        self.T_range = jnp.linspace(300, 3000, 300)
        self.P_range = jnp.logspace(jnp.log10(0.01), jnp.log10(100), 300)
        self.rate_constant = Plog(
            parameters={
                0.01: {"A": 5.02e21, "n": -4.24, "Ea": 898.9},
                0.1: {"A": 5.31e22, "n": -4.24, "Ea": 1184.0},
                0.316: {"A": 1.38e23, "n": -4.22, "Ea": 1376.0},
                1.0: {"A": 3.09e23, "n": -4.17, "Ea": 1621.0},
                3.16: {"A": 5.45e23, "n": -4.09, "Ea": 1911.0},
                10.0: {"A": 6.35e23, "n": -3.97, "Ea": 2222.0},
                31.6: {"A": 3.68e23, "n": -3.75, "Ea": 2501.0},
                100.0: {"A": 7.29e22, "n": -3.41, "Ea": 2660.0},
            },
            name="OH+NO=HONO",
        )

        # ==============================================================================
        # Dataloader
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.1.0", "plog.csv")
        data = np.loadtxt(data_file, delimiter=";")
        data = jnp.array(data)

        self.expected_rate = data

    def test_kinetic_constant(self):
        calculated_rates = self.rate_constant.rate_constant(self.T_range, self.P_range) / 1000
        for i, calculated_rate in enumerate(calculated_rates):
            self.assertTrue(
                jnp.allclose(
                    calculated_rate,
                    self.expected_rate[i],
                    atol=1e-10,
                    rtol=1e-8,
                ),
                "Calculated rate constants for the Arrhenius case don't match reference data",
            )


if __name__ == "__main__":
    unittest.main()
