import os
import unittest

import jax.numpy as jnp
import numpy as np

from diffPLOG2TROE.parametrization import Plog


class TestArrhenius(unittest.TestCase):
    def setUp(self):
        self.T_range = jnp.linspace(300, 3000, 300)
        self.P_range = jnp.logspace(jnp.log10(0.01), jnp.log10(100), 300)
        self.rate_constant = Plog(
            parameters=jnp.array([
                [0.01, 5.02e+21, -4.24, 898.9],
                [0.1, 5.31e+22, -4.24, 1184.0],
                [0.316, 1.38e+23, -4.22, 1376.0],
                [1.0, 3.09e+23, -4.17, 1621.0],
                [3.16, 5.45e+23, -4.09, 1911.0],
                [10.0, 6.35e+23, -3.97, 2222.0],
                [31.6, 3.68e+23, -3.75, 2501.0],
                [100.0, 7.29e+22, -3.41, 2660.0],
            ]),
            name="OH+NO=HONO"
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
        calculated_rates = self.rate_constant.kinetic_constant(self.T_range, self.P_range) / 1000
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
