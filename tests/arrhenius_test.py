import os
import unittest

import jax.numpy as jnp
import numpy as np

from diffPLOG2TROE.parametrization import Arrhenius
from diffPLOG2TROE.parametrization import forward_rate_constant


class TestArrhenius(unittest.TestCase):
    def setUp(self):
        self.T_range = jnp.linspace(300, 3000, 300)
        self.reaction = Arrhenius(parameters={"A": 5.08e04, "n": 2.67, "Ea": 6292}, name="H2+O=H+OH")

        # ==============================================================================
        # Dataloader
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.1.0", "arrhenius.csv")
        data = np.loadtxt(data_file, delimiter=";")
        data = jnp.array(data)

        self.expected_rate = data[:, 1]

    def test_arrhenius_rate_constant_direct(self):
        calculated_rates = self.reaction.rate_constant(self.T_range) / 1000
        self.assertTrue(
            jnp.allclose(
                calculated_rates,
                self.expected_rate,
                atol=1e-10,
                rtol=1e-8,
            ),
            "Calculated rate constants (from the direct function) for the Arrhenius case don't match reference data",
        )

    def test_arrhenius_rate_constant_wrapper(self):
        calculated_rates = forward_rate_constant(self.reaction, self.T_range) / 1000
        self.assertTrue(
            jnp.allclose(
                calculated_rates,
                self.expected_rate,
                atol=1e-10,
                rtol=1e-8,
            ),
            "Calculated rate constants (from the wrapper) for the Arrhenius case don't match reference data",
        )


if __name__ == "__main__":
    unittest.main()
