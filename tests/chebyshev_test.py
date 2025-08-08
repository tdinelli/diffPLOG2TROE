import os
import unittest

import jax.numpy as jnp
import numpy as np

from KiRATE.kinetics import Chebyshev, forward_rate_constant


class TestChebyshev(unittest.TestCase):
    def setUp(self):
        self.T_range = jnp.linspace(300, 2500, 300)
        self.P_range = jnp.logspace(jnp.log10(0.1), jnp.log10(50), 300)
        self.reaction = Chebyshev(
            T_limits=(290, 3000),
            P_limits=(0.0098692326671601278, 98.692326671601279),
            order_T=6,
            order_P=4,
            chebyshev_coefficients=jnp.array(
                [
                    [-1.44280e01, 2.59970e-01, -2.24320e-02, -2.78700e-03],
                    [2.20630e01, 4.88090e-01, -3.96430e-02, -5.48110e-03],
                    [-2.32940e-01, 4.01900e-01, -2.60730e-02, -5.04860e-03],
                    [-2.93660e-01, 2.85680e-01, -9.33730e-03, -4.01020e-03],
                    [-2.26210e-01, 1.69190e-01, 4.85810e-03, -2.38030e-03],
                    [-1.43220e-01, 7.71110e-02, 1.27080e-02, -6.41540e-04],
                ]
            ),
            name="CH4=CH3+H",
        )

        # ==============================================================================
        # Dataloader
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.1.0", "chebyshev.csv")
        data = np.loadtxt(data_file, delimiter=";")
        data = jnp.array(data)

        self.expected_rate = data

    def test_rate_constant_direct(self):
        calculated_rates = self.reaction.rate_constant(self.T_range, self.P_range)

        for i, calculated_rate in enumerate(calculated_rates):
            self.assertTrue(
                jnp.allclose(
                    calculated_rate,
                    self.expected_rate[i],
                    atol=1e-10,
                    rtol=1e-8,
                ),
                "",
            )

    def test_rate_constant_wrapped(self):
        calculated_rates = forward_rate_constant(self.reaction, self.T_range, self.P_range)

        for i, calculated_rate in enumerate(calculated_rates):
            self.assertTrue(
                jnp.allclose(
                    calculated_rate,
                    self.expected_rate[i],
                    atol=1e-10,
                    rtol=1e-8,
                ),
                "",
            )


if __name__ == "__main__":
    unittest.main()
