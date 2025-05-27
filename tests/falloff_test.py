import os
import unittest

import jax.numpy as jnp
import numpy as np

from diffPLOG2TROE.parametrization import FallOff


class TestFallOff(unittest.TestCase):
    def setUp(self):
        self.T_range = jnp.linspace(300, 3000, 300)
        self.P_range = jnp.logspace(jnp.log10(0.01), jnp.log10(100), 300)
        self.rate_constant = FallOff(
            hpl_parameters=jnp.array([2.0e12, 0.9, 4.8749e04]),
            lpl_parameters=jnp.array([2.49e24, -2.3, 4.8749e04]),
            falloff_parameters=jnp.array([0.43, 1.0e-30, 1.0e30]),
            falloff_type="troe",
            name="H2O2(+M)=OH+OH(+M)",
        )

        # ==============================================================================
        # Dataloader
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.1.0", "falloff_troe.csv")
        data = np.loadtxt(data_file, delimiter=";")
        self.expected_rate = jnp.array(data)

    def test_kinetic_constant(self):
        calculated_rates = self.rate_constant.kinetic_constant(self.T_range, self.P_range)
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
