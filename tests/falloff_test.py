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
            efficiencies={
                "H2O": 7.65,
                "N2": 1.5,
                "O2": 1.2,
                "HE": 0.65,
                "H2O2": 7.7,
                "H2": 3.7,
            },
            falloff_type="troe",
            name="H2O2(+M)=OH+OH(+M)",
        )

        # ==============================================================================
        # Dataloader
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.1.0", "falloff_troe_ar.csv")
        data = np.loadtxt(data_file, delimiter=";")
        self.expected_rate_ar = jnp.array(data)

        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.1.0", "falloff_troe_ar_h2o.csv")
        data = np.loadtxt(data_file, delimiter=";")
        self.expected_rate_arh2o = jnp.array(data)

        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.1.0", "falloff_troe_ar_o2_h2o.csv")
        data = np.loadtxt(data_file, delimiter=";")
        self.expected_rate_aro2h2o = jnp.array(data)

    def test_kinetic_constant_ar(self):
        calculated_rates = self.rate_constant.kinetic_constant(
            T=self.T_range,
            P=self.P_range,
            composition={"AR": 1},
        )
        for i, calculated_rate in enumerate(calculated_rates):
            self.assertTrue(
                jnp.allclose(
                    calculated_rate,
                    self.expected_rate_ar[i],
                    atol=1e-10,
                    rtol=1e-8,
                ),
                "",
            )

    def test_kinetic_constant_arh2o(self):
        calculated_rates = self.rate_constant.kinetic_constant(
            T=self.T_range,
            P=self.P_range,
            composition={"AR": 0.5, "H2O": 0.5},
        )
        for i, calculated_rate in enumerate(calculated_rates):
            self.assertTrue(
                jnp.allclose(
                    calculated_rate,
                    self.expected_rate_arh2o[i],
                    atol=1e-10,
                    rtol=1e-8,
                ),
                "",
            )

    def test_kinetic_constant_aro2h2o(self):
        calculated_rates = self.rate_constant.kinetic_constant(
            T=self.T_range,
            P=self.P_range,
            composition={"AR": 0.2, "O2": 0.3, "H2O": 0.5},
        )
        for i, calculated_rate in enumerate(calculated_rates):
            self.assertTrue(
                jnp.allclose(
                    calculated_rate,
                    self.expected_rate_aro2h2o[i],
                    atol=1e-10,
                    rtol=1e-8,
                ),
                "",
            )


if __name__ == "__main__":
    unittest.main()
