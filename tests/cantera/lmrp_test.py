import os
import unittest

import jax.numpy as jnp
import numpy as np

from diffPLOG2TROE.parametrization import CollisionEfficiency, FallOff, MixtureRule


class TestLMRP(unittest.TestCase):
    def setUp(self):
        self.T_range = jnp.linspace(300, 3000, 300)
        self.P_range = jnp.logspace(jnp.log10(0.01), jnp.log10(100), 300)

        # ==============================================================================
        # Extended falloff model (see: https://github.com/Cantera/enhancements/issues/193)
        default_collider = FallOff(
            name="H+O2(+M)=HO2(+M)",
            hpl_parameters={"A": 4.650e12, "n": 0.44, "Ea": 0.0},
            lpl_parameters={"A": 1.737e19, "n": -1.230, "Ea": 0.0},
            falloff_parameters={"A": 0.67, "T3": 1e-30, "T2": 1e30, "T1": 1e30},
            falloff_type="troe",
            efficiencies=[
                CollisionEfficiency(name="H2", value=1.30),
                CollisionEfficiency(name="H2O", value=10.00),
            ],
        )
        self.ar_specific_rate = FallOff(
            name="H+O2(+AR)=HO2(+AR)",
            hpl_parameters={"A": 4.650e12, "n": 0.44, "Ea": 0.0},
            lpl_parameters={"A": 6.81e18, "n": -1.2, "Ea": 0.0},
            falloff_parameters={"A": 0.7, "T3": 1e-30, "T1": 1e30, "T2": 1e30},
            falloff_type="troe",
        )

        he_specific_rate = FallOff(
            name="H+O2(+HE)=HO2(+HE)",
            hpl_parameters={"A": 4.650e12, "n": 0.44, "Ea": 0.0},
            lpl_parameters={"A": 9.192e18, "n": -1.2, "Ea": 0.0},
            falloff_parameters={"A": 0.59, "T3": 1e-30, "T2": 1e30, "T1": 1e30},
            falloff_type="troe",
        )
        self.extended_falloff_reaction = MixtureRule(
            default_rate_constant=default_collider,
            explicit_rate_constants={"AR": self.ar_specific_rate, "HE": he_specific_rate},
            linear=True,
            reduced_pressure=False,
            name="H+O2(+M)=HO2(+M)",
        )

        # ==============================================================================
        # Dataloader
        # 100% AR
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.1.0", "lmrp_ar.csv")
        data = np.loadtxt(data_file, delimiter=";")
        self.expected_rate_extended_falloff_ar = jnp.array(data)

        # 50% AR, 50% H2O
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.1.0", "lmrp_arh2o.csv")
        data = np.loadtxt(data_file, delimiter=";")
        self.expected_rate_extended_falloff_ar_h2o = jnp.array(data)

        # 50% AR, 25% H2O, 25% HE
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.1.0", "lmrp_arh2ohe.csv")
        data = np.loadtxt(data_file, delimiter=";")
        self.expected_rate_extended_falloff_ar_h2o_he = jnp.array(data)

    def test_kinetic_constant_ar_direct(self):
        calculated_rates = self.extended_falloff_reaction.rate_constant(
            T=self.T_range,
            P=self.P_range,
            composition={"AR": 1},
        ) / 1000

        for i, calculated_rate in enumerate(calculated_rates):
            self.assertTrue(
                jnp.allclose(
                    calculated_rate,
                    self.expected_rate_extended_falloff_ar[i],
                    atol=1e-10,
                    rtol=1e-8,
                ),
                "",
            )

    def test_kinetic_constant_ar_h2o_direct(self):
        calculated_rates = (
            self.extended_falloff_reaction.rate_constant(
                T=self.T_range,
                P=self.P_range,
                composition={"H2O": 0.5, "AR": 0.5},
            )
            / 1000
        )

        for i, calculated_rate in enumerate(calculated_rates):
            self.assertTrue(
                jnp.allclose(
                    calculated_rate,
                    self.expected_rate_extended_falloff_ar_h2o[i],
                    atol=1e-10,
                    rtol=1e-8,
                ),
                "",
            )

    def test_kinetic_constant_ar_h2o_he_direct(self):
        calculated_rates = (
            self.extended_falloff_reaction.rate_constant(
                T=self.T_range,
                P=self.P_range,
                composition={"HE": 0.25, "H2O": 0.25, "AR": 0.5},
            )
            / 1000
        )

        for i, calculated_rate in enumerate(calculated_rates):
            self.assertTrue(
                jnp.allclose(
                    calculated_rate,
                    self.expected_rate_extended_falloff_ar_h2o_he[i],
                    atol=1e-10,
                    rtol=1e-8,
                ),
                "",
            )


if __name__ == "__main__":
    unittest.main()
