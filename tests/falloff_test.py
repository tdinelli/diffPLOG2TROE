import os
import unittest

import jax.numpy as jnp
import numpy as np

from KiRATE.kinetics import FallOff
from tests.test_utils import assert_rate_constants_close


class TestFallOff(unittest.TestCase):
    def setUp(self):
        self.T_range = jnp.linspace(300, 3000, 300)
        self.P_range = jnp.logspace(jnp.log10(0.01), jnp.log10(100), 300)  # atm

        self.reaction = FallOff(
            lpl_parameters={"A": 6.366e20, "n": -1.72, "Ea": 524.8},
            hpl_parameters={"A": 4.7e12, "n": 0.44, "Ea": 0.0},
            falloff_parameters={"A": 0.5, "T3": 1.0e-30, "T1": 1.0e30, "T2": 0.0},
            efficiencies={"H2": 2.0, "H2O": 14.0, "O2": 0.78, "CO": 1.9, "CO2": 3.8, "AR": 0.67},
            falloff_type="troe",
            name="H+O2(+M)=HO2(+M)",
        )

        # ==============================================================================
        # Dataloader
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.2.0", "falloff_troe_n2.csv")
        data = np.loadtxt(data_file, delimiter=";")
        self.expected_rate_n2 = jnp.array(data) * 1000

        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.2.0", "falloff_troe_n2_h2o.csv")
        data = np.loadtxt(data_file, delimiter=";")
        self.expected_rate_n2h2o = jnp.array(data) * 1000

        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.2.0", "falloff_troe_n2_o2_h2o.csv")
        data = np.loadtxt(data_file, delimiter=";")
        self.expected_rate_n2o2h2o = jnp.array(data) * 1000

    def test_kinetic_constant_n2(self):
        """Test fall-off rate constant with N2 bath gas."""
        calculated_rates = self.reaction.rate_constant(
            T=self.T_range,
            P=self.P_range,
            composition={"N2": 1},
        )

        assert_rate_constants_close(
            self,
            calculated_rates,
            self.expected_rate_n2,
            T_range=self.T_range,
            P_range=self.P_range,
            test_name="Fall-off Troe (N2 bath gas)",
        )

    def test_kinetic_constant_n2h2o(self):
        """Test fall-off rate constant with N2/H2O mixture."""
        calculated_rates = self.reaction.rate_constant(
            T=self.T_range,
            P=self.P_range,
            composition={"N2": 0.5, "H2O": 0.5},
        )

        assert_rate_constants_close(
            self,
            calculated_rates,
            self.expected_rate_n2h2o,
            T_range=self.T_range,
            P_range=self.P_range,
            test_name="Fall-off Troe (N2/H2O mixture)",
        )

    def test_kinetic_constant_n2o2h2o(self):
        """Test fall-off rate constant with N2/O2/H2O mixture."""
        calculated_rates = self.reaction.rate_constant(
            T=self.T_range,
            P=self.P_range,
            composition={"N2": 0.2, "O2": 0.3, "H2O": 0.5},
        )

        assert_rate_constants_close(
            self,
            calculated_rates,
            self.expected_rate_n2o2h2o,
            T_range=self.T_range,
            P_range=self.P_range,
            test_name="Fall-off Troe (N2/O2/H2O mixture)",
        )

    def test_chemkin_parsing(self):
        chemkin_string = """
            H+O2(+M)=HO2(+M)        4.7e12 0.44 0.0
             LOW / 6.366e20 -1.72 524.8 /
             TROE / 0.5 1.0e-30 1.0e30 0.0 /
            H2 / 2.0 / H2O / 14.0/ O2 / 0.78 /CO/1.9/CO2/3.8/AR/0.67
            """
        test_reaction = FallOff.from_chemkin(input_string=chemkin_string)
        self.assertEqual(test_reaction.hpl.parameters, self.reaction.hpl.parameters)
        self.assertEqual(test_reaction.lpl.parameters, self.reaction.lpl.parameters)
        self.assertEqual(test_reaction.efficiencies["CO"], self.reaction.efficiencies["CO"])
        self.assertEqual(test_reaction.efficiencies["H2"], self.reaction.efficiencies["H2"])
        self.assertEqual(test_reaction.falloff_parameters, self.reaction.falloff_parameters)
        self.assertEqual(test_reaction.falloff_type, self.reaction.falloff_type)
        self.assertEqual(test_reaction.name, self.reaction.name)


if __name__ == "__main__":
    unittest.main()
