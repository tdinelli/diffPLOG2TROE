import os
import unittest

import jax.numpy as jnp
import numpy as np

from KiRATE.kinetics import Arrhenius, MixtureRule, Plog
from tests.test_utils import assert_rate_constants_close


class TestLMRR(unittest.TestCase):
    def setUp(self):
        self.T_range = jnp.linspace(300, 3000, 300)
        self.P_range = jnp.logspace(jnp.log10(0.01), jnp.log10(100), 300)

        default = Plog(
            name="H+O2(+Ar)=HO2(+Ar)",
            parameters={
                1.316e-02: {"A": 9.39968e14, "n": -2.14348e00, "Ea": 7.72730e01},
                1.316e-01: {"A": 1.07254e16, "n": -2.15999e00, "Ea": 1.30239e02},
                3.947e-01: {"A": 3.17830e16, "n": -2.15813e00, "Ea": 1.66994e02},
                1.000e00: {"A": 7.72584e16, "n": -2.15195e00, "Ea": 2.13473e02},
                3.000e00: {"A": 2.11688e17, "n": -2.14062e00, "Ea": 2.79031e02},
                1.000e01: {"A": 6.53093e17, "n": -2.13213e00, "Ea": 3.87493e02},
                3.000e01: {"A": 1.49784e18, "n": -2.10026e00, "Ea": 4.87579e02},
                1.000e02: {"A": 3.82218e18, "n": -2.07057e00, "Ea": 6.65984e02},
            },
        )
        helium = Arrhenius(name="H+O2(+HE)=HO2(+HE)", parameters={"A": 3.37601e-01, "n": 1.82568e-01, "Ea": 3.62408e01})
        nitrogen = Arrhenius(
            name="H+O2(+N2)=HO2(+N2)", parameters={"A": 1.24932e02, "n": -5.93263e-01, "Ea": 5.40921e02}
        )
        hydrogen = Arrhenius(
            name="H+O2(+H2)=HO2(+H2)", parameters={"A": 3.13717e04, "n": -1.25419e00, "Ea": 1.12924e03}
        )
        carbon_dioxyde = Arrhenius(
            name="H+O2(+CO2)=HO2(+CO2)", parameters={"A": 1.62413e08, "n": -2.27622e00, "Ea": 1.97023e03}
        )
        ammonia = Arrhenius(
            name="H+O2(+NH3)=HO2(+NH3)", parameters={"A": 4.97750e00, "n": 1.64855e-01, "Ea": -2.80351e02}
        )
        water = Arrhenius(
            name="H+O2(+H2O)=HO2(+H2O)", parameters={"A": 3.69146e01, "n": -7.12902e-02, "Ea": 3.19087e01}
        )

        self.reaction = MixtureRule(
            name="H+O2(+M)=HO2(+M)",
            linear=True,
            reduced_pressure=True,
            default_rate_constant=default,
            efficiencies={
                "He": helium,
                "N2": nitrogen,
                "H2": hydrogen,
                "CO2": carbon_dioxyde,
                "NH3": ammonia,
                "H2O": water,
            },
        )

        # ==============================================================================
        # Dataloader
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.2.0", "lmrr_n2.csv")
        data = np.loadtxt(data_file, delimiter=";")
        self.expected_rate_n2 = jnp.array(data) * 1000

        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.2.0", "lmrr_n2ar.csv")
        data = np.loadtxt(data_file, delimiter=";")
        self.expected_rate_n2ar = jnp.array(data) * 1000

        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.2.0", "lmrr_n2arhe.csv")
        data = np.loadtxt(data_file, delimiter=";")
        self.expected_rate_n2arhe = jnp.array(data) * 1000

    def test_kinetic_constant_n2(self):
        """Test fall-off rate constant with N2 bath gas."""
        # Direct grid evaluation (most efficient)
        calculated_rates = self.reaction.rate_constant(T=self.T_range, P=self.P_range, composition={"N2": 1})

        assert_rate_constants_close(
            self,
            calculated_rates,
            self.expected_rate_n2,
            T_range=self.T_range,
            P_range=self.P_range,
            test_name="LMR-R (100% N2)",
        )

    def test_kinetic_constant_n2ar(self):
        """Test fall-off rate constant with N2/AR mixture"""
        # Direct grid evaluation (most efficient)
        calculated_rates = self.reaction.rate_constant(T=self.T_range, P=self.P_range, composition={"N2": 0.5, "Ar": 0.5})

        assert_rate_constants_close(
            self,
            calculated_rates,
            self.expected_rate_n2ar,
            T_range=self.T_range,
            P_range=self.P_range,
            test_name="LMR-R (Ar/N2 mixture)",
        )

    def test_kinetic_constant_n2arhe(self):
        """Test fall-off rate constant with N2/Ar/He mixture."""
        # Direct grid evaluation (most efficient)
        calculated_rates = self.reaction.rate_constant(
            T=self.T_range, P=self.P_range, composition={"N2": 0.3, "He": 0.2, "Ar": 0.5}
        )

        assert_rate_constants_close(
            self,
            calculated_rates,
            self.expected_rate_n2arhe,
            T_range=self.T_range,
            P_range=self.P_range,
            test_name="LMR-R (N2/Ar/He mixture)",
        )


if __name__ == "__main__":
    unittest.main()
