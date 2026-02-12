import os
import unittest

import jax.numpy as jnp
import numpy as np

from KiRATE.kinetics import Arrhenius, FallOff, MixtureRule, Plog
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
        helium = Arrhenius(
            name="H+O2(+HE)=HO2(+HE)",
            parameters={"A": 3.37601e-01, "n": 1.82568e-01, "Ea": 3.62408e01},
        )
        nitrogen = Arrhenius(
            name="H+O2(+N2)=HO2(+N2)",
            parameters={"A": 1.24932e02, "n": -5.93263e-01, "Ea": 5.40921e02},
        )
        hydrogen = Arrhenius(
            name="H+O2(+H2)=HO2(+H2)",
            parameters={"A": 3.13717e04, "n": -1.25419e00, "Ea": 1.12924e03},
        )
        carbon_dioxyde = Arrhenius(
            name="H+O2(+CO2)=HO2(+CO2)",
            parameters={"A": 1.62413e08, "n": -2.27622e00, "Ea": 1.97023e03},
        )
        ammonia = Arrhenius(
            name="H+O2(+NH3)=HO2(+NH3)",
            parameters={"A": 4.97750e00, "n": 1.64855e-01, "Ea": -2.80351e02},
        )
        water = Arrhenius(
            name="H+O2(+H2O)=HO2(+H2O)",
            parameters={"A": 3.69146e01, "n": -7.12902e-02, "Ea": 3.19087e01},
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

        default = FallOff(
            name="H + O2 (+M) <=> HO2 (+M)",
            hpl_parameters={"A": 4.66000000e12, "n": 4.40000000e-01, "Ea": 0.00000000},
            lpl_parameters={"A": 4.06620000e19, "n": -1.40000000, "Ea": -1.80537000e02},
            falloff_parameters={"A": 0.5, "T3": 1.0, "T1": 1.0e10, "T2": 1.0e30},
            falloff_type="troe",
        )
        nitrogen = FallOff(
            name="H + O2 (+N2) <=> HO2 (+N2)",
            hpl_parameters={"A": 4.66000000e12, "n": 4.40000000e-01, "Ea": 0.00000000},
            lpl_parameters={"A": 1.91000000e20, "n": -1.55680000, "Ea": 2.53860000e02},
            falloff_parameters={"A": 0.5, "T3": 1.0, "T1": 1.0e10, "T2": 1.0e30},
            falloff_type="troe",
        )
        # Efficiencies from YAML
        hydrogen = Arrhenius(name="H + O2 (+H2) <=> HO2 (+H2)", parameters={"A": 2.0, "n": 0, "Ea": 0})
        water = Arrhenius(name="H + O2 (+H2O) <=> HO2 (+H2O)", parameters={"A": 17.6, "n": 0, "Ea": 0})
        oxygen = Arrhenius(name="H + O2 (+O2) <=> HO2 (+O2)", parameters={"A": 1, "n": 0, "Ea": 0})
        ammonia = Arrhenius(name="H + O2 (+NH3) <=> HO2 (+NH3)", parameters={"A": 0.0927, "n": 0.6872, "Ea": -614})

        # Explicit efficiencies for colliders with explicit rate constants (from YAML)
        nitrogen_eff = Arrhenius(name="N2 efficiency", parameters={"A": 4.6973, "n": -0.157, "Ea": 434.4})
        helium_eff = Arrhenius(name="He efficiency", parameters={"A": 0.2991, "n": 0.170, "Ea": 180.5})
        argon_eff = Arrhenius(name="Ar efficiency", parameters={"A": 0.1706, "n": 0.2090, "Ea": 191.9})

        helium = FallOff(
            name="H + O2 (+HE) <=> HO2 (+HE)",
            hpl_parameters={"A": 4.66000000e12, "n": 4.40000000e-01, "Ea": 0.00000000},
            lpl_parameters={"A": 1.216e19, "n": -1.23000000, "Ea": 0.0},
            falloff_parameters={"A": 0.67, "T3": 1.0e-30, "T1": 1.0e30, "T2": 1.0e30},
            falloff_type="troe",
        )
        argon = Plog(
            name="H + O2 (+AR) <=> HO2 (+AR)",
            parameters={
                1.00000000e-02: {"A": 8.45360e14, "n": -2.19120, "Ea": 1.14078e01},
                1.00000000e-01: {"A": 8.26120e15, "n": -2.18930, "Ea": 2.35054e01},
                1.00000000e00: {"A": 8.39046e16, "n": -2.19250, "Ea": 6.13437e01},
                5.00000000e00: {"A": 4.94439e17, "n": -2.21410, "Ea": 1.48419e02},
                1.00000000e01: {"A": 1.20063e18, "n": -2.23890, "Ea": 2.28993e02},
                2.00000000e01: {"A": 3.35001e18, "n": -2.28080, "Ea": 3.58971e02},
                3.00000000e01: {"A": 9.75402e19, "n": -2.67880, "Ea": 6.10486e02},
                4.00000000e01: {"A": 1.89318e20, "n": -2.72520, "Ea": 7.46275e02},
                5.00000000e01: {"A": 3.11736e20, "n": -2.75890, "Ea": 8.57089e02},
                1.00000000e02: {"A": 1.06227e21, "n": -2.81950, "Ea": 1.19199e03},
                3.00000000e02: {"A": 1.06233e21, "n": -2.66310, "Ea": 1.49349e03},
            },
            k0_parameters={
                "A": float(0.1706 * default.lpl.A),
                "n": float(0.209 + default.lpl.n),
                "Ea": float(191.9 + default.lpl.Ea),
            },
        )

        self.reaction2 = MixtureRule(
            name="H+O2(+M)=HO2(+M)",
            linear=True,
            reduced_pressure=True,
            default_rate_constant=default,
            explicit_rate_constants={"He": helium, "N2": nitrogen, "Ar": argon},
            efficiencies={
                "N2": nitrogen_eff,
                "He": helium_eff,
                "Ar": argon_eff,
                "H2": hydrogen,
                "H2O": water,
                "O2": oxygen,
                "NH3": ammonia,
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

        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.2.0", "lmrr2_n2.csv")
        data = np.loadtxt(data_file, delimiter=";")
        self.expected_rate2_n2 = jnp.array(data) * 1000

        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.2.0", "lmrr2_n2arhe.csv")
        data = np.loadtxt(data_file, delimiter=";")
        self.expected_rate2_n2arhe = jnp.array(data) * 1000

        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.2.0", "lmrr2_n2arhenh3.csv")
        data = np.loadtxt(data_file, delimiter=";")
        self.expected_rate2_n2arhenh3 = jnp.array(data) * 1000

    def test_kinetic_constant_n2(self):
        """"""
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
        """"""
        calculated_rates = self.reaction.rate_constant(
            T=self.T_range, P=self.P_range, composition={"N2": 0.5, "Ar": 0.5}
        )

        assert_rate_constants_close(
            self,
            calculated_rates,
            self.expected_rate_n2ar,
            T_range=self.T_range,
            P_range=self.P_range,
            test_name="LMR-R (Ar/N2 mixture)",
        )

    def test_kinetic_constant_n2arhe(self):
        """"""
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

    def test_kinetic_constant2_n2(self):
        """"""
        calculated_rates = self.reaction2.rate_constant(T=self.T_range, P=self.P_range, composition={"N2": 1.0})

        assert_rate_constants_close(
            self,
            calculated_rates,
            self.expected_rate2_n2,
            T_range=self.T_range,
            P_range=self.P_range,
            test_name="LMR-R2 (100% N2)",
        )

    def test_kinetic_constant2_n2arhe(self):
        """"""
        calculated_rates = self.reaction2.rate_constant(
            T=self.T_range, P=self.P_range, composition={"N2": 0.3, "He": 0.2, "Ar": 0.5}
        )

        assert_rate_constants_close(
            self,
            calculated_rates,
            self.expected_rate2_n2arhe,
            T_range=self.T_range,
            P_range=self.P_range,
            test_name="LMR-R2 (N2/Ar/He mixture)",
        )

    def test_kinetic_constant2_n2arhenh3(self):
        """"""
        calculated_rates = self.reaction2.rate_constant(
            T=self.T_range, P=self.P_range, composition={"N2": 0.3, "He": 0.2, "Ar": 0.3, "NH3": 0.2}
        )

        assert_rate_constants_close(
            self,
            calculated_rates,
            self.expected_rate2_n2arhenh3,
            T_range=self.T_range,
            P_range=self.P_range,
            test_name="LMR-R2 (N2/Ar/He/NH3 mixture)",
        )


if __name__ == "__main__":
    unittest.main()
