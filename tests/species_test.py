import os
import unittest

import jax.numpy as jnp
import numpy as np

from KiRATE.species import Species


class TestSpecies(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with CH4 and CH2O species."""
        self.T_range = jnp.linspace(300, 3000, 300)

        # CH4 thermodynamic data from gri3.0
        ch4_thermo = """
        CH4               L 8/88C   1H   4          G   200.000  3500.000  1000.000    1
         7.48514950E-02 1.33909467E-02-5.73285809E-06 1.22292535E-09-1.01815230E-13    2
        -9.46834459E+03 1.84373180E+01 5.14987613E+00-1.36709788E-02 4.91800599E-05    3
        -4.84743026E-08 1.66693956E-11-1.02466476E+04-4.64130376E+00                   4
        """
        self.ch4 = Species.from_chemkin(ch4_thermo)

        # CH2O thermodynamic data from gri3.0
        ch2o_thermo = """
        CH2O              L 8/88H   2C   1O   1     G   200.000  3500.000  1000.000    1
         1.76069008E+00 9.20000082E-03-4.42258813E-06 1.00641212E-09-8.83855640E-14    2
        -1.39958323E+04 1.36563230E+01 4.79372315E+00-9.90833369E-03 3.73220008E-05    3
        -3.79285261E-08 1.31772652E-11-1.43089567E+04 6.02812900E-01                   4
        """
        self.ch2o = Species.from_chemkin(ch2o_thermo)

        # Load Cantera reference data
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        data_dir = os.path.join(current_dir, "cantera", "cantera_data", "3.2.0")

        # CH4 data: T, Cp, H, S, G, Cp/R, H/(RT), S/R, G/(RT)
        ch4_data = np.loadtxt(
            os.path.join(data_dir, "species_ch4.csv"),
            delimiter=";",
            skiprows=1,
        )
        # Cantera returns J/kmol, KiRATE returns cal/mol
        # Convert: J/kmol / 4.184 J/cal / 1000 kmol/mol = cal/mol
        J_to_cal = 1.0 / 4.184
        kmol_to_mol = 1.0 / 1000.0
        self.ch4_cantera = {
            "T": jnp.array(ch4_data[:, 0]),
            "cp": jnp.array(ch4_data[:, 1]) * J_to_cal * kmol_to_mol,  # cal/mol/K
            "h": jnp.array(ch4_data[:, 2]) * J_to_cal * kmol_to_mol,  # cal/mol
            "s": jnp.array(ch4_data[:, 3]) * J_to_cal * kmol_to_mol,  # cal/mol/K
            "g": jnp.array(ch4_data[:, 4]) * J_to_cal * kmol_to_mol,  # cal/mol
            "cp_over_r": jnp.array(ch4_data[:, 5]),
            "h_RT": jnp.array(ch4_data[:, 6]),
            "s_R": jnp.array(ch4_data[:, 7]),
            "g_RT": jnp.array(ch4_data[:, 8]),
        }

        # CH2O data
        ch2o_data = np.loadtxt(
            os.path.join(data_dir, "species_ch2o.csv"),
            delimiter=";",
            skiprows=1,
        )
        self.ch2o_cantera = {
            "T": jnp.array(ch2o_data[:, 0]),
            "cp": jnp.array(ch2o_data[:, 1]) * J_to_cal * kmol_to_mol,  # cal/mol/K
            "h": jnp.array(ch2o_data[:, 2]) * J_to_cal * kmol_to_mol,  # cal/mol
            "s": jnp.array(ch2o_data[:, 3]) * J_to_cal * kmol_to_mol,  # cal/mol/K
            "g": jnp.array(ch2o_data[:, 4]) * J_to_cal * kmol_to_mol,  # cal/mol
            "cp_over_r": jnp.array(ch2o_data[:, 5]),
            "h_RT": jnp.array(ch2o_data[:, 6]),
            "s_R": jnp.array(ch2o_data[:, 7]),
            "g_RT": jnp.array(ch2o_data[:, 8]),
        }

    def test_ch4_dimensionless_cp(self):
        """Test CH4 dimensionless heat capacity Cp/R against Cantera."""
        calculated = self.ch4.cp_over_r(self.T_range)
        expected = self.ch4_cantera["cp_over_r"]

        self.assertTrue(
            jnp.allclose(calculated, expected, rtol=1e-10, atol=1e-10),
            "CH4 Cp/R doesn't match Cantera reference data",
        )

    def test_ch4_heat_capacity(self):
        """Test CH4 heat capacity Cp against Cantera."""
        # KiRATE returns cal/(mol·K), Cantera converted to cal/(mol·K)
        calculated = self.ch4.cp(self.T_range)
        expected = self.ch4_cantera["cp"]

        self.assertTrue(
            jnp.allclose(calculated, expected, rtol=1e-10, atol=1e-10),
            "CH4 Cp doesn't match Cantera reference data",
        )

    def test_ch4_dimensionless_enthalpy(self):
        """Test CH4 dimensionless enthalpy H/(RT) against Cantera."""
        calculated = self.ch4.h_over_rt(self.T_range)
        expected = self.ch4_cantera["h_RT"]

        self.assertTrue(
            jnp.allclose(calculated, expected, rtol=1e-10, atol=1e-10),
            "CH4 H/(RT) doesn't match Cantera reference data",
        )

    def test_ch4_enthalpy(self):
        """Test CH4 enthalpy H against Cantera."""
        # KiRATE returns cal/mol, Cantera converted to cal/mol
        calculated = self.ch4.h(self.T_range)
        expected = self.ch4_cantera["h"]

        self.assertTrue(
            jnp.allclose(calculated, expected, rtol=1e-10, atol=1e-10),
            "CH4 H doesn't match Cantera reference data",
        )

    def test_ch4_dimensionless_entropy(self):
        """Test CH4 dimensionless entropy S/R against Cantera."""
        calculated = self.ch4.s_over_r(self.T_range)
        expected = self.ch4_cantera["s_R"]

        self.assertTrue(
            jnp.allclose(calculated, expected, rtol=1e-10, atol=1e-10),
            "CH4 S/R doesn't match Cantera reference data",
        )

    def test_ch4_entropy(self):
        """Test CH4 entropy S against Cantera."""
        # KiRATE returns cal/(mol·K), Cantera converted to cal/(mol·K)
        calculated = self.ch4.s(self.T_range)
        expected = self.ch4_cantera["s"]

        self.assertTrue(
            jnp.allclose(calculated, expected, rtol=1e-10, atol=1e-10),
            "CH4 S doesn't match Cantera reference data",
        )

    def test_ch4_dimensionless_gibbs(self):
        """Test CH4 dimensionless Gibbs energy G/(RT) against Cantera."""
        calculated = self.ch4.g_over_rt(self.T_range)
        expected = self.ch4_cantera["g_RT"]

        self.assertTrue(
            jnp.allclose(calculated, expected, rtol=1e-10, atol=1e-10),
            "CH4 G/(RT) doesn't match Cantera reference data",
        )

    def test_ch4_gibbs(self):
        """Test CH4 Gibbs energy G against Cantera."""
        # KiRATE returns cal/mol, Cantera converted to cal/mol
        calculated = self.ch4.g(self.T_range)
        expected = self.ch4_cantera["g"]

        self.assertTrue(
            jnp.allclose(calculated, expected, rtol=1e-10, atol=1e-10),
            "CH4 G doesn't match Cantera reference data",
        )

    def test_ch2o_dimensionless_cp(self):
        """Test CH2O dimensionless heat capacity Cp/R against Cantera."""
        calculated = self.ch2o.cp_over_r(self.T_range)
        expected = self.ch2o_cantera["cp_over_r"]

        self.assertTrue(
            jnp.allclose(calculated, expected, rtol=1e-10, atol=1e-10),
            "CH2O Cp/R doesn't match Cantera reference data",
        )

    def test_ch2o_heat_capacity(self):
        """Test CH2O heat capacity Cp against Cantera."""
        # KiRATE returns cal/(mol·K), Cantera converted to cal/(mol·K)
        calculated = self.ch2o.cp(self.T_range)
        expected = self.ch2o_cantera["cp"]

        self.assertTrue(
            jnp.allclose(calculated, expected, rtol=1e-10, atol=1e-10),
            "CH2O Cp doesn't match Cantera reference data",
        )

    def test_ch2o_dimensionless_enthalpy(self):
        """Test CH2O dimensionless enthalpy H/(RT) against Cantera."""
        calculated = self.ch2o.h_over_rt(self.T_range)
        expected = self.ch2o_cantera["h_RT"]

        self.assertTrue(
            jnp.allclose(calculated, expected, rtol=1e-10, atol=1e-10),
            "CH2O H/(RT) doesn't match Cantera reference data",
        )

    def test_ch2o_enthalpy(self):
        """Test CH2O enthalpy H against Cantera."""
        # KiRATE returns cal/mol, Cantera converted to cal/mol
        calculated = self.ch2o.h(self.T_range)
        expected = self.ch2o_cantera["h"]

        self.assertTrue(
            jnp.allclose(calculated, expected, rtol=1e-10, atol=1e-10),
            "CH2O H doesn't match Cantera reference data",
        )

    def test_ch2o_dimensionless_entropy(self):
        """Test CH2O dimensionless entropy S/R against Cantera."""
        calculated = self.ch2o.s_over_r(self.T_range)
        expected = self.ch2o_cantera["s_R"]

        self.assertTrue(
            jnp.allclose(calculated, expected, rtol=1e-10, atol=1e-10),
            "CH2O S/R doesn't match Cantera reference data",
        )

    def test_ch2o_entropy(self):
        """Test CH2O entropy S against Cantera."""
        # KiRATE returns cal/(mol·K), Cantera converted to cal/(mol·K)
        calculated = self.ch2o.s(self.T_range)
        expected = self.ch2o_cantera["s"]

        self.assertTrue(
            jnp.allclose(calculated, expected, rtol=1e-10, atol=1e-10),
            "CH2O S doesn't match Cantera reference data",
        )

    def test_ch2o_dimensionless_gibbs(self):
        """Test CH2O dimensionless Gibbs energy G/(RT) against Cantera."""
        calculated = self.ch2o.g_over_rt(self.T_range)
        expected = self.ch2o_cantera["g_RT"]

        self.assertTrue(
            jnp.allclose(calculated, expected, rtol=1e-10, atol=1e-10),
            "CH2O G/(RT) doesn't match Cantera reference data",
        )

    def test_ch2o_gibbs(self):
        """Test CH2O Gibbs energy G against Cantera."""
        # KiRATE returns cal/mol, Cantera converted to cal/mol
        calculated = self.ch2o.g(self.T_range)
        expected = self.ch2o_cantera["g"]

        self.assertTrue(
            jnp.allclose(calculated, expected, rtol=1e-10, atol=1e-10),
            "CH2O G doesn't match Cantera reference data",
        )

    def test_thermodynamic_relations(self):
        """Test fundamental thermodynamic relations: G = H - TS, Cv = Cp - R."""
        test_temps = jnp.array([300.0, 1000.0, 2000.0, 3000.0])

        for species, name in [(self.ch4, "CH4"), (self.ch2o, "CH2O")]:
            with self.subTest(species=name):
                # Test G = H - TS
                h = species.h(test_temps)
                s = species.s(test_temps)
                g_calculated = species.g(test_temps)
                g_from_relation = h - test_temps * s

                self.assertTrue(
                    jnp.allclose(g_calculated, g_from_relation, rtol=1e-10, atol=1e-10),
                    f"{name}: G = H - TS relation not satisfied",
                )

                # Test Cv = Cp - R
                cp_over_r = species.cp_over_r(test_temps)
                cv_R = species.cv_over_r(test_temps)
                diff = cp_over_r - cv_R

                self.assertTrue(
                    jnp.allclose(diff, 1.0, rtol=1e-10, atol=1e-10),
                    f"{name}: Cv = Cp - R relation not satisfied",
                )

    def test_vectorized_evaluation(self):
        """Test that vectorized evaluation works correctly."""
        # Single temperature
        single_T = 1000.0
        cp_single = self.ch4.cp_over_r(single_T)
        self.assertEqual(cp_single.shape, ())  # Scalar output

        # Array of temperatures
        array_T = jnp.array([300.0, 1000.0, 2000.0])
        cp_array = self.ch4.cp_over_r(array_T)
        self.assertEqual(cp_array.shape, (3,))  # Vector output

        # Check that single value matches first element of array evaluation
        cp_array_full = self.ch4.cp_over_r(jnp.array([single_T]))
        self.assertTrue(
            jnp.allclose(cp_single, cp_array_full.squeeze(), rtol=1e-10, atol=1e-10),
        )

    def test_temperature_range_switching(self):
        """Test correct coefficient switching at Tmid."""
        # Test at Tmid boundary
        tmid = float(self.ch4.Tmid)

        # Temperature just below Tmid (should use low coeffs)
        T_low = tmid - 1.0
        cp_low = self.ch4.cp_over_r(T_low)

        # Temperature at Tmid (should use high coeffs due to >= condition)
        cp_mid = self.ch4.cp_over_r(tmid)

        # Temperature just above Tmid (should use high coeffs)
        T_high = tmid + 1.0
        cp_high = self.ch4.cp_over_r(T_high)

        # Values should be continuous but derivatives may not be
        self.assertTrue(jnp.isfinite(cp_low))
        self.assertTrue(jnp.isfinite(cp_mid))
        self.assertTrue(jnp.isfinite(cp_high))


if __name__ == "__main__":
    unittest.main()
