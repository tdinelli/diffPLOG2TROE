"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details

Transport property tests with Cantera validation.
"""

import os
import unittest

import jax.numpy as jnp
import numpy as np

from KiRATE.species import Species


class TestSpeciesTransport(unittest.TestCase):
    """Test Species transport properties against Cantera reference data."""

    def setUp(self):
        """Set up test fixtures with species from GRI-Mech 3.0."""
        self.T_range = jnp.linspace(300, 3000, 300)

        # H2O thermodynamic and transport data from GRI-Mech 3.0
        h2o_thermo = """
        H2O               L 8/89H   2O   1          G   200.000  3500.000  1000.000    1
         3.03399249E+00 2.17691804E-03-1.64072518E-07-9.70419870E-11 1.68200992E-14    2
        -3.00042971E+04 4.96677010E+00 4.19864056E+00-2.03643410E-03 6.52040211E-06    3
        -5.48797062E-09 1.77197817E-12-3.02937267E+04-8.49032208E-01                   4
        """
        h2o_transport = "H2O      2  572.400     2.605     1.844     0.000     4.000"
        self.h2o = Species.from_chemkin(h2o_thermo, h2o_transport)

        # CH4 thermodynamic and transport data from GRI-Mech 3.0
        ch4_thermo = """
        CH4               L 8/88C   1H   4          G   200.000  3500.000  1000.000    1
         7.48514950E-02 1.33909467E-02-5.73285809E-06 1.22292535E-09-1.01815230E-13    2
        -9.46834459E+03 1.84373180E+01 5.14987613E+00-1.36709788E-02 4.91800599E-05    3
        -4.84743026E-08 1.66693956E-11-1.02466476E+04-4.64130376E+00                   4
        """
        ch4_transport = "CH4      2  141.400     3.746     0.000     2.600    13.000"
        self.ch4 = Species.from_chemkin(ch4_thermo, ch4_transport)

        # H2 thermodynamic and transport data from GRI-Mech 3.0
        h2_thermo = """
        H2                TPIS78H   2               G   200.000  3500.000  1000.000    1
         3.33727920E+00-4.94024731E-05 4.99456778E-07-1.79566394E-10 2.00255376E-14    2
        -9.50158922E+02-3.20502331E+00 2.34433112E+00 7.98052075E-03-1.94781510E-05    3
         2.01572094E-08-7.37611761E-12-9.17935173E+02 6.83010238E-01                   4
        """
        h2_transport = "H2       1   38.000     2.920     0.000     0.790   280.000"
        self.h2 = Species.from_chemkin(h2_thermo, h2_transport)

        # AR thermodynamic and transport data from GRI-Mech 3.0
        ar_thermo = """
        AR                120186AR  1               G   300.000  5000.000  1000.000    1
         0.02500000E+02 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00    2
        -0.07453750E+04 0.04366000E+02 0.02500000E+02 0.00000000E+00 0.00000000E+00    3
         0.00000000E+00 0.00000000E+00-0.07453750E+04 0.04366000E+02                   4
        """
        ar_transport = "AR       0  136.500     3.330     0.000     0.000     0.000"
        self.ar = Species.from_chemkin(ar_thermo, ar_transport)

        # Load Cantera reference data
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        data_dir = os.path.join(current_dir, "cantera", "cantera_data", "3.2.0")

        # H2O transport data: T, viscosity[Pa·s], thermal_conductivity[W/(m·K)]
        h2o_data = np.loadtxt(
            os.path.join(data_dir, "transport_h2o.csv"),
            delimiter=";",
            skiprows=1,
        )
        self.h2o_cantera = {
            "T": jnp.array(h2o_data[:, 0]),
            "viscosity": jnp.array(h2o_data[:, 1]),  # Pa·s
            "thermal_conductivity": jnp.array(h2o_data[:, 2]),  # W/(m·K)
        }

        # CH4 transport data
        ch4_data = np.loadtxt(
            os.path.join(data_dir, "transport_ch4.csv"),
            delimiter=";",
            skiprows=1,
        )
        self.ch4_cantera = {
            "T": jnp.array(ch4_data[:, 0]),
            "viscosity": jnp.array(ch4_data[:, 1]),  # Pa·s
            "thermal_conductivity": jnp.array(ch4_data[:, 2]),  # W/(m·K)
        }

        # H2 transport data
        h2_data = np.loadtxt(
            os.path.join(data_dir, "transport_h2.csv"),
            delimiter=";",
            skiprows=1,
        )
        self.h2_cantera = {
            "T": jnp.array(h2_data[:, 0]),
            "viscosity": jnp.array(h2_data[:, 1]),  # Pa·s
            "thermal_conductivity": jnp.array(h2_data[:, 2]),  # W/(m·K)
        }

        # AR transport data
        ar_data = np.loadtxt(
            os.path.join(data_dir, "transport_ar.csv"),
            delimiter=";",
            skiprows=1,
        )
        self.ar_cantera = {
            "T": jnp.array(ar_data[:, 0]),
            "viscosity": jnp.array(ar_data[:, 1]),  # Pa·s
            "thermal_conductivity": jnp.array(ar_data[:, 2]),  # W/(m·K)
        }

    def test_h2o_viscosity(self):
        """Test H2O viscosity against Cantera."""
        calculated = self.h2o.viscosity(self.T_range)
        expected = self.h2o_cantera["viscosity"]

        # Check shape match first
        self.assertEqual(
            calculated.shape,
            expected.shape,
            f"H2O viscosity: Shape mismatch - calculated {calculated.shape} vs expected {expected.shape}",
        )

        # Calculate relative errors
        rel_errors = jnp.abs(calculated - expected) / jnp.abs(expected)
        max_rel_error = jnp.max(rel_errors)
        max_error_idx = jnp.argmax(rel_errors)

        # Transport properties from kinetic theory have inherent approximations
        # Different implementations (KiRATE vs Cantera) will have small differences due to:
        #  - Polynomial fitting precision (polyfit vs other methods)
        #  - Floating-point arithmetic order (Python/JAX vs C++)
        #  - Interpolation implementation details
        # A relative tolerance of 0.5% is excellent agreement for transport properties
        rtol, atol = 5e-3, 1e-10
        is_close = jnp.allclose(calculated, expected, atol=atol, rtol=rtol)

        # If not close, build detailed error message
        if not is_close:
            T_at_error = self.T_range[max_error_idx]
            error_msg = (
                f"H2O viscosity doesn't match Cantera reference data:\n"
                f" Max relative error: {max_rel_error:.6e}\n"
                f" at T_idx={max_error_idx} (T={T_at_error:.2f}K)\n"
                f" calculated={calculated[max_error_idx]:.12e}\n"
                f" expected={expected[max_error_idx]:.12e}"
            )
            self.fail(error_msg)

        self.assertTrue(is_close, "H2O viscosity: Values not within tolerance")

    def test_h2o_thermal_conductivity(self):
        """Test H2O thermal conductivity against Cantera."""
        calculated = self.h2o.thermal_conductivity(self.T_range)
        expected = self.h2o_cantera["thermal_conductivity"]

        # Check shape match first
        self.assertEqual(
            calculated.shape,
            expected.shape,
            f"H2O thermal conductivity: Shape mismatch - calculated {calculated.shape} vs expected {expected.shape}",
        )

        # Calculate relative errors
        rel_errors = jnp.abs(calculated - expected) / jnp.abs(expected)
        max_rel_error = jnp.max(rel_errors)
        max_error_idx = jnp.argmax(rel_errors)

        # Thermal conductivity depends on both transport and thermodynamic data
        # Same tolerance as viscosity since it uses Eucken correlation
        rtol, atol = 5e-3, 1e-10
        is_close = jnp.allclose(calculated, expected, atol=atol, rtol=rtol)

        if not is_close:
            T_at_error = self.T_range[max_error_idx]
            error_msg = (
                f"H2O thermal conductivity doesn't match Cantera reference data:\n"
                f" Max relative error: {max_rel_error:.6e}\n"
                f" at T_idx={max_error_idx} (T={T_at_error:.2f}K)\n"
                f" calculated={calculated[max_error_idx]:.12e}\n"
                f" expected={expected[max_error_idx]:.12e}"
            )
            self.fail(error_msg)

        self.assertTrue(is_close, "H2O thermal conductivity: Values not within tolerance")

    def test_ch4_viscosity(self):
        """Test CH4 viscosity against Cantera."""
        calculated = self.ch4.viscosity(self.T_range)
        expected = self.ch4_cantera["viscosity"]

        self.assertEqual(calculated.shape, expected.shape)
        rel_errors = jnp.abs(calculated - expected) / jnp.abs(expected)
        max_rel_error = jnp.max(rel_errors)
        max_error_idx = jnp.argmax(rel_errors)

        rtol, atol = 5e-3, 1e-10
        is_close = jnp.allclose(calculated, expected, atol=atol, rtol=rtol)

        if not is_close:
            T_at_error = self.T_range[max_error_idx]
            self.fail(
                f"CH4 viscosity doesn't match Cantera reference data:\n"
                f" Max relative error: {max_rel_error:.6e}\n"
                f" at T_idx={max_error_idx} (T={T_at_error:.2f}K)\n"
                f" calculated={calculated[max_error_idx]:.12e}\n"
                f" expected={expected[max_error_idx]:.12e}"
            )

        self.assertTrue(is_close)

    def test_ch4_thermal_conductivity(self):
        """Test CH4 thermal conductivity against Cantera."""
        calculated = self.ch4.thermal_conductivity(self.T_range)
        expected = self.ch4_cantera["thermal_conductivity"]

        self.assertEqual(calculated.shape, expected.shape)
        rel_errors = jnp.abs(calculated - expected) / jnp.abs(expected)
        max_rel_error = jnp.max(rel_errors)
        max_error_idx = jnp.argmax(rel_errors)

        rtol, atol = 5e-3, 1e-10
        is_close = jnp.allclose(calculated, expected, atol=atol, rtol=rtol)

        if not is_close:
            T_at_error = self.T_range[max_error_idx]
            self.fail(
                f"CH4 thermal conductivity doesn't match Cantera reference data:\n"
                f" Max relative error: {max_rel_error:.6e}\n"
                f" at T_idx={max_error_idx} (T={T_at_error:.2f}K)\n"
                f" calculated={calculated[max_error_idx]:.12e}\n"
                f" expected={expected[max_error_idx]:.12e}"
            )

        self.assertTrue(is_close)

    def test_h2_viscosity(self):
        """Test H2 viscosity against Cantera."""
        calculated = self.h2.viscosity(self.T_range)
        expected = self.h2_cantera["viscosity"]

        self.assertEqual(calculated.shape, expected.shape)
        rel_errors = jnp.abs(calculated - expected) / jnp.abs(expected)
        max_rel_error = jnp.max(rel_errors)
        max_error_idx = jnp.argmax(rel_errors)

        rtol, atol = 5e-3, 1e-10
        is_close = jnp.allclose(calculated, expected, atol=atol, rtol=rtol)

        if not is_close:
            T_at_error = self.T_range[max_error_idx]
            self.fail(
                f"H2 viscosity doesn't match Cantera reference data:\n"
                f" Max relative error: {max_rel_error:.6e}\n"
                f" at T_idx={max_error_idx} (T={T_at_error:.2f}K)\n"
                f" calculated={calculated[max_error_idx]:.12e}\n"
                f" expected={expected[max_error_idx]:.12e}"
            )

        self.assertTrue(is_close)

    def test_h2_thermal_conductivity(self):
        """Test H2 thermal conductivity against Cantera."""
        calculated = self.h2.thermal_conductivity(self.T_range)
        expected = self.h2_cantera["thermal_conductivity"]

        self.assertEqual(calculated.shape, expected.shape)
        rel_errors = jnp.abs(calculated - expected) / jnp.abs(expected)
        max_rel_error = jnp.max(rel_errors)
        max_error_idx = jnp.argmax(rel_errors)

        rtol, atol = 5e-3, 1e-10
        is_close = jnp.allclose(calculated, expected, atol=atol, rtol=rtol)

        if not is_close:
            T_at_error = self.T_range[max_error_idx]
            self.fail(
                f"H2 thermal conductivity doesn't match Cantera reference data:\n"
                f" Max relative error: {max_rel_error:.6e}\n"
                f" at T_idx={max_error_idx} (T={T_at_error:.2f}K)\n"
                f" calculated={calculated[max_error_idx]:.12e}\n"
                f" expected={expected[max_error_idx]:.12e}"
            )

        self.assertTrue(is_close)

    def test_ar_viscosity(self):
        """Test AR (monatomic) viscosity against Cantera."""
        calculated = self.ar.viscosity(self.T_range)
        expected = self.ar_cantera["viscosity"]

        self.assertEqual(calculated.shape, expected.shape)
        rel_errors = jnp.abs(calculated - expected) / jnp.abs(expected)
        max_rel_error = jnp.max(rel_errors)
        max_error_idx = jnp.argmax(rel_errors)

        rtol, atol = 5e-3, 1e-10
        is_close = jnp.allclose(calculated, expected, atol=atol, rtol=rtol)

        if not is_close:
            T_at_error = self.T_range[max_error_idx]
            self.fail(
                f"AR viscosity doesn't match Cantera reference data:\n"
                f" Max relative error: {max_rel_error:.6e}\n"
                f" at T_idx={max_error_idx} (T={T_at_error:.2f}K)\n"
                f" calculated={calculated[max_error_idx]:.12e}\n"
                f" expected={expected[max_error_idx]:.12e}"
            )

        self.assertTrue(is_close)

    def test_ar_thermal_conductivity(self):
        """Test AR (monatomic) thermal conductivity against Cantera."""
        calculated = self.ar.thermal_conductivity(self.T_range)
        expected = self.ar_cantera["thermal_conductivity"]

        self.assertEqual(calculated.shape, expected.shape)
        rel_errors = jnp.abs(calculated - expected) / jnp.abs(expected)
        max_rel_error = jnp.max(rel_errors)
        max_error_idx = jnp.argmax(rel_errors)

        rtol, atol = 5e-3, 1e-10
        is_close = jnp.allclose(calculated, expected, atol=atol, rtol=rtol)

        if not is_close:
            T_at_error = self.T_range[max_error_idx]
            self.fail(
                f"AR thermal conductivity doesn't match Cantera reference data:\n"
                f" Max relative error: {max_rel_error:.6e}\n"
                f" at T_idx={max_error_idx} (T={T_at_error:.2f}K)\n"
                f" calculated={calculated[max_error_idx]:.12e}\n"
                f" expected={expected[max_error_idx]:.12e}"
            )

        self.assertTrue(is_close)


if __name__ == "__main__":
    unittest.main()
