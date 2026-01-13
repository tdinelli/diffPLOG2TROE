import os
import unittest

import jax.numpy as jnp
import numpy as np

from KiRATE.kinetics import FallOff


class TestFallOff(unittest.TestCase):
    def setUp(self):
        self.T_range = jnp.linspace(300, 3000, 300)
        self.P_range = jnp.logspace(jnp.log10(0.01), jnp.log10(100), 300) * 1.01325  # atm -> bar

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
        calculated_rates = self.reaction.rate_constant(
            T=self.T_range,
            P=self.P_range,
            composition={"N2": 1},
        )

        # Calculate relative errors for reporting
        rel_errors = jnp.abs(calculated_rates - self.expected_rate_n2) / jnp.abs(self.expected_rate_n2)
        max_rel_error = jnp.max(rel_errors)
        max_error_idx = jnp.unravel_index(jnp.argmax(rel_errors), rel_errors.shape)

        # Check if all values are close
        is_close = jnp.allclose(calculated_rates, self.expected_rate_n2, atol=1e-10, rtol=1e-8)

        if not is_close:
            p_idx, t_idx = max_error_idx
            error_msg = (
                f"Max relative error: {max_rel_error:.6e} at "
                f"P_idx={p_idx} (P={self.P_range[p_idx].item():.6f}atm), "
                f"T_idx={t_idx} (T={self.T_range[t_idx].item():.2f}K), "
                f"calculated={calculated_rates[p_idx, t_idx].item():.6e}, "
                f"expected={self.expected_rate_n2[p_idx, t_idx].item():.6e}"
            )
            self.fail(error_msg)

        self.assertTrue(is_close)

    def test_kinetic_constant_n2h2o(self):
        calculated_rates = self.reaction.rate_constant(
            T=self.T_range,
            P=self.P_range,
            composition={"N2": 0.5, "H2O": 0.5},
        )

        # Check shapes match
        self.assertEqual(
            calculated_rates.shape,
            self.expected_rate_n2h2o.shape,
            f"Shape mismatch: calculated {calculated_rates.shape} vs expected {self.expected_rate_n2h2o.shape}",
        )

        # Calculate relative errors for reporting
        rel_errors = jnp.abs(calculated_rates - self.expected_rate_n2h2o) / jnp.abs(self.expected_rate_n2h2o)
        max_rel_error = jnp.max(rel_errors)
        max_error_idx = jnp.unravel_index(jnp.argmax(rel_errors), rel_errors.shape)

        # Check if all values are close
        is_close = jnp.allclose(calculated_rates, self.expected_rate_n2h2o, atol=1e-10, rtol=1e-8)

        if not is_close:
            p_idx, t_idx = max_error_idx
            error_msg = (
                f"Max relative error: {max_rel_error:.6e} at "
                f"P_idx={p_idx} (P={self.P_range[p_idx].item():.6f}atm), "
                f"T_idx={t_idx} (T={self.T_range[t_idx].item():.2f}K), "
                f"calculated={calculated_rates[p_idx, t_idx].item():.6e}, "
                f"expected={self.expected_rate_n2h2o[p_idx, t_idx].item():.6e}"
            )
            self.fail(error_msg)

        self.assertTrue(is_close)

    def test_kinetic_constant_n2o2h2o(self):
        calculated_rates = self.reaction.rate_constant(
            T=self.T_range,
            P=self.P_range,
            composition={"N2": 0.2, "O2": 0.3, "H2O": 0.5},
        )

        # Check shapes match
        self.assertEqual(
            calculated_rates.shape,
            self.expected_rate_n2o2h2o.shape,
            f"Shape mismatch: calculated {calculated_rates.shape} vs expected {self.expected_rate_n2o2h2o.shape}",
        )

        # Calculate relative errors for reporting
        rel_errors = jnp.abs(calculated_rates - self.expected_rate_n2o2h2o) / jnp.abs(self.expected_rate_n2o2h2o)
        max_rel_error = jnp.max(rel_errors)
        max_error_idx = jnp.unravel_index(jnp.argmax(rel_errors), rel_errors.shape)

        # Check if all values are close
        is_close = jnp.allclose(calculated_rates, self.expected_rate_n2o2h2o, atol=1e-10, rtol=1e-8)

        if not is_close:
            p_idx, t_idx = max_error_idx
            error_msg = (
                f"Max relative error: {max_rel_error:.6e} at "
                f"P_idx={p_idx} (P={self.P_range[p_idx].item():.6f}atm), "
                f"T_idx={t_idx} (T={self.T_range[t_idx].item():.2f}K), "
                f"calculated={calculated_rates[p_idx, t_idx].item():.6e}, "
                f"expected={self.expected_rate_n2o2h2o[p_idx, t_idx].item():.6e}"
            )
            self.fail(error_msg)

        self.assertTrue(is_close)


if __name__ == "__main__":
    unittest.main()
