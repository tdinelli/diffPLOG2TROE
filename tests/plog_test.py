import os
import unittest

import jax.numpy as jnp
import numpy as np

from KiRATE.kinetics import Plog


class TestPlog(unittest.TestCase):
    def setUp(self):
        # Define temperature and pressure grids for validation
        self.T_range = jnp.linspace(300, 3000, 300)

        # Pressure range: 0.01 to 100 atm
        self.P_range = jnp.logspace(jnp.log10(0.01), jnp.log10(100), 300)

        # Create PLOG reaction object: OH + NO = HONO
        self.reaction = Plog(
            parameters={
                0.01: {"A": 5.02e21, "n": -4.24, "Ea": 898.9},
                0.1: {"A": 5.31e22, "n": -4.24, "Ea": 1184.0},
                0.316: {"A": 1.38e23, "n": -4.22, "Ea": 1376.0},
                1.0: {"A": 3.09e23, "n": -4.17, "Ea": 1621.0},
                3.16: {"A": 5.45e23, "n": -4.09, "Ea": 1911.0},
                10.0: {"A": 6.35e23, "n": -3.97, "Ea": 2222.0},
                31.6: {"A": 3.68e23, "n": -3.75, "Ea": 2501.0},
                100.0: {"A": 7.29e22, "n": -3.41, "Ea": 2660.0},
            },
            name="OH+NO=HONO",
        )

        # Load reference data from Cantera 3.2.0
        # Each row corresponds to a pressure level, columns are temperatures
        # Shape: (300 pressures, 300 temperatures)
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.2.0", "plog.csv")
        data = np.loadtxt(data_file, delimiter=";")

        # Unit conversion: Cantera returns k in m3/(kmol s), we use cm3/(mol·s)
        # Conversion factor: m3/(kmol s) x 1000 = cm3/(mol s)
        # 1 m3 = 10^6 cm3 and 1 kmol = 1000 mol, so 10^6/1000 = 1000
        data = jnp.array(data) * 1000

        self.expected_rate = data

    def test_rate_constant(self):
        # Compute rate constants over full T-P grid
        # Shape: (300 pressures, 300 temperatures) matching reference data
        calculated_rates = self.reaction.rate_constant(self.T_range, self.P_range)

        # Calculate relative errors for reporting
        rel_errors = jnp.abs(calculated_rates - self.expected_rate) / jnp.abs(self.expected_rate)
        max_rel_error = jnp.max(rel_errors)
        max_error_idx = jnp.unravel_index(jnp.argmax(rel_errors), rel_errors.shape)

        # Check if all values are close
        is_close = jnp.allclose(calculated_rates, self.expected_rate, atol=1e-10, rtol=1e-8)

        if not is_close:
            p_idx, t_idx = max_error_idx
            error_msg = (
                f"Max relative error: {max_rel_error:.6e} at "
                f"P_idx={p_idx} (P={self.P_range[p_idx].item():.6f}atm), "
                f"T_idx={t_idx} (T={self.T_range[t_idx].item():.2f}K), "
                f"calculated={calculated_rates[p_idx, t_idx].item():.6e}, "
                f"expected={self.expected_rate[p_idx, t_idx].item():.6e}"
            )
            self.fail(error_msg)

        self.assertTrue(is_close)


if __name__ == "__main__":
    unittest.main()
