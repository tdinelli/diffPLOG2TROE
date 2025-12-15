import os
import unittest

import jax.numpy as jnp
import numpy as np

from KiRATE.kinetics import CABR


class TestCabr(unittest.TestCase):
    def setUp(self):
        self.T_range = jnp.linspace(300, 3000, 300)
        self.P_range = jnp.logspace(jnp.log10(0.01), jnp.log10(100), 300)

        self.reaction = CABR(
            hpl_parameters={"A": 5.88e-14, "n": 6.721, "Ea": -3022.227},
            lpl_parameters={"A": 282320.078, "n": 1.46878, "Ea": -3270.56495},
            cabr_type="lindemann",
            name="CH+OH(+M)=CH2O+H2(+M)",
        )

        # ==============================================================================
        # Dataloader
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.2.0", "cabr_n2.csv")
        data = np.loadtxt(data_file, delimiter=";")
        self.expected_rate_n2 = jnp.array(data) * 1000

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


if __name__ == "__main__":
    unittest.main()
