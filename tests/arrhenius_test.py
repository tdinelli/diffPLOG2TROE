import os
import unittest

import jax.numpy as jnp
import numpy as np

from KiRATE.kinetics import Arrhenius


class TestArrhenius(unittest.TestCase):
    # @staticmethod
    # def analytical_gradients(reaction_instance, temperature):
    #     """
    #     Compute analytical gradients for Arrhenius rate constant.
    #
    #     For k(T) = A * T^n * exp(-Ea / (R*T)):
    #     - dk/dA = T^n * exp(-Ea / (R*T)) = k/A
    #     - dk/dn = A * T^n * ln(T) * exp(-Ea / (R*T)) = k * ln(T)
    #     - dk/dEa = A * T^n * exp(-Ea / (R*T)) * (-1/(R*T)) = -k/(R*T)
    #     """
    #     A, n, Ea = reaction_instance.A, reaction_instance.n, reaction_instance.Ea
    #
    #     # Common terms for efficiency
    #     T_n = jnp.power(temperature, n)
    #     exp_term = jnp.exp(-Ea / (constants.R_cal_mol * temperature))
    #     k = reaction_instance.rate_constant(temperature)  # Rate constant
    #
    #     # Analytical gradients
    #     dkdA = T_n * exp_term
    #     dkdn = A * T_n * jnp.log(temperature) * exp_term
    #     dkdEa = -k / (constants.R_cal_mol * temperature)
    #
    #     return dkdA, dkdn, dkdEa
    #
    # @staticmethod
    # def analytical_temperature_gradient(reaction_instance, temperature):
    #     """
    #     Compute analytical temperature gradient for Arrhenius rate constant.
    #
    #     For k(T) = A * T^n * exp(-Ea / (R*T)):
    #     dk/dT = k * (n/T - Ea/(R*T^2))
    #     """
    #     k = reaction_instance.rate_constant(temperature)
    #     n, Ea = reaction_instance.n, reaction_instance.Ea
    #
    #     return k * (n / temperature + Ea / (constants.R_cal_mol * temperature**2))

    def setUp(self):
        self.T_range = jnp.linspace(300, 3000, 300)
        self.reaction = Arrhenius(parameters={"A": 1.000e14, "n": 0.0, "Ea": 1.5286e04}, name="H2+O=H+OH")

        # Test temperatures for gradient validation
        self.test_temperatures = jnp.array([300.0, 1000.0, 1500.0, 2000.0, 3000.0])

        # ==============================================================================
        # Dataloader for rate constant validation
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.2.0", "arrhenius.csv")
        data = np.loadtxt(data_file, delimiter=";")
        data = jnp.array(data)

        # Cantera returns the rate constant in m3/kmol·s so the conversion is:
        # cm3/(mol·s) = m3/(kmol·s) x 10^6 / 1000 = m3/(kmol·s) x 10^3
        self.expected_rate = data[:, 1] * 1000

        # ==============================================================================
        # Additional test cases for gradient validation
        self.standard_arrhenius = Arrhenius(parameters={"A": 1e13, "n": 0.0, "Ea": 15000}, name="Standard Arrhenius")
        self.high_n_reaction = Arrhenius(parameters={"A": 1e12, "n": 3.5, "Ea": 25000}, name="High n reaction")
        self.zero_ea_reaction = Arrhenius(parameters={"A": 1e10, "n": 1.2, "Ea": 0.0}, name="Zero activation energy")

    def test_arrhenius_rate_constant(self):
        """Test direct rate constant calculation against reference data."""
        calculated_rates = self.reaction.rate_constant(self.T_range)

        self.assertTrue(
            jnp.allclose(
                calculated_rates,
                self.expected_rate,
                atol=1e-10,
                rtol=1e-10,
            ),
            "Calculated rate constants (from the direct function) for the Arrhenius case don't match reference data",
        )

    def test_log_arrhenius_rate_constant(self):
        calculated_rates = jnp.exp(self.reaction.log_rate_constant(self.T_range))

        self.assertTrue(
            jnp.allclose(
                calculated_rates,
                self.expected_rate,
                atol=1e-10,
                rtol=1e-10,
            ),
            "Calculated rate constants (from the log function) for the Arrhenius case don't match reference data",
        )

    # def test_parameter_gradients_vs_analytical(self):
    #     """Test automatic differentiation parameter gradients against analytical solutions."""
    #
    #     test_reactions = [
    #         ("main_reaction", self.reaction),
    #         ("standard_arrhenius", self.standard_arrhenius),
    #         ("high_n_reaction", self.high_n_reaction),
    #         ("zero_ea_reaction", self.zero_ea_reaction),
    #     ]
    #
    #     for reaction_name, reaction in test_reactions:
    #         with self.subTest(reaction=reaction_name):
    #             for T in self.test_temperatures:
    #                 with self.subTest(temperature=T):
    #                     # Get automatic differentiation gradients
    #                     autodiff_grads = reaction.grad_params(T)
    #
    #                     # Get analytical gradients
    #                     analytical_dkdA, analytical_dkdn, analytical_dkdEa = self.analytical_gradients(reaction, T)
    #
    #                     self.assertTrue(
    #                         jnp.allclose(autodiff_grads.A, analytical_dkdA, rtol=1e-10, atol=1e-10),
    #                         f"dk/dA mismatch for {reaction_name} at T={T}K: "
    #                         f"autodiff={autodiff_grads.A:.6e}, analytical={analytical_dkdA:.6e}",
    #                     )
    #
    #                     self.assertTrue(
    #                         jnp.allclose(autodiff_grads.n, analytical_dkdn, rtol=1e-10, atol=1e-10),
    #                         f"dk/dn mismatch for {reaction_name} at T={T}K: "
    #                         f"autodiff={autodiff_grads.n:.6e}, analytical={analytical_dkdn:.6e}",
    #                     )
    #
    #                     self.assertTrue(
    #                         jnp.allclose(autodiff_grads.Ea, analytical_dkdEa, rtol=1e-10, atol=1e-10),
    #                         f"dk/dEa mismatch for {reaction_name} at T={T}K: "
    #                         f"autodiff={autodiff_grads.Ea:.6e}, analytical={analytical_dkdEa:.6e}",
    #                     )
    #
    # def test_temperature_gradients_vs_analytical(self):
    #     """Test automatic differentiation temperature gradients against analytical solutions."""
    #
    #     test_reactions = [
    #         ("main_reaction", self.reaction),
    #         ("standard_arrhenius", self.standard_arrhenius),
    #         ("high_n_reaction", self.high_n_reaction),
    #         ("zero_ea_reaction", self.zero_ea_reaction),
    #     ]
    #
    #     for reaction_name, reaction in test_reactions:
    #         with self.subTest(reaction=reaction_name):
    #             for T in self.test_temperatures:
    #                 with self.subTest(temperature=T):
    #                     # Get automatic differentiation gradient
    #                     autodiff_dkdT = reaction.grad_temperature(T)
    #
    #                     # Get analytical gradient
    #                     analytical_dkdT = self.analytical_temperature_gradient(reaction, T)
    #
    #                     self.assertTrue(
    #                         jnp.allclose(autodiff_dkdT, analytical_dkdT, rtol=1e-10, atol=1e-10),
    #                         f"dk/dT mismatch for {reaction_name} at T={T}K: "
    #                         f"autodiff={autodiff_dkdT:.6e}, analytical={analytical_dkdT:.6e}",
    #                     )
    #
    # def test_parameter_gradients_vectorized(self):
    #     """Test parameter gradients with vectorized temperature inputs."""
    #
    #     # Test with temperature array
    #     autodiff_grads = self.reaction.grad_params(self.test_temperatures)
    #
    #     # Compute analytical gradients for each temperature
    #     analytical_results = jnp.array([self.analytical_gradients(self.reaction, T) for T in self.test_temperatures])
    #
    #     # Transpose to get separate arrays for each parameter
    #     analytical_dkdA = analytical_results[:, 0]
    #     analytical_dkdn = analytical_results[:, 1]
    #     analytical_dkdEa = analytical_results[:, 2]
    #
    #     # Compare vectorized results
    #     self.assertTrue(
    #         jnp.allclose(autodiff_grads.A, analytical_dkdA, rtol=1e-10, atol=1e-10),
    #         "Vectorized dk/dA mismatch",
    #     )
    #
    #     self.assertTrue(
    #         jnp.allclose(autodiff_grads.n, analytical_dkdn, rtol=1e-10, atol=1e-10),
    #         "Vectorized dk/dn mismatch",
    #     )
    #
    #     self.assertTrue(
    #         jnp.allclose(autodiff_grads.Ea, analytical_dkdEa, rtol=1e-10, atol=1e-10),
    #         "Vectorized dk/dEa mismatch",
    #     )
    #
    # def test_temperature_gradients_vectorized(self):
    #     """Test temperature gradients with vectorized temperature inputs."""
    #
    #     # Test with temperature array
    #     autodiff_dkdT = self.reaction.grad_temperature(self.test_temperatures)
    #
    #     # Compute analytical gradients for each temperature
    #     analytical_dkdT = jnp.array(
    #         [self.analytical_temperature_gradient(self.reaction, T) for T in self.test_temperatures]
    #     )
    #
    #     # Compare vectorized results
    #     self.assertTrue(
    #         jnp.allclose(autodiff_dkdT, analytical_dkdT, rtol=1e-10, atol=1e-10),
    #         "Vectorized dk/dT mismatch",
    #     )

    def test_chemkin_parsing(self):
        """Test that the CHEMKIN parser works correctly"""
        chemkin_string = """
            H2+O = H+OH 1.000e+14   0.0     1.5286e+04
        """
        test_reaction = Arrhenius.from_chemkin(input_string=chemkin_string)

        self.assertEqual(test_reaction.A, self.reaction.A)
        self.assertEqual(test_reaction.n, self.reaction.n)
        self.assertEqual(test_reaction.Ea, self.reaction.Ea)
        self.assertEqual(test_reaction.name, self.reaction.name)


if __name__ == "__main__":
    unittest.main()
