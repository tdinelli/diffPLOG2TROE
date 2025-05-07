import unittest
import jax
import jax.numpy as jnp
from diffPLOG2TROE.utilities.thermodynamic_utilities import calculate_effective_concentration, calculate_concentration


class TestConcentrationCalculation(unittest.TestCase):
    def setUp(self):
        self.T_scalar = 300.0  # K
        self.P_scalar = 1.0  # atm
        self.T_array = jnp.array([200.0, 300.0, 400.0])  # K
        self.P_array = jnp.array([0.5, 1.0, 2.0])  # atm
        self.composition_air = {"N2": 0.78, "O2": 0.21, "Ar": 0.01}
        self.efficiencies_example = {"N2": 1.0, "O2": 0.4, "Ar": 0.7}

    def test_none_inputs(self):
        """Test that function handles None inputs correctly."""
        # When both composition and efficiencies are None
        result = calculate_effective_concentration(self.T_scalar, self.P_scalar)
        expected = calculate_concentration(self.T_scalar, self.P_scalar)
        self.assertAlmostEqual(float(result), float(expected))

        # When only composition is provided
        result = calculate_effective_concentration(
            self.T_scalar, self.P_scalar, composition=self.composition_air, efficiencies=None
        )
        expected = calculate_concentration(self.T_scalar, self.P_scalar)
        self.assertAlmostEqual(float(result), float(expected))

        # When only efficiencies is provided
        result = calculate_effective_concentration(
            self.T_scalar, self.P_scalar, composition=None, efficiencies=self.efficiencies_example
        )
        expected = calculate_concentration(self.T_scalar, self.P_scalar)
        self.assertAlmostEqual(float(result), float(expected))

    def test_scalar_inputs(self):
        """Test with scalar temperature and pressure inputs."""
        result = calculate_effective_concentration(
            self.T_scalar, self.P_scalar, self.composition_air, self.efficiencies_example
        )

        # Calculate expected value manually for validation
        # First, get base concentration
        M = calculate_concentration(self.T_scalar, self.P_scalar)

        # Calculate weighted efficiency
        weighted_eff = (
            self.composition_air["N2"] * self.efficiencies_example["N2"]
            + self.composition_air["O2"] * self.efficiencies_example["O2"]
            + self.composition_air["Ar"] * self.efficiencies_example["Ar"]
        )

        # Calculate total accounted fraction
        total_accounted = sum(self.composition_air.values())

        # Calculate remaining fraction
        remaining = max(0.0, 1.0 - total_accounted)

        # Calculate expected effective concentration
        expected = M * (weighted_eff + remaining)

        self.assertAlmostEqual(float(result), float(expected), places=10)

    def test_array_inputs(self):
        """Test with array temperature and pressure inputs."""
        # Use a single T and array of P
        result_p_array = calculate_effective_concentration(
            self.T_scalar, self.P_array, self.composition_air, self.efficiencies_example
        )
        self.assertEqual(result_p_array.shape, self.P_array.shape)

        # Use array of T and single P
        result_t_array = calculate_effective_concentration(
            self.T_array, self.P_scalar, self.composition_air, self.efficiencies_example
        )
        self.assertEqual(result_t_array.shape, self.T_array.shape)

        # Verify first element of each array
        # T_scalar, P_array[0]
        M = calculate_concentration(self.T_scalar, self.P_array[0])
        weighted_eff = (
            self.composition_air["N2"] * self.efficiencies_example["N2"]
            + self.composition_air["O2"] * self.efficiencies_example["O2"]
            + self.composition_air["Ar"] * self.efficiencies_example["Ar"]
        )
        total_accounted = sum(self.composition_air.values())
        remaining = max(0.0, 1.0 - total_accounted)
        expected = M * (weighted_eff + remaining)
        self.assertAlmostEqual(float(result_p_array[0]), float(expected), places=10)

    def test_meshgrid(self):
        """Test with both T and P as arrays (should create a meshgrid)."""
        result = calculate_effective_concentration(
            self.T_array, self.P_array, self.composition_air, self.efficiencies_example
        )
        # Check if result has the expected shape from meshgrid
        self.assertEqual(result.shape, (len(self.T_array), len(self.P_array)))

    def test_incomplete_composition(self):
        """Test with composition that doesn't sum to 1.0."""
        # Composition with sum < 1.0
        incomplete_comp = {"N2": 0.5, "O2": 0.2}
        result = calculate_effective_concentration(
            self.T_scalar, self.P_scalar, incomplete_comp, self.efficiencies_example
        )

        # Calculate expected value
        M = calculate_concentration(self.T_scalar, self.P_scalar)
        weighted_eff = (
            incomplete_comp["N2"] * self.efficiencies_example["N2"]
            + incomplete_comp["O2"] * self.efficiencies_example["O2"]
        )
        total_accounted = sum(incomplete_comp.values())
        remaining = max(0.0, 1.0 - total_accounted)
        expected = M * (weighted_eff + remaining)

        self.assertAlmostEqual(float(result), float(expected), places=10)

    def test_missing_efficiency(self):
        """Test with missing efficiency values for some species."""
        # Efficiency dict missing some species
        incomplete_eff = {"N2": 1.0}  # Missing O2 and Ar
        result = calculate_effective_concentration(self.T_scalar, self.P_scalar, self.composition_air, incomplete_eff)

        # Calculate expected - missing species should default to 1.0
        M = calculate_concentration(self.T_scalar, self.P_scalar)
        weighted_eff = (
            self.composition_air["N2"] * incomplete_eff["N2"]
            + self.composition_air["O2"] * 1.0  # Default
            + self.composition_air["Ar"] * 1.0  # Default
        )
        total_accounted = sum(self.composition_air.values())
        remaining = max(0.0, 1.0 - total_accounted)
        expected = M * (weighted_eff + remaining)

        self.assertAlmostEqual(float(result), float(expected), places=10)

    def test_extra_composition_species(self):
        """Test with extra species in composition that aren't in efficiencies."""
        # Add a species not present in the efficiencies dict
        extra_comp = {**self.composition_air, "CO2": 0.01}
        # Normalize to ensure total is still 1.0
        total = sum(extra_comp.values())
        extra_comp = {k: v / total for k, v in extra_comp.items()}

        result = calculate_effective_concentration(self.T_scalar, self.P_scalar, extra_comp, self.efficiencies_example)

        # Calculate expected - CO2 should use default efficiency of 1.0
        M = calculate_concentration(self.T_scalar, self.P_scalar)
        weighted_eff = (
            extra_comp["N2"] * self.efficiencies_example["N2"]
            + extra_comp["O2"] * self.efficiencies_example["O2"]
            + extra_comp["Ar"] * self.efficiencies_example["Ar"]
            + extra_comp["CO2"] * 1.0  # Default
        )
        total_accounted = sum(extra_comp.values())
        remaining = max(0.0, 1.0 - total_accounted)
        expected = M * (weighted_eff + remaining)

        self.assertAlmostEqual(float(result), float(expected), places=10)

    def test_jit_compatibility(self):
        """Test that the function works correctly when JIT compiled."""

        # Define a test function that uses our function
        @jax.jit
        def test_fn(T, P, comp, eff):
            return calculate_effective_concentration(T, P, comp, eff)

        # Try with scalar inputs
        result_scalar = test_fn(self.T_scalar, self.P_scalar, self.composition_air, self.efficiencies_example)

        # Calculate expected value manually
        M = calculate_concentration(self.T_scalar, self.P_scalar)
        weighted_eff = (
            self.composition_air["N2"] * self.efficiencies_example["N2"]
            + self.composition_air["O2"] * self.efficiencies_example["O2"]
            + self.composition_air["Ar"] * self.efficiencies_example["Ar"]
        )
        total_accounted = sum(self.composition_air.values())
        remaining = max(0.0, 1.0 - total_accounted)
        expected = M * (weighted_eff + remaining)

        self.assertAlmostEqual(float(result_scalar), float(expected), places=10)

        # Try with array inputs
        result_array = test_fn(self.T_array, self.P_scalar, self.composition_air, self.efficiencies_example)
        self.assertEqual(result_array.shape, self.T_array.shape)


if __name__ == "__main__":
    unittest.main()
