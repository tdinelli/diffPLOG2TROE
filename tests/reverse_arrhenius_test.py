import os
import unittest

import jax.numpy as jnp
import numpy as np

from KiRATE.kinetics import Arrhenius, Reaction
from KiRATE.species import Species
from KiRATE.utilities import parse_stoichiometry


class TestReverseArrhenius(unittest.TestCase):
    def setUp(self):
        self.T_range = jnp.linspace(300, 3000, 300)
        O = Species.from_chemkin(
            """
            O                       O   1               G200.000   6000.000  1000.000      1
             2.55160087E+00-3.83085457E-05 8.43197478E-10 4.01267136E-12-4.17476574E-16    2
             2.92287628E+04 4.87617014E+00 3.15906526E+00-3.21509999E-03 6.49255543E-06    3
            -5.98755115E-09 2.06876117E-12 2.91298453E+04 2.09078344E+00                   4
            """
        )

        H = Species.from_chemkin(
            """
            H                       H   1               G200.000   6000.000  1000.000      1
             2.49985211E+00 2.34582548E-07-1.16171641E-10 2.25708298E-14-1.52992005E-18    2
             2.54738024E+04-4.45864645E-01 2.49975925E+00 6.73824499E-07 1.11807261E-09    3
            -3.70192126E-12 2.14233822E-15 2.54737665E+04-4.45574009E-01                   4
            """
        )

        OH = Species.from_chemkin(
            """
            OH                      H   1O   1          G200.000   6000.000  1000.000      1
             2.84581721E+00 1.09723818E-03-2.89121101E-07 4.09099910E-11-2.31382258E-15    2
             3.71706610E+03 5.80339915E+00 3.97585165E+00-2.28555291E-03 4.33442882E-06    3
            -3.59926640E-09 1.26706930E-12 3.39341137E+03-3.55397262E-02                   4
            """
        )

        O2 = Species.from_chemkin(
            """
            O2                      O   2               G200.000   6000.000  1000.000      1
             3.65980488E+00 6.59877372E-04-1.44158172E-07 2.14656037E-11-1.36503784E-15    2
            -1.21603048E+03 3.42074148E+00 3.78498258E+00-3.02002233E-03 9.92029171E-06    3
            -9.77840434E-09 3.28877702E-12-1.06413589E+03 3.64780709E+00                   4
            """
        )

        rate = Arrhenius(parameters={"A": 1.000e14, "n": 0.0, "Ea": 1.5286e04}, name="H+O2<=>O+OH")

        # Parse stoichiometry
        parsed_stuff = parse_stoichiometry(rate.name)
        reactants = parsed_stuff["reactants"]
        products = parsed_stuff["products"]

        # Create reaction
        self.reaction = Reaction(
            reaction_rate_constant=rate,
            species={"H": H, "O2": O2, "OH": OH, "O": O},
            reactants=reactants,
            products=products,
            name=rate.name,
        )

        # ==============================================================================
        # Dataloader for reverse rate constant validation
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        data_file = os.path.join(current_dir, "cantera", "cantera_data", "3.2.0", "reverse_arrhenius.csv")

        data = np.loadtxt(data_file, delimiter=";")
        data = jnp.array(data)

        # Cantera returns the rate constant in m3/kmol·s so the conversion is:
        # cm3/(mol·s) = m3/(kmol·s) x 10^6 / 1000 = m3/(kmol·s) x 10^3
        self.expected_forward_rate = data[:, 1] * 1000
        self.expected_reverse_rate = data[:, 2] * 1000
        self.expected_equilibrium_constant = data[:, 3]
        self.cantera_data_available = True

    def test_delta_nu_calculation(self):
        """Test that change in moles is calculated correctly."""
        # H + O2 <=> O + OH
        # Reactants: 1 + 1 = 2 moles
        # Products: 1 + 1 = 2 moles
        # delta_nu = 2 - 2 = 0
        self.assertEqual(float(self.reaction.delta_nu), 0.0)

    def test_equilibrium_constant_positive(self):
        """Test that equilibrium constant is positive."""
        K_c = self.reaction.equilibrium_constant(self.T_range)
        self.assertTrue(jnp.all(K_c > 0))

    def test_detailed_balance(self):
        """Test detailed balance: k_r = k_f / K_c."""
        k_f = self.reaction.forward_rate_constant(self.T_range)
        K_c = self.reaction.equilibrium_constant(self.T_range)
        k_r = self.reaction.reverse_rate_constant(self.T_range)

        # Verify detailed balance
        k_r_expected = k_f / K_c
        self.assertTrue(
            jnp.allclose(
                k_r,
                k_r_expected,
                atol=1e-10,
                rtol=1e-10,
            ),
            "TODO",
        )

    def test_reverse_rate_vs_cantera(self):
        """Compare KiRATE reverse rate constants against Cantera reference data."""
        # Compute KiRATE values
        k_f_kirate = self.reaction.forward_rate_constant(self.T_range)
        K_c_kirate = self.reaction.equilibrium_constant(self.T_range)
        k_r_kirate = self.reaction.reverse_rate_constant(self.T_range)

        # Compare forward rate constants (should match since same Arrhenius params)
        self.assertTrue(
            jnp.allclose(
                k_f_kirate,
                self.expected_forward_rate,
                atol=1e-10,
                rtol=1e-10,
            ),
            "Forward rate constants do not match Cantera",
        )

        # Compare equilibrium constants
        self.assertTrue(
            jnp.allclose(K_c_kirate, self.expected_equilibrium_constant, rtol=1e-10, atol=1e-10),
            "Equilibrium constants do not match Cantera",
        )

        # Compare reverse rate constants
        self.assertTrue(
            jnp.allclose(k_r_kirate, self.expected_reverse_rate, rtol=1e-10, atol=1e-10),
            "Reverse rate constants do not match Cantera",
        )

    def test_temperature_vectorization(self):
        """Test that reverse rate constant handles vectorized temperatures."""
        # Scalar
        T_scalar = 1000.0
        k_r_scalar = self.reaction.reverse_rate_constant(T_scalar)
        self.assertEqual(k_r_scalar.shape, ())

        # 1D array
        T_array = jnp.array([300.0, 1000.0, 2000.0, 3000.0])
        k_r_array = self.reaction.reverse_rate_constant(T_array)
        self.assertEqual(k_r_array.shape, (4,))

        # Verify consistency
        for i, T in enumerate([300.0, 1000.0, 2000.0, 3000.0]):
            k_r_single = self.reaction.reverse_rate_constant(T)
            self.assertTrue(jnp.allclose(k_r_array[i], k_r_single, rtol=1e-10, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
