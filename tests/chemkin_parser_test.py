"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details

Unit tests for CHEMKIN parser utilities.
"""

import unittest

from KiRATE.utilities.chemkin_parser import (
    check_reaction_name,
    parse_cabr,
    parse_falloff,
    parse_plog,
    parse_reaction_line,
    parse_species,
    parse_stoichiometry,
)


class TestParseReactionLine(unittest.TestCase):
    """Test basic reaction line parsing."""

    def test_simple_reaction(self):
        """Test parsing a simple reaction with Arrhenius parameters."""
        reaction_name, params = parse_reaction_line("H+O2=>OH+O 2.0e+14 0.0 16800.0")

        self.assertEqual(reaction_name, "H+O2=>OH+O")
        self.assertEqual(params["A"], 2.0e14)
        self.assertEqual(params["n"], 0.0)
        self.assertEqual(params["Ea"], 16800.0)

    def test_reaction_with_whitespace(self):
        """Test parsing with various whitespace."""
        reaction_name, params = parse_reaction_line("  2CH3 (+M) = C2H6 (+M)   1e16   0.0   0.0  ", m_is_allowed=True)

        # Parser doesn't remove spaces before (+M)
        self.assertEqual(reaction_name, "2CH3 (+M)=C2H6 (+M)")
        self.assertEqual(params["A"], 1e16)
        self.assertEqual(params["n"], 0.0)
        self.assertEqual(params["Ea"], 0.0)

    def test_reversible_arrows(self):
        """Test different arrow types."""
        # Irreversible
        name1, _ = parse_reaction_line("A=>B 1e10 0.0 0.0")
        self.assertEqual(name1, "A=>B")

        # Reversible (=)
        name2, _ = parse_reaction_line("A=B 1e10 0.0 0.0")
        self.assertEqual(name2, "A=B")

        # Reversible (<=>)
        name3, _ = parse_reaction_line("A<=>B 1e10 0.0 0.0")
        self.assertEqual(name3, "A<=>B")

    def test_comments(self):
        """Test that comments are properly stripped."""
        reaction_name, params = parse_reaction_line("H+O2=>OH+O 2.0e+14 0.0 16800.0 ! This is a comment")

        self.assertEqual(reaction_name, "H+O2=>OH+O")
        self.assertEqual(params["A"], 2.0e14)

    def test_invalid_input(self):
        """Test error handling for invalid inputs."""
        with self.assertRaises(ValueError):
            parse_reaction_line("")

        with self.assertRaises(ValueError):
            parse_reaction_line("H+O2=>OH+O")  # Missing parameters

        with self.assertRaises(ValueError):
            parse_reaction_line("H+O2=>OH+O abc def ghi")  # Non-numeric parameters


class TestCheckReactionName(unittest.TestCase):
    """Test reaction name validation."""

    def test_valid_reactions(self):
        """Test reactions that should pass validation."""
        # No third-body
        check_reaction_name("H+O2=>OH+O")

        # Generic third-body M
        check_reaction_name("H+O2+M=HO2+M", m_is_allowed=True)
        check_reaction_name("2CH3(+M)=C2H6(+M)", m_is_allowed=True)

        # Species containing M in name
        check_reaction_name("H+B2M2=CH3+IC4H8")

    def test_invalid_m_usage(self):
        """Test that M is properly detected when not allowed."""
        # Standalone M should be rejected
        with self.assertRaises(ValueError):
            check_reaction_name("H+O2+M=HO2+M", m_is_allowed=False)

        # (+M) should be rejected
        with self.assertRaises(ValueError):
            check_reaction_name("2CH3(+M)=C2H6(+M)", m_is_allowed=False)

    def test_m_in_species_name(self):
        """Test that M within species names is allowed."""
        # These should NOT raise errors (M is part of species name)
        check_reaction_name("H+B2M2=CH3+IC4H8")
        check_reaction_name("CH3+CH2M=C2H5M")
        check_reaction_name("HM+O2=HO2+M", m_is_allowed=True)  # Last M is third-body


class TestParseStoichiometry(unittest.TestCase):
    """Test stoichiometry parsing."""

    def test_simple_reaction(self):
        """Test parsing a simple reaction."""
        result = parse_stoichiometry("H+O2=>OH+O")

        self.assertSetEqual(set(result["species"]), {"H", "O2", "OH", "O"})
        self.assertEqual(result["reactants"], {"H": 1.0, "O2": 1.0})
        self.assertEqual(result["products"], {"OH": 1.0, "O": 1.0})
        self.assertFalse(result["reversible"])

    def test_stoichiometric_coefficients(self):
        """Test reactions with stoichiometric coefficients."""
        result = parse_stoichiometry("2H+O2=H2O+O")

        self.assertEqual(result["reactants"]["H"], 2.0)
        self.assertEqual(result["reactants"]["O2"], 1.0)

    def test_generic_third_body(self):
        """Test generic third-body M removal."""
        # +M notation
        result = parse_stoichiometry("H+O2+M=HO2+M")
        self.assertSetEqual(set(result["species"]), {"H", "O2", "HO2"})

        # (+M) notation
        result = parse_stoichiometry("2CH3(+M)=C2H6(+M)")
        self.assertSetEqual(set(result["species"]), {"CH3", "C2H6"})

    def test_explicit_collider(self):
        """Test explicit collider notation."""
        result = parse_stoichiometry("H+O2(+N2)=HO2(+N2)")

        self.assertSetEqual(set(result["species"]), {"H", "O2", "HO2", "N2"})
        self.assertEqual(result["reactants"], {"H": 1.0, "O2": 1.0, "N2": 1.0})
        self.assertEqual(result["products"], {"HO2": 1.0, "N2": 1.0})

    def test_explicit_collider_variants(self):
        """Test various explicit collider species."""
        # Argon
        result = parse_stoichiometry("2H(+AR)=>H2(+AR)")
        self.assertIn("AR", result["species"])
        self.assertEqual(result["reactants"]["AR"], 1.0)

        # H2O
        result = parse_stoichiometry("H+OH(+H2O)=H2O(+H2O)")
        self.assertIn("H2O", result["species"])
        self.assertEqual(result["reactants"]["H2O"], 1.0)
        self.assertEqual(result["products"]["H2O"], 2.0)  # H2O as product + collider

    def test_species_with_parentheses(self):
        """Test species with parentheses (excited states, isomers)."""
        result = parse_stoichiometry("CH2(S)+H2=CH4")

        self.assertIn("CH2(S)", result["species"])
        self.assertEqual(result["reactants"]["CH2(S)"], 1.0)

    def test_reversibility(self):
        """Test reversibility detection."""
        # Irreversible
        result1 = parse_stoichiometry("A=>B")
        self.assertFalse(result1["reversible"])

        # Reversible (=)
        result2 = parse_stoichiometry("A=B")
        self.assertTrue(result2["reversible"])

        # Reversible (<=>)
        result3 = parse_stoichiometry("A<=>B")
        self.assertTrue(result3["reversible"])

    def test_duplicate_species(self):
        """Test accumulation of duplicate species."""
        result = parse_stoichiometry("O+O=O2")

        self.assertEqual(result["reactants"]["O"], 2.0)


class TestParsePlog(unittest.TestCase):
    """Test PLOG reaction parsing."""

    def test_single_term_plog(self):
        """Test parsing PLOG with one term per pressure."""
        input_string = """
            H+O2=O+OH  0.0 0.0 0.0
             PLOG / 0.01  1.0E+12  0.0  10000.0 /
             PLOG / 1.0   1.0E+13  0.5  12000.0 /
             PLOG / 100.0 1.0E+14  1.0  15000.0 /
        """

        reaction_name, params = parse_plog(input_string)

        self.assertEqual(reaction_name, "H+O2=O+OH")
        self.assertEqual(len(params), 3)
        self.assertIn(0.01, params)
        self.assertIn(1.0, params)
        self.assertIn(100.0, params)

        # Each pressure should have a list with one entry
        self.assertEqual(len(params[0.01]), 1)
        self.assertEqual(params[0.01][0]["A"], 1.0e12)
        self.assertEqual(params[0.01][0]["n"], 0.0)
        self.assertEqual(params[0.01][0]["Ea"], 10000.0)

    def test_multi_term_plog(self):
        """Test parsing PLOG with multiple terms per pressure."""
        input_string = """
            O+C10H7CH3=CH3C10H6OH    1.0e+17  -1.64  4750.0
             PLOG / 0.01  1.44e+14  -0.93   1700.0 /
             PLOG / 0.01  4.07e+15  -6.73  -14031.0 /
             PLOG / 0.01  1.07e+35  -6.92   13025.0 /
             PLOG / 1.0   1.38e+17  -1.64   4750.0 /
             PLOG / 1.0   6.20e+13  -0.78   3522.0 /
        """

        reaction_name, params = parse_plog(input_string)

        self.assertEqual(reaction_name, "O+C10H7CH3=CH3C10H6OH")

        # 0.01 atm should have 3 terms
        self.assertEqual(len(params[0.01]), 3)
        self.assertAlmostEqual(params[0.01][0]["A"], 1.44e14)
        self.assertAlmostEqual(params[0.01][1]["A"], 4.07e15)
        self.assertAlmostEqual(params[0.01][2]["A"], 1.07e35)

        # 1.0 atm should have 2 terms
        self.assertEqual(len(params[1.0]), 2)

    def test_plog_parameter_extraction(self):
        """Test correct extraction of A, n, Ea parameters."""
        input_string = """
            TEST_REACTION  0.0 0.0 0.0
             PLOG / 1.0  2.5e+13  0.5  1500.0 /
        """

        _, params = parse_plog(input_string)

        self.assertAlmostEqual(params[1.0][0]["A"], 2.5e13)
        self.assertAlmostEqual(params[1.0][0]["n"], 0.5)
        self.assertAlmostEqual(params[1.0][0]["Ea"], 1500.0)


class TestParseFalloff(unittest.TestCase):
    """Test fall-off reaction parsing."""

    def test_troe_falloff(self):
        """Test parsing Troe fall-off reaction."""
        input_string = """
            2CH3(+M)=C2H6(+M)  9.03e+16  -1.18  654.0
             LOW / 3.18e+41  -7.03  2762.0 /
            TROE / 0.6041  6927  132  2762 /
        """

        reaction_name, hpl, lpl, falloff_type, falloff_params, eff = parse_falloff(input_string)

        self.assertEqual(reaction_name, "2CH3(+M)=C2H6(+M)")
        self.assertEqual(falloff_type, "troe")
        self.assertAlmostEqual(hpl["A"], 9.03e16)
        self.assertAlmostEqual(lpl["A"], 3.18e41)
        self.assertAlmostEqual(falloff_params["A"], 0.6041)
        self.assertIn("T3", falloff_params)

    def test_lindemann_falloff(self):
        """Test parsing Lindemann fall-off (no broadening factor)."""
        input_string = """
        H+O2(+M)=HO2(+M)  1.48e+12  0.6  0.0
         LOW / 6.37e+20  -1.72  524.8 /
        """

        reaction_name, hpl, lpl, falloff_type, falloff_params, eff = parse_falloff(input_string)

        self.assertEqual(falloff_type, "lindemann")
        self.assertIsNone(falloff_params)

    def test_falloff_efficiencies(self):
        """Test parsing third-body efficiencies."""
        input_string = """
        2CH3(+M)=C2H6(+M)  9.03e+16  -1.18  654.0
         LOW / 3.18e+41  -7.03  2762.0 /
         H2/2/ CO/2/ CO2/3/ H2O/5/
        """

        _, _, _, _, _, efficiencies = parse_falloff(input_string)

        self.assertEqual(efficiencies["H2"], 2.0)
        self.assertEqual(efficiencies["CO"], 2.0)
        self.assertEqual(efficiencies["CO2"], 3.0)
        self.assertEqual(efficiencies["H2O"], 5.0)


class TestParseCABR(unittest.TestCase):
    """Test CABR (Chemically Activated BiRadical) parsing."""

    def test_cabr_basic(self):
        """Test basic CABR reaction parsing (Lindemann default)."""
        input_string = """
        H+CH2(+M)=CH3(+M)  1.0e+14  0.0  0.0
         HIGH / 1.0e+10  0.0  0.0 /
        """

        reaction_name, lpl, hpl, falloff_type, falloff_params, eff = parse_cabr(input_string)

        self.assertEqual(reaction_name, "H+CH2(+M)=CH3(+M)")
        self.assertEqual(falloff_type, "lindemann")
        self.assertIsNone(falloff_params)
        # For CABR: HIGH keyword gives LPL, main line gives HPL
        self.assertAlmostEqual(lpl["A"], 1.0e10)
        self.assertAlmostEqual(hpl["A"], 1.0e14)


class TestParseSpecies(unittest.TestCase):
    """Test NASA polynomial thermodynamic data parsing."""

    def test_basic_species(self):
        """Test parsing a basic species with NASA polynomials."""
        thermo_string = """
        AR                120186AR  1               G  0300.00   5000.00  1000.00      1
         2.50000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00    2
        -7.45375000E+02 4.37967000E+00 2.50000000E+00 0.00000000E+00 0.00000000E+00    3
         0.00000000E+00 0.00000000E+00-7.45375000E+02 4.37967000E+00                   4
        """

        name, composition, phase, Tmin, Tmax, Tmid, high_coeffs, low_coeffs = parse_species(thermo_string)

        self.assertEqual(name, "AR")
        self.assertEqual(composition["Ar"], 1)
        self.assertEqual(phase, "G")
        self.assertEqual(Tmin, 300.0)
        self.assertEqual(Tmax, 5000.0)
        self.assertEqual(Tmid, 1000.0)
        self.assertEqual(len(high_coeffs), 7)
        self.assertEqual(len(low_coeffs), 7)

    def test_species_with_extended_composition(self):
        """Test parsing species with extended composition using &."""
        thermo_string = """
        C10H7CH3                120186C  10H   8    G   300.000  5000.000 1000.000    &
        C 10 H 8
         2.50000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00    2
        -7.45375000E+02 4.37967000E+00 2.50000000E+00 0.00000000E+00 0.00000000E+00    3
         0.00000000E+00 0.00000000E+00-7.45375000E+02 4.37967000E+00                   4
        """

        name, composition, phase, Tmin, Tmax, Tmid, high_coeffs, low_coeffs = parse_species(thermo_string)

        self.assertEqual(name, "C10H7CH3")
        self.assertEqual(composition["C"], 10)
        self.assertEqual(composition["H"], 8)

    def test_missing_tmid_error(self):
        """Test that missing Tmid raises informative error."""
        thermo_string = """
        BAD_SPECIES             120186C   1H   4    G   300.000  5000.000              1
         2.50000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00    2
        -7.45375000E+02 4.37967000E+00 2.50000000E+00 0.00000000E+00 0.00000000E+00    3
         0.00000000E+00 0.00000000E+00-7.45375000E+02 4.37967000E+00                   4
        """

        with self.assertRaises(ValueError) as context:
            parse_species(thermo_string)

        self.assertIn("Missing intermediate temperature", str(context.exception))
        self.assertIn("Tmid", str(context.exception))


if __name__ == "__main__":
    unittest.main()
