"""
Copyright (c) 2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import equinox as eqx
import jax.numpy as jnp
from jax import lax, vmap
from jaxtyping import Array, Float64

from KiRATE.species.atomic_weights_db import get_molecular_weight
from KiRATE.utilities import parse_species
from KiRATE.utilities.physical_constants import constants


class Species(eqx.Module):
    """
    Chemical species with NASA 7-coefficient polynomial thermodynamic properties.

    This class represents a chemical species and provides methods to calculate
    its thermodynamic properties (heat capacity, enthalpy, entropy, Gibbs energy)
    using the NASA 7-coefficient polynomial formulation. The NASA polynomials
    provide temperature-dependent thermodynamic properties using two sets of
    coefficients covering different temperature ranges:

    - Low temperature range: [Tmin, Tmid]
    - High temperature range: [Tmid, Tmax]

    The thermodynamic properties are computed using the following formulas:

    .. math::
        \\frac{C_p}{R} &= a_1 + a_2 T + a_3 T^2 + a_4 T^3 + a_5 T^4 \\\\
        \\frac{H}{RT} &= a_1 + \\frac{a_2}{2} T + \\frac{a_3}{3} T^2 + \\frac{a_4}{4} T^3
                       + \\frac{a_5}{5} T^4 + \\frac{a_6}{T} \\\\
        \\frac{S}{R} &= a_1 \\ln T + a_2 T + \\frac{a_3}{2} T^2 + \\frac{a_4}{3} T^3
                      + \\frac{a_5}{4} T^4 + a_7

    Parameters
    ----------
    name : str
        Species identifier (e.g., "H2O", "CH4", "O2")
    elemental_composition : dict[str, int]
        Element symbols mapped to atom counts (e.g., {"H": 2, "O": 1} for H2O)
    phase : str
        Phase indicator: "G" (gas), "L" (liquid), or "S" (solid)
    Tmin : float
        Minimum valid temperature [K] for polynomial evaluation
    Tmax : float
        Maximum valid temperature [K] for polynomial evaluation
    Tmid : float
        Temperature [K] separating low-T and high-T polynomial ranges
    low_coeffs : list[float]
        7 NASA polynomial coefficients for T < Tmid: [a1, a2, a3, a4, a5, a6, a7]
    high_coeffs : list[float]
        7 NASA polynomial coefficients for T >= Tmid: [a1, a2, a3, a4, a5, a6, a7]

    Attributes
    ----------
    _name : str
        Chemical species identifier (static field)
    _molecular_weight : Float64[Array, ""]
        Molecular weight [kg/mol], computed from elemental composition
    _elemental_composition : dict[str, int]
        Elemental composition mapping elements to atom counts (static field)
    _phase : str
        Phase indicator: "G" (gas), "L" (liquid), or "S" (solid) (static field)
    _Tmin : Float64[Array, ""]
        Minimum valid temperature [K]
    _Tmax : Float64[Array, ""]
        Maximum valid temperature [K]
    _Tmid : Float64[Array, ""]
        Temperature [K] separating low/high polynomial ranges
    _low_coeffs : Float64[Array, "7"]
        NASA polynomial coefficients for T < Tmid
    _high_coeffs : Float64[Array, "7"]
        NASA polynomial coefficients for T >= Tmid

    Notes
    -----
    **NASA Polynomial Formulation:**

    The NASA 7-coefficient polynomial is a widely-used empirical representation
    of thermodynamic properties. Each temperature range uses 7 coefficients:

    - Coefficients a1-a5: Define Cp/R polynomial
    - Coefficient a6: Integration constant for H/(RT)
    - Coefficient a7: Integration constant for S/R

    **Temperature Range Selection:**

    The class automatically selects the appropriate coefficient set based on
    temperature using JAX's `lax.cond` for efficient conditional branching
    with automatic differentiation support.

    **Vectorization:**

    All thermodynamic property methods support vectorized evaluation over
    temperature arrays using JAX's `vmap`, enabling efficient computation
    across multiple temperature points.

    References
    ----------
    .. [1] McBride, B. J., Gordon, S., and Reno, M. A. "Coefficients for
           Calculating Thermodynamic and Transport Properties of Individual
           Species." NASA Technical Memorandum 4513 (1993).
    .. [2] Kee, R. J., Rupley, F. M., and Miller, J. A. "CHEMKIN-II: A Fortran
           Chemical Kinetics Package for the Analysis of Gas-Phase Chemical
           Kinetics." Sandia Report SAND89-8009 (1989).
    .. [3] Gordon, S., and McBride, B. J. "Computer Program for Calculation of
           Complex Chemical Equilibrium Compositions and Applications."
           NASA Reference Publication 1311 (1994).
    """

    _molecular_weight: Float64[Array, ""]
    _Tmin: Float64[Array, ""]
    _Tmax: Float64[Array, ""]
    _Tmid: Float64[Array, ""]
    _low_coeffs: Float64[Array, "7"]
    _high_coeffs: Float64[Array, "7"]
    _elemental_composition: dict[str, int] = eqx.field(static=True, default_factory=dict)
    _phase: str = eqx.field(static=True, default="G")
    _name: str = eqx.field(static=True, default="")

    def __init__(
        self,
        name: str,
        elemental_composition: dict[str, int],
        phase: str,
        Tmin: float,
        Tmax: float,
        Tmid: float,
        low_coeffs: list[float],
        high_coeffs: list[float],
    ) -> None:
        """
        Initialize a Species with NASA polynomial coefficients.

        Parameters
        ----------
        name : str
            Species identifier (e.g., "H2O", "CH4", "O2")
        elemental_composition : dict[str, int]
            Element symbols mapped to atom counts (e.g., {"H": 2, "O": 1} for H2O)
        phase : str
            Phase indicator: "G" (gas), "L" (liquid), or "S" (solid)
        Tmin : float
            Minimum valid temperature [K] for polynomial evaluation
        Tmax : float
            Maximum valid temperature [K] for polynomial evaluation
        Tmid : float
            Temperature [K] separating low-T and high-T polynomial ranges
        low_coeffs : list[float]
            7 NASA polynomial coefficients for T < Tmid: [a1, a2, a3, a4, a5, a6, a7]
        high_coeffs : list[float]
            7 NASA polynomial coefficients for T >= Tmid: [a1, a2, a3, a4, a5, a6, a7]

        Raises
        ------
        ValueError
            If elemental composition contains unknown elements or invalid atom counts

        Notes
        -----
        The molecular weight is automatically computed from the elemental composition
        using the atomic weights database.

        **NASA Polynomial Coefficients:**

        Each set of 7 coefficients defines the thermodynamic properties via:

        .. math::
            \\frac{C_p}{R} &= a_1 + a_2 T + a_3 T^2 + a_4 T^3 + a_5 T^4 \\\\
            \\frac{H}{RT} &= a_1 + \\frac{a_2}{2} T + \\frac{a_3}{3} T^2 + \\frac{a_4}{4} T^3
                           + \\frac{a_5}{5} T^4 + \\frac{a_6}{T} \\\\
            \\frac{S}{R} &= a_1 \\ln T + a_2 T + \\frac{a_3}{2} T^2 + \\frac{a_4}{3} T^3
                          + \\frac{a_5}{4} T^4 + a_7

        where:
            - a1-a5: Polynomial coefficients for Cp/R
            - a6: Integration constant for H/(RT)
            - a7: Integration constant for S/R
        """
        self._name = name
        self._phase = phase
        self._elemental_composition = elemental_composition
        self._Tmin = jnp.float64(Tmin)
        self._Tmax = jnp.float64(Tmax)
        self._Tmid = jnp.float64(Tmid)
        self._low_coeffs = jnp.array(low_coeffs, dtype=jnp.float64)
        self._high_coeffs = jnp.array(high_coeffs, dtype=jnp.float64)
        self._molecular_weight = get_molecular_weight(elemental_composition)

    @classmethod
    def from_chemkin(
        cls,
        thermo_data: str,
    ) -> "Species":
        """
        Create a Species instance from CHEMKIN NASA polynomial thermodynamic data.

        This class method provides a convenient way to construct Species objects
        directly from CHEMKIN-style thermodynamic data strings, which are the
        standard format used in combustion and chemical kinetics databases.

        Parameters
        ----------
        thermo_data : str
            4-line CHEMKIN NASA thermodynamic data format. The format is:

            .. code-block:: text

                SPECIES_NAME      DATE  ELEMENTS      PHASE  TMIN   TMAX   TMID      1
                 a1_high a2_high a3_high a4_high a5_high                              2
                 a6_high a7_high a1_low  a2_low  a3_low  a4_low  a5_low               3
                 a6_low  a7_low                                                       4

            Each line is fixed-width or space-separated with specific formatting.
            The coefficients define NASA 7-coefficient polynomials for two
            temperature ranges (high-T first, then low-T).

        Returns
        -------
        Species
            New Species instance with parsed name, composition, phase, temperature
            ranges, and NASA polynomial coefficients.

        Raises
        ------
        ValueError
            If the input string cannot be parsed or contains invalid data format.

        Notes
        -----
        **CHEMKIN Format Details:**

        - Line 1: Species name, date code, elemental composition, phase, Tmin, Tmax, Tmid
        - Line 2: First 5 high-temperature coefficients (a1-a5)
        - Line 3: Last 2 high-T coefficients (a6-a7) + first 3 low-T coefficients (a1-a3)
        - Line 4: Last 4 low-temperature coefficients (a4-a7)

        The parser handles various formatting variations including:
        - Leading/trailing whitespace
        - Numbers without spaces between them (e.g., "1.23E+02-4.56E-03")
        - Fortran 'D' notation for exponents (e.g., "1.23D+02")
        """
        # Parse thermodynamic data
        name, composition, phase, Tmin, Tmax, Tmid, high_coeffs, low_coeffs = parse_species(thermo_data)

        return cls(
            name=name,
            elemental_composition=composition,
            phase=phase,
            Tmin=Tmin,
            Tmax=Tmax,
            Tmid=Tmid,
            low_coeffs=low_coeffs,
            high_coeffs=high_coeffs,
        )

    # =======================================================================
    # Helper methods for NASA polynomial evaluation
    @staticmethod
    def _eval_polynomial(coeffs: Float64[Array, "7"], T: Float64[Array, ""]) -> Float64[Array, ""]:
        """
        Evaluate NASA Cp/R polynomial using Horner's method.

        Computes: a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4

        Parameters
        ----------
        coeffs : Float64[Array, "7"]
            NASA 7-coefficient array [a1, a2, a3, a4, a5, a6, a7]
        T : Float64[Array, ""]
            Temperature [K] (scalar)

        Returns
        -------
        Float64[Array, ""]
            Dimensionless heat capacity Cp/R at temperature T

        Notes
        -----
        Uses Horner's method for efficient polynomial evaluation with
        fewer operations and better numerical stability.
        """
        return coeffs[0] + T * (coeffs[1] + T * (coeffs[2] + T * (coeffs[3] + T * coeffs[4])))

    @staticmethod
    def _eval_h_polynomial(coeffs: Float64[Array, "7"], T: Float64[Array, ""]) -> Float64[Array, ""]:
        """
        Evaluate NASA H/(RT) enthalpy polynomial.

        Computes: a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T

        Parameters
        ----------
        coeffs : Float64[Array, "7"]
            NASA 7-coefficient array [a1, a2, a3, a4, a5, a6, a7]
        T : Float64[Array, ""]
            Temperature [K] (scalar)

        Returns
        -------
        Float64[Array, ""]
            Dimensionless enthalpy H/(RT) at temperature T

        Notes
        -----
        This polynomial is the integral of the Cp/R polynomial divided by T,
        plus the integration constant a6/T.
        """
        return (
            coeffs[0]
            + coeffs[1] * T / 2.0
            + coeffs[2] * jnp.pow(T, 2) / 3.0
            + coeffs[3] * jnp.pow(T, 3) / 4.0
            + coeffs[4] * jnp.pow(T, 4) / 5.0
            + coeffs[5] / T
        )

    @staticmethod
    def _eval_s_polynomial(coeffs: Float64[Array, "7"], T: Float64[Array, ""]) -> Float64[Array, ""]:
        """
        Evaluate NASA S/R entropy polynomial.

        Computes: a1*ln(T) + a2*T + a3*T^2/2 + a4*T^3/3 + a5*T^4/4 + a7

        Parameters
        ----------
        coeffs : Float64[Array, "7"]
            NASA 7-coefficient array [a1, a2, a3, a4, a5, a6, a7]
        T : Float64[Array, ""]
            Temperature [K] (scalar)

        Returns
        -------
        Float64[Array, ""]
            Dimensionless entropy S/R at temperature T

        Notes
        -----
        This polynomial is the integral of the Cp/R polynomial divided by T,
        plus the integration constant a7.
        """
        return (
            coeffs[0] * jnp.log(T)
            + coeffs[1] * T
            + coeffs[2] * jnp.pow(T, 2) / 2.0
            + coeffs[3] * jnp.pow(T, 3) / 3.0
            + coeffs[4] * jnp.pow(T, 4) / 4.0
            + coeffs[6]
        )

    def _compute_thermo_property(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
        eval_func: callable,
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Generic method to compute thermodynamic properties with temperature-dependent coefficients.

        This internal method implements the common pattern used by all dimensionless
        thermodynamic property calculations (Cp/R, H/(RT), S/R). It handles:
        1. Converting input to JAX array
        2. Selecting appropriate coefficients based on temperature via lax.cond
        3. Vectorizing computation over temperature array using vmap
        4. Squeezing result for scalar inputs

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) [K]. Can be scalar, 0-D array, or 1-D array.
        eval_func : callable
            Polynomial evaluation function with signature: eval_func(coeffs, T) -> result
            Should accept coefficient array and single temperature scalar.
            Examples: _eval_polynomial, _eval_h_polynomial, _eval_s_polynomial

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Computed property value(s). Shape matches input temperature shape.

        Notes
        -----
        This method uses JAX's `lax.cond` for lazy evaluation of the conditional
        branch, which is more efficient for JIT compilation and autodiff than
        using `jnp.where`, which evaluates both branches.
        """
        T_array = jnp.atleast_1d(jnp.asarray(T, dtype=jnp.float64))

        def compute_single(t):
            """Compute property for a single temperature using conditional branching."""
            return lax.cond(
                t < self._Tmid,
                lambda _: eval_func(self._low_coeffs, t),
                lambda _: eval_func(self._high_coeffs, t),
                None,
            )

        result = vmap(compute_single)(T_array)
        return result.squeeze() if T_array.shape == (1,) else result

    # =======================================================================
    # Thermodynamic property methods
    @eqx.filter_jit
    def cp_R(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Dimensionless heat capacity at constant pressure: Cp/R.

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) [K]

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Dimensionless heat capacity Cp/R
        """
        return self._compute_thermo_property(T, self._eval_polynomial)

    @eqx.filter_jit
    def cp(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Heat capacity at constant pressure [J/(mol·K)].

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) [K]

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Heat capacity Cp [J/(mol·K)]
        """
        return self.cp_R(T) * constants.R_J_mol_K

    @eqx.filter_jit
    def h_RT(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Dimensionless enthalpy: H/(RT).

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) [K]

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Dimensionless enthalpy H/(RT)
        """
        return self._compute_thermo_property(T, self._eval_h_polynomial)

    @eqx.filter_jit
    def h(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Enthalpy [J/mol].

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) [K]

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Enthalpy H [J/mol]
        """
        T_array = jnp.asarray(T, dtype=jnp.float64)
        return self.h_RT(T) * constants.R_J_mol_K * T_array

    @eqx.filter_jit
    def s_R(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Dimensionless entropy: S/R.

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) [K]

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Dimensionless entropy S/R
        """
        return self._compute_thermo_property(T, self._eval_s_polynomial)

    @eqx.filter_jit
    def s(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Entropy [J/(mol·K)].

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) [K]

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Entropy S [J/(mol·K)]
        """
        return self.s_R(T) * constants.R_J_mol_K

    @eqx.filter_jit
    def g_RT(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Dimensionless Gibbs free energy: G/(RT).

        Computed as: G/(RT) = H/(RT) - S/R

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) [K]

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Dimensionless Gibbs energy G/(RT)
        """
        return self.h_RT(T) - self.s_R(T)

    @eqx.filter_jit
    def g(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Gibbs free energy [J/mol].

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) [K]

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Gibbs energy G [J/mol]
        """
        T_array = jnp.asarray(T, dtype=jnp.float64)
        return self.g_RT(T) * constants.R_J_mol_K * T_array

    @eqx.filter_jit
    def u_RT(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Dimensionless internal energy: U/(RT).

        Computed as: U/(RT) = H/(RT) - 1

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) [K]

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Dimensionless internal energy U/(RT)
        """
        return self.h_RT(T) - 1.0

    @eqx.filter_jit
    def u(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Internal energy [J/mol].

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) [K]

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Internal energy U [J/mol]
        """
        T_array = jnp.asarray(T, dtype=jnp.float64)
        return self.u_RT(T) * constants.R_J_mol_K * T_array

    @eqx.filter_jit
    def cv_R(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Dimensionless heat capacity at constant volume: Cv/R.

        Computed as: Cv/R = Cp/R - 1

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) [K]

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Dimensionless heat capacity Cv/R
        """
        return self.cp_R(T) - 1.0

    @eqx.filter_jit
    def cv(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Heat capacity at constant volume [J/(mol·K)].

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) [K]

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Heat capacity Cv [J/(mol·K)]
        """
        return self.cv_R(T) * constants.R_J_mol_K

    # =======================================================================
    # String representations for display and debugging
    def __repr__(self) -> str:
        """
        Return detailed string representation for debugging.

        Provides a multi-line, human-readable representation showing species
        name, elemental composition, molecular weight, phase, and temperature
        ranges for the NASA polynomials.

        Returns
        -------
        str
            Multi-line formatted string with all species information
        """
        return (
            f"Species(\n"
            f"  name='{self._name}',\n"
            f"  formula={self._elemental_composition},\n"
            f"  MW={float(self._molecular_weight):.6f} kg/mol,\n"
            f"  phase='{self._phase}',\n"
            f"  T_range=[{float(self._Tmin):.1f}, {float(self._Tmax):.1f}] K,\n"
            f"  T_mid={float(self._Tmid):.1f} K\n"
            f")"
        )

    def __str__(self) -> str:
        """
        Return concise string representation.

        Provides a short, human-readable representation suitable for display
        in lists or logs.

        Returns
        -------
        str
            Species name followed by phase in parentheses (e.g., "H2O (G)")
        """
        return f"{self._name} ({self._phase})"

    # =======================================================================
    # Properties for species information access
    @property
    def name(self) -> str:
        """
        Species name identifier.

        Returns
        -------
        str
            The chemical species name (e.g., "H2O", "CH4", "O2")
        """
        return self._name

    @property
    def phase(self) -> str:
        """
        Phase indicator for the species.

        Returns
        -------
        str
            One of: "G" (gas), "L" (liquid), or "S" (solid)
        """
        return self._phase

    @property
    def molecular_weight(self) -> Float64[Array, ""]:
        """
        Molecular weight of the species.

        Returns
        -------
        Float64[Array, ""]
            Molecular weight in kg/mol, computed from elemental composition
        """
        return self._molecular_weight

    @property
    def elemental_composition(self) -> dict[str, int]:
        """
        Elemental composition of the species.

        Returns
        -------
        dict[str, int]
            Dictionary mapping element symbols to atom counts.
            For example, H2O returns {"H": 2, "O": 1}
        """
        return self._elemental_composition

    @property
    def Tmin(self) -> Float64[Array, ""]:
        """
        Minimum valid temperature for NASA polynomial evaluation.

        Returns
        -------
        Float64[Array, ""]
            Minimum temperature in Kelvin. Thermodynamic properties should
            not be evaluated below this temperature.
        """
        return self._Tmin

    @property
    def Tmax(self) -> Float64[Array, ""]:
        """
        Maximum valid temperature for NASA polynomial evaluation.

        Returns
        -------
        Float64[Array, ""]
            Maximum temperature in Kelvin. Thermodynamic properties should
            not be evaluated above this temperature.
        """
        return self._Tmax

    @property
    def Tmid(self) -> Float64[Array, ""]:
        """
        Midpoint temperature separating low and high polynomial ranges.

        Returns
        -------
        Float64[Array, ""]
            Midpoint temperature in Kelvin. The low-temperature coefficients
            are used for T < Tmid, and high-temperature coefficients for T >= Tmid.
        """
        return self._Tmid

    @property
    def low_coeffs(self) -> Float64[Array, "7"]:
        """
        NASA 7-coefficient polynomial for low temperature range.

        Returns
        -------
        Float64[Array, "7"]
            Coefficients [a1, a2, a3, a4, a5, a6, a7] for T < Tmid
        """
        return self._low_coeffs

    @property
    def high_coeffs(self) -> Float64[Array, "7"]:
        """
        NASA 7-coefficient polynomial for high temperature range.

        Returns
        -------
        Float64[Array, "7"]
            Coefficients [a1, a2, a3, a4, a5, a6, a7] for T >= Tmid
        """
        return self._high_coeffs
