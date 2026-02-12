"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

from collections.abc import Callable

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

    def reformulate_thermo(
        self,
        Tmax: float | Float64[Array, ""] | None = None,
        Tmid: float | Float64[Array, ""] | None = None,
    ) -> "Species":
        """
        Reformulate thermodynamic coefficients for smooth transitions at intermediate temperature.

        Creates new NASA7 coefficients using least-squares fitting to ensure smooth
        thermodynamic properties (Cp, H, S) across the intermediate temperature boundary.
        This eliminates discontinuities that can occur with standard NASA7 coefficients.

        Implementation follows OpenSMOKE++ ReformulationOfThermodynamics algorithm.

        Algorithm Overview
        -------------------
        1. **Fit smooth polynomial**: Fit Cp/R data using spline basis [1, T, T², T³, T⁴, (T-Tknot)⁴]
           - The (T-Tknot)⁴ spline ensures all derivatives (up to 3rd) are continuous at Tknot
           - Least-squares fitting solves: (X^T X) coeffs = X^T y (normal equations, not QR-based)

        2. **Extract NASA7 coefficients**:
           - Low-T coefficients: Use fitted [a0, a1, a2, a3, a4] directly
           - High-T coefficients: Apply spline transformation to get [AHT, BHT, CHT, DHT, EHT]
           - Integration constants: Compute a6 and a7 for both ranges using original species' H/(RT) and S/R

        3. **Temperature grid**: Split at ORIGINAL species' Tmid (not at candidate Tknot)
           - First n points: Tmin to original_Tmid (low-T region)
           - Next n points: original_Tmid to Tmax (high-T region)
           - This ensures consistent evaluation of Cp/R across candidate Tknot values

        Key Mathematical Details
        -------------------------
        For a candidate intermediate temperature :math:`T_{\\text{knot}}`, the fitted polynomial is:

        .. math::

            \\frac{C_p}{R} = a_0 + a_1 T + a_2 T^2 + a_3 T^3 + a_4 T^4 + \\alpha(T-T_{\\text{knot}})^4

        The high-T coefficients are computed by expanding :math:`(T-T_{\\text{knot}})^4` in the :math:`T` basis:

        .. math::

            (T-T_{\\text{knot}})^4 = T^4 - 4T_{\\text{knot}}T^3 + 6T_{\\text{knot}}^2T^2 - 4T_{\\text{knot}}^3T + T_{\\text{knot}}^4

        This gives:

        .. math::

            a_1^{\\text{high}} &= a_0 + \\alpha T_{\\text{knot}}^4 \\\\
            a_2^{\\text{high}} &= a_1 - 4\\alpha T_{\\text{knot}}^3 \\\\
            a_3^{\\text{high}} &= a_2 + 6\\alpha T_{\\text{knot}}^2 \\\\
            a_4^{\\text{high}} &= a_3 - 4\\alpha T_{\\text{knot}} \\\\
            a_5^{\\text{high}} &= a_4 + \\alpha

        Integration constants :math:`a_6` and :math:`a_7` are set to match the original species'
        :math:`H/(RT)` and :math:`S/R` at :math:`T_{\\text{knot}}`:

        .. math::

            a_6^{\\text{low}} &= (h_{RT}^{\\text{orig}} - h_{RT}^{\\text{low}}) T_{\\text{knot}} \\\\
            a_6^{\\text{high}} &= (h_{RT}^{\\text{orig}} - h_{RT}^{\\text{high}}) T_{\\text{knot}} \\\\
            a_7^{\\text{low}} &= s_R^{\\text{orig}} - s_R^{\\text{low}} \\\\
            a_7^{\\text{high}} &= s_R^{\\text{orig}} - s_R^{\\text{high}}

        Tmid Search Strategy
        --------------------
        If Tmid is not specified, automatically searches for optimal value:

        - **Search range**: :math:`[T_{\\text{min}} + 0.2(T_{\\text{max}} - T_{\\text{min}}), T_{\\text{max}} - 0.2(T_{\\text{max}} - T_{\\text{min}})]`
          (excludes 20% at each boundary to avoid edge effects)
        - **Stepping**: Adaptive - divides search range into ~15 candidates
          (scales naturally with species temperature range)
        - **Error metric**: Cumulative relative error :math:`= \\sum_i \\frac{|C_p^{\\text{fitted}} - C_p^{\\text{orig}}|}{C_p^{\\text{orig}}}`
        - **Best Tmid**: Minimizes the error metric across the temperature range

        Parameters
        ----------
        Tmax : float, optional
            Maximum temperature for fitting [K]. If None, uses self._Tmax.
        Tmid : float, optional
            Fixed intermediate temperature [K]. If None, automatically searches for optimal Tmid
            that minimizes the fitting error.

        Returns
        -------
        Species
            New Species instance with reformulated NASA7 coefficients that have:
            - Continuous Cp/R, H/(RT), S/R across Tmid
            - Continuous derivatives (dCp/dT, dH/dT, dS/dT) through spline basis
            - Original H/(RT) and S/R values at Tmid

        Notes
        -----
        The reformulated species will NOT match the original Cp/R away from Tmid.
        This is intentional - the reformulation trades off point-wise accuracy for
        smooth transitions, which is beneficial for numerical integration and stability.

        The implementation uses normal equations (X^T X coeffs = X^T y) rather than
        QR-based least-squares, matching OpenSMOKE++ for numerical stability with
        polynomial fits at high powers.
        """
        Tmax = jnp.float64(Tmax) if Tmax is not None else self._Tmax
        Tmin = self._Tmin
        Tmid_original = self._Tmid
        temperatures = self._build_temperature_grid(Tmin, Tmax, Tmid_original, 30)

        if Tmid is not None:
            # Fixed intermediate temperature case
            Tknot = jnp.float64(Tmid)
            coeffs_ls = self._fit_spline_coefficients(temperatures, Tknot)
            new_low_coeffs, new_high_coeffs = self._compute_reformulated_coeffs(coeffs_ls, Tknot)

            return Species(
                name=self._name,
                elemental_composition=self._elemental_composition,
                phase=self._phase,
                Tmin=Tmin,
                Tmax=Tmax,
                Tmid=Tknot,
                low_coeffs=new_low_coeffs,
                high_coeffs=new_high_coeffs,
            )
        else:
            # Auto-search for optimal intermediate temperature
            # Search range: centered around the actual species temperature bounds
            # Exclude 20% at each boundary to avoid edge effects
            range_width = float(Tmax - Tmin)
            T_min_search = float(Tmin) + 0.2 * range_width
            T_max_search = float(Tmax) - 0.2 * range_width

            best_Tmid = None
            best_error = jnp.inf

            # Adaptive stepping: divide search range into consistent number of candidates
            # This scales naturally with the species' temperature range
            n_candidates = 15
            step = (T_max_search - T_min_search) / n_candidates

            Tknot_candidate = T_min_search
            while Tknot_candidate <= T_max_search:
                coeffs_ls = self._fit_spline_coefficients(temperatures, Tknot_candidate)
                error = self._compute_fitting_error(temperatures, Tknot_candidate, coeffs_ls)

                if error < best_error:
                    best_error = error
                    best_Tmid = Tknot_candidate

                Tknot_candidate += step

            # Final fit with optimal Tmid
            coeffs_ls = self._fit_spline_coefficients(temperatures, best_Tmid)
            new_low_coeffs, new_high_coeffs = self._compute_reformulated_coeffs(coeffs_ls, best_Tmid)

            return Species(
                name=self._name,
                elemental_composition=self._elemental_composition,
                phase=self._phase,
                Tmin=Tmin,
                Tmax=Tmax,
                Tmid=best_Tmid,
                low_coeffs=new_low_coeffs,
                high_coeffs=new_high_coeffs,
            )

    def _fit_spline_coefficients(
        self,
        temperatures: Float64[Array, "n"],
        Tknot: Float64[Array, "n"],
    ) -> Float64[Array, "6"]:
        """
        Fit spline coefficients using least-squares normal equations.

        Internal helper method that performs the core fitting operation:
        builds design matrix, solves normal equations to get fitted coefficients.

        Parameters
        ----------
        temperatures : Float64[Array, "n"]
            Temperature points for fitting [K]
        Tknot : Float64[Array, ""]
            Intermediate temperature for spline correction [K]

        Returns
        -------
        Float64[Array, "6"]
            Fitted coefficients [a0, a1, a2, a3, a4, alpha]
        """
        y = self.cp_R(temperatures)
        X = self._build_design_matrix(temperatures, Tknot)

        # Assemble and solve the normal equations like OpenSMOKE: solve (X^T X) * params = X^T * y
        XTX = X.T @ X
        XTy = X.T @ y
        return jnp.linalg.solve(XTX, XTy)

    def _compute_fitting_error(
        self,
        temperatures: Float64[Array, "n"],
        Tknot: Float64[Array, ""],
        coeffs_ls: Float64[Array, "6"],
    ) -> float:
        """
        Compute cumulative relative error for fitted coefficients.

        Calculates error metric: :math:`\\sum | Cp_{fitted} - Cp_{original}| / Cp_{original}`

        Parameters
        ----------
        temperatures : Float64[Array, "n"]
            Temperature points [K]
        Tknot : Float64[Array, ""]
            Intermediate temperature [K]
        coeffs_ls : Float64[Array, "6"]
            Fitted coefficients [a0, a1, a2, a3, a4, alpha]

        Returns
        -------
        float
            Cumulative relative error
        """
        X = self._build_design_matrix(temperatures, Tknot)
        y = self.cp_R(temperatures)
        cp_fitted = X @ coeffs_ls

        return float(jnp.sum(jnp.abs(cp_fitted - y) / jnp.abs(y)))

    def _build_temperature_grid(
        self,
        Tmin: Float64[Array, ""],
        Tmax: Float64[Array, ""],
        Tmid_original: Float64[Array, ""],
        n: int = 15,
    ) -> Float64[Array, "2n-1"]:
        """
        Build temperature grid split at original Tmid.

        Helper method to construct consistent temperature grid for fitting.
        Temperature points are split at the original species' Tmid to ensure
        consistent evaluation across all candidate Tknot values.

        Parameters
        ----------
        Tmin : Float64[Array, ""]
            Minimum temperature [K]
        Tmax : Float64[Array, ""]
            Maximum temperature [K]
        Tmid_original : Float64[Array, ""]
            Original species' intermediate temperature [K]
        n : int
            Number of points in each temperature range (default: 15)

        Returns
        -------
        Float64[Array, "2n-1"]
            Combined temperature grid
        """
        return jnp.concatenate(
            [
                jnp.linspace(Tmin, Tmid_original, n),
                jnp.linspace(Tmid_original, Tmax, n)[1:],
            ]
        )

    def _compute_reformulated_coeffs(
        self,
        coeffs_ls: Float64[Array, ""],
        Tknot: Float64[Array, ""],
    ) -> tuple[list[float], list[float]]:
        """
        Compute NASA7 coefficients ensuring continuity with original H/(RT) and S/R.

        Following OpenSMOKE++ approach, extracts and transforms the fitted spline
        coefficients into NASA7 polynomial sets, then computes a6 and a7 integration
        constants to match the original species' H/(RT) and S/R at Tknot.

        Coefficient Transformation
        ---------------------------
        Input fitted coefficients represent:

        .. math::

            \\frac{C_p}{R} = a_0 + a_1 T + a_2 T^2 + a_3 T^3 + a_4 T^4 + \\alpha(T-T_{\\text{knot}})^4

        For :math:`T \\leq T_{\\text{knot}}` (low-T region):

        .. math::

            a_1^{\\text{low}} = a_0, \\quad a_2^{\\text{low}} = a_1, \\quad a_3^{\\text{low}} = a_2,
            \\quad a_4^{\\text{low}} = a_3, \\quad a_5^{\\text{low}} = a_4

        (Direct assignment since spline term is zero)

        For :math:`T > T_{\\text{knot}}` (high-T region), apply transformation based on
        :math:`(T-T_{\\text{knot}})^4` expansion:

        .. math::

            a_1^{\\text{high}} &= a_0 + \\alpha T_{\\text{knot}}^4 \\\\
            a_2^{\\text{high}} &= a_1 - 4\\alpha T_{\\text{knot}}^3 \\\\
            a_3^{\\text{high}} &= a_2 + 6\\alpha T_{\\text{knot}}^2 \\\\
            a_4^{\\text{high}} &= a_3 - 4\\alpha T_{\\text{knot}} \\\\
            a_5^{\\text{high}} &= a_4 + \\alpha

        This transformation ensures that both polynomial sets, when evaluated with
        :math:`(T-T_{\\text{knot}})^4 = 0` at :math:`T_{\\text{knot}}`, give the same
        :math:`C_p/R` value (continuity).

        Integration Constants
        ---------------------
        For :math:`H/(RT)`:

        .. math::

            \\frac{H}{RT} = \\int \\frac{C_p/R}{T} \\, dT + \\frac{a_6}{T}

        Compute both low and high range integration constants to match original:

        .. math::

            a_6^{\\text{low}} &= \\left(\\frac{H}{RT}\\bigg|_{\\text{orig}} - \\frac{H}{RT}\\bigg|_{\\text{low}}\\right) T_{\\text{knot}} \\\\
            a_6^{\\text{high}} &= \\left(\\frac{H}{RT}\\bigg|_{\\text{orig}} - \\frac{H}{RT}\\bigg|_{\\text{high}}\\right) T_{\\text{knot}}

        For :math:`S/R`:

        .. math::

            \\frac{S}{R} = \\int \\frac{C_p/R}{T} \\, dT + a_7

        Compute both low and high range constants:

        .. math::

            a_7^{\\text{low}} &= \\frac{S}{R}\\bigg|_{\\text{orig}} - \\frac{S}{R}\\bigg|_{\\text{low}} \\\\
            a_7^{\\text{high}} &= \\frac{S}{R}\\bigg|_{\\text{orig}} - \\frac{S}{R}\\bigg|_{\\text{high}}

        This ensures :math:`H/(RT)` and :math:`S/R` are continuous and match the original at
        :math:`T_{\\text{knot}}`, while :math:`dH/dT = C_p/R/T` and :math:`dS/dT = C_p/R/T`
        are also continuous (through :math:`C_p/R` continuity).

        Parameters
        ----------
        coeffs_ls : Float64[Array, "6"]
            Fitted coefficients [a0, a1, a2, a3, a4, alpha] from least-squares fit
        Tknot : Float64[Array, ""]
            Intermediate temperature [K] where spline basis changes

        Returns
        -------
        tuple[list[float], list[float]]
            (low_coeffs, high_coeffs) - NASA7 coefficient lists for low and high T ranges
            Each list contains [a1, a2, a3, a4, a5, a6, a7]
        """
        a0, a1, a2, a3, a4, alpha = coeffs_ls

        Tknot2 = Tknot**2
        Tknot3 = Tknot**3
        Tknot4 = Tknot**4

        # For T <= Tknot: use base polynomial coefficients (no spline correction)
        a1_low = a0
        a2_low = a1
        a3_low = a2
        a4_low = a3
        a5_low = a4

        # For T > Tknot: apply spline correction to ensure smooth derivatives
        # Derivative of (T-Tknot)^4 is 4*(T-Tknot)^3, 12*(T-Tknot)^2, 12*(T-Tknot), 4
        # So: dCp/R|_high = da1_high + 2*da2_high*T + 3*da3_high*T^2 + 4*da4_high*T^3
        #                  = d(alpha*(T-Tknot)^4)/dT = 4*alpha*(T-Tknot)^3
        # This gives: da1_high = 4*alpha*(T-Tknot)^3 at Tknot, but we need continuous coefficients
        # Using Taylor expansion: (T-Tknot)^4 coefficients in T-basis are:
        a1_high = a0 + alpha * Tknot4
        a2_high = a1 - 4.0 * alpha * Tknot3
        a3_high = a2 + 6.0 * alpha * Tknot2
        a4_high = a3 - 4.0 * alpha * Tknot
        a5_high = a4 + alpha

        # Get original species' H/(RT) and S/R at Tknot
        h_RT_original = self.h_RT(Tknot)
        s_R_original = self.s_R(Tknot)

        # Build coefficient arrays for evaluation using class methods
        coeffs_low = jnp.array([a1_low, a2_low, a3_low, a4_low, a5_low, 0.0, 0.0])
        coeffs_high = jnp.array([a1_high, a2_high, a3_high, a4_high, a5_high, 0.0, 0.0])

        # Compute H/(RT) at Tknot from low-T and high-T coefficients using class method
        h_RT_low = self._eval_h_polynomial(coeffs_low, Tknot)
        h_RT_high = self._eval_h_polynomial(coeffs_high, Tknot)

        # Compute a6 to ensure H/(RT) continuity with original species
        a6_low = (h_RT_original - h_RT_low) * Tknot
        a6_high = (h_RT_original - h_RT_high) * Tknot

        # Compute S/R at Tknot from low-T and high-T coefficients using class method
        s_R_low = self._eval_s_polynomial(coeffs_low, Tknot)
        s_R_high = self._eval_s_polynomial(coeffs_high, Tknot)

        # Compute a7 to ensure S/R continuity with original species
        a7_low = s_R_original - s_R_low
        a7_high = s_R_original - s_R_high

        low_coeffs = [
            float(a1_low),
            float(a2_low),
            float(a3_low),
            float(a4_low),
            float(a5_low),
            float(a6_low),
            float(a7_low),
        ]
        high_coeffs = [
            float(a1_high),
            float(a2_high),
            float(a3_high),
            float(a4_high),
            float(a5_high),
            float(a6_high),
            float(a7_high),
        ]

        return low_coeffs, high_coeffs

    @staticmethod
    def _build_design_matrix(
        temperatures: Float64[Array, "n"],
        Tknot: Float64[Array, ""],
    ) -> Float64[Array, "n 6"]:
        """
        Build design matrix for least-squares fitting of NASA coefficients.

        Constructs the matrix :math:`X` for the least-squares problem where:

        .. math::

            \\frac{C_p}{R} = X \\cdot \\mathbf{coeffs}

        This implements the spline basis from OpenSMOKE++ for smooth thermodynamic fits.

        The design matrix has 6 columns for basis functions:

        .. math::

            X[:, 0:5] &= [1, T, T^2, T^3, T^4] \\quad \\text{(standard polynomial basis)} \\\\
            X[:, 5] &= \\begin{cases}
                0 & \\text{if } T \\leq T_{\\text{knot}} \\\\
                (T - T_{\\text{knot}})^4 & \\text{if } T > T_{\\text{knot}}
            \\end{cases} \\quad \\text{(smooth spline correction)}

        The spline basis function :math:`(T-T_{\\text{knot}})^4` ensures:

        - :math:`C_p/R` is continuous at :math:`T_{\\text{knot}}` (function value matches)
        - :math:`\\frac{dC_p}{dT}` is continuous at :math:`T_{\\text{knot}}` (first derivative matches)
        - :math:`\\frac{d^2C_p}{dT^2}` is continuous at :math:`T_{\\text{knot}}` (second derivative matches)
        - :math:`\\frac{d^3C_p}{dT^3}` is continuous at :math:`T_{\\text{knot}}` (third derivative matches)

        This automatically enforces smooth derivatives without explicit constraints.

        Parameters
        ----------
        temperatures : Float64[Array, "n"]
            Temperature points for fitting [K]
        Tknot : Float64[Array, ""]
            Intermediate temperature for spline correction [K]

        Returns
        -------
        Float64[Array, "n 6"]
            Design matrix with shape (n_temperatures, 6)

        Notes
        -----
        The spline term :math:`(T-T_{\\text{knot}})^4` is zero for :math:`T \\leq T_{\\text{knot}}`, ensuring:

        - For :math:`T \\leq T_{\\text{knot}}`: :math:`C_p/R = a_0 + a_1 T + a_2 T^2 + a_3 T^3 + a_4 T^4`
        - For :math:`T > T_{\\text{knot}}`: :math:`C_p/R = a_0 + a_1 T + \\cdots + \\alpha(T-T_{\\text{knot}})^4`

        When :math:`(T-T_{\\text{knot}})^4` is expanded to polynomial basis for :math:`T > T_{\\text{knot}}`,
        the high-T coefficients are computed as spline-corrected versions of the low-T coefficients.
        """
        n_points = len(temperatures)
        X = jnp.zeros((n_points, 6))

        T_powers = jnp.stack(
            [
                jnp.ones_like(temperatures),
                temperatures,
                temperatures**2,
                temperatures**3,
                temperatures**4,
            ],
            axis=1,
        )

        # Spline correction: zero for T < Tknot, (T-Tknot)^4 for T >= Tknot
        spline_correction = jnp.where(temperatures >= Tknot, (temperatures - Tknot) ** 4, 0.0)

        X = jnp.concatenate([T_powers, spline_correction[:, None]], axis=1)
        return X

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
        eval_func: Callable,
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Generic method to compute thermodynamic properties with temperature-dependent coefficients.

        This internal method implements the common pattern used by all dimensionless
        thermodynamic property calculations (Cp/R, H/(RT), S/R). It handles:
        1. Converting input to JAX array
        2. Checking temperature bounds with eqx.error_if
        3. Selecting appropriate coefficients based on temperature via lax.cond
        4. Vectorizing computation over temperature array using vmap
        5. Squeezing result for scalar inputs

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

        Raises
        ------
        RuntimeError
            If any temperature is outside the valid range [Tmin, Tmax]

        Notes
        -----
        This method uses JAX's `lax.cond` for lazy evaluation of the conditional
        branch, which is more efficient for JIT compilation and autodiff than
        using `jnp.where`, which evaluates both branches.

        Temperature bounds are checked using `eqx.error_if` which provides
        JIT-compatible error handling.
        """
        T_array = jnp.atleast_1d(jnp.asarray(T, dtype=jnp.float64))

        def compute_single(t):
            """Compute property for a single temperature using conditional branching."""
            # Check temperature bounds
            t = eqx.error_if(
                t,
                t < self._Tmin,
                f"Temperature {t} K is below minimum valid temperature {self._Tmin} K for species {self._name}",
            )
            t = eqx.error_if(
                t,
                t > self._Tmax,
                f"Temperature {t} K is above maximum valid temperature {self._Tmax} K for species {self._name}",
            )

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
        Heat capacity at constant pressure [cal/(mol·K)].

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) [K]

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Heat capacity Cp [cal/(mol·K)]
        """
        return self.cp_R(T) * constants.R_cal_mol

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
        Enthalpy [cal/mol].

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) [K]

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Enthalpy H [cal/mol]
        """
        T_array = jnp.asarray(T, dtype=jnp.float64)
        return self.h_RT(T) * constants.R_cal_mol * T_array

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
        Entropy [cal/(mol·K)].

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) [K]

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Entropy S [cal/(mol·K)]
        """
        return self.s_R(T) * constants.R_cal_mol

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
        Gibbs free energy [cal/mol].

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) [K]

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Gibbs energy G [cal/mol]
        """
        T_array = jnp.asarray(T, dtype=jnp.float64)
        return self.g_RT(T) * constants.R_cal_mol * T_array

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
        Internal energy [cal/mol].

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) [K]

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Internal energy U [cal/mol]
        """
        T_array = jnp.asarray(T, dtype=jnp.float64)
        return self.u_RT(T) * constants.R_cal_mol * T_array

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
        Heat capacity at constant volume [cal/(mol·K)].

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) [K]

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Heat capacity Cv [cal/(mol·K)]
        """
        return self.cv_R(T) * constants.R_cal_mol

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
        Return CHEMKIN NASA7 thermodynamic data string representation.

        Generates a 4-line CHEMKIN-format string representing the species
        thermodynamic properties using NASA 7-coefficient polynomials.

        Returns
        -------
        str
            4-line CHEMKIN NASA7 format with species name, comment, elemental
            composition, phase, temperature ranges, and polynomial coefficients.

        Notes
        -----
        The format follows CHEMKIN standard with fixed-width fields:

        .. code-block:: text

            SPECIES_NAME      DATE  ELEMENTS      PHASE  TMIN   TMAX   TMID      1
            a1_high a2_high a3_high a4_high a5_high                              2
            a6_high a7_high a1_low  a2_low  a3_low  a4_low  a5_low               3
            a6_low  a7_low                                                       4
        """
        # Format elemental composition string (e.g., "O   1C   1")
        # CHEMKIN format: Element symbol left-aligned in 2 chars, count right-aligned in 3 chars
        comp_str = ""
        for elem in sorted(self._elemental_composition.keys()):
            count = self._elemental_composition[elem]
            comp_str += f"{elem:<2}{count:>3}"

        # Build line 1 with fixed-width columns matching CHEMKIN parser expectations
        # Columns 1-18: Species name (left-aligned, padded with spaces)
        # Columns 19-24: Date field (blank)
        # Columns 25-44: Elemental composition (left-aligned, padded)
        # Column 45: Phase
        # Columns 46-55: Tmin (10 chars, right-aligned)
        # Columns 56-65: Tmax (10 chars, right-aligned)
        # Columns 66-75: Tmid (10 chars, right-aligned)
        # Columns 76-79: Spaces
        # Column 80: Line number
        line1 = (
            f"{self._name:<18}"  # Columns 1-18: Species name
            f"      "  # Columns 19-24: Blank date field
            f"{comp_str:<20}"  # Columns 25-44: Elemental composition
            f"{self._phase:1}"  # Column 45: Phase
            f"{float(self._Tmin):>10.2f}"  # Columns 46-55: Tmin
            f"{float(self._Tmax):>10.2f}"  # Columns 56-65: Tmax
            f"{float(self._Tmid):>10.2f}"  # Columns 66-75: Tmid
            f"    1"  # Columns 76-80: Spaces + line number
        )

        # Line 2: High-temperature coefficients a1-a5
        line2 = ""
        for coeff in self._high_coeffs[:5]:
            line2 += f"{float(coeff):>15.8E}"
        line2 += "    2"

        # Line 3: High-temperature coefficients a6-a7 + low-temperature a1-a3
        line3 = ""
        for coeff in self._high_coeffs[5:7]:
            line3 += f"{float(coeff):>15.8E}"
        for coeff in self._low_coeffs[:3]:
            line3 += f"{float(coeff):>15.8E}"
        line3 += "    3"

        # Line 4: Low-temperature coefficients a4-a7
        line4 = ""
        for coeff in self._low_coeffs[3:7]:
            line4 += f"{float(coeff):>15.8E}"
        line4 += "                   4"

        return f"{line1}\n{line2}\n{line3}\n{line4}"

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
