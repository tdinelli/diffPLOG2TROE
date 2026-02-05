"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import equinox as eqx
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float64

from KiRATE.kinetics.arrhenius import Arrhenius
from KiRATE.kinetics.broadening_functions import compute_broadening_factor
from KiRATE.kinetics.utils import validate_broadening_parameters, validate_efficiencies
from KiRATE.utilities import calculate_effective_concentration, parse_falloff


class FallOff(eqx.Module):
    """
    Fall-off rate constant calculator for pressure-dependent reactions.

    This class implements the fall-off formalism for reactions that transition
    between low-pressure (third-order) and high-pressure (second-order)
    kinetic behavior. The fall-off formulation is given by:

    .. math::
        k(T, P, \\mathbf{x}) = k_\\infty(T) \\cdot \\frac{P_r}{1 + P_r} \\cdot F(T, P_r)

    where:
        - :math:`k(T, P, \\mathbf{x})` is the pressure-dependent rate constant
        - :math:`k_\\infty(T)` is the high-pressure limit (Arrhenius)
        - :math:`k_0(T)` is the low-pressure limit (Arrhenius)
        - :math:`P_r = k_0(T) \\cdot [M] / k_\\infty (T)` is the reduced pressure
        - :math:`[M]` is the effective third-body concentration with efficiencies
        - :math:`F(T, Pr)` is the broadening factor (Lindemann, Troe, SRI, or Tsang)

    Parameters
    ----------
    hpl_parameters : dict[str, float]
        High-pressure limit Arrhenius parameters:
        {"A": pre-exponential factor, "n": temperature exponent, "Ea": activation energy}
    lpl_parameters : dict[str, float]
        Low-pressure limit Arrhenius parameters (same format as hpl_parameters)
    falloff_type : str
        Type of broadening factor: "lindemann", "troe", "sri", or "tsang"
    falloff_parameters : dict[str, float], optional
        Broadening factor parameters (type-dependent).
    efficiencies : dict[str, float], optional
        Third-body collision efficiencies: {"species_name": efficiency}
        Default efficiency is 1.0 for species not listed
    name : str, optional
        Human-readable reaction name, by default ""

    Attributes
    ----------
    _hpl : Arrhenius
        High-pressure limit Arrhenius object
    _lpl : Arrhenius
        Low-pressure limit Arrhenius object
    _falloff_type : str
        Broadening factor type (static field)
    _falloff_parameters : dict[str, Float64[Array, ""]] | None
        Broadening factor parameters as JAX arrays
    _efficiencies : dict[str, Float64[Array, ""]] | None
        Third-body efficiencies as JAX arrays
    _name : str
        Reaction name (static field)

    Notes
    -----
    The fall-off formulation captures the transition from collision-dominated
    (low-pressure) to stabilization-limited (high-pressure) kinetics. At low
    pressures, the rate is proportional to [M] (third-order), while at high
    pressures it becomes independent of [M] (second-order).

    References
    ----------
    .. [1] F. Lindemann. Discussion on “the radiation theory of chemical action”.
           Trans. Faraday Soc., 17:598, 1922.
           URL: https://dx.doi.org/10.1039/TF9221700598, doi:10.1039/TF9221700598.
    .. [2] R. G. Gilbert, K. Luther, and J. Troe. Theory of thermal unimolecular
           reactions in the fall-off range. II. weak collision rate constants.
           Berichte der Bunsengesellschaft für physikalische Chemie, 87(2):169–175,
           1983. URL:
           https://doi.org/10.1002/bbpc.19830870218, doi:10.1002/bbpc.19830870218.
    .. [3] W. Tsang and J. T. Herron. Chemical kinetic data base for propellant
           combustion I. reactions involving NO, NO2, HNO, HNO2, HCN and N2O.
           Journal of Physical and Chemical Reference Data, 20(3):779–798, 1991.
           URL: https://dx.doi.org/10.1063/1.555890, doi:10.1063/1.555890.
    .. [4] P. H. Stewart, C. W. Larson, and D. Golden. Pressure and temperature
           dependence of reactions proceeding via a bound complex. 2. application
           to 2 CH3 -> C2H5 + H. Combustion and Flame, 75(1):25–40, 1989. URL:
           https://doi.org/10.1016/0010-2180(89)90084-9,
           doi:10.1016/0010-2180(89)90084-9.
    .. [5] R. J. Kee, F. M. Rupley, and J. A. Miller. Chemkin-II: a fortran chemical
           kinetics package for the analysis of gas-phase chemical kinetics.
           Technical Report SAND89-8009, Sandia National Laboratories, 1989. URL:
           https://www.osti.gov/biblio/5681118.
    """

    _hpl: Arrhenius
    _lpl: Arrhenius
    _falloff_type: str = eqx.field(static=True)
    _falloff_parameters: dict[str, Float64[Array, ""]] | None = None
    _efficiencies: dict[str, Float64[Array, ""]] | None = None
    _name: str = eqx.field(static=True, default="")

    def __init__(
        self,
        hpl_parameters: dict[str, float],
        lpl_parameters: dict[str, float],
        falloff_type: str,
        falloff_parameters: dict[str, float] | None = None,
        efficiencies: dict[str, float] | None = None,
        name: str = "",
    ) -> None:
        """
        Initialize the Fall-off rate constant calculator.

        Parameters
        ----------
        hpl_parameters : dict[str, float]
            High-pressure limit Arrhenius parameters:
            {"A": pre-exponential, "n": temperature exponent, "Ea": activation energy}
        lpl_parameters : dict[str, float]
            Low-pressure limit Arrhenius parameters (same format)
        falloff_type : str
            Broadening factor type: "lindemann", "troe", "sri", or "tsang"
        falloff_parameters : dict[str, float], optional
            Broadening parameters (type-dependent)
        efficiencies : dict[str, float], optional
            Third-body collision efficiencies, e.g., {"AR": 0.7, "H2O": 6.0}
            Species not listed default to efficiency = 1.0
        name : str, optional
            Human-readable reaction name, by default ""

        Raises
        ------
        ValueError
            If broadening parameters are invalid for the specified falloff_type
        UserWarning
            If efficiency values are outside typical ranges
        """
        self._name = name
        self._falloff_type = falloff_type

        # Create Arrhenius objects for high-pressure and low-pressure limits
        self._hpl = Arrhenius(parameters=hpl_parameters, name=f"{name} (HPL)")
        self._lpl = Arrhenius(parameters=lpl_parameters, name=f"{name} (LPL)")

        # Validate and store broadening factor parameters
        if falloff_parameters is not None:
            validate_broadening_parameters(falloff_type, falloff_parameters)
            # Convert to JAX arrays for differentiability
            self._falloff_parameters = {key: jnp.float64(value) for key, value in falloff_parameters.items()}
        else:
            self._falloff_parameters = None

        # Validate and store third-body efficiencies
        if efficiencies is not None:
            validate_efficiencies(efficiencies)
            # Convert to JAX arrays for differentiability
            self._efficiencies = {key: jnp.float64(value) for key, value in efficiencies.items()}
        else:
            self._efficiencies = None

    @classmethod
    def from_chemkin(cls, input_string: str) -> "FallOff":
        """
        Create a FallOff instance from a CHEMKIN format string.

        This class method provides a convenient way to construct FallOff objects
        directly from CHEMKIN-style input strings, which are the standard format
        used in combustion and chemical kinetics databases.

        Parameters
        ----------
        input_string : str
            CHEMKIN-formatted string containing reaction name, high-pressure limit,
            low-pressure limit, broadening parameters, and optionally third-body
            efficiencies. The format is:

            .. code-block:: text

                REACTION_NAME    A_hpl   n_hpl   Ea_hpl
                  LOW  /         A_lpl   n_lpl   Ea_lpl /
                  TROE / alpha   T3   T1   T2           / ! or SRI / a b c d e /
                  SPECIES / efficiency / ... / ...      / ! optional

        Returns
        -------
        FallOff
            New FallOff instance with parsed parameters, reaction name, and efficiencies.

        Raises
        ------
        ValueError
            If the input string cannot be parsed or contains invalid parameters.
        """
        # Parse CHEMKIN input string to extract all parameters
        reaction_name, hpl_params, lpl_params, falloff_type, falloff_params, efficiencies = parse_falloff(input_string)

        # Construct FallOff object with parsed parameters
        return cls(
            hpl_parameters=hpl_params,
            lpl_parameters=lpl_params,
            falloff_type=falloff_type,
            falloff_parameters=falloff_params,
            efficiencies=efficiencies,
            name=reaction_name,
        )

    # ==================================================================================
    # Rate constant methods
    @eqx.filter_jit
    def rate_constant(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "nt"],
        P: float | Float64[Array, ""] | Float64[Array, "np"],
        composition: dict[str, float] | None = None,
    ) -> Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]:
        """
        Calculate the fall-off rate constant at given temperature(s) and pressure(s).

        This method implements the fall-off formalism with broadening factor
        corrections, automatically handling third-body effects and vectorized
        evaluation.

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "nt"]
            Temperature(s) in Kelvin. Accepts scalars or 1D arrays.
        P : float | Float64[Array, ""] | Float64[Array, "np"]
            Pressure(s) in atmospheres. Accepts scalars or 1D arrays.
        composition : dict[str, float], optional
            Gas composition as mole fractions: {"species": x_i}
            Used to calculate effective third-body concentration with efficiencies.
            If None, assumes uniform composition with default efficiencies.

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]
            Fall-off rate constant(s) with shape matching input broadcasting:

            - Scalar T, Scalar P -> Scalar output
            - Vector T, Scalar P -> Vector output (length nt)
            - Scalar T, Vector P -> Vector output (length np)
            - Vector T, Vector P -> Matrix output (shape: np x nt)

        Notes
        -----
        **Fall-off Formula:**

        .. math::
            k(T, P, \\mathbf{x}) = k_\\infty \\cdot \\frac{P_r}{1 + P_r} \\cdot F(T, P_r)

        where :math:`P_r = k_0 \\cdot [M] / k_\\infty` is the reduced pressure.

        **Third-Body Effects:**

        The effective concentration [M]_eff accounts for species-specific
        collision efficiencies:

        .. math::
            [M]_{eff} = [M] \\cdot \\sum_i \\epsilon_i x_i
        """
        T = jnp.asarray(T, dtype=jnp.float64)
        P = jnp.asarray(P, dtype=jnp.float64)

        # Convert composition to JAX arrays for differentiability
        jax_composition = (
            {key: jnp.float64(value) for key, value in composition.items()} if composition is not None else None
        )

        # Compute high-pressure and low-pressure limit rate constants
        k_hpl = self._hpl.rate_constant(T)  # High-pressure limit [cm3/mol/s]
        k_lpl = self._lpl.rate_constant(T)  # Low-pressure limit [cm6/mol2/s]

        if jnp.isscalar(P) or P.ndim == 0:  # Scalar pressure - evaluate directly
            return self._single_P_rate_constant(T, P, k_lpl, k_hpl, jax_composition)
        else:  # Vector pressure - vectorize over pressure dimension
            vec_func = vmap(lambda p: self._single_P_rate_constant(T, p, k_lpl, k_hpl, jax_composition))
            return vec_func(P)

    @eqx.filter_jit
    def _single_P_rate_constant(
        self,
        T: Float64[Array, ""] | Float64[Array, "nt"],
        P: Float64[Array, ""],
        lpl: Float64[Array, ""] | Float64[Array, "nt"],
        hpl: Float64[Array, ""] | Float64[Array, "nt"],
        composition: dict[str, Float64[Array, ""]] | None = None,
    ) -> Float64[Array, ""] | Float64[Array, "nt"]:
        """
        Calculate fall-off rate constant at a single pressure and one or more temperatures.

        This internal method implements the core fall-off calculation
        with broadening factor corrections. It is called by
        `rate_constant()` and should not be invoked directly by users.

        Parameters
        ----------
        T : Float64[Array, ""] | Float64[Array, "nt"]
            Temperature(s) in Kelvin (scalar or 1D array)
        P : Float64[Array, ""]
            Single pressure value in atmospheres (scalar)
        lpl : Float64[Array, ""] | Float64[Array, "nt"]
            Pre-computed low-pressure limit rate constant(s) k0(T)
            Units: cm6/mol2/s
        hpl : Float64[Array, ""] | Float64[Array, "nt"]
            Pre-computed high-pressure limit rate constant(s) k_inf(T)
            Units: cm3/mol/s for typical bimolecular reactions
        composition : dict[str, Float64[Array, ""]], optional
            Gas composition as mole fractions with JAX arrays

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "nt"]
            Fall-off rate constant(s) at the specified pressure and temperature(s)

        Notes
        -----
        **Algorithm Steps:**

        1. **Effective Concentration Calculation:**
           Computes :math:`[M]_{eff}` accounting for third-body efficiencies:

           .. math::
               [M]_{eff} = \\frac{P}{R \\cdot T} \\cdot \\sum_i \\epsilon_i x_i

        2. **Reduced Pressure Calculation:**
           Determines the dimensionless pressure parameter:

           .. math::
               P_r = \\frac{k_0(T) \\cdot [M]_{eff}}{k_\\infty(T)}

        3. **Broadening Factor Evaluation:**
           Computes :math:`F(T, Pr)` using the specified formulation (Troe, SRI, etc.)

        4. **Final Rate Constant:**
           Applies the Lindemann formula:

           .. math::
               k(T, P) = k_\\infty(T) \\cdot \\frac{P_r}{1 + P_r} \\cdot F(T, P_r)

        **Physical Interpretation:**

        - At low :math:`P_r` (low pressure): :math:`k \\approx k_0·[M] \\cdot F` (third-order, collision-limited)
        - At high :math:`P_r` (high pressure): :math:`k \\approx k_\\infty \\cdot F` (second-order, stabilization-limited)
        - The broadening factor F corrects for the non-Lindemann behavior in the transition region
        """
        # Step 1: Calculate effective third-body concentration [M]_eff
        # Uses ideal gas law with species-specific collision efficiencies
        # Units: mol/cm3
        M = calculate_effective_concentration(T, P, composition, self._efficiencies)

        # Step 2: Calculate reduced pressure P_r
        # Dimensionless parameter characterizing pressure regime:
        # - Pr << 1: Low-pressure (third-order) regime
        # - Pr >> 1: High-pressure (second-order) regime
        # - Pr ~= 1: Transition (fall-off) regime
        Pr = (lpl * M) / hpl

        # Step 3: Compute broadening factor F(T, Pr)
        # Corrects for non-Lindemann behavior using specific formulas
        F = compute_broadening_factor(self._falloff_type, T, Pr, self._falloff_parameters)

        # Step 4: Apply FallOff formula
        return hpl * (Pr / (1 + Pr)) * F

    # ==================================================================================
    # String Representations and Debugging
    def __str__(self) -> str:
        """
        Return a CHEMKIN-compatible string representation.

        Formats the FallOff parameters in the standard CHEMKIN input format,
        which is widely used in combustion and chemical kinetics software.

        Returns
        -------
        str
            Multi-line string with reaction name, high-pressure limit, low-pressure
            limit, falloff parameters, and efficiencies in CHEMKIN format.
        """
        representation = (
            f"{self._name}\t\t{float(self._hpl.A):.5E} {float(self._hpl.n):.5E} {float(self._hpl.Ea):.5E}\n"
        )
        representation += f"  LOW / {float(self._lpl.A):.5E} {float(self._lpl.n):.5E} {float(self._lpl.Ea):.5E} /\n"

        if self._falloff_type == "troe" and self._falloff_parameters is not None:
            representation += " TROE / {:.5E} {:.5E} {:.5E} {:.5E} /\n".format(
                float(self._falloff_parameters["A"]),
                float(self._falloff_parameters["T3"]),
                float(self._falloff_parameters["T1"]),
                float(self._falloff_parameters["T2"]),
            )
        elif self._falloff_type == "sri" and self._falloff_parameters is not None:
            representation += " SRI / {:.5E} {:.5E} {:.5E} {:.5E} {:.5E} /\n".format(
                float(self._falloff_parameters["a"]),
                float(self._falloff_parameters["b"]),
                float(self._falloff_parameters["c"]),
                float(self._falloff_parameters["d"]),
                float(self._falloff_parameters["e"]),
            )
        elif self._falloff_type == "tsang" and self._falloff_parameters is not None:
            # This is not CHEMKIN standard but i believe it would have been
            # implemented like this
            representation += " TSANG / {:.5E} {:.5E} /\n".format(
                float(self._falloff_parameters["A"]),
                float(self._falloff_parameters["B"]),
            )

        if self._efficiencies is not None:
            for species, efficiency in self._efficiencies.items():
                representation += f" {species} / {float(efficiency):.5F} /"

        return representation

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging and development.

        Provides a multi-line, human-readable representation showing all
        parameter values with appropriate precision.

        Returns
        -------
        str
            Multi-line formatted string showing all FallOff parameters.
        """
        lines = ["FallOff("]
        # Display reaction name and broadening type
        lines.append(f" name = {self._name}")
        lines.append(f" type = {self._falloff_type}")

        # Display high-pressure limit parameters (reuse Arrhenius __repr__ for consistency)
        hpl_repr = repr(self._hpl)
        lines.append(" HPL  = " + hpl_repr.replace("\n", "\n         "))

        # Display low-pressure limit parameters
        lpl_repr = repr(self._lpl)
        lines.append(" LPL  = " + lpl_repr.replace("\n", "\n         "))

        # Display broadening parameters if present (Troe, SRI, Tsang)
        if self._falloff_parameters is not None:
            lines.append(" falloff_parameters = {")
            for key, value in self._falloff_parameters.items():
                lines.append(f"  {key:4s} = {float(value):.5E}")
            lines.append(" }")

        # Display third-body efficiencies if specified
        if self._efficiencies is not None:
            lines.append(" efficiencies = {")
            for species, efficiency in self._efficiencies.items():
                lines.append(f"  {species:4s} = {float(efficiency):.5E}")
            lines.append(" }")
        lines.append(")")
        return "\n".join(lines)

    # ==================================================================================
    # Properties for parameters access
    @property
    def name(self) -> str:
        """
        Human-readable reaction name.

        Returns
        -------
        str
            The reaction name string.
        """
        return self._name

    @property
    def falloff_type(self) -> str:
        """
        Type of broadening factor used for fall-off correction.

        Returns
        -------
        str
            One of: "lindemann", "troe", "sri", or "tsang"
        """
        return self._falloff_type

    @property
    def hpl(self) -> Arrhenius:
        """
        High-pressure limit (:math:`k_\\infty`) Arrhenius object.

        Returns
        -------
        Arrhenius
            Arrhenius object representing the high-pressure limit rate constant.
        """
        return self._hpl

    @property
    def lpl(self) -> Arrhenius:
        """
        Low-pressure limit (:math:`k_0`) Arrhenius object.

        Returns
        -------
        Arrhenius
            Arrhenius object representing the low-pressure limit rate constant.
        """
        return self._lpl

    @property
    def falloff_parameters(self) -> dict[str, Float64[Array, ""]] | None:
        """
        Broadening factor parameters as JAX arrays.

        Returns
        -------
        dict[str, Float64[Array, ""]] or None
            Dictionary of broadening parameters (type-dependent):
        """
        return self._falloff_parameters

    @property
    def efficiencies(self) -> dict[str, Float64[Array, ""]] | None:
        """
        Third-body collision efficiencies as JAX arrays.

        Returns
        -------
        dict[str, Float64[Array, ""]] or None
            Dictionary mapping species names to collision efficiencies.
            All values are JAX float64 arrays. Returns None if no
            efficiencies were specified (default efficiency 1.0 for all).
        """
        return self._efficiencies
