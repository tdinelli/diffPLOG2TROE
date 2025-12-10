"""
Copyright (c) 2025 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""
from typing import Optional

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
        k(T, P, [M]) = k_\\infty(T) \\cdot \\frac{P_r}{1 + P_r} \\cdot F(T, P_r)

    where:
        - k(T, P, [M]) is the pressure-dependent rate constant
        - :math:`k_\\infty(T)` is the high-pressure limit (Arrhenius)
        - :math:`k_0(T)` is the low-pressure limit (Arrhenius)
        - :math:`P_r = k_0(T) \\cdot [M] / k_\\infty (T)` is the reduced pressure
        - [M] is the effective third-body concentration with efficiencies
        - F(T, Pr) is the broadening factor (Lindemann, Troe, SRI, or Tsang)

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
        Broadening factor parameters (type-dependent):

        - **Troe**: {"A": alpha, "T3": T3, "T1": T1, "T2": T2}
        - **SRI**: {"a": a, "b": b, "c": c, "d": d, "e": e}
        - **Tsang**: {"A": A, "B": B}
        - **Lindemann**: None (F = 1.0)

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
    _falloff_parameters : Optional[dict[str, Float64[Array, ""]]]
        Broadening factor parameters as JAX arrays
    _efficiencies : Optional[dict[str, Float64[Array, ""]]]
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
    .. [1] Gilbert, R. G., et al. "Theory of thermal unimolecular reactions in the
           fall-off range. I. Strong collision rate constants." Ber. Bunsenges. Phys.
           Chem. 87.2 (1983): 169-177.
    .. [2] Troe, J. "Predictive possibilities of unimolecular rate theory." J. Phys.
           Chem. 83.1 (1979): 114-126.
    .. [3] Stewart, P. H., et al. "Pressure and temperature dependence of reactions
           proceeding via a bound complex. 2." J. Phys. Chem. 93.8 (1989): 3557-3561.
    """

    _hpl: Arrhenius
    _lpl: Arrhenius
    _falloff_type: str = eqx.field(static=True)
    _falloff_parameters: Optional[dict[str, Float64[Array, ""]]] = None
    _efficiencies: Optional[dict[str, Float64[Array, ""]]] = None
    _name: str = eqx.field(static=True, default="")

    def __init__(
        self,
        hpl_parameters: dict[str, float],
        lpl_parameters: dict[str, float],
        falloff_type: str,
        falloff_parameters: Optional[dict[str, float]] = None,
        efficiencies: Optional[dict[str, float]] = None,
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

        Notes
        -----
        - All parameters are converted to JAX float64 arrays for differentiability
        - HPL and LPL Arrhenius objects are created internally
        - Validation is performed on broadening parameters and efficiencies
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

    @eqx.filter_jit
    def rate_constant(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "nt"],
        P: float | Float64[Array, ""] | Float64[Array, "np"],
        composition: Optional[dict[str, float]] = None,
    ) -> Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]:
        """
        Calculate the fall-off rate constant at given temperature(s) and pressure(s).

        This method implements the Lindemann-Hinshelwood fall-off formalism with
        broadening factor corrections, automatically handling third-body effects
        and vectorized evaluation.

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "nt"]
            Temperature(s) in Kelvin. Accepts scalars or 1D arrays.
        P : float | Float64[Array, ""] | Float64[Array, "np"]
            Pressure(s) in bar. Accepts scalars or 1D arrays.
        composition : dict[str, float], optional
            Gas composition as mole fractions: {"species": x_i}
            Used to calculate effective third-body concentration with efficiencies.
            If None, assumes uniform composition with default efficiencies.

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]
            Fall-off rate constant(s) with shape matching input broadcasting:

            - Scalar T, Scalar P = Scalar output
            - Vector T, Scalar P = Vector output (length nt)
            - Scalar T, Vector P = Vector output (length np)
            - Vector T, Vector P = Matrix output (shape: np x nt)

            Units depend on reaction order (typically cm3/(mol s) for bimolecular HPL).

        Notes
        -----
        **Fall-off Formula:**

        .. math::
            k(T, P) = k_\\infty \\cdot \\frac{P_r}{1 + P_r} \\cdot F(T, P_r)

        where :math:`P_r = k_0 \\cdot [M] / k_\\infty` is the reduced pressure.

        **Third-Body Effects:**

        The effective concentration [M]_eff accounts for species-specific
        collision efficiencies:

        .. math::
            [M]_{eff} = [M] \\cdot \\sum_i \\epsilon_i x_i

        **Performance:**

        - Fully differentiable w.r.t. T, P, and all parameters
        - Vectorization over temperature and pressure arrays
        """
        T = jnp.asarray(T, dtype=jnp.float64)
        P = jnp.asarray(P, dtype=jnp.float64)

        # Convert composition to JAX arrays for differentiability
        jax_composition = (
            {key: jnp.float64(value) for key, value in composition.items()} if composition is not None else None
        )

        # Compute high-pressure and low-pressure limit rate constants
        k_hpl = self._hpl.rate_constant(T)  # High-pressure limit [cm3/(mol s)]
        k_lpl = self._lpl.rate_constant(T)  # Low-pressure limit [cm6/(mol2 s)]

        if jnp.isscalar(P) or P.ndim == 0:
            # Scalar pressure - evaluate directly
            return self._single_P_rate_constant(
                T,
                P,
                k_lpl,
                k_hpl,
                jax_composition,
            )
        else:
            # Vector pressure - vectorize over pressure dimension
            vec_func = vmap(
                lambda p: self._single_P_rate_constant(
                    T,
                    p,
                    k_lpl,
                    k_hpl,
                    jax_composition,
                )
            )
            return vec_func(P)

    @eqx.filter_jit
    def _single_P_rate_constant(
        self,
        T: Float64[Array, ""] | Float64[Array, "nt"],
        P: Float64[Array, ""],
        lpl: Float64[Array, ""] | Float64[Array, "nt"],
        hpl: Float64[Array, ""] | Float64[Array, "nt"],
        composition: Optional[dict[str, Float64[Array, ""]]] = None,
    ) -> Float64[Array, ""] | Float64[Array, "nt"]:
        """
        Calculate fall-off rate constant at a single pressure and one or more temperatures.

        This internal method implements the core Lindemann-Hinshelwood fall-off calculation
        with broadening factor corrections. It is called by `rate_constant()` and should not
        be invoked directly by users.

        Parameters
        ----------
        T : Float64[Array, ""] | Float64[Array, "nt"]
            Temperature(s) in Kelvin (scalar or 1D array)
        P : Float64[Array, ""]
            Single pressure value in bar (scalar)
        lpl : Float64[Array, ""] | Float64[Array, "nt"]
            Pre-computed low-pressure limit rate constant(s) k0(T)
            Units: cm6/(mol2 s)
        hpl : Float64[Array, ""] | Float64[Array, "nt"]
            Pre-computed high-pressure limit rate constant(s) k_inf(T)
            Units: cm3/(mol s) for typical bimolecular reactions
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
           Computes [M]_eff accounting for third-body efficiencies:

           .. math::
               [M]_{eff} = \\frac{P}{R \\cdot T} \\cdot \\sum_i \\epsilon_i x_i

        2. **Reduced Pressure Calculation:**
           Determines the dimensionless pressure parameter:

           .. math::
               P_r = \\frac{k_0(T) \\cdot [M]_{eff}}{k_\\infty(T)}

        3. **Broadening Factor Evaluation:**
           Computes F(T, Pr) using the specified formulation (Troe, SRI, etc.)

        4. **Final Rate Constant:**
           Applies the Lindemann formula:

           .. math::
               k(T, P) = k_\\infty(T) \\cdot \\frac{P_r}{1 + P_r} \\cdot F(T, P_r)

        **Physical Interpretation:**

        - At low P_r (low pressure): :math:`k \\approx k_0·[M] \\cdot F` (third-order, collision-limited)
        - At high P_r (high pressure): :math:`k \\approx k_\\infty \\cdot F` (second-order, stabilization-limited)
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
        # - Pr \\approx 1: Transition (fall-off) regime
        Pr = (lpl * M) / hpl

        # Step 3: Compute broadening factor F(T, Pr)
        # Corrects for non-Lindemann behavior using specific formulas
        F = compute_broadening_factor(self._falloff_type, T, Pr, self._falloff_parameters)

        # Step 4: Apply FallOff formula
        return hpl * (Pr / (1 + Pr)) * F

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
                  TROE / alpha   T3   T1   T2 /     ! or SRI / a b c d e /
                  SPECIES / efficiency /            ! optional

        Returns
        -------
        FallOff
            New FallOff instance with parsed parameters, reaction name, and efficiencies.

        Raises
        ------
        ValueError
            If the input string cannot be parsed or contains invalid parameters.

        Notes
        -----
        The parser automatically detects the broadening factor type (TROE, SRI, TSANG,
        or Lindemann) based on the keywords present in the input string. If no
        broadening parameters are specified, Lindemann formulation (F=1) is assumed.

        The CHEMKIN format uses the following conventions:
        - Activation energy Ea is in cal/mol
        - Pre-exponential factors A have units depending on reaction order
        - Efficiencies default to 1.0 for species not listed
        """
        # Parse CHEMKIN input string to extract all parameters
        reaction_name, hpl_params, lpl_params, falloff_type, falloff_params, efficiencies = parse_falloff(
            input_string
        )

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
        representation = "{}\t\t{:.5E} {:.5E} {:.5E}\n".format(
            self._name, float(self._hpl.A), float(self._hpl.n), float(self._hpl.Ea)
        )
        representation += "  LOW / {:.5E} {:.5E} {:.5E} /\n".format(
            float(self._lpl.A), float(self._lpl.n), float(self._lpl.Ea)
        )

        if self._falloff_type == "troe" and self._falloff_parameters is not None:
            representation += " TROE / {:.5E} {:.5E} {:.5E} {:.5E} /".format(
                float(self._falloff_parameters["A"]),
                float(self._falloff_parameters["T3"]),
                float(self._falloff_parameters["T1"]),
                float(self._falloff_parameters["T2"]),
            )
        elif self._falloff_type == "sri" and self._falloff_parameters is not None:
            representation += " SRI / {:.5E} {:.5E} {:.5E} {:.5E} {:.5E} /".format(
                float(self._falloff_parameters["a"]),
                float(self._falloff_parameters["b"]),
                float(self._falloff_parameters["c"]),
                float(self._falloff_parameters["d"]),
                float(self._falloff_parameters["e"]),
            )
        elif self._falloff_type == "tsang" and self._falloff_parameters is not None:
            # This is not CHEMKIN standard but i believe it would have been
            # implemented like this
            representation += " TSANG / {:.5E} {:.5E} /".format(
                float(self._falloff_parameters["A"]),
                float(self._falloff_parameters["B"]),
            )

        if self._efficiencies is not None:
            representation += "\n"
            for species, efficiency in self._efficiencies.items():
                representation += " {} / {:.5E} /".format(species, float(efficiency))

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
            The reaction name string, typically in chemical equation format
            with (+M) notation indicating pressure-dependence.
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
    def falloff_parameters(self) -> Optional[dict[str, Float64[Array, ""]]]:
        """
        Broadening factor parameters as JAX arrays.

        Returns
        -------
        dict[str, Float64[Array, ""]] or None
            Dictionary of broadening parameters (type-dependent):

        Notes
        -----
        These parameters are fully differentiable and can be optimized using
        gradient-based methods. They control the shape of the broadening factor
        F(T, P_r) which corrects for non-Lindemann behavior in the transition region.
        """
        if self._falloff_parameters is not None:
            return self._falloff_parameters
        else:
            return None

    @property
    def efficiencies(self) -> Optional[dict[str, Float64[Array, ""]]]:
        """
        Third-body collision efficiencies as JAX arrays.

        Returns
        -------
        dict[str, Float64[Array, ""]] or None
            Dictionary mapping species names to collision efficiencies.
            All values are JAX float64 arrays. Returns None if no
            efficiencies were specified (default efficiency 1.0 for all).

        Notes
        -----
        Collision efficiencies (:math:`\\eps_i`) account for the varying effectiveness of
        different bath gas species in stabilizing the activated complex. The
        effective third-body concentration is:

        .. math::
            [M]_{eff} = [M] \\cdot \\sum_i \\epsilon_i x_i

        Typical values:
        - Noble gases (Ar, He): 0.5-0.9 (less efficient)
        - Polar molecules (H2O): 5-15 (more efficient)
        - Default species: 1.0

        These values are differentiable and can be optimized if needed.
        """
        if self._efficiencies is not None:
            return self._efficiencies
        else:
            return None
