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
from KiRATE.utilities import calculate_effective_concentration


class CABR(eqx.Module):
    """
    Chemically Activated Bimolecular Reaction (CABR) rate constant calculator.

    This class implements chemically activated reactions where the rate constant
    decreases with increasing pressure, opposite to fall-off reactions. These reactions
    occur when an energized intermediate formed by bimolecular collision either
    dissociates to products (at low pressure) or is stabilized by collisions (at high
    pressure).

    The CABR formulation is given by:

    .. math::
        k(T, P, [M]) = k_0(T) \\cdot \\frac{1}{1 + P_r} \\cdot F(T, P_r)

    where:
        - k(T, P, [M]) is the pressure-dependent rate constant
        - :math:`k_0 (T)` is the low-pressure limit (Arrhenius) - **dominant at low P**
        - :math:`k_\\infty (T)` is the high-pressure limit (Arrhenius)
        - :math:`P_r = k_0(T) \\cdot [M] / k_\\infty (T)` is the reduced pressure
        - [M] is the effective third-body concentration with efficiencies
        - F(T, Pr) is the broadening factor (Lindemann, Troe, SRI, or Tsang)

    Physical Interpretation:
        At low pressure: Activated complex dissociates to products (fast)
        At high pressure: Complex is stabilized by collisions before dissociation (slow)

    Parameters
    ----------
    hpl_parameters : dict[str, float]
        High-pressure limit Arrhenius parameters:
        {"A": pre-exponential factor, "n": temperature exponent, "Ea": activation energy}
    lpl_parameters : dict[str, float]
        Low-pressure limit Arrhenius parameters (same format as hpl_parameters)
    cabr_type : str
        Type of broadening factor: "lindemann", "troe", "sri", or "tsang"
    cabr_parameters : dict[str, float], optional
        Broadening factor parameters (type-dependent)
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
    _cabr_type : str
        Broadening factor type (static field)
    _cabr_parameters : Optional[dict[str, Float64[Array, ""]]]
        Broadening factor parameters as JAX arrays
    _efficiencies : Optional[dict[str, Float64[Array, ""]]]
        Third-body efficiencies as JAX arrays
    _name : str
        Reaction name (static field)

    Notes
    -----
    The rate decreases with pressure because increasing collisions stabilize the
    activated intermediate before it can dissociate to products.

    References
    ----------
    .. [1] Cantera Documentation: Chemically-Activated Reactions.
           https://cantera.org/stable/reference/kinetics/rate-constants.html
    .. [2] Gilbert, R. G., et al. "Theory of thermal unimolecular reactions in the
           fall-off range." Ber. Bunsenges. Phys. Chem. 87.2 (1983): 169-177.
    """

    _hpl: Arrhenius
    _lpl: Arrhenius
    _cabr_type: str = eqx.field(static=True)
    _cabr_parameters: Optional[dict[str, Float64[Array, ""]]] = None
    _efficiencies: Optional[dict[str, Float64[Array, ""]]] = None
    _name: str = eqx.field(static=True, default="")

    def __init__(
        self,
        hpl_parameters: dict[str, float],
        lpl_parameters: dict[str, float],
        cabr_type: str,
        cabr_parameters: Optional[dict[str, float]] = None,
        efficiencies: Optional[dict[str, float]] = None,
        name: str = "",
    ) -> None:
        """
        Initialize the CABR rate constant calculator.

        Parameters
        ----------
        hpl_parameters : dict[str, float]
            High-pressure limit Arrhenius parameters:
            {"A": pre-exponential, "n": temperature exponent, "Ea": activation energy}
        lpl_parameters : dict[str, float]
            Low-pressure limit Arrhenius parameters (same format)
        cabr_type : str
            Broadening factor type: "lindemann", "troe", "sri", or "tsang"
        cabr_parameters : dict[str, float], optional
            Broadening parameters (type-dependent)
        efficiencies : dict[str, float], optional
            Third-body collision efficiencies, e.g., {"AR": 0.7, "H2O": 6.0}
            Species not listed default to efficiency = 1.0
        name : str, optional
            Human-readable reaction name, by default ""

        Raises
        ------
        ValueError
            If broadening parameters are invalid for the specified cabr_type
        UserWarning
            If efficiency values are outside typical ranges
        """
        self._name = name
        self._cabr_type = cabr_type

        # Create Arrhenius objects for high-pressure and low-pressure limits
        self._hpl = Arrhenius(parameters=hpl_parameters, name=f"{name} (HPL)")
        self._lpl = Arrhenius(parameters=lpl_parameters, name=f"{name} (LPL)")

        # Validate and store broadening factor parameters
        if cabr_parameters is not None:
            validate_broadening_parameters(cabr_type, cabr_parameters)
            # Convert to JAX arrays for differentiability
            self._cabr_parameters = {key: jnp.float64(value) for key, value in cabr_parameters.items()}
        else:
            self._cabr_parameters = None

        # Validate and store third-body efficiencies
        if efficiencies is not None:
            validate_efficiencies(efficiencies)
            # Convert to JAX arrays for differentiability
            self._efficiencies = {key: jnp.float64(value) for key, value in efficiencies.items()}
        else:
            self._efficiencies = None

    # ==================================================================================
    # CHEMKIN string parser
    @classmethod
    def from_chemkin(cls, input_string: str) -> "CABR":
        """
        Parse a CHEMKIN-format CABR entry (not yet implemented).

        Parameters
        ----------
        input_string : str
            CHEMKIN-formatted CABR reaction string

        Raises
        ------
        NotImplementedError
            This method is a placeholder for future implementation
        """
        raise NotImplementedError("CHEMKIN parsing for CABR is not yet implemented")

    # ==================================================================================
    # Rate constant methods
    @eqx.filter_jit
    def rate_constant(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "nt"],
        P: float | Float64[Array, ""] | Float64[Array, "np"],
        composition: Optional[dict[str, float]] = None,
    ) -> Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]:
        """
        Calculate the CABR rate constant at given temperature(s) and pressure(s).

        This method implements the chemically activated bimolecular reaction formalism
        where the rate constant decreases with increasing pressure, opposite to fall-off
        reactions.

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
            CABR rate constant(s) with shape matching input broadcasting:

            - Scalar T, Scalar P -> Scalar output
            - Vector T, Scalar P -> Vector output (length nt)
            - Scalar T, Vector P -> Vector output (length np)
            - Vector T, Vector P -> Matrix output (shape: np x nt)

            Units depend on reaction order (typically cm3/(mol s) for bimolecular).

        Notes
        -----
        **CABR Formula:**

        .. math::
            k(T, P) = k_0 \\cdot \\frac{1}{1 + P_r} \\cdot F(T, P_r)

        where :math:`P_r = k_0 \\cdot [M] / k_\\infty` is the reduced pressure.

        **Pressure Dependence:**

        - At low P (P_r << 1): :math:`k \\approx k_0 \\cdot F` (maximum rate, complex dissociates)
        - At high P (P_r >> 1): :math:`k \\approx (k_0 / P_r) \\cdot F = (k_\\infty / [M]) \\cdot F` (minimum rate, complex stabilized)

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
            {key: jnp.float64(value) for key, value in composition.items()}
            if composition is not None
            else None
        )

        # Compute high-pressure and low-pressure limit rate constants
        k_hpl = self._hpl.rate_constant(T)  # High-pressure limit [cm3/(mol s)]
        k_lpl = self._lpl.rate_constant(T)  # Low-pressure limit [cm6/(mol2 s)]

        if jnp.isscalar(P) or P.ndim == 0: # Scalar pressure - evaluate directly
            return self._single_P_rate_constant(
                T,
                P,
                k_lpl,
                k_hpl,
                jax_composition,
            )
        else: # Vector pressure - vectorize over pressure dimension
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
        Calculate CABR rate constant at a single pressure and one or more temperatures.

        This internal method implements the core chemically activated bimolecular
        reaction calculation. It is called by `rate_constant()` and should not be
        invoked directly by users.

        Parameters
        ----------
        T : Float64[Array, ""] | Float64[Array, "nt"]
            Temperature(s) in Kelvin (scalar or 1D array)
        P : Float64[Array, ""]
            Single pressure value in bar (scalar)
        lpl : Float64[Array, ""] | Float64[Array, "nt"]
            Pre-computed low-pressure limit rate constant(s) k0(T)
        hpl : Float64[Array, ""] | Float64[Array, "nt"]
            Pre-computed high-pressure limit rate constant(s) k_inf(T)
            Units: cm3/(mol s) for typical bimolecular reactions
        composition : dict[str, Float64[Array, ""]], optional
            Gas composition as mole fractions with JAX arrays

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "nt"]
            CABR rate constant(s) at the specified pressure and temperature(s)

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
           Applies the CABR formula (inverse of fall-off):

           .. math::
               k(T, P) = k_0(T) \\cdot \\frac{1}{1 + P_r} \\cdot F(T, P_r)
        """
        # Step 1: Calculate effective third-body concentration [M]_eff
        # Units: mol/cm3
        M = calculate_effective_concentration(T, P, composition, self._efficiencies)

        # Step 2: Calculate reduced pressure Pr
        # Dimensionless parameter characterizing pressure regime:
        # - Pr << 1: Low-pressure regime (activated complex dissociates)
        # - Pr >> 1: High-pressure regime (complex stabilized)
        Pr = (lpl * M) / hpl

        # Step 3: Compute broadening factor F(T, Pr)
        # Corrects for non-Lindemann behavior using specific formulas
        F = compute_broadening_factor(self._cabr_type, T, Pr, self._cabr_parameters)

        # Step 4: Apply CABR formula (inverse of fall-off)
        return lpl * (1 / (1 + Pr)) * F

    # ==================================================================================
    # String Representations and Debugging
    def __str__(self) -> str:
        """
        Return a CHEMKIN-compatible string representation.

        Formats the CABR parameters in the standard CHEMKIN input format. For CABR
        reactions, the LOW parameter becomes the main rate (base line) and HIGH
        specifies the high-pressure limit.

        Returns
        -------
        str
            Multi-line string with reaction name, low-pressure limit (main rate),
            high-pressure limit, broadening parameters, and efficiencies in CHEMKIN format.

        Notes
        -----
        CHEMKIN CABR format uses:
        - Main line: name and low-pressure limit parameters (k₀)
        - HIGH keyword: high-pressure limit parameters (k_∞)
        - TROE/SRI/TSANG keyword: optional broadening parameters
        - Species/efficiency pairs: optional third-body efficiencies
        """
        # Main line: reaction name and LOW-pressure limit (which dominates at low P)
        representation = "{}\t\t{:.5E} {:.5E} {:.5E}\n".format(
            self._name, float(self._lpl.A), float(self._lpl.n), float(self._lpl.Ea)
        )

        # HIGH keyword: high-pressure limit parameters
        representation += " HIGH / {:.5E} {:.5E} {:.5E} /\n".format(
            float(self._hpl.A), float(self._hpl.n), float(self._hpl.Ea)
        )

        # Broadening factor parameters
        if self._cabr_type == "troe" and self._cabr_parameters is not None:
            representation += " TROE / {:.5E} {:.5E} {:.5E} {:.5E} /".format(
                float(self._cabr_parameters["A"]),
                float(self._cabr_parameters["T3"]),
                float(self._cabr_parameters["T1"]),
                float(self._cabr_parameters["T2"]),
            )
        elif self._cabr_type == "sri" and self._cabr_parameters is not None:
            representation += " SRI / {:.5E} {:.5E} {:.5E} {:.5E} {:.5E} /".format(
                float(self._cabr_parameters["a"]),
                float(self._cabr_parameters["b"]),
                float(self._cabr_parameters["c"]),
                float(self._cabr_parameters["d"]),
                float(self._cabr_parameters["e"]),
            )
        elif self._cabr_type == "tsang" and self._cabr_parameters is not None:
            representation += " TSANG / {:.5E} {:.5E} /".format(
                float(self._cabr_parameters["A"]),
                float(self._cabr_parameters["B"]),
            )

        # Third-body efficiencies
        if self._efficiencies is not None:
            representation += "\n"
            for species, efficiency in self._efficiencies.items():
                representation += " {} / {:.5E} /".format(species, float(efficiency))

        return representation

    def __repr__(self) -> str:
        raise NotImplementedError("'__repr__' method not implemented yet!")

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
    def cabr_type(self) -> str:
        """
        Type of broadening factor used for CABR correction.

        Returns
        -------
        str
            One of: "lindemann", "troe", "sri", or "tsang"
        """
        return self._cabr_type

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
    def cabr_parameters(self) -> Optional[dict[str, Float64[Array, ""]]]:
        """
        Broadening factor parameters as JAX arrays.

        Returns
        -------
        dict[str, Float64[Array, ""]] or None
            Dictionary of broadening parameters
        """
        if self._cabr_parameters is not None:
            return self._cabr_parameters
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
        """
        if self._efficiencies is not None:
            return self._efficiencies
        else:
            return None
