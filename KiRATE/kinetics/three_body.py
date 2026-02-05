"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from KiRATE.kinetics.arrhenius import Arrhenius
from KiRATE.kinetics.utils import validate_efficiencies
from KiRATE.utilities import calculate_effective_concentration, parse_threebody


class Threebody(eqx.Module):
    """
    Three-body (termolecular) reaction rate constant calculator.

    This class implements rate constants for three-body reactions where a third
    collision partner (M) is required to stabilize the reaction products by
    carrying away excess energy. The rate constant depends linearly on pressure
    through the effective third-body concentration.

    The rate constant is given by:

    .. math::
        k(T, P) = k_0(T) \\cdot [M]_{eff}

    where:
        - :math:`k_0(T)` is the temperature-dependent third-order rate constant
        - :math:`[M]_{eff}` is the effective third-body concentration accounting
          for species-specific collision efficiencies

    The effective concentration includes collision efficiency factors:

    .. math::
        [M]_{eff} = [M] \\cdot \\left( \\sum_i \\alpha_i x_i + (1 - \\sum_i x_i) \\right)

    where :math:`\\alpha_i` is the collision efficiency of species i and :math:`x_i`
    is its mole fraction.

    Attributes
    ----------
    _k0 : Arrhenius
        Third-order rate constant :math:`k_0(T)` with units [cm6/mol2/s]
    _efficiencies : dict[str, Float64[Array, ""]] or None
        Species-specific collision efficiencies (dimensionless)
    _name : str
        Human-readable reaction name
    """

    _k0: Arrhenius
    _efficiencies: dict[str, Float64[Array, ""]] | None = None
    _name: str = eqx.field(static=True, default="")

    def __init__(
        self,
        parameters: dict[str, float],
        efficiencies: dict[str, float] | None = None,
        name: str = "",
    ) -> None:
        """
        Initialize the three-body reaction rate constant calculator.

        Parameters
        ----------
        parameters : dict[str, float]
            Third-order Arrhenius rate constant parameters:
            {"A": pre-exponential factor, "n": temperature exponent, "Ea": activation energy}

        efficiencies : dict[str, float], optional
            Third-body collision efficiencies, e.g., {"AR": 0.7, "H2O": 6.0}
            Species not listed default to efficiency = 1.0.

        name : str, optional
            Human-readable reaction name, by default ""
            Example: "H+OH+M=H2O+M"

        Raises
        ------
        ValueError
            If efficiency values are negative
        UserWarning
            If efficiency values are outside typical ranges (0.1 to 20)

        Notes
        -----
        The third-order rate constant :math:`k_0(T)` follows the modified Arrhenius form:

        .. math::
            k_0(T) = A \\cdot T^n \\cdot \\exp\\left(-\\frac{E_a}{RT}\\right)

        All parameters are stored as JAX arrays for automatic differentiation
        compatibility.
        """
        self._name = name

        # Create Arrhenius object for the third-order rate constant
        self._k0 = Arrhenius(parameters=parameters, name=f"{name} (k0)")

        # Validate and store third-body efficiencies
        if efficiencies is not None:
            validate_efficiencies(efficiencies)
            # Convert to JAX arrays for differentiability
            self._efficiencies = {key: jnp.float64(value) for key, value in efficiencies.items()}
        else:
            self._efficiencies = None

    @classmethod
    def from_chemkin(cls, input_string: str) -> "Threebody":
        """
        Create a Threebody instance from a CHEMKIN format string.

        This class method provides a convenient way to construct Threebody objects
        directly from CHEMKIN-style input strings, which are the standard format
        used in combustion and chemical kinetics databases.

        Parameters
        ----------
        input_string : str
            CHEMKIN-formatted string containing reaction name, rate constant parameters,
            and optionally third-body efficiencies. The format is:

            .. code-block:: text

                REACTION_NAME    A   n   Ea
                SPECIES / efficiency / ... / ...      / ! optional

            Example:
                H+OH+M=H2O+M  2.2E+22  -2.0  0.0
                H2O/6.0/ AR/0.38/

        Returns
        -------
        Threebody
            New Threebody instance with parsed parameters, reaction name, and efficiencies.

        Raises
        ------
        ValueError
            If the input string cannot be parsed or contains invalid parameters.
        """
        name, k0_params, efficiencies = parse_threebody(input_string)

        return cls(name=name, parameters=k0_params, efficiencies=efficiencies)

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
        Calculate the three-body rate constant at given temperature(s) and pressure(s).

        This method computes the **full pressure-dependent rate constant** by multiplying
        the third-order Arrhenius rate constant :math:`k_0(T)` with the effective third-body
        concentration [M]_eff, which accounts for species-specific collision efficiencies.

        .. important::
            **Design Choice - KiRATE vs Cantera:**

            This method returns the **full pressure-dependent rate constant**:

            .. math::
                k(T, P) = k_0(T) \\cdot [M]_{eff}

            This differs from Cantera's ``forward_rate_constants`` for three-body reactions,
            which returns **only** :math:`k_0(T)`. Cantera handles the :math:`[M]` multiplication
            internally at the reactor/ODE level.

            **Rationale for KiRATE's approach:**
                - More intuitive: users get a directly usable rate constant
                - Consistent with other pressure-dependent classes (``FallOff``, ``CABR``)
                - Self-contained: no need for separate [M] calculations
                - Result can be used directly: ``rate = k(T,P) × [A] × [B]``

            **To match Cantera's behavior:** Access only the temperature-dependent component
            via ``reaction.k0.rate_constant(T)``, which returns :math:`k_0(T)` without [M].

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
            Mole fractions should sum to 1.0.

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]
            Full pressure-dependent rate constant(s) with shape matching input broadcasting:

            - Scalar T, Scalar P → Scalar output
            - Vector T, Scalar P → Vector output (length nt)
            - Scalar T, Vector P → Vector output (length np)
            - Vector T, Vector P → Matrix output (shape: nt × np)

        Notes
        -----
        **Three-Body Rate Formula:**

        .. math::
            k(T, P) = k_0(T) \\cdot [M]_{eff}

        where:
            - :math:`k_0(T)` is the third-order Arrhenius rate constant [cm⁶/mol²/s]:

              .. math::
                  k_0(T) = A \\cdot T^n \\cdot \\exp\\left(-\\frac{E_a}{RT}\\right)

            - :math:`[M]_{eff}` is the effective third-body concentration [mol/cm³]

        **Third-Body Effects:**

        The effective concentration accounts for species-specific collision efficiencies:

        .. math::
            [M]_{eff} = [M] \\cdot \\left( \\sum_i \\alpha_i x_i + (1 - \\sum_i x_i) \\right)

        where :math:`\\alpha_i` is the collision efficiency of species i and :math:`x_i`
        is its mole fraction. Species not listed in the efficiencies dictionary are
        assigned a default efficiency of 1.0.

        **Usage in Rate Expressions:**

        The returned rate constant can be used directly in rate laws:

        .. math::
            \\text{rate} = k(T, P) \\cdot [A] \\cdot [B]

        **Important:** Do NOT multiply by [M] again, as it's already included in k(T,P).
        """
        T = jnp.asarray(T, dtype=jnp.float64)
        P = jnp.asarray(P, dtype=jnp.float64)

        # Convert composition to JAX arrays for differentiability
        jax_composition = (
            {key: jnp.float64(value) for key, value in composition.items()} if composition is not None else None
        )

        k_0 = self._k0.rate_constant(T)  # Third-order rate constant [cm6/mol2/s]
        M = calculate_effective_concentration(T, P, jax_composition, self._efficiencies)

        if jnp.isscalar(P) or P.ndim == 0:  # Scalar pressure - evaluate directly
            return k_0 * M
        else:  # Vector pressure so M is a matrix of shape (n_T × n_P)
            return k_0[:, None] * M

    # ==================================================================================
    # String representations and debugging
    def __str__(self) -> str:
        """
        Return a CHEMKIN-compatible string representation.

        Formats the Arrhenius parameters in the standard CHEMKIN input format,
        which is widely used in combustion and chemical kinetics software.

        Returns
        -------
        str
            Tab-separated string with reaction name and parameters:
            Format: "{name}\\t\\t{A:.5e} {n:.5f} {Ea:.5e}"
        """
        representation = f"{self._name}\t\t{float(self._k0.A):.5E} {float(self._k0.n):.5E} {float(self._k0.Ea):.5E}"
        if self._efficiencies is not None:
            representation += "\n"
            for species, efficiency in self._efficiencies.items():
                representation += f" {species} / {float(efficiency):.5E} /"

        return representation

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging and development.

        Provides a multi-line, human-readable representation showing all
        parameter values with appropriate precision.

        Returns
        -------
        str
            Multi-line formatted string showing all Arrhenius parameters.
        """
        params = [
            ("name", self._name, "s"),
            ("A", float(self._k0.A), ".5e"),
            ("n", float(self._k0.n), ".5e"),
            ("Ea", float(self._k0.Ea), ".5e"),
        ]

        lines = ["Threebody("]
        for key, value, fmt in params:
            lines.append(f" {key:4s} = {value:{fmt}}")

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
    def A(self) -> Float64[Array, ""]:
        """
        Pre-exponential factor (frequency factor).

        Returns
        -------
        Float64[Array, ""]
            Pre-exponential factor A from the third-order Arrhenius equation.
        """
        return self._k0.A

    @property
    def n(self) -> Float64[Array, ""]:
        """
        Temperature exponent (modified Arrhenius parameter).

        Returns
        -------
        Float64[Array, ""]
            Temperature exponent n from the Arrhenius equation (dimensionless).
        """
        return self._k0.n

    @property
    def Ea(self) -> Float64[Array, ""]:
        """
        Activation energy.

        Returns
        -------
        Float64[Array, ""]
            Activation energy Ea in cal/mol.
        """
        return self._k0.Ea

    @property
    def k0(self) -> "Arrhenius":
        """
        Third-order rate constant as an Arrhenius object.

        Returns
        -------
        Arrhenius
            Arrhenius object representing the third-order rate constant k₀(T).
        """
        return self._k0

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
        if self._efficiencies is not None:
            return self._efficiencies
        else:
            return None

    @property
    def name(self) -> str:
        """
        Human-readable reaction name.

        Returns
        -------
        str
            The reaction name.
        """
        return self._name
