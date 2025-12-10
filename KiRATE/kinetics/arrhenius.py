"""
Copyright (c) 2025 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import equinox as eqx
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float64

from KiRATE.kinetics.utils import validate_arrhenius_parameters
from KiRATE.utilities import constants, parse_reaction_line


class Arrhenius(eqx.Module):
    """
    Arrhenius rate constant calculator for chemical reactions.

    This class implements the Arrhenius equation to calculate temperature-dependent
    rate constants for chemical reactions. The Arrhenius equation is given by:

    .. math::
        k(T) = A \\cdot T^n \\cdot \\exp\\left(\\frac{-Ea}{R \\cdot T}\\right)

    where:
        - k(T) is the rate constant at temperature T [units vary with reaction order]
        - A is the pre-exponential factor (frequency factor) [units vary with reaction order]
        - T is the absolute temperature [K]
        - n is the temperature exponent [dimensionless]
        - Ea is the activation energy [cal/mol]
        - R is the universal gas constant [cal/mol/K]

    This implementation supports both standard Arrhenius (n=0) and modified Arrhenius
    (n!=0) forms, with automatic differentiation capabilities.

    Parameters
    ----------
    parameters : dict[str, float]
        Dictionary containing Arrhenius parameters with keys:

        - "A": Pre-exponential factor (units depend on reaction order)
        - "n": Temperature exponent (dimensionless)
        - "Ea": Activation energy (cal/mol)
    name : str, optional
        Human-readable name for the reaction, by default ""

    Attributes
    ----------
    _A : Float64[Array, ""]
        Pre-exponential factor stored as JAX scalar
    _n : Float64[Array, ""]
        Temperature exponent stored as JAX scalar
    _Ea : Float64[Array, ""]
        Activation energy stored as JAX scalar
    _name : str
        Reaction name for identification (static field)
    """

    _A: Float64[Array, ""]
    _n: Float64[Array, ""]
    _Ea: Float64[Array, ""]
    _name: str = eqx.field(static=True, default="")

    def __init__(self, parameters: dict[str, float], name: str = "") -> None:
        """
        Initialize the Arrhenius rate constant calculator.

        Parameters
        ----------
        parameters : dict[str, float]
            Dictionary containing Arrhenius parameters:
            - "A": Pre-exponential factor (units depend on reaction order)
            - "n": Temperature exponent (dimensionless)
            - "Ea": Activation energy (cal/mol)
        name : str, optional
            Human-readable name for the reaction, by default ""

        Raises
        ------
        ValueError
            If required parameters are missing or contain invalid values
        UserWarning
            If parameter values are outside typical ranges (validation warnings)
        """
        self._name = name

        # Parameters validation
        validate_arrhenius_parameters(parameters)

        self._A = jnp.float64(parameters["A"])
        self._n = jnp.float64(parameters["n"])
        self._Ea = jnp.float64(parameters["Ea"])

    @classmethod
    def from_chemkin(cls, input_string: str) -> "Arrhenius":
        """
        Create an Arrhenius instance from a CHEMKIN format string.

        This class method provides a convenient way to construct Arrhenius objects
        directly from CHEMKIN-style input strings, which are commonly used in chemical
        kinetics databases and modeling software.

        Parameters
        ----------
        input_string : str
            CHEMKIN-formatted string containing reaction name and parameters.

        Returns
        -------
        Arrhenius
            New Arrhenius instance with parsed parameters and reaction name.

        Raises
        ------
        ValueError
            If the input string cannot be parsed or contains invalid parameters.
        """
        name, params = parse_reaction_line(input_string)

        return cls(name=name, parameters=params)

    @eqx.filter_jit
    def rate_constant(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Calculate the Arrhenius rate constant at given temperature(s).

        This method implements the core Arrhenius equation with automatic broadcasting
        for vector inputs.

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) in Kelvin. Accepts:

            - Python float: Single temperature
            - JAX scalar array: Single temperature as array
            - JAX 1D array: Multiple temperatures

            Note: Python floats are automatically promoted by JAX, but explicit
            array conversion may be needed for gradient operations.

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Rate constant(s) at the specified temperature(s).

            - Units depend on reaction order and pre-exponential factor A
            - Shape matches input temperature array
            - Always returned as JAX arrays for consistency

        Notes
        -----
        - This method is JIT-compiled
        - This method supports automatic differentiation w.r.t. both T and parameters
        """
        T = jnp.asarray(T, dtype=jnp.float64)

        return self._A * jnp.power(T, self._n) * jnp.exp(-self._Ea / constants.R_cal_mol / T)

    # ==================================================================================
    # Automatic Differentiation Methods
    @eqx.filter_jit
    def grad_temperature(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Calculate the derivative of the rate constant with respect to temperature
        (dk/dT) using automatic differentiation.

        For the modified arrhenius expression it is possible to derive an analytical
        formula:

        .. math::
            \\frac{dk(T)}{dT} = k(T) \\cdot \\left(\\frac{n}{T} - \\frac{Ea}{R \\cdot T^2}\\right)

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) in Kelvin at which to evaluate the gradient.

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Temperature gradient dk/dT at the specified temperature(s).
        """
        T_jax = jnp.asarray(T, dtype=jnp.float64)  # Ensure differentiability with eqx.filter_grad

        if T_jax.ndim == 0:
            return eqx.filter_grad(self.rate_constant)(T_jax)
        else:
            return vmap(lambda t: eqx.filter_grad(self.rate_constant)(t))(T_jax)

    @eqx.filter_jit
    def grad_params(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> "Arrhenius":
        """
        Calculate the gradient of the rate constant with respect to parameters using
        automatic differentiation.

        Computes the parameter sensitivity vector:

        .. math::
            \\nabla_{\\theta} k(T) = \\begin{bmatrix}
                \\frac{\\partial k}{\\partial A} \\\\
                \\frac{\\partial k}{\\partial n} \\\\
                \\frac{\\partial k}{\\partial E_a}
            \\end{bmatrix}

        where :math:`\\theta = (A, n, E_a)` are the Arrhenius parameters.

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) in Kelvin at which to evaluate the parameter gradients.

            - **Scalar**: Returns gradients at a single temperature
            - **Vector**: Returns gradients of the sum :math:`\\sum_i k(T_i)`, useful for
                          fitting to multiple experimental points simultaneously

        Returns
        -------
        Arrhenius
            An Arrhenius object with gradients stored in place of parameters:

            - ``result.A``: :math:`\\frac{\\partial k}{\\partial A}` - Sensitivity to pre-exponential factor [dimensionless if A has same units as k]
            - ``result.n``: :math:`\\frac{\\partial k}{\\partial n}` - Sensitivity to temperature exponent [same units as k]
            - ``result.Ea``: :math:`\\frac{\\partial k}{\\partial E_a}` - Sensitivity to activation energy [k·mol/cal]

        Notes
        -----
        For the Arrhenius equation :math:`k(T) = A T^n \\exp(-E_a / RT)`, the analytical gradients are:

        .. math::
            \\frac{\\partial k}{\\partial A} &= \\frac{k}{A} \\\\
            \\frac{\\partial k}{\\partial n} &= k \\ln(T) \\\\
            \\frac{\\partial k}{\\partial E_a} &= -\\frac{k}{RT}

        When T is a vector, this method computes :math:`\\nabla_{\\theta} \\sum_i k(T_i)`, which is
        equivalent to summing individual gradients: :math:`\\sum_i \\nabla_{\\theta} k(T_i)`.
        """
        T_jax = jnp.asarray(T, dtype=jnp.float64)

        # This is needed to perform the differentiation wrt to the params
        # since they are stored as fields in this class
        wrapper_function = lambda m, t: m.rate_constant(t)

        if T_jax.ndim == 0:
            return eqx.filter_grad(wrapper_function)(self, T_jax)
        else:
            return vmap(lambda t: eqx.filter_grad(wrapper_function)(self, t))(T_jax)

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
        return f"{self._name}\t\t{float(self._A):.5E} {float(self._n):.5E} {float(self._Ea):.5E}"

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
            ("A", float(self._A), ".5e"),
            ("n", float(self._n), ".5e"),
            ("Ea", float(self._Ea), ".5e"),
        ]

        lines = ["Arrhenius("]
        for key, value, fmt in params:
            lines.append(f" {key:4s} = {value:{fmt}}")
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
            Pre-exponential factor A from the Arrhenius equation. Units depend
            on the reaction order.
        """
        return self._A

    @property
    def n(self) -> Float64[Array, ""]:
        """
        Temperature exponent (modified Arrhenius parameter).

        Returns
        -------
        Float64[Array, ""]
            Temperature exponent n from the Arrhenius equation (dimensionless).
        """
        return self._n

    @property
    def Ea(self) -> Float64[Array, ""]:
        """
        Activation energy.

        Returns
        -------
        Float64[Array, ""]
            Activation energy Ea in cal/mol.
        """
        return self._Ea

    @property
    def name(self) -> str:
        """
        Human-readable reaction name.

        Returns
        -------
        str
            The reaction name string, typically in chemical equation format.

        Notes
        -----
        - Static field that doesn't participate in JAX transformations
        - Used for identification and output formatting
        - Can contain operators: =, =>, <=> for different reaction types
        - Empty string by default if not specified during initialization
        """
        return self._name
