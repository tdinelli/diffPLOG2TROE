from typing import Dict, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from ..physical_constants import PhysicalConstants as constants
from .rate_interpreter import parse_rate_constant


class Arrhenius(eqx.Module):
    """
    Represents a modified Arrhenius rate expression for chemical kinetics.

    The modified Arrhenius equation is commonly used in chemical kinetics to model
    the temperature dependence of reaction rate constants:

    k(T) = A * T^n * exp(-Ea/(R*T))

    Where:
    - A: Pre-exponential factor [units depend on reaction order]
    - n: Temperature exponent [dimensionless]
    - Ea: Activation energy [cal/mol]
    - R: Gas constant [cal/(mol*K)]
    - T: Temperature [K]

    For numerical stability, this implementation internally stores ln(A) rather than A.

    Attributes:
        lnA (Float64): Natural logarithm of the pre-exponential factor
        n (Float64): Temperature exponent
        EaR (Float64): Activation energy divided by the gas constant (Ea/R)
        name (str): Name or identifier for the reaction
    """

    lnA: Float64
    n: Float64
    EaR: Float64
    name: str

    def __init__(self, rate_constant: Dict) -> None:
        """
        Initialize an Arrhenius rate expression from a dictionary of parameters.

        The dictionary is expected to contain rate constant information in a standard
        format that can be parsed by the parse_rate_constant function.

        Args:
            rate_constant (Dict): Dictionary containing rate constant information with:
                - "name": Reaction name or identifier
                - Rate constant parameters (accessed via parse_rate_constant)

        Example:
            >>> arrhenius = Arrhenius({
            ...     "name": "H + O2 = O + OH",
            ...     "type": "arrhenius",
            ...     "rate-constant": {"coefficients": [2.65e+16, -0.671, 17041.0]}
            ... })
        """
        self.name = rate_constant["name"]
        parameters = parse_rate_constant(rate_constant)
        self.lnA = jnp.log(parameters[0])
        self.n = parameters[1]
        self.EaR = parameters[2] / constants.R_cal_mol

    @eqx.filter_jit
    def kinetic_constant(self, T: Union[Float64, Array]) -> Union[Float64, Array]:
        """
        Calculate the rate constant at the specified temperature(s).

        This method implements the modified Arrhenius equation:
        k(T) = A * T^n * exp(-Ea/(R*T))

        This is computed in logarithmic form for numerical stability:
        ln(k(T)) = ln(A) + n*ln(T) - Ea/(R*T)

        The method is just-in-time compiled with JAX for performance and can
        handle both single temperature values and arrays of temperatures.

        Args:
            T (Float64 or Array): Temperature(s) in Kelvin at which to evaluate
                                 the rate constant

        Returns:
            Float64 or Array: Rate constant(s) evaluated at the specified temperature(s)

        Example:
            >>> arr = Arrhenius({"name": "Example", "type": "arrhenius",
            ...                  "rate-constant": {"coefficients": [1.0e+13, 0.0, 0.0]}})
            >>> arr.kinetic_constant(1000.0)
            1.0e+13
            >>> arr.kinetic_constant(jnp.array([300.0, 1000.0, 2000.0]))
            Array([1.0e+13, 1.0e+13, 1.0e+13], dtype=float64)
        """
        return jnp.exp(self.lnA + self.n * jnp.log(T) - self.EaR / T)

    def __str__(self) -> str:
        """
        Return a string representation following the CHEMKIN formalism of the reaction stored whithin the object.

        Returns:
            str: Formatted string with reaction name and Arrhenius parameters

        Example:
            >>> arr = Arrhenius({"name": "H+O2=OH+O", "type": "arrhenius",
            ...                  "rate-constant": {"coefficients": [2.65e+16, -0.671, 17041.0]}})
            >>> print(arr)
            H+O2=OH+O       2.65000e+16 -6.71000e-01 1.70410e+04
        """
        return "{}\t\t{:.5e} {:.5e} {:.5e}".format(self.name, jnp.exp(self.lnA), self.n, self.EaR * constants.R_cal_mol)
