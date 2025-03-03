import warnings
from typing import Dict, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from jax import jit
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


@jit
def refit_arrhenius(
    rate_constant: Array, temperature: Array, three_params: bool = False, residual_threshold: Float64 = 1.0
) -> Tuple[Float64, Float64, Float64]:
    """
    Refit Arrhenius parameters from rate constant data using least squares regression.

    This function calculates parameters for the Arrhenius equation by fitting experimental
    rate constant data at different temperatures. It can fit either the standard Arrhenius
    equation or the modified Arrhenius equation with a temperature exponent.

    Standard Arrhenius equation:
        k = A * exp(-Ea/(R*T))

    Modified Arrhenius equation:
        k = A * T^n * exp(-Ea/(R*T))

    The fitting is performed by solving the least squares problem for:
        ln(k) = ln(A) + n*ln(T) - (Ea/R)*(1/T)

    Parameters
    ----------
    rate_constant : Array
        Experimental rate constants [units consistent with your kinetic]
    temperature : Array
        Temperatures corresponding to each rate constant [K]
    three_params : bool, default=False
        If True, fits the modified Arrhenius equation with temperature exponent (A, n, Ea)
        If False, fits the standard Arrhenius equation (A, Ea)
    residual_threshold : float, default=1.0
        Threshold for the sum of squared residuals. If the fit produces residuals
        above this value, a warning will be issued.

    Returns
    -------
    Tuple[Float64, Float64, Float64]
        A tuple containing:
        - ln(A): Natural logarithm of the pre-exponential factor
        - n: Temperature exponent (0 for standard Arrhenius)
        - Ea/R: Activation energy divided by gas constant [K]

    Warns
    -----
    UserWarning
        If the sum of squared residuals exceeds the specified threshold,
        indicating a potentially poor fit to the Arrhenius model.

    Notes
    -----
    The units of the pre-exponential factor A depend on the reaction order
    and the units used for the rate constants.

    Consider plotting ln(k) vs 1/T to visually verify the linearity of your data
    when residuals are high.
    """
    log_k = jnp.log(rate_constant)
    inv_T = 1.0 / temperature

    if three_params:
        log_T = jnp.log(temperature)
        X = jnp.vstack([jnp.ones_like(log_T), log_T, -inv_T]).T
    else:
        X = jnp.vstack([jnp.ones_like(inv_T), -inv_T]).T

    beta, residuals, _, _ = jnp.linalg.lstsq(X, log_k, rcond=None)

    if residuals > residual_threshold:
        warnings.warn(f"High residuals value ({residuals:.4f}) detected in Arrhenius fit. ", UserWarning)

    # Return ln(A), n, Ea/R with n=0 for two-parameter model
    return beta[0], beta[1] if three_params else 0.0, beta[-1]
