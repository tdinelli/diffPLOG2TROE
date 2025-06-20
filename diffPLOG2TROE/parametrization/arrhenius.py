from typing import Dict, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from ..utilities.physical_constants import constants
from .parametrization_utils import validate_arrhenius_parameters


class Arrhenius(eqx.Module):
    """
    Implementation of the Arrhenius equation for calculating reaction rate constants.

    The Arrhenius equation relates the rate constant (k) of a chemical reaction to the
    temperature (T) according to the following law:

    k = A * T^n * exp(-Ea/(R*T))

    where:
    - A is the pre-exponential factor with units depending on reaction order
    - n is the dimensionless temperature exponent
    - Ea is the activation energy in cal/mol
    - R is the gas constant (1.987 cal/mol/K)
    - T is the absolute temperature in K

    This implementation stores the parameters in a computationally efficient form:
    - lnA: natural logarithm of A (avoids repeated log operations)
    - n: temperature exponent (unchanged)
    - EaR: Ea/R (pre-computed to avoid repeated division)

    Parameters
    ----------
    parameters : Dict[str, Float64]
        Dictionary containing:
        - "A": Pre-exponential factor. Units depend on reaction order:
          * 1st order: [1/s]
          * 2nd order: [cm³/mol/s] or [L/mol/s]
          * 3rd order: [cm⁶/mol²/s] or [L²/mol²/s]
        - "n": Temperature exponent (dimensionless)
        - "Ea": Activation energy in [cal/mol]
    name : str, optional
        Optional identifier for the reaction, by default ""

    Attributes
    ----------
    lnA : Float64
        Natural logarithm of the pre-exponential factor A
    n : Float64
        Temperature exponent
    EaR : Float64
        Activation energy divided by gas constant (Ea/R) in K
    name : str
        Optional reaction identifier

    References
    ----------
    .. [1] Arrhenius, S. (1889). "Über die Reaktionsgeschwindigkeit bei der Inversion
           von Rohrzucker durch Säuren". Zeitschrift für physikalische Chemie, 4U(1), 96-116.
    .. [2] Kee, R. J., Rupley, F. M., & Miller, J. A. (1989). "Chemkin-II: A Fortran
           chemical kinetics package for the analysis of gas-phase chemical kinetics"
           (No. SAND-89-8009). Sandia National Labs.
    """

    lnA: Float64
    n: Float64
    EaR: Float64
    name: str

    def __init__(self, parameters: Dict[str, Float64], name: str = "") -> None:
        """
        Initialize an Arrhenius instance with given parameters.

        The constructor validates input parameters and converts them to an
        efficient internal representation for fast computation.

        Parameters
        ----------
        parameters : Dict[str, Float64]
            Dictionary containing Arrhenius parameters:
            - "A": Pre-exponential factor (must be positive)
            - "n": Temperature exponent (dimensionless)
            - "Ea": Activation energy in cal/mol
        name : str, optional
            Optional identifier for the reaction, by default ""

        Raises
        ------
        ValueError
            If parameters are invalid according to validate_arrhenius_parameters()
        """
        # ==============================================================================
        # Validate input parameters
        validate_arrhenius_parameters(parameters)

        # ==============================================================================
        # Store reaction name
        self.name = name

        # ==============================================================================
        # Natural logarithm of the pre-exponential factor
        self.lnA = jnp.log(parameters["A"])

        # ==============================================================================
        # Temperature exponent
        self.n = parameters["n"]

        # ==============================================================================
        # Activation energy pre-divide by gas constant
        self.EaR = parameters["Ea"] / constants.R_cal_mol

    @eqx.filter_jit
    def rate_constant(self, T: Union[Float64, Float64[Array, "dim"]]) -> Union[Float64, Float64[Array, "dim"]]:
        """
        Calculate the reaction rate constant at given temperature(s).

        Implements the Arrhenius equation in the computationally efficient form:
        k = exp(ln(A) + n*ln(T) - (Ea/R)/T)

        This method is JIT-compiled for optimal performance and supports both
        scalar and vectorized temperature inputs.

        Parameters
        ----------
        T : Union[Float64, Array]
            Temperature(s) in Kelvin. Must be positive.
            Can be a scalar or array for vectorized computation.

        Returns
        -------
        Union[Float64, Array]
            Rate constant(s) with units depending on reaction order:
            - 1st order: [1/s]
            - 2nd order: [cm³/mol/s] or [L/mol/s]
            - 3rd order: [cm⁶/mol²/s] or [L²/mol²/s]

        Warnings
        --------
        No explicit bounds checking is performed on temperature for performance.
        Negative or zero temperatures will produce invalid results.
        """
        return jnp.exp(self.lnA + self.n * jnp.log(T) - self.EaR / T)

    def __str__(self) -> str:
        """
        Return a string representation in CHEMKIN format.

        The CHEMKIN format is widely used in combustion and chemical kinetics
        software. The format is: "name  A  n  Ea" where Ea is in cal/mol.

        Returns
        -------
        str
            String representation of Arrhenius parameters in CHEMKIN format.
            Format: "{name}        {A:.5e} {n:.5e} {Ea:.5e}"
        """
        A_original = jnp.exp(self.lnA)
        Ea_original = self.EaR * constants.R_cal_mol
        return f"{self.name}\t\t{A_original:.5e} {self.n:.5e} {Ea_original:.5e}"

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging.

        Returns
        -------
        str
            Detailed representation showing internal parameter storage.
        """
        return (
            f"Arrhenius(name='{self.name}', "
            f"A={jnp.exp(self.lnA):.3e}, "
            f"n={self.n:.3f}, "
            f"Ea={self.EaR * constants.R_cal_mol:.1f} cal/mol)"
        )
