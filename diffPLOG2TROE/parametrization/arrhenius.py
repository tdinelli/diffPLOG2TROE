import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float64

from ..utilities.custom_types import Array64f_3, ScalarOrVector
from ..utilities.physical_constants import constants
from .parametrization_utils import validate_arrhenius_parameters


class Arrhenius(eqx.Module):
    """
    Implementation of the Arrhenius equation for calculating reaction rate constants.

    The Arrhenius equation relates the rate constant (k) of a chemical reaction to the
    temperature (T) according to the following law:

    k = A * T^n * exp(-Ea/(R*T))

    where:
    - A is the pre-exponential factor. Units can be [1/s] if the reaction is first order, [cm3/mol/s] if second order.
    - n is the temperature exponent. This is unitless.
    - Ea is the activation energy. Here we use [cal/mol].
    - R is the gas constant. The value to be used depends on the units chosen for Ea.

    Attributes
    ----------
    lnA : Float64
        Natural logarithm of the pre-exponential factor A.
    n : Float64
        Temperature exponent.
    EaR : Float64
        Activation energy divided by the gas constant (Ea/R).
    name : str
        Optional identifier for the reaction.

    Notes
    -----
    This implementation uses JAX for numerical operations and Equinox for
    creating differentiable modules.
    """

    lnA: Float64
    n: Float64
    EaR: Float64
    name: str

    def __init__(self, parameters: Array64f_3, name: str = "") -> None:
        """
        Initialize an Arrhenius instance with given parameters.

        Parameters
        ----------
        parameters : Array
            Array of [A, n, Ea], where:
            - A is the pre-exponential factor in [1/s] or [cm³/mol/s]
            - n is the dimensionless temperature exponent
            - Ea is the activation energy in [cal/mol]
        name : str, optional
            Optional identifier for the reaction, by default ""

        Raises
        ------
        ValueError
            If the pre-exponential factor A equals 0, or some of the parameters are not finite.
        """
        self.name = name
        validate_arrhenius_parameters(parameters)

        # ==============================================================================
        # Pre-exponential factor
        self.lnA = jnp.log(parameters[0])

        # ==============================================================================
        # Temperature exponent
        self.n = parameters[1]

        # ==============================================================================
        # Activation energy
        self.EaR = parameters[2] / constants.R_cal_mol

    @eqx.filter_jit
    def kinetic_constant(self, T: ScalarOrVector) -> ScalarOrVector:
        """
        Calculate rate constant at given temperature(s).

        Parameters
        ----------
        T : Union[Float64, Array]
            Temperature or array of temperatures in [K].

        Returns
        -------
        Union[Float64, Array]
            Rate constant(s) calculated using the Arrhenius equation.
            Units depend on the reaction order:
            - For first-order reactions: [1/s]
            - For second-order reactions: [cm3/mol/s]
            - For third-order reactions: [cm6/mol2/s]

        Notes
        -----
        The calculation uses the Arrhenius equation in the form:
        k = exp(lnA + n*ln(T) - EaR/T)

        This method is JIT-compiled for performance.
        """
        return jnp.exp(self.lnA + self.n * jnp.log(T) - self.EaR / T)

    def __str__(self) -> str:
        """
        Return a string representation in CHEMKIN format.

        Returns
        -------
        str
            String representation of the Arrhenius parameters in CHEMKIN format.
        """
        return "{}\t\t{:.5e} {:.5e} {:.5e}".format(self.name, jnp.exp(self.lnA), self.n, self.EaR * constants.R_cal_mol)
