from typing import Dict, Optional, Tuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float64

from .refitting_utilities import refit_arrhenius, validate_fitting_data
from ..utilities.custom_types import Array64f, Array64f_3, ScalarOrVector
from ..utilities.physical_constants import constants


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
            If the pre-exponential factor A equals 0.
        """
        self.name = name
        self._validate_parameters(parameters)
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

    @classmethod
    def from_data(
        cls,
        rate_constant: Array64f,
        temperature: Array64f,
        weights: Optional[Array64f] = None,
        three_params: bool = True,
        name: str = "",
    ) -> Tuple["Arrhenius", Dict[str, Float64]]:
        """
        Create an Arrhenius instance by fitting to experimental data.

        Parameters
        ----------
        rates : Array
            Array of measured rate constants.
        temps : Array
            Array of temperatures corresponding to the measured rate constants.
        three_params : bool, optional
            If True, fits a three-parameter Arrhenius model (A, n, Ea).
            If False, fits a two-parameter model (A, Ea) with n=0, by default True.
        weights : Optional[Array], optional
            Optional weights for data points. If provided, weighted least squares is used.
        name : str, optional
            Optional identifier for the reaction, by default "".

        Returns
        -------
        Arrhenius
            An Arrhenius instance with parameters fitted to the provided data.

        Raises
        ------
        ValueError
            If input data validation fails.

        Notes
        -----
        Uses least squares regression to fit the Arrhenius parameters.
        Includes uncertainty estimation based on covariance matrix.
        """
        # ==============================================================================
        # Validate input data
        validate_fitting_data(rate_constant, temperature, weights, three_params)

        # ==============================================================================
        # Perform fitting with uncertainty estimation
        fit_result = refit_arrhenius(rate_constant, temperature, weights, three_params)

        # ==============================================================================
        # Convert back to physical parameters
        params = jnp.array([jnp.exp(fit_result["lnA"]), fit_result["n"], fit_result["EaR"] * constants.R_cal_mol])

        return (
            cls(parameters=params, name=name),
            {
                "lnA_uncertainty": fit_result["lnA_uncertainty"],
                "n_uncertainty": fit_result["n_uncertainty"],
                "EaR_uncertainty": fit_result["EaR_uncertainty"],
                "RSS": fit_result["RSS"],
                "DoF": fit_result["DoF"],
            },
        )

    def __str__(self) -> str:
        """
        Return a string representation in CHEMKIN format.

        Returns
        -------
        str
            String representation of the Arrhenius parameters in CHEMKIN format.
        """
        return "{}\t\t{:.5e} {:.5e} {:.5e}".format(self.name, jnp.exp(self.lnA), self.n, self.EaR * constants.R_cal_mol)

    @staticmethod
    def _validate_parameters(parameters: Array64f_3) -> None:
        """
        Validate Arrhenius parameters to ensure consistency in calculations.

        Parameters
        ----------
        parameters : Float64[Array, "3"]
            Array of [A, n, Ea] Arrhenius parameters.

        Raises
        ------
        ValueError
            If parameters are invalid.
        """
        A, n, Ea = parameters
        if A <= 0:
            raise ValueError("Pre-exponential factor must be positive")
        if not jnp.isfinite(A):
            raise ValueError("Pre-exponential factor must be finite")
        if not jnp.isfinite(n):
            raise ValueError("Temperature exponent must be finite")
        if not jnp.isfinite(Ea):
            raise ValueError("Activation energy must be finite")
