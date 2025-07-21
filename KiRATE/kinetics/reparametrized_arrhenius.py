from typing import Dict, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from ..utilities.physical_constants import constants
from .utils import validate_arrhenius_parameters


class ReparametrizedArrhenius(eqx.Module):
    """
    Implementation of the reparametrized Arrhenius equation in the centered
    form for improved numerical stability.

    The centered Arrhenius equation relates the rate constant (k) to temperature (T)
    using a reference temperature T_ref to reduce parameter correlation:

    k = k_ref * (T/T_ref)^n * exp(-Ea/R * (1/T - 1/T_ref))

    where:
    - k_ref is the rate constant at reference temperature T_ref
    - n is the dimensionless temperature exponent
    - Ea is the activation energy in cal/mol
    - R is the gas constant (1.987 cal/mol/K)
    - T_ref is the reference temperature in K
    - T is the absolute temperature in K

    Parameters
    ----------
    parameters : Dict[str, Float64]
        Dictionary containing:
        - "k_ref": Rate constant at reference temperature with units depending on reaction order
        - "n": Temperature exponent (dimensionless)
        - "Ea": Activation energy in [cal/mol]
        - "T_ref": Reference temperature in [K]
    name : str, optional
        Optional identifier for the reaction, by default ""

    Attributes
    ----------
    lnk_ref : Float64
        Natural logarithm of the rate constant at reference temperature
    n : Float64
        Temperature exponent
    EaR : Float64
        Activation energy divided by gas constant (Ea/R) in K
    T_ref : Float64
        Reference temperature in K
    name : str
        Optional reaction identifier
    """

    lnk_ref: Float64
    n: Float64
    EaR: Float64
    T_ref: Float64
    name: str

    def __init__(self, parameters: Dict[str, Float64], name: str = "") -> None:
        """
        Initialize a CenteredArrhenius instance with given parameters.

        Parameters
        ----------
        parameters : Dict[str, Float64]
            Dictionary containing centered Arrhenius parameters:
            - "k_ref": Rate constant at reference temperature (must be positive)
            - "n": Temperature exponent (dimensionless)
            - "Ea": Activation energy in cal/mol
            - "T_ref": Reference temperature in K (must be positive)
        name : str, optional
            Optional identifier for the reaction, by default ""

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        # ====================================================================
        # TODO: This needs to be checked
        # Validate basic parameters (reuse existing validation)
        # temp_params = {"A": parameters["k_ref"], "n": parameters["n"], "Ea": parameters["Ea"]}
        # validate_arrhenius_parameters(temp_params)

        # ====================================================================
        # Additional validation for T_ref
        if parameters["T_ref"] <= 0:
            raise ValueError("Reference temperature T_ref must be positive")

        self.name = name
        self.lnk_ref = jnp.log(parameters["k_ref"])
        self.n = parameters["n"]
        self.EaR = parameters["Ea"] / constants.R_cal_mol
        self.T_ref = parameters["T_ref"]

    @eqx.filter_jit
    def rate_constant(self, T: Union[Float64, Float64[Array, "dim"]]) -> Union[Float64, Float64[Array, "dim"]]:
        """
        Calculate the reaction rate constant at given temperature(s) using centered form.

        Implements the centered Arrhenius equation:
        k = k_ref * (T/T_ref)^n * exp(-Ea/R * (1/T - 1/T_ref))

        Parameters
        ----------
        T : Union[Float64, Array]
            Temperature(s) in Kelvin. Must be positive.

        Returns
        -------
        Union[Float64, Array]
            Rate constant(s) with same units as k_ref
        """
        return jnp.exp(self.lnk_ref + self.n * jnp.log(T / self.T_ref) - self.EaR * (1.0 / T - 1.0 / self.T_ref))

    def to_standard_form(self) -> Dict[str, Float64]:
        """
        Convert centered parameters to standard Arrhenius form.

        Returns
        -------
        Dict[str, Float64]
            Dictionary with standard Arrhenius parameters:
            - "A": Pre-exponential factor
            - "n": Temperature exponent
            - "Ea": Activation energy in cal/mol
        """
        A_standard = jnp.exp(self.lnk_ref) * (self.T_ref ** (-self.n)) * jnp.exp(self.EaR / self.T_ref)
        return {"A": A_standard, "n": self.n, "Ea": self.EaR * constants.R_cal_mol}

    @classmethod
    def from_standard_form(
        cls,
        parameters: Dict[str, Float64],
        T_ref: Float64,
        name: str = "",
    ) -> "ReparametrizedArrhenius":
        """
        Create CenteredArrhenius instance from standard Arrhenius parameters.

        Parameters
        ----------
        parameters : Dict[str, Float64]
            Standard Arrhenius parameters (A, n, Ea)
        T_ref : Float64
            Reference temperature in K
        name : str, optional
            Optional identifier for the reaction

        Returns
        -------
        ReparametrizedArrhenius
            Instance with centered parametrization
        """
        EaR = parameters["Ea"] / constants.R_cal_mol
        k_ref = parameters["A"] * (T_ref ** parameters["n"]) * jnp.exp(-EaR / T_ref)

        centered_params = {"k_ref": k_ref, "n": parameters["n"], "Ea": parameters["Ea"], "T_ref": T_ref}

        return cls(centered_params, name)

    def __str__(self) -> str:
        """Return string representation in centered format."""
        k_ref_original = jnp.exp(self.lnk_ref)
        Ea_original = self.EaR * constants.R_cal_mol
        return f"{self.name}\t\t{k_ref_original:.5e} {self.n:.5e} {Ea_original:.5e} {self.T_ref:.1f}"

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return (
            "ReparametrizedArrhenius(\n"
            f" name  = {self.name}\n"
            f" k_ref = {jnp.exp(self.lnk_ref):.3e}\n"
            f" n     = {self.n:.3f}\n"
            f" Ea    = {self.EaR * constants.R_cal_mol:.3e} cal/mol\n"
            f" T_ref = {self.T_ref:.3f} K\n"
            ")"
        )
