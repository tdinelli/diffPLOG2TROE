"""
Copyright (c) 2025 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import equinox as eqx
import jax.numpy as jnp

from ..types.common import Either, ParamsDict, Real
from ..utilities.physical_constants import constants
from .utils import validate_arrhenius_parameters


class Arrhenius(eqx.Module):
    _lnA: Real
    _n: Real
    _EaR: Real
    _name: str

    def __init__(self, parameters: ParamsDict, name: str = "") -> None:
        # ==============================================================================
        # Validate input parameters
        validate_arrhenius_parameters(parameters)

        # ==============================================================================
        # Store reaction name
        self._name = name

        # ==============================================================================
        # Natural logarithm of the pre-exponential factor
        self._lnA = jnp.log(parameters["A"])

        # ==============================================================================
        # Temperature exponent
        self._n = parameters["n"]

        # ==============================================================================
        # Activation energy pre-divide by gas constant
        self._EaR = parameters["Ea"] / constants.R_cal_mol

    @eqx.filter_jit
    def rate_constant(self, T: Either) -> Either:
        return jnp.exp(self._lnA + self._n * jnp.log(T) - self._EaR / T)

    def __str__(self) -> str:
        A_original = jnp.exp(self._lnA)
        Ea_original = self._EaR * constants.R_cal_mol
        return f"{self._name}\t\t{A_original:.5e} {self._n:.5e} {Ea_original:.5e}"

    def __repr__(self) -> str:
        return (
            f"Arrhenius(name='{self._name}', "
            f"A={jnp.exp(self._lnA):.3e}, "
            f"n={self._n:.3f}, "
            f"Ea={self._EaR * constants.R_cal_mol:.1f} cal/mol)"
        )

    @property
    def A(self) -> Real:
        return jnp.exp(self._lnA)

    @property
    def n(self) -> Real:
        return self._n

    @property
    def Ea(self) -> Real:
        return self._EaR * constants.R_cal_mol
