"""
Copyright (c) 2025 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import equinox as eqx
import jax.numpy as jnp

from ..types.common import ParamsDict, Scalar, ScalarOrVector
from ..utilities.physical_constants import constants
from .utils import validate_arrhenius_parameters


class Arrhenius(eqx.Module):
    _lnA: Scalar
    _n: Scalar
    _EaR: Scalar
    _name: str = eqx.field(static=True, default="")

    def __init__(self, parameters: ParamsDict, name: str = "") -> None:
        # ==============================================================================
        # Validate input parameters
        eqx.filter_pure_callback(validate_arrhenius_parameters, parameters, result_shape_dtypes=None)

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
    def rate_constant(self, T: ScalarOrVector) -> ScalarOrVector:
        return jnp.exp(self._lnA + self._n * jnp.log(T) - self._EaR / T)

    def __str__(self) -> str:
        A_original = jnp.exp(self._lnA)
        Ea_original = self._EaR * constants.R_cal_mol
        return f"{self._name}\t\t{A_original:.5e} {self._n:.5e} {Ea_original:.5e}"

    def __repr__(self) -> str:
        return (
            f"Arrhenius("
            f"\n name = {self._name}"
            f"\n A    = {jnp.exp(self._lnA):.3e}"
            f"\n n    = {self._n:.3f}"
            f"\n Ea   = {self._EaR * constants.R_cal_mol:.3e}"
            "\n)"
        )

    # ==================================================================================
    # Getters
    @property
    def A(self) -> Scalar:
        return jnp.exp(self._lnA)

    @property
    def n(self) -> Scalar:
        return self._n

    @property
    def Ea(self) -> Scalar:
        return self._EaR * constants.R_cal_mol
