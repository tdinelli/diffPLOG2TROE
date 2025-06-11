"""Thermal collision efficiency implementation"""

import equinox as eqx
import jax.numpy as jnp

from ..utilities.custom_types import Array64f_3, ScalarOrVector
from ..utilities.physical_constants import constants
from .parametrization_utils import validate_arrhenius_parameters


class CollisionEfficiency(eqx.Module):
    name: str

    def __init__(self, parameters: Array64f_3, name: str = "") -> None:
        self.name = name
        validate_arrhenius_parameters(parameters)

        self.lnA = jnp.log(parameters[0])

        self.n = parameters[1]

        self.EaR = parameters[2] / constants.R_cal_mol

    @eqx.filter_jit
    def kinetic_constant(self, T: ScalarOrVector) -> ScalarOrVector:
        return jnp.exp(self.lnA + self.n * jnp.log(T) - self.EaR / T)
