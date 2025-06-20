from typing import Optional, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from ..utilities.custom_types import Array64f_3, ScalarOrVector
from ..utilities.physical_constants import constants
from .parametrization_utils import validate_arrhenius_parameters


class CollisionEfficiency(eqx.Module):
    lnA: Float64
    name: str

    n: Optional[Float64] = None
    EaR: Optional[Float64] = None

    def __init__(self, parameters: Union[Float64, Array64f_3], name: str = "") -> None:
        self.name = name

        if isinstance(parameters, Array):
            validate_arrhenius_parameters(parameters)
            self.lnA = jnp.log(parameters[0])
            self.n = parameters[1]
            self.EaR = parameters[2] / constants.R_cal_mol
        else:
            self.lnA = parameters

    def __call__(self, T: Optional[ScalarOrVector] = None) -> ScalarOrVector:
        if T is None and self.n is None:
            return self.lnA
        else:
            return self.arrhenius_like(T)

    @eqx.filter_jit
    def arrhenius_like(self, T: ScalarOrVector) -> ScalarOrVector:
        return jnp.exp(self.lnA + self.n * jnp.log(T) - self.EaR / T)

    def __str__(self) -> str:
        if self.n is None:
            return f"{self.name} / {self.lnA:.5e} /"
        else:
            return f"{self.name} / {jnp.exp(self.lnA):.5e} {self.n:.5e} {self.EaR * constants.R_cal_mol:.5e} /"
