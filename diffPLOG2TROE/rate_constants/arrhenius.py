from typing import Dict, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from ..physical_constants import PhysicalConstants
from .rate_interpreter import parse_rate_constant


class Arrhenius(eqx.Module):
    lnA: Float64
    n: Float64
    EaR: Float64
    name: str

    def __init__(self, rate_constant: Dict) -> None:
        self.name = rate_constant["name"]
        parameters = parse_rate_constant(rate_constant)
        self.lnA = jnp.log(parameters[0])
        self.n = parameters[1]
        self.EaR = parameters[2] / PhysicalConstants.R_cal_mol

    @eqx.filter_jit
    def kinetic_constant(self, T: Union[Float64, Array]) -> Union[Float64, Array]:
        return jnp.exp(self.lnA + self.n * jnp.log(T) - self.EaR / T)

    def __str__(self) -> str:
        return "{}\t\t{:.5e} {:.5e} {:.5e}".format(
            self.name, jnp.exp(self.lnA), self.n, self.EaR * PhysicalConstants.R_cal_mol
        )
