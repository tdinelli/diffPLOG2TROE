from typing import Dict, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from .rate_interpreter import parse_rate_constant


class Arrhenius(eqx.Module):
    R = jnp.float64(1.987)
    lnA: Float64
    beta: Float64
    EaR: Float64
    name: str

    def __init__(self, rate_constant: Dict) -> None:
        self.name = rate_constant["name"]
        parameters = parse_rate_constant(rate_constant)
        self.lnA = jnp.log(parameters[0])
        self.beta = parameters[1]
        self.EaR = parameters[2] / self.R

    @eqx.filter_jit
    def kinetic_constant(self, T: Union[Float64, Array]) -> Union[Float64, Array]:
        return jnp.exp(self.lnA + self.beta * jnp.log(T) - self.EaR / T)

    def __str__(self) -> str:
        return "{}    {:.3E}\t{}\t{:.3E}".format(self.name, jnp.exp(self.lnA), self.beta, self.EaR * self.R)
