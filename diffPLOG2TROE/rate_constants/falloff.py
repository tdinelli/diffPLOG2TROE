from typing import Dict, Union

import equinox as eqx
import jax.numpy as jnp
from jax import lax, vmap
from jaxtyping import Array, Float64

from .arrhenius import Arrhenius
from .falloff_functions import lindemann, sri, troe
from .rate_interpreter import parse_rate_constant


class FallOff(eqx.Module):
    hpl: Arrhenius
    lpl: Arrhenius
    falloff_type: int
    falloff_coefficients: Array
    R_IDEAL_GAS = jnp.float64(0.08206)
    name: str

    def __init__(self, rate_constant: Dict) -> None:
        self.name = rate_constant["name"]
        hpl_coeff, lpl_coeff, self.falloff_coefficients, self.falloff_type = parse_rate_constant(rate_constant)
        self.hpl = Arrhenius(
            {"name": rate_constant["name"], "type": "arrhenius", "rate-constant": {"coefficients": hpl_coeff}}
        )
        self.lpl = Arrhenius(
            {"name": rate_constant["name"], "type": "arrhenius", "rate-constant": {"coefficients": lpl_coeff}}
        )

    def _calculate_concentration(self, P: Float64, T: Union[Float64, Array]) -> Union[Float64, Array]:
        return (P / (self.R_IDEAL_GAS * T)) * jnp.float64(0.001)

    def _compute_falloff_factor(self, T: Union[Float64, Array], Pr: Union[Float64, Array]) -> Union[Float64, Array]:
        operand = (T, Pr, self.falloff_coefficients)
        return lax.switch(
            self.falloff_type,  # 0: Lindemann, 1: Troe, 2: Sri
            [
                lambda _: lindemann(T),
                lambda x: troe(*x),
                lambda x: sri(*x),
            ],
            operand,
        )

    @eqx.filter_jit
    def _single_P_kinetic_constant(self, T: Union[Float64, Array], P: Float64) -> Union[Float64, Array]:
        k_hpl = self.hpl.kinetic_constant(T)
        k_lpl = self.lpl.kinetic_constant(T)
        M = self._calculate_concentration(P, T)
        Pr = k_lpl * M / k_hpl
        F = self._compute_falloff_factor(T, Pr)
        return k_hpl * (Pr / (1 + Pr)) * F

    @eqx.filter_jit
    def kinetic_constant(self, T: Union[Float64, Array], P: Union[Float64, Array]) -> Union[Float64, Array]:
        if jnp.isscalar(P) or P.ndim == 0:
            return self._single_P_kinetic_constant(T, P)
        else:
            vectorized_k = vmap(lambda p: self._single_P_kinetic_constant(T, p))
            return vectorized_k(P)

    def __str__(self) -> str:
        representation = "{}\t\t{:.5e} {:.5f} {:.5e}\n".format(
            self.name, jnp.exp(self.hpl.lnA), self.hpl.beta, self.hpl.EaR * 1.982
        )
        representation += " LOW  / \t\t{:.5e} {:.5f} {:.5e} /\n".format(
            jnp.exp(self.lpl.lnA), self.lpl.beta, self.lpl.EaR * 1.982
        )
        representation += " TROE / {:.5e} {:.5e} {:.5e} {:.5e} /".format(
            self.falloff_coefficients[0],
            self.falloff_coefficients[1],
            self.falloff_coefficients[2],
            self.falloff_coefficients[3],
        )
        return representation
