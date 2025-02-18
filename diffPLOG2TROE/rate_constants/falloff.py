from typing import Dict, Union

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float64

from .rate_interpreter import parse_rate_constant
from .arrhenius import Arrhenius
from .falloff_functions import sri, troe


class FallOff(eqx.Module):
    hpl: Arrhenius
    lpl: Arrhenius
    falloff_type: int
    falloff_coefficients: Array
    R_IDEAL_GAS = jnp.float64(0.08206)  # L⋅atm/(mol⋅K)

    def __init__(self, rate_constant: Dict) -> None:
        hpl, lpl, self.falloff_coefficients, self.falloff_type = parse_rate_constant(rate_constant)
        self.hpl = Arrhenius(
            {"name": rate_constant["name"], "type": "arrhenius", "rate-constant": {"coefficients": hpl}}
        )
        self.lpl = Arrhenius(
            {"name": rate_constant["name"], "type": "arrhenius", "rate-constant": {"coefficients": lpl}}
        )

    @staticmethod
    def _calculate_concentration(
        P: Union[Float64, Array], T: Union[Float64, Array], R: Float64 = jnp.float64(0.08206)
    ) -> Union[Float64, Array]:
        """Calculate concentration in mol/cm³ from pressure (atm) and temperature (K)."""
        return (P / (R * T)) * jnp.float64(0.001)  # Convert L -> cm³

    def kinetic_constant(self, T: Union[Float64, Array], P: Union[Float64, Array]) -> Union[Float64, Array]:
        is_troe = self.falloff_type == 1
        is_sri = self.falloff_type == 2
        k_hpl = self.hpl.kinetic_constant(T)
        k_lpl = self.lpl.kinetic_constant(T)
        M = self._calculate_concentration(P, T, self.R_IDEAL_GAS)
        Pr = k_lpl * M / k_hpl
        operand = (T, Pr, self.falloff_coefficients)
        F = lax.cond(
            is_troe,
            lambda x: troe(*x),
            lambda x: lax.cond(
                is_sri,
                lambda y: sri(*y),
                # lambda y: lindemann(*y),
                lambda _: jnp.ones_like(T, dtype=jnp.float64),
                x,
            ),
            operand,
        )
        return k_hpl * (Pr / (1 + Pr)) * F

    def __repr__(self) -> str:
        return f"<FallOff: type={self.falloff_type}, coeffs={self.falloff_coefficients}>"
