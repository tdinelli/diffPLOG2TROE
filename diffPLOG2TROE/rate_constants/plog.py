from typing import Dict, Union

import equinox as eqx
import jax.numpy as jnp
from jax import lax, debug
from jaxtyping import Array, Float64

from .arrhenius import Arrhenius
from .rate_interpreter import parse_rate_constant


class Plog(eqx.Module):
    k_levels: Array
    p_levels: Array
    lnp_levels: Array
    number_of_pressure_levels: int

    def __init__(self, rate_constant: Dict) -> None:
        self.p_levels, k_levels = parse_rate_constant(rate_constant)
        self.lnp_levels = jnp.log(self.p_levels)
        self.k_levels = jnp.empty(len(self.p_levels), dtype=jnp.float64)
        for i, level in enumerate(k_levels):
            debug.print("{x}", x=level)
            self.k_levels.at[i].set(
                Arrhenius(
                    {"name": rate_constant["name"], "type": "arrhenius", "rate-constant": {"coefficients": level}}
                )
            )
        self.number_of_pressure_levels = len(self.p_levels)

    def _find_index(self, p_index: int, i: int, P: Float64) -> int:
        return lax.cond(P <= self.p_levels[i], lambda _: i, lambda _: p_index, None)

    def _low_pressure_case(self, T: Union[Float64, Array]):
        return self.k_levels[0].kinetic_constant(T)

    def _high_pressure_case(self, T: Union[Float64, Array]):
        return self.k_levels[-1].kinetic_constant(T)

    def _intermediate_pressure_case(self, p_index: int, T: Union[Float64, Array], P: Union[Float64, Array]):
        log_k1 = jnp.log(self.k_levels[p_index - 1].kinetic_constant(T))
        log_k2 = jnp.log(self.k_levels[p_index].kinetic_constant(T))

        # Logarithmic interpolation
        return jnp.exp(
            log_k1
            + (log_k2 - log_k1)
            * (jnp.log(P) - self.lnp_levels[p_index - 1])
            / (self.lnp_levels[p_index] - self.lnp_levels[p_index - 1])
        )

    def kinetic_constant(self, T: Union[Float64, Array], P: Union[Float64, Array]) -> Union[Float64, Array]:
        p_index = lax.fori_loop(0, self.number_of_pressure_levels, lambda idx, i: self._find_index(idx, i, P), 0)
        return lax.cond(
            P <= self.p_levels[0],
            lambda _: self._low_pressure_case(T),
            lambda _: lax.cond(
                P >= self.p_levels[-1],
                lambda _: self._high_pressure_case(T),
                lambda _: self._intermediate_pressure_case(p_index, T, P),
                None,
            ),
            None,
        )

    def __repr__(self) -> str:
        return f"<PLOG:>"
