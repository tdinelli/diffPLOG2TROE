from typing import Dict, List, Union

import equinox as eqx
import jax.numpy as jnp
from jax import lax, vmap
from jaxtyping import Array, Float64

from ..physical_constants import PhysicalConstants as constants
from .arrhenius import Arrhenius
from .rate_interpreter import parse_rate_constant


class Plog(eqx.Module):
    k_levels: List[Arrhenius]
    p_levels: Array
    lnp_levels: Array
    number_of_pressure_levels: int
    name: str

    def __init__(self, rate_constant: Dict) -> None:
        self.name = rate_constant["name"]
        self.p_levels, k_levels = parse_rate_constant(rate_constant)
        self.lnp_levels = jnp.log(self.p_levels)
        self.k_levels = [
            Arrhenius({"name": rate_constant["name"], "type": "arrhenius", "rate-constant": {"coefficients": level}})
            for level in k_levels
        ]
        self.number_of_pressure_levels = len(self.p_levels)

    def _find_index(self, p_index: int, i: int, P: Float64) -> int:
        return lax.cond(P <= self.p_levels[i], lambda _: i, lambda _: p_index, None)

    def _compute_k(self, T: Union[Float64, Array], idx: int) -> Union[Float64, Array]:
        """
        Helper function to compute kinetic constant for a specific index.
        Keep in mind the branch concept when reading this one in the future.
        """
        branches = [lambda i=i: self.k_levels[i].kinetic_constant(T) for i in range(self.number_of_pressure_levels)]
        return lax.switch(idx, branches)

    def _intermediate_pressure_case(self, p_index: int, T: Union[Float64, Array], P: Float64) -> Union[Float64, Array]:
        k1 = self._compute_k(T, p_index - 1)
        k2 = self._compute_k(T, p_index)
        log_k1 = jnp.log(k1)
        log_k2 = jnp.log(k2)
        return jnp.exp(
            log_k1
            + (log_k2 - log_k1)
            * (jnp.log(P) - self.lnp_levels[p_index - 1])
            / (self.lnp_levels[p_index] - self.lnp_levels[p_index - 1])
        )

    def _low_pressure_case(self, T: Union[Float64, Array]) -> Union[Float64, Array]:
        return self._compute_k(T, 0)

    def _high_pressure_case(self, T: Union[Float64, Array]) -> Union[Float64, Array]:
        return self._compute_k(T, self.number_of_pressure_levels - 1)

    @eqx.filter_jit
    def _single_P_kinetic_constant(self, T: Union[Float64, Array], P: Float64) -> Union[Float64, Array]:
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

    @eqx.filter_jit
    def kinetic_constant(self, T: Union[Float64, Array], P: Union[Float64, Array]) -> Union[Float64, Array]:
        if jnp.isscalar(P) or P.ndim == 0:
            return self._single_P_kinetic_constant(T, P)
        else:
            vectorized_k = vmap(lambda p: self._single_P_kinetic_constant(T, p))
            return vectorized_k(P)

    def __str__(self) -> str:
        str_obj = "{}\t\t{:.5e} {:.5f} {:.5e}\n".format(self.name, 0.0, 0.0, 0.0)
        for i in range(self.number_of_pressure_levels):
            arrhenius = self.k_levels[i]
            str_obj += " PLOG / {:.5e}\t{:.5e} {:.5f} {:.5e} /\n".format(
                self.p_levels[i], jnp.exp(arrhenius.lnA), arrhenius.n, arrhenius.EaR * constants.R_cal_mol
            )
        return str_obj
