from typing import List, Union

import equinox as eqx
import jax.numpy as jnp
from jax import lax, vmap
from jaxtyping import Array, Float64

from ..utilities.physical_constants import constants
from .arrhenius import Arrhenius


class Plog(eqx.Module):
    k_levels: List[Arrhenius]
    p_levels: Array
    lnp_levels: Array
    num_p_levels: int
    name: str

    def __init__(self, parameters: Array, name: str = "") -> None:
        self.name = name

        # ==============================================================================
        # Ensuring the pressure levels are sorted in ascending order correctly
        parameters = jnp.sort(parameters, axis=0)

        self.p_levels = parameters[:, 0]
        self.lnp_levels = jnp.log(self.p_levels)
        self.num_p_levels = len(self.p_levels)
        self.k_levels = [Arrhenius(parameters=level, name=name) for level in parameters[:, 1:]]

    def _find_index(self, p_index: int, i: int, P: Float64) -> int:
        """Find index of pressure level for interpolation."""
        return lax.cond(
            P <= self.p_levels[i],
            lambda _: i,
            lambda _: p_index,
            None,
        )

    def _compute_k(self, T: Union[Float64, Array], idx: int) -> Union[Float64, Array]:
        """
        Compute kinetic constant for a specific pressure level index.
        Uses JAX's lax.switch for efficient branch handling.
        """
        branches = [lambda i=i: self.k_levels[i].kinetic_constant(T) for i in range(self.num_p_levels)]
        return lax.switch(idx, branches)

    def _interpolate_k(self, p_index: int, T: Union[Float64, Array], P: Float64) -> Union[Float64, Array]:
        """
        Interpolate rate constant between two pressure levels in log-log space.
        Used when pressure is between two tabulated pressure levels.
        """
        k1 = self._compute_k(T, p_index - 1)
        k2 = self._compute_k(T, p_index)
        log_k1 = jnp.log(k1)
        log_k2 = jnp.log(k2)

        # Log-log interpolation
        return jnp.exp(
            log_k1
            + (log_k2 - log_k1)
            * (jnp.log(P) - self.lnp_levels[p_index - 1])
            / (self.lnp_levels[p_index] - self.lnp_levels[p_index - 1])
        )

    def _low_p_k(self, T: Union[Float64, Array]) -> Union[Float64, Array]:
        """Return rate constant for pressure below lowest tabulated level."""
        return self._compute_k(T, 0)

    def _high_p_k(self, T: Union[Float64, Array]) -> Union[Float64, Array]:
        """Return rate constant for pressure above highest tabulated level."""
        return self._compute_k(T, self.num_p_levels - 1)

    @eqx.filter_jit
    def _single_P_kinetic_constant(self, T: Union[Float64, Array], P: Float64) -> Union[Float64, Array]:
        p_index = lax.fori_loop(0, self.num_p_levels, lambda idx, i: self._find_index(idx, i, P), 0)

        return lax.cond(
            P <= self.p_levels[0],
            lambda _: self._low_p_k(T),
            lambda _: lax.cond(
                P >= self.p_levels[-1],
                lambda _: self._high_p_k(T),
                lambda _: self._interpolate_k(p_index, T, P),
                None,
            ),
            None,
        )

    # @eqx.filter_jit
    # def kinetic_constant(self, T: Union[Float64, Array], P: Union[Float64, Array]) -> Union[Float64, Array]:
    #     """Calculate rate constant for given temperature(s) and pressure(s)."""
    #     if jnp.isscalar(P) or P.ndim == 0:
    #         return self._single_p_k(T, P)
    #     else:
    #         vectorized_k = vmap(lambda p: self._single_p_k(T, p))
    #         return vectorized_k(P)
    @eqx.filter_jit
    def kinetic_constant(self, T: Union[Float64, Array], P: Union[Float64, Array]) -> Union[Float64, Array]:
        """
        Note: for future development in principle we could precompute the vectorized functions in the constructor of the
              class to make things even more fast.
        """
        if (jnp.isscalar(T) or T.ndim == 0) and (jnp.isscalar(P) or P.ndim == 0):  # Both are scalars
            return self._single_P_kinetic_constant(T, P)
        elif not (jnp.isscalar(T) or T.ndim == 0) and (jnp.isscalar(P) or P.ndim == 0):  # T is array, P is scalar
            return vmap(lambda t: self._single_P_kinetic_constant(t, P))(T)
        elif (jnp.isscalar(T) or T.ndim == 0) and not (jnp.isscalar(P) or P.ndim == 0):  # P is array, T is scalar
            return vmap(lambda p: self._single_P_kinetic_constant(T, p))(P)
        else:  # Both are arrays. Handle broadcasting based on array shapes
            return vmap(lambda p: self._single_P_kinetic_constant(T, p))(P)

    def __str__(self) -> str:
        """Return string representation in CHEMKIN format."""
        str_obj = "{}\t\t{:.5e} {:.5f} {:.5e}\n".format(self.name, 0.0, 0.0, 0.0)
        for i in range(self.num_p_levels):
            arrhenius = self.k_levels[i]
            str_obj += " PLOG / {:.5e}\t{:.5e} {:.5f} {:.5e} /\n".format(
                self.p_levels[i], jnp.exp(arrhenius.lnA), arrhenius.n, arrhenius.EaR * constants.R_cal_mol
            )
        return str_obj
