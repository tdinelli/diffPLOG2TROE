from typing import List

import equinox as eqx
import jax.numpy as jnp
from jax import lax, vmap
from jaxtyping import Float64

from ..utilities.custom_types import Array64f, Matrix64f, ScalarOrVector
from ..utilities.physical_constants import constants
from .arrhenius import Arrhenius


class Plog(eqx.Module):
    k_levels: List[Arrhenius]
    p_levels: Array64f
    lnp_levels: Array64f
    num_p_levels: int
    name: str

    def __init__(self, parameters: Matrix64f, name: str = "") -> None:
        self.name = name

        # ==============================================================================
        # Sort pressure levels in ascending order
        parameters = parameters[jnp.argsort(parameters[:, 0])]

        self.p_levels = parameters[:, 0]
        self.lnp_levels = jnp.log(self.p_levels)
        self.num_p_levels = len(self.p_levels)

        self.k_levels = [Arrhenius(parameters=i) for i in parameters[:, 1:]]

    @eqx.filter_jit
    def kinetic_constant(self, T: ScalarOrVector, P: ScalarOrVector) -> ScalarOrVector:
        """Compute kinetic constant for given temperature and pressure."""
        if jnp.isscalar(P) or P.ndim == 0:  # P is scalar
            return self._single_P_kinetic_constant(T, P)
        else:  # P is array
            vec_func = vmap(lambda p: self._single_P_kinetic_constant(T, p))
            return vec_func(P)

    def _single_P_kinetic_constant(self, T: ScalarOrVector, P: Float64) -> ScalarOrVector:
        all_lnk = jnp.log(jnp.array([k_level.kinetic_constant(T) for k_level in self.k_levels]))

        # ==============================================================================
        # Identify the region of the table
        is_below_min = P <= self.p_levels[0]
        is_above_max = P >= self.p_levels[-1]

        k = lax.cond(
            is_below_min,
            lambda _: all_lnk[0],
            lambda _: lax.cond(
                is_above_max,
                lambda _: all_lnk[-1],
                lambda _: self._interpolated_constant(all_lnk, P),
                None,
            ),
            None,
        )
        return jnp.exp(k)

    def _interpolated_constant(self, all_lnk: ScalarOrVector, P: Float64) -> ScalarOrVector:
        # ==============================================================================
        # Log-log interpolation for pressures within range
        upper_idx = self._find_index(P)  # Position of the current pressure value in the pressure levels of the plog
        lower_idx = upper_idx - 1

        upper_lnp = self.lnp_levels[upper_idx]
        lower_lnp = self.lnp_levels[lower_idx]

        upper_lnk = all_lnk[upper_idx]
        lower_lnk = all_lnk[lower_idx]

        return self._log_log_interpolation(lower_lnk, upper_lnk, lower_lnp, upper_lnp, P)

    def _find_index(self, P: ScalarOrVector) -> Array64f:
        # ==============================================================================
        # Get the first insertion point where P <= p_levels[i]
        indices = jnp.searchsorted(self.p_levels, P, side="left")

        # ==============================================================================
        # If P is greater than all values in p_levels, set index to the last element
        indices = jnp.where(indices == self.num_p_levels, self.num_p_levels - 1, indices)

        return indices

    @staticmethod
    def _log_log_interpolation(
        log_k1: ScalarOrVector,
        log_k2: ScalarOrVector,
        log_P1: ScalarOrVector,
        log_P2: ScalarOrVector,
        P: Float64,
    ) -> ScalarOrVector:
        return log_k1 + (log_k2 - log_k1) * (jnp.log(P) - log_P1) / (log_P2 - log_P1)

    def __str__(self) -> str:
        """Return string representation in CHEMKIN format."""
        str_obj = f"{self.name}\t\t{0.0:.5e} {0.0:.5f} {0.0:.5e}\n"
        for i in range(self.num_p_levels):
            arrhenius = self.k_levels[i]
            str_obj += f" PLOG / {self.p_levels[i]:.5e}\t{jnp.exp(arrhenius.lnA):.5e} {arrhenius.n:.5f} {arrhenius.EaR * constants.R_cal_mol:.5e} /\n"
        return str_obj
