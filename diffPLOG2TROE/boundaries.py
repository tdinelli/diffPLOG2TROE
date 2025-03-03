from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float64

from .rate_constants.arrhenius import refit_arrhenius


class Boundaries:
    def __init__(
        self,
        rate_constant: Array,
        uncertainty_factor: Float64,
        uncertainty_type: str,
        T_range: Tuple[Float64, Float64],
        n_T: int = 100,
    ) -> None:
        self.rate_constant = rate_constant
        self.f = uncertainty_factor
        self.uncertainty_type = uncertainty_type
        self.T_range = jnp.linspace(T_range[0], T_range[1], n_T)

    def compute_boundaries(self) -> Tuple[Array, Array]:
        if self.uncertainty_type == "symmetric":
            k_lb = (10 ** (-self.f)) * self.rate_constant
            k_ub = (10**self.f) * self.rate_constant

            lnA_lb, beta_lb, EaR_lb = refit_arrhenius(k_lb, self.T_range, True)
            lnA_ub, beta_ub, EaR_ub = refit_arrhenius(k_ub, self.T_range, True)

            lb_params = jnp.array([lnA_lb, beta_lb, EaR_lb])
            ub_params = jnp.array([lnA_ub, beta_ub, EaR_ub])

            return lb_params, ub_params
        else:
            raise ValueError(f"Unknown uncertainty type {self.uncertainty_type}")
