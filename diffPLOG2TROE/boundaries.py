from typing import Dict, Literal, Optional, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array, Float64

from .physical_constants import constants
from .rate_constants import Arrhenius, refit_arrhenius


class Boundaries:
    def __init__(
        self,
        rate_constant: Array,
        uncertainty_factor: Union[Float64, Array],
        uncertainty_type: Literal["symmetric", "asymmetric"] = "symmetric",
        T_range: Optional[Tuple[Float64, Float64]] = None,
        n_T: int = 100,
        lower_factor: Optional[Float64] = None,
    ) -> None:
        self.rate_constant = rate_constant
        self.uncertainty_factor = uncertainty_factor

        if uncertainty_type not in ["symmetric", "asymmetric"]:
            raise ValueError(f"Uncertainty type must be 'symmetric' or 'asymmetric', got {uncertainty_type}")
        self.uncertainty_type = uncertainty_type

        if T_range is None:
            T_range = (300.0, 2500.0)

        if T_range[0] <= 0 or T_range[1] <= T_range[0]:
            raise ValueError(f"Invalid temperature range: {T_range}")

        self.T_range_array = jnp.linspace(T_range[0], T_range[1], n_T)

        self.lower_factor = lower_factor if lower_factor is not None else uncertainty_factor

    def _compute_symmetric_boundaries(self) -> Tuple[Array, Array]:
        """Calculate symmetric logarithmic boundaries."""
        k_lb = self.rate_constant / (10**self.uncertainty_factor)
        k_ub = self.rate_constant * (10**self.uncertainty_factor)

        return self._fit_boundaries(k_lb, k_ub)

    def _compute_asymmetric_boundaries(self) -> Tuple[Array, Array]:
        """Calculate asymmetric boundaries using separate factors."""
        k_lb = self.rate_constant / (10**self.lower_factor)
        k_ub = self.rate_constant * (10**self.uncertainty_factor)

        return self._fit_boundaries(k_lb, k_ub)

    def _fit_boundaries(self, k_lb: Array, k_ub: Array) -> Tuple[Array, Array]:
        """Fit Arrhenius parameters to the boundary rate constants."""
        lnA_lb, beta_lb, EaR_lb = refit_arrhenius(k_lb, self.T_range_array, three_params=True)
        lnA_ub, beta_ub, EaR_ub = refit_arrhenius(k_ub, self.T_range_array, three_params=True)

        lb_params = jnp.array([jnp.exp(lnA_lb), beta_lb, EaR_lb * constants.R_cal_mol])
        ub_params = jnp.array([jnp.exp(lnA_ub), beta_ub, EaR_ub * constants.R_cal_mol])

        return lb_params, ub_params

    def compute_boundaries(self) -> Tuple[Array, Array]:
        """
        Compute the upper and lower boundaries for the rate constants.

        Returns:
            Tuple[Array, Array]: Lower and upper bound parameters as [A, n, Ea/R]
        """
        if self.uncertainty_type == "symmetric":
            return self._compute_symmetric_boundaries()
        elif self.uncertainty_type == "asymmetric":
            return self._compute_asymmetric_boundaries()
        else:
            raise ValueError(f"Unknown uncertainty type {self.uncertainty_type}")

    def get_boundary_rate_constants(self, temperatures: Array) -> Dict[str, Array]:
        lb_params, ub_params = self.compute_boundaries()

        lb_params = jnp.array([lb_params[0], lb_params[1], lb_params[2]])
        ub_params = jnp.array([ub_params[0], ub_params[1], ub_params[2]])

        lb_arrhenius = Arrhenius(name="", params=lb_params)
        ub_arrhenius = Arrhenius(name="", params=ub_params)

        k_lb = lb_arrhenius.kinetic_constant(temperatures)
        k_ub = ub_arrhenius.kinetic_constant(temperatures)

        return {"lower": k_lb, "upper": k_ub, "nominal": self.rate_constant}
