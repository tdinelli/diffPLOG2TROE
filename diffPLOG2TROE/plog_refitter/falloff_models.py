from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array

from ..kinetic_constants import FallOff
from ..physical_constants import constants


class ModelBuilder:
    def __init__(
        self,
        falloff_type: str,
        name: str = "",
    ):
        self.falloff_type = falloff_type
        self.name = name

    def compute_rate_constants(
        self,
        params: Array,
        param_names: List[str],
        T_eval: Array,
        P_eval: Optional[Array] = None,
    ) -> Array:
        """Compute rate constants for the given parameters."""
        param_dict = {name: value for name, value in zip(param_names, params)}
        falloff_model = self._build_falloff_model(param_dict)
        k_values = falloff_model.kinetic_constant(T_eval, P_eval)[0]

        return k_values

    def _build_falloff_model(self, param_dict: Dict[str, float]) -> FallOff:
        """Build a FallOff model from the current parameters."""
        # Extract high pressure limit parameters
        A_high = jnp.exp(param_dict["A_high"])
        n_high = param_dict["n_high"]
        E_high = param_dict["E_high"] * constants.R_cal_mol
        hpl_params = jnp.array([A_high, n_high, E_high])

        # Extract low pressure limit parameters
        A_low = jnp.exp(param_dict["A_low"])
        n_low = param_dict["n_low"]
        E_low = param_dict["E_low"] * constants.R_cal_mol
        lpl_params = jnp.array([A_low, n_low, E_low])

        # Handle falloff parameters based on type
        if self.falloff_type == "troe":
            falloff_params = jnp.array(
                [param_dict["A"], param_dict["T3"], param_dict["T1"], param_dict.get("T2", 0.0), 0.0]
            )
        elif self.falloff_type == "sri":
            falloff_params = jnp.array(
                [param_dict["a"], param_dict["b"], param_dict["c"], param_dict.get("d", 1.0), param_dict.get("e", 0.0)]
            )
        else:  # lindemann
            falloff_params = jnp.empty(5)

        return FallOff(
            hpl_params=hpl_params,
            lpl_params=lpl_params,
            falloff_params=falloff_params,
            falloff_type=self.falloff_type,
            name=self.name,
        )

    def get_falloff_models(self, optimized_params: Array, param_names: list) -> Union[FallOff, Tuple[FallOff, FallOff]]:
        """Get the falloff model(s) with optimized parameters."""
        param_dict = {name: value for name, value in zip(param_names, optimized_params)}
        return self._build_falloff_model(param_dict)
