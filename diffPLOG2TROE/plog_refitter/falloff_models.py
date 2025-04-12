from typing import Dict, Optional, Tuple, Union, List
import jax.numpy as jnp
from jaxtyping import Array

from ..physical_constants import constants
from ..kinetic_constants import FallOff


class ModelBuilder:
    def __init__(
        self,
        fitting_mode: str,
        primary_falloff_type: str,
        secondary_falloff_type: Optional[str] = None,
        name: str = "",
    ):
        self.fitting_mode = fitting_mode
        self.primary_falloff_type = primary_falloff_type
        self.secondary_falloff_type = secondary_falloff_type if fitting_mode == "duplicate" else None
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
        if self.fitting_mode == "single":
            falloff_model = self._build_falloff_model(param_dict)
            k_values = falloff_model.kinetic_constant(T_eval, P_eval)
        else:  # duplicate mode
            primary_falloff, secondary_falloff = self._build_duplicate_falloff_models(param_dict)

            k_primary = primary_falloff.kinetic_constant(T_eval, P_eval)
            k_secondary = secondary_falloff.kinetic_constant(T_eval, P_eval)
            k_values = k_primary + k_secondary

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
        if self.primary_falloff_type == "troe":
            falloff_type = 1  # Troe type index
            falloff_params = jnp.array(
                [param_dict["A"], param_dict["T3"], param_dict["T1"], param_dict.get("T2", 0.0), 0.0]
            )
        elif self.primary_falloff_type == "sri":
            falloff_type = 2  # SRI type index
            falloff_params = jnp.array(
                [param_dict["a"], param_dict["b"], param_dict["c"], param_dict.get("d", 1.0), param_dict.get("e", 0.0)]
            )
        else:  # lindemann
            falloff_type = 0
            falloff_params = jnp.empty(5)

        return FallOff(
            hpl_params=hpl_params,
            lpl_params=lpl_params,
            falloff_params=falloff_params,
            falloff_type=falloff_type,
            name=self.name,
        )

    def _build_duplicate_falloff_models(self, param_dict: Dict[str, float]) -> Tuple[FallOff, FallOff]:
        """Build two FallOff models for duplicate reaction mode."""
        # Primary reaction parameters
        primary_hpl = jnp.array(
            [jnp.exp(param_dict["A_high"]), param_dict["n_high"], param_dict["E_high"] * constants.R_cal_mol]
        )

        primary_lpl = jnp.array(
            [jnp.exp(param_dict["A_low"]), param_dict["n_low"], param_dict["E_low"] * constants.R_cal_mol]
        )

        # Primary falloff parameters
        if self.primary_falloff_type == "troe":
            primary_type = 1
            primary_falloff_params = jnp.array(
                [param_dict["A"], param_dict["T3"], param_dict["T1"], param_dict.get("T2", 0.0), 0.0]
            )
        elif self.primary_falloff_type == "sri":
            primary_type = 2
            primary_falloff_params = jnp.array(
                [param_dict["a"], param_dict["b"], param_dict["c"], param_dict.get("d", 1.0), param_dict.get("e", 0.0)]
            )
        else:  # lindemann
            primary_type = 0
            primary_falloff_params = jnp.empty(5)

        # Secondary reaction parameters
        secondary_hpl = jnp.array(
            [
                jnp.exp(param_dict["secondary_A_high"]),
                param_dict["secondary_n_high"],
                param_dict["secondary_E_high"] * constants.R_cal_mol,
            ]
        )

        secondary_lpl = jnp.array(
            [
                jnp.exp(param_dict["secondary_A_low"]),
                param_dict["secondary_n_low"],
                param_dict["secondary_E_low"] * constants.R_cal_mol,
            ]
        )

        # Secondary falloff parameters
        if self.secondary_falloff_type == "troe":
            secondary_type = 1
            secondary_falloff_params = jnp.array(
                [
                    param_dict["secondary_A"],
                    param_dict["secondary_T3"],
                    param_dict["secondary_T1"],
                    param_dict.get("secondary_T2", 0.0),
                    0.0,
                ]
            )
        elif self.secondary_falloff_type == "sri":
            secondary_type = 2
            secondary_falloff_params = jnp.array(
                [
                    param_dict["secondary_a"],
                    param_dict["secondary_b"],
                    param_dict["secondary_c"],
                    param_dict.get("secondary_d", 1.0),
                    param_dict.get("secondary_e", 0.0),
                ]
            )
        else:  # lindemann
            secondary_type = 0
            secondary_falloff_params = jnp.empty(5)

        primary_falloff = FallOff(
            hpl_params=primary_hpl,
            lpl_params=primary_lpl,
            falloff_params=primary_falloff_params,
            falloff_type=primary_type,
            name=f"{self.name}_primary",
        )

        secondary_falloff = FallOff(
            hpl_params=secondary_hpl,
            lpl_params=secondary_lpl,
            falloff_params=secondary_falloff_params,
            falloff_type=secondary_type,
            name=f"{self.name}_secondary",
        )

        return primary_falloff, secondary_falloff

    def get_falloff_models(self, optimized_params: Array, param_names: list) -> Union[FallOff, Tuple[FallOff, FallOff]]:
        """Get the falloff model(s) with optimized parameters."""
        param_dict = {name: value for name, value in zip(param_names, optimized_params)}

        if self.fitting_mode == "single":
            return self._build_falloff_model(param_dict)
        else:
            return self._build_duplicate_falloff_models(param_dict)
