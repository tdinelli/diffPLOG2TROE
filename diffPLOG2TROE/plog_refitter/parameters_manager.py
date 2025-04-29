import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array

from ..parametrization import Plog, refit_arrhenius
from ..physical_constants import constants


class ParameterManager:
    def __init__(
        self,
        falloff_type: str,
        T_range: Array,
        P_range: Array,
        plog: Plog,
        k_plog: Array,
        logger: logging.Logger,
        param_config: Optional[Dict[str, Union[bool, float, Dict[str, Any]]]] = None,
    ):
        self.falloff_type = falloff_type
        self.T_range = T_range
        self.P_range = P_range
        self.plog = plog
        self.k_plog = k_plog
        self.logger = logger

        # ====================================================================
        # Configure parameters
        # ====================================================================
        self.param_names, self.param_mask, self.initial_values = self._configure_parameters(param_config)

    def _configure_parameters(self, param_config: Optional[Dict[str, Any]]):
        param_names = self._get_falloff_param_names(self.falloff_type)

        if param_config is None:
            param_mask = jnp.ones(len(param_names), dtype=bool)
            initial_values = [None] * len(param_names)
        else:  # Some of the parameters have a special treatment
            mask_list = []
            initial_values = []

            for name in param_names:
                config = param_config.get(name, {"optimize": True})

                if isinstance(config, dict):
                    optimize = config.get("optimize", True)
                    mask_list.append(optimize)
                    initial_values.append(float(config["value"]) if "value" in config else None)
                elif isinstance(config, bool):  # Handle cases where config is just a boolean (optimize flag)
                    mask_list.append(config)
                    initial_values.append(None)
                elif isinstance(config, (int, float)):  # Assume config is a direct value to use
                    mask_list.append(False)  # Don't optimize
                    initial_values.append(float(config))
                else:
                    raise ValueError(f"Invalid configuration for parameter {name}: {config}")

            param_mask = jnp.array(mask_list, dtype=bool)

        # ====================================================================
        # Log parameter configuration if logger is available
        # ====================================================================
        if self.logger:
            self._log_parameter_config(param_names, param_mask, initial_values)

        return param_names, param_mask, initial_values

    def estimate_initial_params(self, plog_data) -> Array:
        """Estimate initial parameter values based on PLOG data."""
        self.logger.info("\nEstimating initial parameters:")
        params_dict = {}

        T_mean = jnp.mean(self.T_range)

        # ====================================================================
        # Extract high and low pressure limits from PLOG data. Here for the
        # high pressure limit we are using the last value provided in the plog
        # keep in mind that this is not always the best estimate so it is
        # maybe better to have a user defined one. The low pressure limit
        # is estimated using the lowest pressure value divided by the total
        # concentration.
        # ====================================================================
        # HPL
        lnA_high, n_high, EaR_high = plog_data.k_levels[-1].lnA, plog_data.k_levels[-1].n, plog_data.k_levels[-1].EaR

        # ====================================================================
        # LPL
        M = (plog_data.p_levels[0] / (constants.R_L_atm_K_mol * self.T_range)) * jnp.float64(0.001)
        low_k = self.k_plog[0] / M
        lnA_low, n_low, EaR_low = refit_arrhenius(low_k, self.T_range, True)

        # Default values for different falloff parameters based on temperature range
        falloff_defaults = {
            "troe": {"A": 0.5, "T3": T_mean * 0.7, "T1": T_mean * 0.2, "T2": T_mean * 1.5},
            "sri": {"a": 1.0, "b": 0.5 * T_mean, "c": T_mean, "d": 1.0, "e": 0.0},
            "lindemann": {},
        }

        params_dict = self._estimate_single_reaction_params(
            lnA_low, n_low, EaR_low, lnA_high, n_high, EaR_high, falloff_defaults, params_dict
        )
        params = jnp.array([params_dict[name] for name in self.param_names], dtype=jnp.float64)

        return params

    def _estimate_single_reaction_params(
        self, lnA_low, n_low, EaR_low, lnA_high, n_high, EaR_high, falloff_defaults, params_dict
    ):
        """Process parameters for a single reaction."""
        # Process Arrhenius parameters
        arrh_params = [
            ("A_low", lnA_low),
            ("n_low", n_low),
            ("E_low", EaR_low),
            ("A_high", lnA_high),
            ("n_high", n_high),
            ("E_high", EaR_high),
        ]

        # Fill in base Arrhenius parameters
        for i, (name, default_value) in enumerate(arrh_params):
            if self.initial_values[i] is not None:
                if name.startswith("A"):
                    params_dict[name] = jnp.log(self.initial_values[i])
                elif name.startswith("E"):
                    params_dict[name] = self.initial_values[i] / constants.R_cal_mol
                else:
                    params_dict[name] = self.initial_values[i]
            else:
                params_dict[name] = default_value

        # Process falloff-specific parameters
        falloff_params = self._get_falloff_specific_params(self.falloff_type)

        # Fill in falloff parameters
        offset = 6  # After 6 Arrhenius parameters
        for i, param in enumerate(falloff_params):
            idx = i + offset
            if idx < len(self.initial_values) and self.initial_values[idx] is not None:
                params_dict[param] = self.initial_values[idx]
            else:
                params_dict[param] = falloff_defaults[self.falloff_type].get(param, 1.0)

        if self.logger:
            self._log_reaction_params(params_dict, self.falloff_type)

        return params_dict

    def _log_reaction_params(self, params_dict, falloff_type):
        """Log the estimated parameters for a single reaction."""
        if falloff_type == "troe":
            self.logger.info("Troe parameters")
            self.logger.info(
                "  High pressure limit (A, n, Ea): {:.3e}, {:.3f}, {:.3e}".format(
                    jnp.exp(params_dict["A_high"]),
                    params_dict["n_high"],
                    params_dict["E_high"] * constants.R_cal_mol,
                )
            )
            self.logger.info(
                "  Low pressure limit (A, n, Ea): {:.3e}, {:.3f}, {:.3e}".format(
                    jnp.exp(params_dict["A_low"]),
                    params_dict["n_low"],
                    params_dict["E_low"] * constants.R_cal_mol,
                )
            )
            self.logger.info(
                "  Troe parameters (A, T3, T1, T2): {:.3f}, {:.3e}, {:.3e}, {:.3e}\n".format(
                    params_dict["A"], params_dict["T3"], params_dict["T1"], params_dict["T2"]
                )
            )
        elif falloff_type == "sri":
            self.logger.info("\nSRI parameters:")
            self.logger.info(
                "  High pressure limit (A, n, Ea): {:.3e}, {:.3f}, {:.3e}".format(
                    jnp.exp(params_dict["A_high"]),
                    params_dict["n_high"],
                    params_dict["E_high"] * constants.R_cal_mol,
                )
            )
            self.logger.info(
                "  Low pressure limit (A, n, Ea): {:.3e}, {:.3f}, {:.3e}".format(
                    jnp.exp(params_dict["A_low"]),
                    params_dict["n_low"],
                    params_dict["E_low"] * constants.R_cal_mol,
                )
            )
            self.logger.info(
                "  SRI parameters (a, b, c, d, e): {:.3f}, {:.3e}, {:.3e}, {:.3f}, {:.3f}\n".format(
                    params_dict["a"], params_dict["b"], params_dict["c"], params_dict["d"], params_dict["e"]
                )
            )
        else:  # lindemann
            self.logger.info("\nLindemann parameters:")
            self.logger.info(
                "  High pressure limit (A, n, Ea): {:.3e}, {:.3f}, {:.3e}".format(
                    jnp.exp(params_dict["A_high"]),
                    params_dict["n_high"],
                    params_dict["E_high"] * constants.R_cal_mol,
                )
            )
            self.logger.info(
                "  Low pressure limit (A, n, Ea): {:.3e}, {:.3f}, {:.3e}".format(
                    jnp.exp(params_dict["A_low"]),
                    params_dict["n_low"],
                    params_dict["E_low"] * constants.R_cal_mol,
                )
            )

    def _log_parameter_config(
        self, param_names: List[str], param_mask: Array, initial_values: Optional[List[float]]
    ) -> None:
        """Log parameter configuration details."""
        self.logger.info("\nParameters configuration:")

        primary_count = len(self._get_falloff_param_names(self.falloff_type))
        if isinstance(initial_values, list):
            for i in range(primary_count):
                name, mask = param_names[i], param_mask[i]
                value = initial_values[i]
                self._log_parameter_entry(name, bool(mask), value)

    def _log_parameter_entry(self, name: str, mask: bool, value: Optional[float]) -> None:
        """Log a single parameter configuration entry."""
        if mask and value is not None:
            self.logger.info(f"  {name}: optimizing, starting from {value:.5e}")
        elif mask:
            self.logger.info(f"  {name}: optimizing from default initial value")
        else:
            self.logger.info(f"  {name}: fixed at {value:.5e}")

    def _compute_bounds(
        self, param_name: str, param_value: float, uncertainty_factor: float, uncertainty_type: str
    ) -> Tuple[float, float]:
        if uncertainty_type == "symmetric":
            lower_bound = param_value / (10**uncertainty_factor)
            upper_bound = param_value * (10**uncertainty_factor)
        else:
            # In asymmetric case, we could implement different factors for upper and lower bounds if needed
            lower_bound = param_value / (10**uncertainty_factor)
            upper_bound = param_value * (10**uncertainty_factor)

        # Apply parameter-specific constraints
        if param_name.startswith("A_"):
            lower_bound = max(lower_bound, 1.0e-30)
        elif param_name.startswith("n_"):
            lower_bound = max(lower_bound, -3.0)
            upper_bound = min(upper_bound, 3.0)
        elif param_name.startswith("E_"):
            lower_bound = max(lower_bound, -10000.0)  # Allow slightly negative
            upper_bound = min(upper_bound, 200000.0)  # Reasonable upper limit

        # Troe parameters
        elif param_name == "A":
            lower_bound = 0.0
            upper_bound = 1.0
        elif param_name == "T3":
            # T3 parameter (T*) typically positive
            lower_bound = max(lower_bound, 1.0)
        elif param_name == "T1":
            # T1 parameter (T***) typically positive
            lower_bound = max(lower_bound, 1.0)
        elif param_name == "T2":
            # T2 parameter (T**) typically positive or zero
            lower_bound = max(lower_bound, 0.0)

        # SRI parameters
        elif param_name == "a":
            lower_bound = max(lower_bound, 0.1)
        elif param_name == "b":
            lower_bound = max(lower_bound, 0.0)
        elif param_name == "c":
            lower_bound = max(lower_bound, 0.0)

        if lower_bound < upper_bound:
            return lower_bound, upper_bound
        else:
            return upper_bound, lower_bound

    def _group_parameters_for_bounds(self) -> Dict[str, List[int]]:
        param_groups = {}
        def find_params(pattern):
            return [i for i, name in enumerate(self.param_names) if pattern in name]

        param_groups["Arrhenius parameters (high pressure limit)"] = find_params("_high")
        param_groups["Arrhenius parameters (low pressure limit)"] = find_params("_low")

        # Falloff-specific parameters
        if self.falloff_type == "troe":
            param_groups["Troe parameters"] = [
                i for i, name in enumerate(self.param_names) if name in ["A", "T3", "T1", "T2"]
            ]
        elif self.falloff_type == "sri":
            param_groups["SRI parameters"] = [
                i for i, name in enumerate(self.param_names) if name in ["a", "b", "c", "d", "e"]
            ]

        return param_groups
    def get_parameters_bounds(
        self, initial_params: Array, uncertainty_factor: float = 1.0, uncertainty_type: str = "symmetric"
    ) -> Tuple[Array, Array]:
        self.logger.info(f"\nCalculating optimization bounds (uncertainty factor: {uncertainty_factor})")

        # Initialize bounds arrays with initial values
        lower_bounds = initial_params.copy()
        upper_bounds = initial_params.copy()

        # Process parameters in logical groups
        param_groups = self._group_parameters_for_bounds()

        for group_name, param_indices in param_groups.items():
            self.logger.info(f"\nBounds for {group_name}:")

            # Process each parameter in the group
            for idx in param_indices:
                param_name = self.param_names[idx]
                param_value = initial_params[idx]

                # Skip bound calculation for non-optimized parameters
                if not self.param_mask[idx]:
                    # Fixed parameter - set tight bounds
                    lower_bounds = lower_bounds.at[idx].set(param_value * 0.999)
                    upper_bounds = upper_bounds.at[idx].set(param_value * 1.001)
                    self.logger.info(f"  {param_name}: fixed at {param_value:.3e}")
                    continue

                # Calculate bounds based on parameter type
                lb, ub = self._compute_bounds(param_name, param_value, uncertainty_factor, uncertainty_type)

                lower_bounds = lower_bounds.at[idx].set(lb)
                upper_bounds = upper_bounds.at[idx].set(ub)

                self.logger.info(f"  {param_name}: {lb:.3e} ≤ {param_value:.3e} ≤ {ub:.3e}")

        return lower_bounds, upper_bounds

    @staticmethod
    def _get_falloff_param_names(falloff_type: str) -> List[str]:
        # Base Arrhenius parameters for high and low pressure limits
        base_params = ["A_low", "n_low", "E_low", "A_high", "n_high", "E_high"]

        # Add falloff-specific parameters
        if falloff_type == "troe":
            return base_params + ["A", "T3", "T1", "T2"]
        elif falloff_type == "sri":
            return base_params + ["a", "b", "c", "d", "e"]
        else:  # LINDEMANN
            return base_params

    @staticmethod
    def _get_falloff_specific_params(falloff_type):
        """Get the falloff-specific parameter names for a given falloff type."""
        if falloff_type == "troe":
            return ["A", "T3", "T1", "T2"]
        elif falloff_type == "sri":
            return ["a", "b", "c", "d", "e"]
        else:  # lindemann
            return []
