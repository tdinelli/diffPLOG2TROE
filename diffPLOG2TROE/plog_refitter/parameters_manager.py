import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array

from ..physical_constants import constants
from ..kinetic_constants import Plog, refit_arrhenius


class ParameterManager:
    def __init__(
        self,
        fitting_mode: str,
        primary_falloff_type: str,
        T_range: Array,
        P_range: Array,
        plog: Plog,
        k_plog: Array,
        logger: logging.Logger,
        secondary_falloff_type: Optional[str] = None,
        param_config: Optional[Dict[str, Union[bool, float, Dict[str, Any]]]] = None,
    ):
        self.fitting_mode = fitting_mode
        self.primary_falloff_type = primary_falloff_type
        self.secondary_falloff_type = secondary_falloff_type if fitting_mode == "duplicate" else None
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
        primary_params = self._get_falloff_param_names(self.primary_falloff_type)

        secondary_params = []
        if self.fitting_mode == "duplicate" and self.secondary_falloff_type:
            secondary_params = self._get_falloff_param_names(self.secondary_falloff_type, prefix="secondary_")

        param_names = primary_params + secondary_params

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
                elif isinstance(config, bool): # Handle cases where config is just a boolean (optimize flag)
                    mask_list.append(config)
                    initial_values.append(None)
                elif isinstance(config, (int, float)): # Assume config is a direct value to use
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

    @staticmethod
    def _get_falloff_param_names(falloff_type: str, prefix: str = "") -> List[str]:
        def add_prefix(name):
            # Add prefix if provided (e.g., "secondary_" for duplicate reaction parameters)
            return f"{prefix}{name}" if prefix else name

        # Base Arrhenius parameters for high and low pressure limits
        base_params = [add_prefix(p) for p in ["A_low", "n_low", "E_low", "A_high", "n_high", "E_high"]]

        # Add falloff-specific parameters
        if falloff_type == "troe":
            return base_params + [add_prefix(p) for p in ["A", "T3", "T1", "T2"]]
        elif falloff_type == "sri":
            return base_params + [add_prefix(p) for p in ["a", "b", "c", "d", "e"]]
        else:  # LINDEMANN
            return base_params

    def _log_parameter_config(
        self, param_names: List[str], param_mask: Array, initial_values: List[Optional[float]]
    ) -> None:
        """Log parameter configuration details."""
        self.logger.info("\nParameters configuration:")

        if self.fitting_mode == "duplicate":
            self.logger.info(f"\nPrimary reaction ({self.primary_falloff_type}):")

        primary_count = len(self._get_falloff_param_names(self.primary_falloff_type))
        for i in range(primary_count):
            name, mask = param_names[i], param_mask[i]
            value = initial_values[i]
            self._log_parameter_entry(name, mask, value)

        # ====================================================================
        # Log secondary parameters if applicable
        # ====================================================================
        if self.fitting_mode == "duplicate" and self.secondary_falloff_type:
            self.logger.info(f"\nSecondary reaction ({self.secondary_falloff_type}):")
            for i in range(primary_count, len(param_names)):
                name, mask = param_names[i], param_mask[i]
                value = initial_values[i]
                self._log_parameter_entry(name, mask, value)

    def _log_parameter_entry(self, name: str, mask: bool, value: Optional[float]) -> None:
        """Log a single parameter configuration entry."""
        if mask and value is not None:
            self.logger.info(f"  {name}: optimizing, starting from {value:.5e}")
        elif mask:
            self.logger.info(f"  {name}: optimizing from default initial value")
        else:
            self.logger.info(f"  {name}: fixed at {value:.5e}")

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

        if self.fitting_mode == "single":
            params_dict = self._estimate_single_reaction_params(
                lnA_low, n_low, EaR_low, lnA_high, n_high, EaR_high, falloff_defaults, params_dict
            )
        else:  # duplicate mode
            # Prepare base rate parameters for duplicate reactions
            # Each reaction gets half the total rate contribution
            params_dict = self._estimate_duplicate_reaction_params(falloff_defaults, params_dict)

        # Convert dictionary to array in the correct parameter order
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
        falloff_type = self.primary_falloff_type
        falloff_params = self._get_falloff_specific_params(falloff_type)

        # Fill in falloff parameters
        offset = 6  # After 6 Arrhenius parameters
        for i, param in enumerate(falloff_params):
            idx = i + offset
            if idx < len(self.initial_values) and self.initial_values[idx] is not None:
                params_dict[param] = self.initial_values[idx]
            else:
                params_dict[param] = falloff_defaults[falloff_type].get(param, 1.0)

        if self.logger:
            self._log_single_reaction_params(params_dict, falloff_type)

        return params_dict

    def _estimate_duplicate_reaction_params(self, falloff_defaults, params_dict):
        """Process parameters for duplicate reactions."""
        # Estimate high pressure limit from highest pressure PLOG split equally
        split_high_k = self.k_plog[-1] / 2.0
        estimated_lnA_high, estimated_n_high, estimated_EaR_high = refit_arrhenius(split_high_k, self.T_range, True)

        # Estimate low pressure limit from lowest pressure PLOG split equally
        M = (self.plog.p_levels[0] / (constants.R_L_atm_K_mol * self.T_range)) * jnp.float64(0.001)
        split_low_k = (self.k_plog[0] / M) / 2.0
        estimated_lnA_low, estimated_n_low, estimated_EaR_low = refit_arrhenius(split_low_k, self.T_range, True)

        # Initialize primary reaction parameters
        primary_type = self.primary_falloff_type
        primary_arrh_params = [
            ("A_low", estimated_lnA_low),
            ("n_low", estimated_n_low),
            ("E_low", estimated_EaR_low),
            ("A_high", estimated_lnA_high),
            ("n_high", estimated_n_high),
            ("E_high", estimated_EaR_high),
        ]

        # Fill in primary reaction Arrhenius parameters
        for i, (name, default_value) in enumerate(primary_arrh_params):
            if self.initial_values[i] is not None:
                if name.startswith("A"):
                    params_dict[name] = jnp.log(self.initial_values[i])
                elif name.startswith("E"):
                    params_dict[name] = self.initial_values[i] / constants.R_cal_mol
                else:
                    params_dict[name] = self.initial_values[i]
            else:
                params_dict[name] = default_value

        # Process primary falloff-specific parameters
        primary_falloff_params = self._get_falloff_specific_params(primary_type)

        # Fill in primary falloff parameters
        offset = 6  # After 6 Arrhenius parameters
        for i, param in enumerate(primary_falloff_params):
            idx = i + offset
            if idx < len(self.initial_values) and self.initial_values[idx] is not None:
                params_dict[param] = self.initial_values[idx]
            else:
                params_dict[param] = falloff_defaults[primary_type].get(param, 1.0)

        # Now handle secondary reaction (if in duplicate mode)
        if self.secondary_falloff_type:
            secondary_type = self.secondary_falloff_type

            # Get number of parameters in primary reaction to determine offset
            primary_param_count = 6 + len(primary_falloff_params)

            # Initialize secondary reaction Arrhenius parameters
            secondary_arrh_params = [
                ("secondary_A_low", estimated_lnA_low),
                ("secondary_n_low", estimated_n_low),
                ("secondary_E_low", estimated_EaR_low),
                ("secondary_A_high", estimated_lnA_high),
                ("secondary_n_high", estimated_n_high),
                ("secondary_E_high", estimated_EaR_high),
            ]

            # Fill in secondary reaction Arrhenius parameters
            for i, (name, default_value) in enumerate(secondary_arrh_params):
                idx = i + primary_param_count
                if idx < len(self.initial_values) and self.initial_values[idx] is not None:
                    if name.startswith("secondary_A"):
                        params_dict[name] = jnp.log(self.initial_values[idx])
                    elif name.startswith("secondary_E"):
                        params_dict[name] = self.initial_values[idx] / constants.R_cal_mol
                    else:
                        params_dict[name] = self.initial_values[idx]
                else:
                    params_dict[name] = default_value

            # Process secondary falloff-specific parameters
            secondary_falloff_params = self._get_falloff_specific_params(secondary_type, prefix="secondary_")

            # Fill in secondary falloff parameters
            offset = primary_param_count + 6  # After primary params and 6 secondary Arrhenius params
            for i, param in enumerate(secondary_falloff_params):
                idx = i + offset
                if idx < len(self.initial_values) and self.initial_values[idx] is not None:
                    params_dict[param] = self.initial_values[idx]
                else:
                    # Extract the base parameter name (without prefix)
                    base_param = param.replace("secondary_", "")
                    params_dict[param] = falloff_defaults[secondary_type].get(base_param, 1.0)

        # Log the estimated parameters
        if self.logger:
            self._log_duplicate_reaction_params(params_dict)

        return params_dict

    def _get_falloff_specific_params(self, falloff_type, prefix=""):
        """Get the falloff-specific parameter names for a given falloff type."""
        if falloff_type == "troe":
            return [f"{prefix}A", f"{prefix}T3", f"{prefix}T1", f"{prefix}T2"]
        elif falloff_type == "sri":
            return [f"{prefix}a", f"{prefix}b", f"{prefix}c", f"{prefix}d", f"{prefix}e"]
        else:  # lindemann
            return []

    def _log_single_reaction_params(self, params_dict, falloff_type):
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

    def _log_duplicate_reaction_params(self, params_dict):
        """Log the estimated parameters for duplicate reactions."""
        # Log primary reaction
        primary_type = self.primary_falloff_type
        self.logger.info(f"\n{primary_type.capitalize()} reaction parameters (primary):")
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

        # Log falloff-specific parameters for primary reaction
        if primary_type == "troe":
            self.logger.info(
                "  Troe parameters (A, T3, T1, T2): {:.3f}, {:.3e}, {:.3e}, {:.3e}\n".format(
                    params_dict["A"], params_dict["T3"], params_dict["T1"], params_dict["T2"]
                )
            )
        elif primary_type == "sri":
            self.logger.info(
                "  SRI parameters (a, b, c, d, e): {:.3f}, {:.3e}, {:.3e}, {:.3f}, {:.3f}\n".format(
                    params_dict["a"], params_dict["b"], params_dict["c"], params_dict["d"], params_dict["e"]
                )
            )
        else:
            self.logger.info("")  # Add empty line

        # Log secondary reaction
        if self.secondary_falloff_type:
            secondary_type = self.secondary_falloff_type
            self.logger.info(f"{secondary_type.capitalize()} reaction parameters (secondary):")
            self.logger.info(
                "  High pressure limit (A, n, Ea): {:.3e}, {:.3f}, {:.3e}".format(
                    jnp.exp(params_dict["secondary_A_high"]),
                    params_dict["secondary_n_high"],
                    params_dict["secondary_E_high"] * constants.R_cal_mol,
                )
            )
            self.logger.info(
                "  Low pressure limit (A, n, Ea): {:.3e}, {:.3f}, {:.3e}".format(
                    jnp.exp(params_dict["secondary_A_low"]),
                    params_dict["secondary_n_low"],
                    params_dict["secondary_E_low"] * constants.R_cal_mol,
                )
            )

            # Log falloff-specific parameters for secondary reaction
            if secondary_type == "troe":
                self.logger.info(
                    "  Troe parameters (A, T3, T1, T2): {:.3f}, {:.3e}, {:.3e}, {:.3e}\n".format(
                        params_dict["secondary_A"],
                        params_dict["secondary_T3"],
                        params_dict["secondary_T1"],
                        params_dict["secondary_T2"],
                    )
                )
            elif secondary_type == "sri":
                self.logger.info(
                    "  SRI parameters (a, b, c, d, e): {:.3f}, {:.3e}, {:.3e}, {:.3f}, {:.3f}\n".format(
                        params_dict["secondary_a"],
                        params_dict["secondary_b"],
                        params_dict["secondary_c"],
                        params_dict["secondary_d"],
                        params_dict["secondary_e"],
                    )
                )

    def _group_parameters_for_bounds(self) -> Dict[str, List[int]]:
        param_groups = {}

        def find_params(pattern, prefix=""):
            # Helper function to find parameter indices by pattern
            return [i for i, name in enumerate(self.param_names) if name.startswith(prefix) and pattern in name]

        if self.fitting_mode == "single":
            # Single reaction groups
            param_groups["Arrhenius parameters (high pressure limit)"] = find_params("_high")
            param_groups["Arrhenius parameters (low pressure limit)"] = find_params("_low")

            # Falloff-specific parameters
            if self.primary_falloff_type == "troe":
                param_groups["Troe parameters"] = [
                    i for i, name in enumerate(self.param_names) if name in ["A", "T3", "T1", "T2"]
                ]
            elif self.primary_falloff_type == "sri":
                param_groups["SRI parameters"] = [
                    i for i, name in enumerate(self.param_names) if name in ["a", "b", "c", "d", "e"]
                ]
        else:
            # Duplicate reactions - primary reaction
            param_groups["Primary reaction (high pressure limit)"] = find_params("_high")
            param_groups["Primary reaction (low pressure limit)"] = find_params("_low")

            # Primary falloff parameters
            falloff_params = self._get_falloff_specific_params(self.primary_falloff_type)
            if falloff_params:
                group_name = f"Primary reaction {self.primary_falloff_type} parameters"
                param_groups[group_name] = [i for i, name in enumerate(self.param_names) if name in falloff_params]

            # Secondary reaction parameters
            if self.secondary_falloff_type:
                param_groups["Secondary reaction (high pressure limit)"] = find_params("_high", "secondary_")
                param_groups["Secondary reaction (low pressure limit)"] = find_params("_low", "secondary_")

                # Secondary falloff parameters
                falloff_params = self._get_falloff_specific_params(self.secondary_falloff_type, prefix="secondary_")
                if falloff_params:
                    group_name = f"Secondary reaction {self.secondary_falloff_type} parameters"
                    param_groups[group_name] = [i for i, name in enumerate(self.param_names) if name in falloff_params]

        return param_groups

    def _calculate_parameter_bounds(
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
        if param_name.startswith("A_") or (param_name == "secondary_A"):
            # A-factors (pre-exponential)
            # A-factors are usually positive and can vary by many orders of magnitude
            # Ensure a reasonable lower bound
            lower_bound = max(lower_bound, 1.0e-30)

            # For low-pressure limit, often higher
            if "_low" in param_name:
                lower_bound = max(lower_bound, 1.0e-20)
        elif param_name.startswith("n_"):
            # n values (temperature exponents)
            # n values typically between -3 and +3
            lower_bound = max(lower_bound, -3.0)
            upper_bound = min(upper_bound, 3.0)

            # For low-pressure limit, often negative
            if "_low" in param_name:
                lower_bound = max(lower_bound, -1.5)
                upper_bound = min(upper_bound, 1.0)

            # For high-pressure limit, often closer to zero
            if "_high" in param_name:
                lower_bound = max(lower_bound, -1.0)
                upper_bound = min(upper_bound, 1.0)

        # E values (activation energies)
        elif param_name.startswith("E_"):
            # Activation energies can be zero or positive
            # For some reactions, negative values are possible but rare
            lower_bound = max(lower_bound, -10000.0)  # Allow slightly negative

            # Upper bound based on physical considerations
            upper_bound = min(upper_bound, 200000.0)  # Reasonable upper limit

        # Troe parameters
        elif param_name == "A" or param_name == "secondary_A":
            # Troe alpha parameter typically between 0 and 1
            lower_bound = 0.0
            upper_bound = 1.0

        elif param_name == "T3" or param_name == "secondary_T3":
            # T3 parameter (T*) typically positive
            lower_bound = max(lower_bound, 1.0)

        elif param_name == "T1" or param_name == "secondary_T1":
            # T1 parameter (T***) typically positive
            lower_bound = max(lower_bound, 1.0)

        elif param_name == "T2" or param_name == "secondary_T2":
            # T2 parameter (T**) typically positive or zero
            lower_bound = max(lower_bound, 0.0)

        # SRI parameters
        elif param_name == "a" or param_name == "secondary_a":
            # 'a' typically positive
            lower_bound = max(lower_bound, 0.1)

        elif param_name == "b" or param_name == "secondary_b":
            # 'b' typically positive
            lower_bound = max(lower_bound, 0.0)

        elif param_name == "c" or param_name == "secondary_c":
            # 'c' typically positive
            lower_bound = max(lower_bound, 0.0)

        # For parameters not handled above, use the default bounds

        return lower_bound, upper_bound

    def calculate_optimization_bounds(
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
                lb, ub = self._calculate_parameter_bounds(param_name, param_value, uncertainty_factor, uncertainty_type)

                lower_bounds = lower_bounds.at[idx].set(lb)
                upper_bounds = upper_bounds.at[idx].set(ub)

                self.logger.info(f"  {param_name}: {lb:.3e} ≤ {param_value:.3e} ≤ {ub:.3e}")

        return lower_bounds, upper_bounds
