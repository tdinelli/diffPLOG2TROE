import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
import optax
from jax import jit, value_and_grad
from jaxtyping import Array, Float64

from .physical_constants import constants
from .rate_constants import FallOff, Plog, refit_arrhenius


class PlogRefitter(eqx.Module):
    plog: Plog
    param_mask: Array
    param_names: List[str]
    T_range: Array
    P_range: Array
    k_plog: Array
    logger: logging.Logger
    initial_values: List[Optional[float]]
    loss_name: str
    primary_falloff_type: str
    secondary_falloff_type: Optional[str]
    fitting_mode: str

    def __init__(
        self,
        plog_dict: Dict[str, Any],
        T_range: Tuple[Float64, Float64],
        P_range: Tuple[Float64, Float64],
        n_T: int = 100,
        n_P: int = 100,
        param_config: Optional[Dict[str, Union[bool, float, Dict[str, Any]]]] = None,
        fitting_mode: str = "single",
        primary_falloff_type: str = "troe",
        secondary_falloff_type: Optional[str] = "lindemann",
        loss_name: str = "bo",
        log_name: str = "refitter.log",
    ) -> None:
        self.fitting_mode = fitting_mode
        self.primary_falloff_type = primary_falloff_type
        self.secondary_falloff_type = secondary_falloff_type if fitting_mode == "duplicate" else None
        self.loss_name = loss_name

        # Set up logging
        self.logger = self._setup_logging(log_name)
        self._log_initialization(T_range, P_range)

        # Generate training data from the PLOG expression
        self.plog = Plog(plog_dict)
        self.T_range = jnp.linspace(T_range[0], T_range[1], n_T)
        self.P_range = jnp.logspace(jnp.log10(P_range[0]), jnp.log10(P_range[1]), n_P)
        self.k_plog = self.plog.kinetic_constant(self.T_range, self.P_range)

        # Configure parameters based on the optimization strategy
        self.param_names, self.param_mask, self.initial_values = self._configure_parameters(param_config)

    def _compute_rate_constants(self, params: Array, T_eval: Array, P_eval: Optional[Array] = None) -> Array:
        if self.fitting_mode == "single":
            falloff_model = self._build_falloff_model(params)
            k_values = falloff_model.kinetic_constant(T_eval, P_eval)
        else:  # duplicate mode
            primary_falloff, secondary_falloff = self._build_duplicate_falloff_models(params)

            k_primary = primary_falloff.kinetic_constant(T_eval, P_eval)
            k_secondary = secondary_falloff.kinetic_constant(T_eval, P_eval)
            k_values = k_primary + k_secondary

        return k_values

    def _build_falloff_model(self, params: Array) -> FallOff:
        """Build a FallOff model from the current parameters."""
        param_dict = {name: value for name, value in zip(self.param_names, params)}

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
            name=self.plog.name,
        )

    def _build_duplicate_falloff_models(self, params: Array) -> Tuple[FallOff, FallOff]:
        """Build two FallOff models for duplicate reaction mode."""
        param_dict = {name: value for name, value in zip(self.param_names, params)}

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
            name=f"{self.plog.name}_primary",
        )

        secondary_falloff = FallOff(
            hpl_params=secondary_hpl,
            lpl_params=secondary_lpl,
            falloff_params=secondary_falloff_params,
            falloff_type=secondary_type,
            name=f"{self.plog.name}_secondary",
        )

        return primary_falloff, secondary_falloff

    def _loss_function(self, params: Array) -> Float64:
        masked_params = jnp.where(self.param_mask, params, jnp.array([p for p in self.initial_values]))
        k_fitted = self._compute_rate_constants(masked_params, self.T_range, self.P_range)
        k_plog = self.k_plog

        if self.loss_name == "mse":
            # Mean squared error
            loss = jnp.mean((k_fitted - k_plog) ** 2)
        elif self.loss_name == "rmse":
            # Root mean squared error
            loss = jnp.sqrt(jnp.mean((k_fitted - k_plog) ** 2))
        elif self.loss_name == "mae":
            # Mean absolute error
            loss = jnp.mean(jnp.abs(k_fitted - k_plog))
        elif self.loss_name == "mape":
            # Mean absolute percentage error
            loss = jnp.mean(jnp.abs((k_plog - k_fitted) / jnp.maximum(k_plog, 1e-10))) * 100
        elif self.loss_name == "rmsle":
            # Root mean squared logarithmic error
            log_k_plog = jnp.log(jnp.maximum(k_plog, 1e-30))
            log_k_fitted = jnp.log(jnp.maximum(k_fitted, 1e-30))
            loss = jnp.sqrt(jnp.mean((log_k_fitted - log_k_plog) ** 2))
        elif self.loss_name == "bo":
            # Bilger-Otomo metric (weighted log scale)

            # First term: average logarithmic ratio
            ratio = k_fitted / k_plog
            log_ratio = jnp.log10(ratio)
            log_term = jnp.mean(log_ratio**2)

            # Second term: slope deviation
            T_normalized = (self.T_range - jnp.min(self.T_range)) / (jnp.max(self.T_range) - jnp.min(self.T_range))
            d_log_k_plog = jnp.gradient(jnp.log10(jnp.maximum(k_plog, 1e-30)), T_normalized, axis=1)
            d_log_k_fitted = jnp.gradient(jnp.log10(jnp.maximum(k_fitted, 1e-30)), T_normalized, axis=1)
            slope_term = jnp.mean((d_log_k_fitted - d_log_k_plog) ** 2)

            # Combine terms with weighting
            loss = log_term + 0.5 * slope_term
        else:
            # Default to RMSE
            loss = jnp.sqrt(jnp.mean((k_fitted - k_plog) ** 2))

        return loss

    def _project_params(self, params: Array, lower_bounds: Array, upper_bounds: Array) -> Array:
        param_dict = {name: value for name, value in zip(self.param_names, params)}

        projected_params = optax.projections.projection_box(
            param_dict,
            lower=dict(zip(self.param_names, lower_bounds)),
            upper=dict(zip(self.param_names, upper_bounds)),
        )

        return jnp.array([projected_params[name] for name in self.param_names])

    def optimize(
        self,
        max_iterations: int = 100,
        uncertainty_factor: float = 1.0,
        uncertainty_type: str = "symmetric",
        optimizer: str = "adam",
        tol: float = 1e-6,
        learning_rate: float = 0.01,
    ) -> Dict[str, Any]:
        initial_params = self._estimate_initial_params()
        lower_bounds, upper_bounds = self._calculate_optimization_bounds(
            initial_params, uncertainty_factor, uncertainty_type
        )

        self.logger.info(f"\nStarting optimization with Optax {optimizer}")
        self.logger.info(f"Maximum iterations: {max_iterations}")
        self.logger.info(f"Tolerance: {tol}")
        self.logger.info(f"Learning rate: {learning_rate}")

        params = self._project_params(initial_params, lower_bounds, upper_bounds)

        @jit
        def loss_fn(params):
            return self._loss_function(params)

        loss_and_grad_fn = jit(value_and_grad(loss_fn))

        # Configure the optimizer
        if optimizer == "adam":
            opt = optax.adam(learning_rate)
        elif optimizer == "adamw":
            opt = optax.adamw(learning_rate)
        elif optimizer == "sgd":
            opt = optax.sgd(learning_rate)
        elif optimizer == "rmsprop":
            opt = optax.rmsprop(learning_rate)
        else:
            self.logger.warning(f"Unknown optimizer {optimizer}, defaulting to adam")
            opt = optax.adam(learning_rate)

        # Initialize optimizer state
        opt_state = opt.init(params)

        # Store optimization history
        loss_history = []
        param_history = []

        # Run optimization loop
        self.logger.info("\nOptimization progress:")
        self.logger.info("Iteration\tLoss")

        for i in range(max_iterations):
            # Compute loss and gradients
            loss, grads = loss_and_grad_fn(params)

            # Check for NaN gradients
            if jnp.any(jnp.isnan(grads)):
                self.logger.warning(f"NaN gradients detected at iteration {i}, stopping optimization")
                break

            # Store history
            loss_history.append(float(loss))
            param_history.append(params)

            # Log progress every 10 iterations
            if i % 10 == 0 or i == max_iterations - 1:
                self.logger.info(f"{i}\t{loss:.6e}")

            # Check convergence
            if i > 0 and jnp.abs(loss_history[-1] - loss_history[-2]) < tol:
                self.logger.info(f"Converged at iteration {i}")
                break

            # Update parameters
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            # Project parameters to respect bounds
            params = self._project_params(params, lower_bounds, upper_bounds)

        # Calculate final loss
        final_loss = self._loss_function(params)

        self.logger.info("\nOptimization complete")
        self.logger.info(f"Final loss: {final_loss:.6e}")

        # Return results
        return {
            "initial_params": initial_params,
            "optimized_params": params,  # These are already projected to respect bounds
            "loss": final_loss,
            "loss_history": jnp.array(loss_history),
            "param_history": jnp.array(param_history),
            "iterations": len(loss_history),
            "success": len(loss_history) < max_iterations or jnp.abs(loss_history[-1] - loss_history[-2]) < tol,
            "bounds": list(zip(lower_bounds, upper_bounds)),
        }

    def _estimate_initial_params(self) -> Array:
        self.logger.info("\nEstimating initial parameters:")
        params_dict = {}

        T_mean = jnp.mean(self.T_range)

        # Extract high and low pressure limits from PLOG data
        lnA_high, n_high, EaR_high = self.plog.k_levels[-1].lnA, self.plog.k_levels[-1].n, self.plog.k_levels[-1].EaR
        M = (self.plog.p_levels[0] / (constants.R_L_atm_K_mol * self.T_range)) * jnp.float64(0.001)
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
            params_dict = self._estimate_duplicate_reaction_params(
                lnA_low, n_low, EaR_low, lnA_high, n_high, EaR_high, falloff_defaults, params_dict
            )

        # Convert dictionary to array in the correct parameter order
        params = jnp.array([params_dict[name] for name in self.param_names], dtype=jnp.float64)
        return params

    def _calculate_optimization_bounds(
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
        # Default bounds - apply uncertainty factor
        if uncertainty_type == "symmetric":
            lower_bound = param_value / (10**uncertainty_factor)
            upper_bound = param_value * (10**uncertainty_factor)
        else:
            # In asymmetric case, we could implement different factors for upper and lower bounds if needed
            lower_bound = param_value / (10**uncertainty_factor)
            upper_bound = param_value * (10**uncertainty_factor)

        # Apply parameter-specific constraints

        if param_name.startswith("A_") or (param_name == "A" or param_name == "secondary_A"):
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
            lower_bound = max(lower_bound, 0.0)
            upper_bound = min(upper_bound, 1.0)

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

    # Add a method to use Boundaries for uncertainty quantification
    # def calculate_uncertainty_bounds(self, optimized_params: Array, uncertainty_factor: float = 0.5) -> Dict[str, Any]:
    #     self.logger.info(f"\nCalculating uncertainty bounds (factor: {uncertainty_factor})")
    #
    #     # Create a sample temperature range for evaluation
    #     T_eval = jnp.linspace(self.T_range[0], self.T_range[-1], 50)
    #
    #     # Reconstruct the kinetic model with optimized parameters
    #     # (This would depend on your specific implementation)
    #     k_optimized = self._compute_rate_constants(optimized_params, T_eval)
    #
    #     # Use Boundaries class to calculate uncertainty bounds
    #     boundaries = Boundaries(
    #         rate_constant=k_optimized,
    #         uncertainty_factor=uncertainty_factor,
    #         uncertainty_type="symmetric",
    #         T_range=(self.T_range[0], self.T_range[-1]),
    #     )
    #
    #     # Calculate boundary rate constants
    #     boundary_rates = boundaries.get_boundary_rate_constants(T_eval)
    #
    #     # Log the uncertainty analysis
    #     self.logger.info("Reaction rate uncertainty (at selected temperatures):")
    #     for i, T in enumerate(T_eval[::10]):  # Log every 10th point
    #         k_nom = boundary_rates["nominal"][i]
    #         k_low = boundary_rates["lower"][i]
    #         k_high = boundary_rates["upper"][i]
    #         self.logger.info(f"  T = {T:.1f} K: {k_low:.3e} ≤ {k_nom:.3e} ≤ {k_high:.3e}")
    #
    #     return {
    #         "temperatures": T_eval,
    #         "nominal_rates": boundary_rates["nominal"],
    #         "lower_bounds": boundary_rates["lower"],
    #         "upper_bounds": boundary_rates["upper"],
    #         "uncertainty_factor": uncertainty_factor,
    #     }

    def _estimate_single_reaction_params(
        self, lnA_low, n_low, EaR_low, lnA_high, n_high, EaR_high, falloff_defaults, params_dict
    ):
        """Process parameters for a single reaction."""
        # Process Arrhenius parameters
        arrh_params = [
            ("A_low", lnA_low, True),
            ("n_low", n_low, False),
            ("E_low", EaR_low, True),
            ("A_high", lnA_high, True),
            ("n_high", n_high, False),
            ("E_high", EaR_high, True),
        ]

        # Fill in base Arrhenius parameters
        for i, (name, default_value, is_exponential) in enumerate(arrh_params):
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

        self._log_single_reaction_params(params_dict, falloff_type)

        return params_dict

    def _estimate_duplicate_reaction_params(
        self, lnA_low, n_low, EaR_low, lnA_high, n_high, EaR_high, falloff_defaults, params_dict
    ):
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
            ("A_low", estimated_lnA_low, True),
            ("n_low", estimated_n_low, False),
            ("E_low", estimated_EaR_low, True),
            ("A_high", estimated_lnA_high, True),
            ("n_high", estimated_n_high, False),
            ("E_high", estimated_EaR_high, True),
        ]

        # Fill in primary reaction Arrhenius parameters
        for i, (name, default_value, is_exponential) in enumerate(primary_arrh_params):
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
                ("secondary_A_low", estimated_lnA_low, True),
                ("secondary_n_low", estimated_n_low, False),
                ("secondary_E_low", estimated_EaR_low, True),
                ("secondary_A_high", estimated_lnA_high, True),
                ("secondary_n_high", estimated_n_high, False),
                ("secondary_E_high", estimated_EaR_high, True),
            ]

            # Fill in secondary reaction Arrhenius parameters
            for i, (name, default_value, is_exponential) in enumerate(secondary_arrh_params):
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

    def _log_initialization(self, T_range: Tuple[float, float], P_range: Tuple[float, float]) -> None:
        """Log initial configuration settings."""
        self.logger.info("=" * 89)

        if self.fitting_mode == "single":
            self.logger.info(f"Plog to {self.primary_falloff_type} refitter")
        else:
            primary = self.primary_falloff_type
            secondary = self.secondary_falloff_type
            self.logger.info(f"Plog to Duplicate Reactions ({primary} + {secondary}) refitter")

        self.logger.info(f" Temperature range [K]: {T_range}")
        self.logger.info(f" Pressure range [atm]: {P_range}")
        self.logger.info(f" Loss function: {self.loss_name}")

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

    def _configure_parameters(
        self, param_config: Optional[Dict[str, Any]]
    ) -> Tuple[List[str], Array, List[Optional[float]]]:
        # Get parameter names based on falloff type and fitting mode
        primary_params = self._get_falloff_param_names(self.primary_falloff_type)

        # Add secondary reaction parameters if in duplicate mode
        secondary_params = []
        if self.fitting_mode == "duplicate" and self.secondary_falloff_type:
            secondary_params = self._get_falloff_param_names(self.secondary_falloff_type, prefix="secondary_")

        # Combined parameter list
        param_names = primary_params + secondary_params

        # Process parameter configuration
        if param_config is None:
            # Default: optimize all parameters
            param_mask = jnp.ones(len(param_names), dtype=bool)
            initial_values = [None] * len(param_names)
        else:
            # Process each parameter according to configuration
            mask_list = []
            initial_values = []

            for name in param_names:
                config = param_config.get(name, {"optimize": True})

                if isinstance(config, dict):
                    optimize = config.get("optimize", True)
                    mask_list.append(optimize)
                    initial_values.append(float(config["value"]) if "value" in config else None)
                elif isinstance(config, bool):
                    # Handle cases where config is just a boolean (optimize flag)
                    mask_list.append(config)
                    initial_values.append(None)
                elif isinstance(config, (int, float)):
                    # Assume config is a direct value to use
                    mask_list.append(False)  # Don't optimize
                    initial_values.append(float(config))
                else:
                    raise ValueError(f"Invalid configuration for parameter {name}: {config}")

            param_mask = jnp.array(mask_list, dtype=bool)

        # Log parameter configuration
        self._log_parameter_config(param_names, param_mask, initial_values)

        return param_names, param_mask, initial_values

    def _log_parameter_config(
        self, param_names: List[str], param_mask: Array, initial_values: List[Optional[float]]
    ) -> None:
        """Log parameter configuration details."""
        self.logger.info("\nParameters configuration:")

        # Group parameters by reaction type for clearer logging
        if self.fitting_mode == "duplicate":
            self.logger.info(f"\nPrimary reaction ({self.primary_falloff_type}):")

        # Log primary parameters
        primary_count = len(self._get_falloff_param_names(self.primary_falloff_type))
        for i in range(primary_count):
            name, mask = param_names[i], param_mask[i]
            value = initial_values[i]
            self._log_parameter_entry(name, mask, value)

        # Log secondary parameters if applicable
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

    @staticmethod
    def _setup_logging(log_name: str) -> logging.Logger:
        logger = logging.getLogger("PlogRefitter")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter("%(message)s")

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        log_path = Path.cwd()
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / log_name, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger
