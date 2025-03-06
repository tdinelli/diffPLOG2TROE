from typing import Any, Dict, Optional, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Float64

from ..rate_constants import Plog
from .falloff_models import ModelBuilder
from .log_manager import log_initialization, setup_logging
from .optimizer import PlogOptimizer
from .parameters_manager import ParameterManager


class PlogRefitter:
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
        loss_name: str = "rmsle",
        log_name: str = "refitter.log",
    ) -> None:
        # Set up logging
        self.logger = setup_logging(log_name)
        log_initialization(
            self.logger,
            fitting_mode,
            primary_falloff_type,
            secondary_falloff_type if fitting_mode == "duplicate" else None,
            T_range,
            P_range,
            loss_name,
        )

        # Generate training data from the PLOG expression
        self.plog = Plog(plog_dict)
        self.T_range = jnp.linspace(T_range[0], T_range[1], n_T)
        self.P_range = jnp.logspace(jnp.log10(P_range[0]), jnp.log10(P_range[1]), n_P)
        self.k_plog = self.plog.kinetic_constant(self.T_range, self.P_range)

        # Store configuration
        self.fitting_mode = fitting_mode
        self.primary_falloff_type = primary_falloff_type
        self.secondary_falloff_type = secondary_falloff_type
        self.loss_name = loss_name

        # Initialize parameter manager
        self.param_manager = ParameterManager(
            fitting_mode=fitting_mode,
            primary_falloff_type=primary_falloff_type,
            secondary_falloff_type=secondary_falloff_type,
            T_range=self.T_range,
            P_range=self.P_range,
            plog=Plog(plog_dict),
            k_plog=self.k_plog,
            param_config=param_config,
            logger=self.logger,
        )

        # Initialize model builder
        self.model_builder = ModelBuilder(
            fitting_mode=fitting_mode,
            primary_falloff_type=primary_falloff_type,
            secondary_falloff_type=secondary_falloff_type,
            name=self.plog.name,
        )

        # Estimate initial parameters
        self.initial_params = self.param_manager.estimate_initial_params(self.plog)

        # Initialize optimizer
        self.optimizer = PlogOptimizer(
            model_builder=self.model_builder,
            param_names=self.param_manager.param_names,
            param_mask=self.param_manager.param_mask,
            initial_params=self.initial_params,
            T_range=self.T_range,
            P_range=self.P_range,
            k_plog=self.k_plog,
            loss_name=loss_name,
            logger=self.logger,
        )

    def optimize(
        self,
        max_iterations: int = 100,
        uncertainty_factor: float = 1.0,
        uncertainty_type: str = "symmetric",
        tol: float = 1e-6,
        learning_rate: float = 1e-3
    ) -> Dict[str, Any]:
        # Calculate parameter bounds
        lower_bounds, upper_bounds = self.param_manager.calculate_optimization_bounds(
            self.initial_params, uncertainty_factor, uncertainty_type
        )

        results = self.optimizer.optimize(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            max_iterations=max_iterations,
            tol=tol,
            learning_rate=learning_rate
        )
        return results
