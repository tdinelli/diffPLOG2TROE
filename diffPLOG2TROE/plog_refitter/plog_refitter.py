from typing import Any, Dict, Optional, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Float64

from ..kinetic_constants import Plog
from .falloff_models import ModelBuilder
from .log_manager import log_initialization, setup_logging
from .optimizer import PlogOptimizer
from .parameters_manager import ParameterManager


class PlogRefitter:
    def __init__(
        self,
        plog_dict: Dict[str, Any],  # Think about fixing this it must be a plog object directly in my vision
        T_range: Tuple[Float64, Float64],
        P_range: Tuple[Float64, Float64],
        n_T: int = 100,
        n_P: int = 100,
        param_config: Optional[Dict[str, Union[bool, float, Dict[str, Any]]]] = None,
        falloff_type: str = "troe",
        lossfunction_name: str = "rmsle",
        log_name: Optional[str] = None,
    ) -> None:
        # ====================================================================
        # Set up logging
        # ====================================================================
        self.logger = setup_logging(log_name)
        log_initialization(
            self.logger,
            falloff_type,
            T_range,
            P_range,
            lossfunction_name,
        )

        # ====================================================================
        # Generate training data from the PLOG expression
        # ====================================================================
        self.plog = Plog(plog_dict)
        self.T_range = jnp.linspace(T_range[0], T_range[1], n_T)
        self.P_range = jnp.logspace(jnp.log10(P_range[0]), jnp.log10(P_range[1]), n_P)
        self.k_plog = self.plog.kinetic_constant(self.T_range, self.P_range)

        # ====================================================================
        # Store configuration
        # ====================================================================
        self.falloff_type = falloff_type
        self.loss_name = lossfunction_name

        # ====================================================================
        # Initialize parameter manager
        # ====================================================================
        self.param_manager = ParameterManager(
            falloff_type=falloff_type,
            T_range=self.T_range,
            P_range=self.P_range,
            plog=Plog(plog_dict),
            k_plog=self.k_plog,
            param_config=param_config,
            logger=self.logger,
        )

        # ====================================================================
        # Initialize model builder
        # ====================================================================
        self.model_builder = ModelBuilder(
            falloff_type=falloff_type,
            name=self.plog.name,
        )

        # ====================================================================
        # Estimate initial parameters
        # ====================================================================
        self.initial_params = self.param_manager.estimate_initial_params(self.plog)

        # ====================================================================
        # Initialize optimizer
        # ====================================================================
        self.optimizer = PlogOptimizer(
            model_builder=self.model_builder,
            param_names=self.param_manager.param_names,
            param_mask=self.param_manager.param_mask,
            initial_params=self.initial_params,
            T_range=self.T_range,
            P_range=self.P_range,
            k_plog=self.k_plog,
            loss_name=lossfunction_name,
            logger=self.logger,
        )

    def optimize(
        self,
        max_iterations: int = 100,
        uncertainty_factor: float = 1.0,
        uncertainty_type: str = "symmetric",
    ) -> Dict[str, Any]:
        # ====================================================================
        # Compute parameter bounds
        # ====================================================================
        lower_bounds, upper_bounds = self.param_manager.get_parameters_bounds(
            self.initial_params, uncertainty_factor, uncertainty_type
        )

        results = self.optimizer.optimize(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            max_iterations=max_iterations,
        )
        return results
