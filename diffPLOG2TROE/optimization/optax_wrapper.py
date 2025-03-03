from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import optax
from optax.projections import projection_box
from jax import jit, value_and_grad
from jaxtyping import Array, Float64


@dataclass
class OptimizerConfig:
    """Base configuration for all optimizers"""

    algorithm: str
    max_steps: int
    patience: int = 10
    ftol: Float64 = 1e-6
    steps_file: int = 100
    learning_rate: Float64 = 1e-3


@dataclass
class FirstOrderConfig(OptimizerConfig):
    """Configuration for first-order optimizers (Adam, SGD, AdaBelief)"""

    use_scheduler: bool = False
    schedule_type: str = "cosine"
    warmup_steps: int = 1000
    decay_steps: int = 10000
    peak_value: Optional[Float64] = None
    end_value: Optional[Float64] = None
    init_value: Float64 = 0.0

    def __post_init__(self):
        """Set derived values after initialization"""
        if self.peak_value is None:
            self.peak_value = self.learning_rate
        if self.end_value is None:
            self.end_value = self.learning_rate / 100


@dataclass
class LBFGSConfig(OptimizerConfig):
    """Configuration for L-BFGS optimizer"""

    line_search: str = "zoom"


class OptaxWrapper:
    """Wrapper for Optax optimizers with standardized configuration handling."""

    def __init__(
        self, opt_options: Dict[str, Any], bounds: Optional[Tuple[Array, Array]] = None, logger: Any = None
    ) -> None:
        """
        Initialize the optimizer wrapper with configuration options.

        Args:
            opt_options: Dictionary of optimization options
            logger: Optional logger instance
        """
        # Validate required options
        if "algorithm" not in opt_options:
            raise ValueError("'algorithm' must be specified in optimization options")
        if "max_steps" not in opt_options:
            raise ValueError("'max_steps' must be specified in optimization options")

        # Extract algorithm type
        self.algorithm = opt_options["algorithm"]
        self.logger = logger

        # Bounds
        self.bounds = bounds

        # Parse configuration based on algorithm type
        if self.algorithm in ["adam", "adabelief", "sgd"]:
            self.config = self._create_first_order_config(opt_options)

            # Extract common parameters from specific config
            self.max_steps = self.config.max_steps
            self.early_stop_patience = self.config.patience
            self.early_stop_delta = self.config.ftol
            self.log_interval = self.config.steps_file
            self.learning_rate = self.config.learning_rate

        elif self.algorithm == "lbfgs":
            self.config = self._create_lbfgs_config(opt_options)

            # Extract common parameters from specific config
            self.max_steps = self.config.max_steps
            self.early_stop_patience = self.config.patience
            self.early_stop_delta = self.config.ftol
            self.log_interval = self.config.steps_file
            self.learning_rate = self.config.learning_rate

        elif self.algorithm == "trust_region":
            raise ValueError("Trust region not implemented yet!")
        else:
            raise ValueError(f"Unknown algorithm {self.algorithm}")

        # Runtime state
        self.step = 0
        self.best_loss = jnp.inf
        self.best_params = None
        self.steps_without_improvement = 0

    def _create_first_order_config(self, opt_options: Dict[str, Any]) -> FirstOrderConfig:
        """Create configuration for first-order optimizers"""
        return FirstOrderConfig(
            algorithm=opt_options["algorithm"],
            max_steps=opt_options["max_steps"],
            patience=opt_options.get("patience", 10),
            ftol=opt_options.get("ftol", 1e-6),
            steps_file=opt_options.get("steps_file", 100),
            learning_rate=opt_options.get("learning_rate", 1e-3),
            use_scheduler=opt_options.get("use_scheduler", False),
            schedule_type=opt_options.get("schedule_type", "cosine"),
            warmup_steps=opt_options.get("warmup_steps", 1000),
            decay_steps=opt_options.get("decay_steps", 10000),
            peak_value=opt_options.get("peak_value"),
            end_value=opt_options.get("end_value"),
            init_value=opt_options.get("init_value", 0.0),
        )

    def _create_lbfgs_config(self, opt_options: Dict[str, Any]) -> LBFGSConfig:
        """Create configuration for L-BFGS optimizer"""
        return LBFGSConfig(
            algorithm=opt_options["algorithm"],
            max_steps=opt_options["max_steps"],
            patience=opt_options.get("patience", 10),
            ftol=opt_options.get("ftol", 1e-6),
            steps_file=opt_options.get("steps_file", 100),
            learning_rate=opt_options.get("learning_rate", 1e-3),
            line_search=opt_options.get("line_search", "zoom"),
        )

    def optimize(self, loss_fn: Callable, base_params: Array, active_indices: Array) -> Dict[str, Any]:
        """
        Optimize parameters using the selected algorithm.

        Args:
            loss_fn: Function that takes full parameters and returns loss value
            base_params: Full parameter array (including fixed parameters)
            active_indices: Indices of parameters to optimize

        Returns:
            Dictionary with optimization results containing:
            - params: Optimized full parameter array
            - active_params: Optimized active parameters
            - loss: Final loss value
            - iterations: Number of iterations performed
            - early_stopped: Whether optimization stopped early
        """
        self._log(f"Starting optimization with {self.algorithm} algorithm")

        # Create & call optimizer based on algorithm type
        optimizer_methods = {
            "adam": self._optax_optimize,
            "adabelief": self._optax_optimize,
            "sgd": self._optax_optimize,
            "lbfgs": self._lbfgs_optimize,
            "trust_region": None,  # Placeholder for future implementation
        }

        if self.algorithm not in optimizer_methods:
            raise ValueError(f"Unsupported optimization algorithm: {self.algorithm}")

        if optimizer_methods[self.algorithm] is None:
            raise ValueError(f"{self.algorithm} not implemented yet!")

        return optimizer_methods[self.algorithm](loss_fn, base_params, active_indices)

    def _optax_optimize(self, loss_fn: Callable, base_params: Array, active_indices: Array) -> Dict[str, Any]:
        """
        Optimize using standard Optax optimizers.

        Args:
            loss_fn: Function that takes full parameters and returns loss value
            base_params: Full parameter array (including fixed parameters)
            active_indices: Indices of parameters to optimize

        Returns:
            Dictionary with optimization results
        """
        self.step = 0
        self.best_loss = jnp.inf
        self.best_params = base_params.copy()
        self.steps_without_improvement = 0

        active_params = base_params[active_indices]

        optimizer = self._create_optax_optimizer()
        opt_state = optimizer.init(active_params)

        def loss_wrapper(active_params):
            """Compute loss for active parameters by updating the full parameter array"""
            full_params = base_params.at[active_indices].set(active_params)
            return loss_fn(full_params)

        use_bounds = self.bounds is not None
        if use_bounds:
            bounds_low, bounds_high = self.bounds
            active_bounds_low = bounds_low[active_indices]
            active_bounds_high = bounds_high[active_indices]

        @jit
        def update(params, opt_state, base_params_ref):
            """Single optimization step for standard optimizers"""
            loss_value, grads = value_and_grad(loss_wrapper)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            if use_bounds:
                new_params = projection_box(new_params, active_bounds_low, active_bounds_high)

            grad_norm = optax.global_norm(grads)
            return (
                new_params,
                new_opt_state,
                loss_value,
                grad_norm,
                base_params_ref.at[active_indices].set(new_params),
            )

        current_base_params = base_params.copy()

        for step in range(self.max_steps):
            self.step = step

            # Update parameters
            active_params, opt_state, loss_value, grad_norm, current_base_params = update(
                active_params, opt_state, current_base_params
            )

            # Check for improvement
            if loss_value < self.best_loss - self.early_stop_delta:
                self.best_loss = loss_value
                self.best_params = current_base_params.copy()
                self.steps_without_improvement = 0
            else:
                self.steps_without_improvement += 1

            # Log progress
            if step % self.log_interval == 0:
                self._log(f"  Step {step}: loss = {loss_value:.6e}, grad_norm = {grad_norm:.6e}")

            # Check early stopping conditions
            if self.steps_without_improvement >= self.early_stop_patience:
                self._log(f"Early stopping triggered after {step} steps")
                break

            if grad_norm < self.early_stop_delta:
                self._log(f"Gradient norm below tolerance at step {step}")
                break

        return {
            "params": self.best_params,
            "active_params": self.best_params[active_indices],
            "loss": self.best_loss,
            "iterations": self.step,
            "early_stopped": self.steps_without_improvement >= self.early_stop_patience,
        }

    def _lbfgs_optimize(self, loss_fn: Callable, base_params: Array, active_indices: Array) -> Dict[str, Any]:
        """
        Optimize using L-BFGS algorithm.

        Args:
            loss_fn: Function that takes full parameters and returns loss value
            base_params: Full parameter array (including fixed parameters)
            active_indices: Indices of parameters to optimize

        Returns:
            Dictionary with optimization results
        """
        self.step = 0
        self.best_loss = jnp.inf
        self.best_params = base_params.copy()
        self.steps_without_improvement = 0

        active_params = base_params[active_indices]

        optimizer = self._create_optax_optimizer()
        opt_state = optimizer.init(active_params)

        def loss_wrapper(active_params):
            """Compute loss for active parameters by updating the full parameter array"""
            full_params = base_params.at[active_indices].set(active_params)
            return loss_fn(full_params)

        use_bounds = self.bounds is not None
        if use_bounds:
            bounds_low, bounds_high = self.bounds
            active_bounds_low = bounds_low[active_indices]
            active_bounds_high = bounds_high[active_indices]

        @jit
        def update(params, opt_state, base_params_ref):
            loss_value, grads = value_and_grad(loss_wrapper)(params)
            updates, new_opt_state = optimizer.update(
                updates=grads, state=opt_state, params=params, value=loss_value, grad=grads, value_fn=loss_wrapper
            )
            new_params = optax.apply_updates(params, updates)
            if use_bounds:
                new_params = projection_box(new_params, active_bounds_low, active_bounds_high)
            grad_norm = optax.global_norm(grads)
            updated_base_params = base_params_ref.at[active_indices].set(new_params)
            return (new_params, new_opt_state, loss_value, grad_norm, updated_base_params)

        current_base_params = base_params.copy()

        for step in range(self.max_steps):
            self.step = step

            active_params, opt_state, loss_value, grad_norm, current_base_params = update(
                active_params, opt_state, current_base_params
            )

            # Check for improvement
            if loss_value < self.best_loss - self.early_stop_delta:
                self.best_loss = loss_value
                self.best_params = current_base_params.copy()
                self.steps_without_improvement = 0
            else:
                self.steps_without_improvement += 1

            # Log progress
            if step % self.log_interval == 0:
                self._log(f"  Step {step}: loss = {loss_value:.6e}, grad_norm = {grad_norm:.6e}")

            # Check early stopping conditions - L-BFGS often converges faster
            if self.steps_without_improvement >= self.early_stop_patience:
                self._log(f"Early stopping triggered after {step} steps")
                break

            # L-BFGS typically converges when the gradient norm is sufficiently small
            if grad_norm < self.early_stop_delta:
                self._log(f"Gradient norm below tolerance at step {step}")
                break

        self._log(f"Found minimum at: {self.best_loss:.6e}")
        # Return optimization results
        return {
            "params": self.best_params,
            "active_params": self.best_params[active_indices],
            "loss": self.best_loss,
            "iterations": self.step,
            "early_stopped": self.steps_without_improvement >= self.early_stop_patience,
        }

    def _create_optax_optimizer(self) -> optax.GradientTransformation:
        """
        Create an Optax optimizer based on configuration.

        Returns:
            Optax optimizer instance
        """
        if self.algorithm in ["adam", "adabelief", "sgd"] and isinstance(self.config, FirstOrderConfig):
            if self.config.use_scheduler:
                lr = self._create_learning_rate_schedule()
            else:
                lr = self.learning_rate

            # Map algorithm to optimizer creation
            optimizer_map = {
                "adam": lambda: optax.adam(learning_rate=lr),
                "adabelief": lambda: optax.adabelief(learning_rate=lr),
                "sgd": lambda: optax.sgd(learning_rate=lr),
            }

            return optimizer_map[self.algorithm]()
        elif self.algorithm == "lbfgs" and isinstance(self.config, LBFGSConfig):
            # Configure line search method
            if self.config.line_search == "zoom":
                linesearch = optax.scale_by_zoom_linesearch(
                    max_linesearch_steps=70, verbose=True, initial_guess_strategy="one"
                )
            elif self.config.line_search == "backtracking":
                linesearch = optax.scale_by_backtracking_linesearch(
                    max_backtracking_steps=70,
                    verbose=True,
                )
            else:
                raise ValueError(f"Unsupported line search method: {self.config.line_search}")

            return optax.lbfgs(scale_init_precond=True, linesearch=linesearch)
        else:
            raise ValueError(f"Invalid optimizer configuration for {self.algorithm}")

    def _create_learning_rate_schedule(self) -> Union[Float64, Callable]:
        """
        Create a learning rate schedule based on configuration.

        Returns:
            Learning rate schedule or scalar learning rate
        """
        # This method is only applicable for first-order methods
        if not isinstance(self.config, FirstOrderConfig):
            return self.learning_rate

        config = self.config

        if not config.use_scheduler:
            return self.learning_rate

        schedule_creators = {
            "cosine": lambda: optax.warmup_cosine_decay_schedule(
                init_value=config.init_value,
                peak_value=config.peak_value,
                warmup_steps=config.warmup_steps,
                decay_steps=config.decay_steps,
                end_value=config.end_value,
            ),
            "exponential": self._create_exponential_schedule,
            "linear": lambda: optax.linear_schedule(
                init_value=config.peak_value, end_value=config.end_value, transition_steps=config.decay_steps
            ),
        }

        if config.schedule_type not in schedule_creators:
            self._log(f"Unknown schedule type: {config.schedule_type}, using fixed learning rate")
            return self.learning_rate

        return schedule_creators[config.schedule_type]()

    def _create_exponential_schedule(self) -> Callable:
        """Create an exponential learning rate schedule"""
        if not isinstance(self.config, FirstOrderConfig):
            raise ValueError("Exponential schedule only applicable for first-order methods")

        config = self.config

        if config.warmup_steps > 0:
            warmup = optax.linear_schedule(
                init_value=config.init_value,
                end_value=config.peak_value,
                transition_steps=config.warmup_steps,
            )

            decay = optax.exponential_decay(
                init_value=config.peak_value,
                transition_steps=config.decay_steps // 10,
                decay_rate=0.9,
                end_value=config.end_value,
            )

            return optax.join_schedules(schedules=[warmup, decay], boundaries=[config.warmup_steps])
        else:
            return optax.exponential_decay(
                init_value=config.peak_value,
                transition_steps=config.decay_steps // 10,
                decay_rate=0.9,
                end_value=config.end_value,
            )

    def _log(self, message: str) -> None:
        """
        Log a message using the configured logger.

        Args:
            message: Message to log
        """
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
