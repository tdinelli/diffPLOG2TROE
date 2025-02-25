from typing import Any, Callable, Dict, Union

import jax.numpy as jnp
import optax
from jax import jit, value_and_grad
from jaxtyping import Array, Float64


class OptaxWrapper:
    """
    Wrapper around optimization algorithms with a unified interface.

    This class provides a common interface to various optimizers including:
    - Standard Optax optimizers (Adam, AdaBelief, SGD)
    - Quasi-Newton methods (L-BFGS)
    - Trust region methods (planned for future implementation)

    Attributes:
        algorithm (str): Name of the optimization algorithm to use
        max_steps (int): Maximum number of optimization steps
        early_stop_patience (int): Number of steps without improvement before early stopping
        early_stop_delta (float): Minimum improvement threshold to reset patience counter
        log_interval (int): Frequency of logging optimization progress
        learning_rate (float): Base learning rate for optimizers
    """

    def __init__(self, opt_options: Dict[str, Any], logger: Any = None) -> None:
        """
        Initialize optimizer wrapper with the given options.

        Args:
            opt_options: Dictionary containing optimization configuration
            logger: Optional logger instance for recording progress

        Configuration options:
            Common options:
              - algorithm: Optimization algorithm name ('adam', 'sgd', 'adabelief', 'lbfgs', etc.)
              - max_steps: Maximum number of optimization steps
              - patience: Steps without improvement before early stopping
              - ftol: Tolerance for improvement in loss value
              - steps_file: Interval for logging progress
              - learning_rate: Base learning rate
              - clip_norm: Gradient norm clipping threshold (for applicable algorithms)

            For gradient-based optimizers (adam, sgd, adabelief):
              - use_scheduler: Whether to use learning rate scheduling
              - schedule_type: Type of schedule ('cosine', 'exponential', 'linear')
              - warmup_steps: Steps for warmup phase
              - decay_steps: Steps for decay phase
              - peak_value: Maximum learning rate
              - end_value: Final learning rate
              - init_value: Initial learning rate (for warmup)

            For L-BFGS:
              - history_size: Number of past iterations to store
              - line_search: Line search method ('zoom', 'strong_wolfe', etc.)
        """
        # Extract base configuration
        self.algorithm = opt_options["algorithm"]
        self.max_steps = opt_options["max_steps"]
        self.early_stop_patience = opt_options.get("patience", 10)
        self.early_stop_delta = opt_options.get("ftol", 1e-6)
        self.clip_norm = opt_options.get("clip_norm", 2.0)  # Not used yet
        self.log_interval = opt_options.get("steps_file", 100)
        self.logger = logger
        self.learning_rate = opt_options.get("learning_rate", 1e-3)

        # Optimizer-specific configuration
        self.optimizer_config = {}

        # Configure first-order gradient-based methods (Adam, SGD, AdaBelief)
        if self.algorithm in ["adam", "adabelief", "sgd"]:
            self.optimizer_config.update(
                {
                    "use_scheduler": opt_options.get("use_scheduler", False),
                    "schedule_type": opt_options.get("schedule_type", "cosine"),
                    "warmup_steps": opt_options.get("warmup_steps", 1000),
                    "decay_steps": opt_options.get("decay_steps", 10000),
                    "peak_value": opt_options.get("peak_value", self.learning_rate),
                    "end_value": opt_options.get("end_value", self.learning_rate / 100),
                    "init_value": opt_options.get("init_value", 0.0),
                }
            )
        elif self.algorithm == "lbfgs":
            self.optimizer_config.update(
                {
                    "history_size": opt_options.get("history_size", 10),
                    "line_search": opt_options.get("line_search", "zoom"),
                }
            )
        elif self.algorithm == "trust_region":
            raise ValueError("Trust region not implemented yet!")
        else:
            raise ValueError(f"Unknown algorithm {self.algorithm}")

        # Runtime state
        self.step = 0
        self.best_loss = jnp.float64("inf")
        self.best_params = None
        self.steps_without_improvement = 0

    def _create_optax_optimizer(self) -> optax.GradientTransformation:
        """
        Create an Optax optimizer based on configuration.

        Returns:
            Optax optimizer instance
        """
        # Handle learning rate scheduling for applicable optimizers
        if self.algorithm in ["adam", "adabelief", "sgd"] and self.optimizer_config["use_scheduler"]:
            lr = self._create_learning_rate_schedule()
        else:
            lr = self.learning_rate

        # Create optimizer based on algorithm
        if self.algorithm == "adam":
            optimizer = optax.adam(learning_rate=lr)
        elif self.algorithm == "adabelief":
            optimizer = optax.adabelief(learning_rate=lr)
        elif self.algorithm == "sgd":
            optimizer = optax.sgd(learning_rate=lr)
        elif self.algorithm == "lbfgs":
            optimizer = optax.lbfgs(learning_rate=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.algorithm}")

        # Add gradient clipping if requested (for applicable algorithms) not yet implemented
        # if self.clip_norm > 0 and self.algorithm not in ["lbfgs"]:
        #     optimizer = optax.chain(
        #         optax.clip_by_global_norm(self.clip_norm),
        #         optimizer,
        #     )

        return optimizer

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
        self.best_loss = jnp.float64("inf")
        self.best_params = base_params.copy()
        self.steps_without_improvement = 0

        active_params = base_params[active_indices]

        optimizer = self._create_optax_optimizer()
        opt_state = optimizer.init(active_params)

        def loss_wrapper(active_params):
            """Compute loss for active parameters by updating the full parameter array"""
            full_params = base_params.at[active_indices].set(active_params)
            return loss_fn(full_params)

        @jit
        def update(params, opt_state, base_params_ref):
            """Single optimization step for standard optimizers"""
            loss_value, grads = value_and_grad(loss_wrapper)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
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

    def optimize(
        self,
        loss_fn: Callable,
        base_params: Array,
        active_indices: Array,
    ) -> Dict[str, Any]:
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

        if self.algorithm in ["adam", "adabelief", "sgd", "lbfgs"]:
            return self._optax_optimize(loss_fn, base_params, active_indices)
        elif self.algorithm == "trust_region":
            # Placeholder for future trust region implementation
            raise ValueError("Trust region not implemented yet!")
        else:
            raise ValueError(f"Unsupported optimization algorithm: {self.algorithm}")

    def _create_learning_rate_schedule(self) -> Union[Float64, Callable]:
        """
        Create a learning rate schedule based on configuration.

        Returns:
            Learning rate schedule or scalar learning rate
        """
        config = self.optimizer_config

        if not config["use_scheduler"]:
            return self.learning_rate

        if config["schedule_type"] == "cosine":
            return optax.warmup_cosine_decay_schedule(
                init_value=config["init_value"],
                peak_value=config["peak_value"],
                warmup_steps=config["warmup_steps"],
                decay_steps=config["decay_steps"],
                end_value=config["end_value"],
            )
        elif config["schedule_type"] == "exponential":
            if config["warmup_steps"] > 0:
                warmup = optax.linear_schedule(
                    init_value=config["init_value"],
                    end_value=config["peak_value"],
                    transition_steps=config["warmup_steps"],
                )

                decay = optax.exponential_decay(
                    init_value=config["peak_value"],
                    transition_steps=config["decay_steps"] // 10,
                    decay_rate=0.9,
                    end_value=config["end_value"],
                )

                return optax.join_schedules(schedules=[warmup, decay], boundaries=[config["warmup_steps"]])
            else:
                return optax.exponential_decay(
                    init_value=config["peak_value"],
                    transition_steps=config["decay_steps"] // 10,
                    decay_rate=0.9,
                    end_value=config["end_value"],
                )
        elif config["schedule_type"] == "linear":
            return optax.linear_schedule(
                init_value=config["peak_value"], end_value=config["end_value"], transition_steps=config["decay_steps"]
            )
        else:
            self._log(f"Unknown schedule type: {config['schedule_type']}, using fixed learning rate")
            return self.learning_rate

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
