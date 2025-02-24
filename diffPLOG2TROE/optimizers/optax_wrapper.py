from typing import Any, Callable, Dict, Optional, Union

import jax
import jax.numpy as jnp
import optax
from jax import value_and_grad
from jaxtyping import Array, Float64


class OptaxWrapper:
    """Wrapper around Optax optimizers."""

    def __init__(
        self,
        learning_rate: Union[float, Callable] = 1e-3,
        optimizer_name: str = "adam",
        max_steps: int = 100000,
        early_stop_patience: int = 10000,
        early_stop_delta: Float64 = 1e-6,
        clip_norm: Float64 = 2.0,
        log_interval: int = 1000,
        logger=None,
    ) -> None:
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.max_steps = max_steps
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        self.clip_norm = clip_norm
        self.log_interval = log_interval
        self.logger = logger

        # Runtime state
        self.step = 0
        self.best_loss = jnp.float64("inf")
        self.best_params = jnp.array([])
        self.steps_without_improvement = 0

    def _log(self, message: str) -> None:
        """Helper method to log messages"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def _create_optimizer(self) -> optax.GradientTransformation:
        """Create Optax optimizer based on configuration"""
        if callable(self.learning_rate):
            lr = self.learning_rate
        elif isinstance(self.learning_rate, (int, float)):
            lr = self.learning_rate
        else:
            raise ValueError(f"Unsupported learning rate type: {type(self.learning_rate)}")

        # Set up optimizer
        if self.optimizer_name == "adam":
            optimizer = optax.adam(learning_rate=lr)
        elif self.optimizer_name == "adabelief":
            optimizer = optax.adabelief(learning_rate=lr)
        elif self.optimizer_name == "sgd":
            optimizer = optax.sgd(learning_rate=lr)
        elif self.optimizer_name == "adamw":
            optimizer = optax.adamw(learning_rate=lr)
        elif self.optimizer_name == "lion":
            optimizer = optax.lion(learning_rate=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        # Add gradient clipping if requested
        if self.clip_norm > 0:
            optimizer = optax.chain(optimizer, optax.clip_by_global_norm(self.clip_norm))

        return optimizer

    def create_cosine_schedule(
        self, peak_value: Float64 = 1e-3, warmup_steps: int = 5000, decay_steps: int = 100000, end_value: Float64 = 1e-5
    ) -> Callable:
        """Create a cosine learning rate schedule with warmup"""
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=peak_value,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=end_value,
        )

    @staticmethod
    def _create_update_fn(loss_fn, optimizer, active_indices, base_params):
        """Create JAX-jitted update function for optimization"""

        @jax.jit
        def update(active_params, opt_state):
            def loss_wrapper(params):
                full_params = base_params.copy()
                full_params = full_params.at[active_indices].set(params)
                return loss_fn(full_params)

            loss_value, grads = value_and_grad(loss_wrapper)(active_params)
            updates, opt_state = optimizer.update(grads, opt_state, active_params)
            active_params = optax.apply_updates(active_params, updates)
            grad_norm = optax.global_norm(grads)
            return active_params, opt_state, loss_value, grad_norm

        return update

    def optimize(
        self,
        loss_fn: Callable,
        base_params: Array,
        active_indices: Array,
        initial_active_params: Optional[Array] = None,
    ) -> Dict[str, Any]:
        self.step = 0
        self.best_loss = jnp.float64("inf")
        self.best_params = jnp.array(base_params)
        self.steps_without_improvement = 0

        if initial_active_params is None:
            initial_active_params = base_params[active_indices].copy()

        # Create optimizer
        optimizer = self._create_optimizer()
        opt_state = optimizer.init(initial_active_params)
        # Create update function
        update_fn = self._create_update_fn(loss_fn, optimizer, active_indices, base_params)
        # Initial active parameters
        current_active_params = initial_active_params

        # Run optimization
        for step in range(self.max_steps):
            self.step = step
            current_active_params, opt_state, loss_value, grad_norm = update_fn(current_active_params, opt_state, step)

            # Check for improvement
            if loss_value < self.best_loss - self.early_stop_delta:
                self.best_loss = loss_value
                self.best_params = base_params.at[active_indices].set(current_active_params)
                self.steps_without_improvement = 0
            else:
                self.steps_without_improvement += 1

            # Log progress
            if step % self.log_interval == 0:
                self._log(f"  Step {step}: loss = {loss_value:.6e}, grad_norm = {grad_norm:.6e}")


            if self.steps_without_improvement >= self.early_stop_patience:
                self._log(f"Early stopping triggered after {step} steps")
                break

        final_result = self.best_params

        return {
            "params": final_result,
            "active_params": final_result[active_indices],
            "loss": self.best_loss,
            "iterations": self.step,
            "early_stopped": self.steps_without_improvement >= self.early_stop_patience,
        }
