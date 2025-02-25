from typing import Any, Callable, Dict, Union

import jax
import jax.numpy as jnp
import optax
from jax import value_and_grad
from jaxtyping import Array


class OptaxWrapper:
    """Wrapper around Optax optimizers."""

    def __init__(
        self,
        optax_options: Dict[str, Any],
        logger=None,
    ) -> None:
        self.algorithm = optax_options["algorithm"]
        self.max_steps = optax_options["max_steps"]
        self.early_stop_patience = optax_options["patience"]
        self.early_stop_delta = optax_options["ftol"]
        self.clip_norm = optax_options["clip_norm"] or 2
        self.log_interval = optax_options["steps_file"]
        self.logger = logger

        self.learning_rate = optax_options["learning_rate"]
        self.use_scheduler = optax_options.get("use_scheduler", False)

        if self.use_scheduler:
            self.schedule_type = optax_options.get("schedule_type", "cosine")
            self.warmup_steps = optax_options.get("warmup_steps", 1000)
            self.decay_steps = optax_options.get("decay_steps", 10000)
            self.peak_value = optax_options.get("peak_value", self.learning_rate)
            self.end_value = optax_options.get("end_value", self.learning_rate)
            self.init_value = optax_options.get("init_value", 0.0)

        # Runtime state
        self.step = 0
        self.best_loss = float("inf")
        self.best_params = None
        self.steps_without_improvement = 0

    def _log(self, message: str) -> None:
        """Helper method to log messages"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def _create_optimizer(self) -> optax.GradientTransformation:
        """Create Optax optimizer based on configuration"""
        lr = self._create_learning_rate_schedule()

        if self.algorithm == "adam":
            optimizer = optax.adam(learning_rate=lr)
        elif self.algorithm == "adabelief":
            optimizer = optax.adabelief(learning_rate=lr)
        elif self.algorithm == "sgd":
            optimizer = optax.sgd(learning_rate=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.algorithm}")

        # Add gradient clipping if requested
        if self.clip_norm > 0:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.clip_norm),
                optimizer,
            )

        return optimizer

    def _create_learning_rate_schedule(self) -> Union[float, Callable]:
        """Create a learning rate schedule based on configuration"""
        if not self.use_scheduler:
            return self.learning_rate

        if self.schedule_type == "cosine":
            return optax.warmup_cosine_decay_schedule(
                init_value=self.init_value,
                peak_value=self.peak_value,
                warmup_steps=self.warmup_steps,
                decay_steps=self.decay_steps,
                end_value=self.end_value,
            )
        elif self.schedule_type == "exponential":
            if self.warmup_steps > 0:
                warmup = optax.linear_schedule(
                    init_value=self.init_value, end_value=self.peak_value, transition_steps=self.warmup_steps
                )

                decay = optax.exponential_decay(
                    init_value=self.peak_value,
                    transition_steps=self.decay_steps // 10,  # Decay period
                    decay_rate=0.9,
                    end_value=self.end_value,
                )

                return optax.join_schedules(schedules=[warmup, decay], boundaries=[self.warmup_steps])
            else:
                return optax.exponential_decay(
                    init_value=self.peak_value,
                    transition_steps=self.decay_steps // 10,
                    decay_rate=0.9,
                    end_value=self.end_value,
                )
        elif self.schedule_type == "linear":
            return optax.linear_schedule(
                init_value=self.peak_value, end_value=self.end_value, transition_steps=self.decay_steps
            )
        else:
            self._log(f"Unknown schedule type: {self.schedule_type}, using fixed learning rate")
            return self.learning_rate

    def optimize(
        self,
        loss_fn: Callable,
        base_params: Array,
        active_indices: Array,
    ) -> Dict[str, Any]:
        # Reset state for this optimization run
        self.step = 0
        self.best_loss = jnp.float64("inf")
        self.best_params = base_params.copy()
        self.steps_without_improvement = 0

        # Extract active parameters to optimize
        active_params = base_params[active_indices]

        # Create optimizer
        optimizer = self._create_optimizer()
        opt_state = optimizer.init(active_params)

        # Create update function with JAX JIT
        @jax.jit
        def update(params, opt_state, base_params_ref):
            def loss_wrapper(active_params):
                # Update full parameters with the active parameters
                full_params = base_params_ref.at[active_indices].set(active_params)
                return loss_fn(full_params)

            loss_value, grads = value_and_grad(loss_wrapper)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            grad_norm = optax.global_norm(grads)
            return new_params, new_opt_state, loss_value, grad_norm, base_params_ref.at[active_indices].set(new_params)

        # Run optimization
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

            # Check early stopping condition
            if self.steps_without_improvement >= self.early_stop_patience:
                self._log(f"Early stopping triggered after {step} steps")
                break

        # Return optimization results
        return {
            "params": self.best_params,
            "active_params": self.best_params[active_indices],
            "loss": self.best_loss,
            "iterations": self.step,
            "early_stopped": self.steps_without_improvement >= self.early_stop_patience,
        }
