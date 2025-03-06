from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import jit
from jaxtyping import Array, Float64

from .falloff_models import ModelBuilder


class PlogOptimizer:
    def __init__(
        self,
        model_builder: ModelBuilder,
        param_names: List[str],
        param_mask: Array,
        initial_params: Array,
        T_range: Array,
        P_range: Array,
        k_plog: Array,
        loss_name: str = "rmsle",
        logger=None,
    ):
        self.model_builder = model_builder
        self.param_names = param_names
        self.active_indices = jnp.where(param_mask)[0]
        self.active_param_names = [self.param_names[int(i)] for i in self.active_indices]
        self.initial_params = initial_params

        # Pre-compute full params template for faster updates
        self.full_params_template = self.initial_params.copy()
        self.T_range = jnp.asarray(T_range)
        self.P_range = jnp.asarray(P_range)
        self.k_plog = jnp.asarray(k_plog)
        self.loss_function = self._get_loss_function(loss_name)
        self.logger = logger

        # Pre-compile the value_and_grad function
        self.value_and_grad_fn = jax.value_and_grad(self._compute_loss)

    @partial(jit, static_argnums=(0,))
    def _compute_loss(self, params: Array) -> Float64:
        """
        Compute the loss for the given parameters.
        Here params are masked we need to transform them into the full one.
        """
        full_params = self.full_params_template.at[self.active_indices].set(params)
        k_fitted = self.model_builder.compute_rate_constants(full_params, self.param_names, self.T_range, self.P_range)

        # Compute loss
        return self.loss_function(k_fitted, self.k_plog)

    @staticmethod
    @jit
    def _fix_bounds(lower_bounds: Array, upper_bounds: Array) -> Tuple[Array, Array]:
        """JAX-optimized bound fixing function"""
        wrong_order = lower_bounds > upper_bounds
        new_lower = jnp.where(wrong_order, upper_bounds, lower_bounds)
        new_upper = jnp.where(wrong_order, lower_bounds, upper_bounds)
        return new_lower, new_upper

    @partial(jit, static_argnums=(0,))
    def _project_params(self, params: Array, lower_bounds: Array, upper_bounds: Array) -> Array:
        """Project parameters to respect bounds using optax's projection_box."""
        projected_params = optax.projections.projection_box(
            params,
            lower_bounds,
            upper_bounds,
        )
        return projected_params

    @partial(jit, static_argnums=(0,))
    def _optimization_step(self, active_params, opt_state, lower_bounds, upper_bounds, loss_prev):
        """Single optimization step, JIT-compiled for speed"""
        loss, grads = self.value_and_grad_fn(active_params)

        # converged = jnp.abs(loss - loss_prev) < self.tol
        converged = False

        # Always compute updates, but only apply if not converged
        updates, new_opt_state = self.optimizer.update(
            grads,
            opt_state,
            params=active_params,
            value=loss,
            grad=grads,
            value_fn=self._compute_loss,
        )

        # Apply updates
        new_params = optax.apply_updates(active_params, updates)

        # Project parameters to respect bounds
        new_params = self._project_params(
            new_params,
            lower_bounds,
            upper_bounds,
        )

        return new_params, new_opt_state, loss, converged

    def optimize(
        self,
        lower_bounds: Array,
        upper_bounds: Array,
        max_iterations: int = 100,
        tol: float = 1e-6,
        learning_rate: float = 1e-3,
    ) -> Dict[str, Any]:
        if self.logger:
            self.logger.info("\nStarting optimization with L-BFGS")
            self.logger.info(f"Maximum iterations: {max_iterations}")
            self.logger.info(f"Function tolerance: {tol}")

        self.tol = tol

        fixed_lower, fixed_upper = self._fix_bounds(lower_bounds, upper_bounds)

        # Get active bounds and params
        active_lower = fixed_lower[self.active_indices]
        active_upper = fixed_upper[self.active_indices]
        active_params = self.initial_params[self.active_indices]

        # Initial projection
        active_params = self._project_params(
            active_params,
            active_lower,
            active_upper,
        )

        # Initialize optimizer
        # linesearch = optax.scale_by_backtracking_linesearch(max_backtracking_steps=50)
        # linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=50)
        self.optimizer = optax.lbfgs()
        opt_state = self.optimizer.init(active_params)

        loss_history = []
        param_history = []

        if self.logger:
            self.logger.info("\nOptimization progress:")
            self.logger.info("Iteration\tLoss")

        # Initial loss calculation
        loss_prev = jnp.inf
        loss = self._compute_loss(active_params)
        loss_history.append(float(loss))
        param_history.append(active_params)

        if self.logger:
            self.logger.info(f"0\t{loss:.6e}")

        for i in range(1, max_iterations + 1):
            # Perform optimization step
            active_params, opt_state, loss, converged = self._optimization_step(
                active_params,
                opt_state,
                active_lower,
                active_upper,
                loss_prev,
            )

            # Record history
            loss_history.append(float(loss))
            loss_prev = loss_history[-1]
            param_history.append(active_params)

            # Log progress
            if self.logger and (i % 10 == 0 or i == max_iterations):
                self.logger.info(f"{i}\t{loss:.6e}")

            # Check for convergence
            if converged:
                if self.logger:
                    self.logger.info(f"Converged at iteration {i}")
                break

        final_loss = loss_history[-1]  # Already computed in the loop

        if self.logger:
            self.logger.info("\nOptimization complete")
            self.logger.info(f"Final loss: {final_loss:.6e}")

        return {
            "initial_params": self.initial_params,
            "optimized_params": active_params,
            "loss": final_loss,
            "loss_history": jnp.array(loss_history),
            "param_history": jnp.array(param_history),
            "iterations": len(loss_history) - 1,  # Subtract 1 because we include initial state
            "success": converged or i < max_iterations,
            "bounds": (fixed_lower, fixed_upper),
        }

    def _get_loss_function(self, loss_name: str) -> Callable:
        """Get a loss function by name, preferring optax built-ins when available."""
        if loss_name == "mse":

            @jit
            def mse(y_pred, y_true):
                return optax.l2_loss(y_pred, y_true).mean()

            return mse

        elif loss_name == "rmse":

            @jit
            def rmse(y_pred, y_true):
                return jnp.sqrt(optax.l2_loss(y_pred, y_true).mean())

            return rmse

        elif loss_name == "rmsle":

            @jit
            def rmsle(y_pred, y_true):
                epsilon = 1e-10
                log_y_true = jnp.log(y_true + epsilon)
                log_y_pred = jnp.log(y_pred + epsilon)
                return jnp.sqrt(jnp.mean(jnp.square(log_y_pred - log_y_true)))

            return rmsle

        elif loss_name == "huber":

            @jit
            def huber_loss(y_pred, y_true, delta=1.0):
                abs_error = jnp.abs(jnp.log(y_pred) - jnp.log(y_true))
                quadratic = jnp.minimum(abs_error, delta)
                linear = abs_error - quadratic
                return jnp.mean(0.5 * quadratic**2 + delta * linear)

            return huber_loss

        elif loss_name == "log_cosh":

            @jit
            def log_cosh_loss(y_pred, y_true):
                return jnp.mean(jnp.cosh(jnp.log(y_pred) - jnp.log(y_true)))

            return log_cosh_loss

        elif loss_name == "robust_rmsle":

            @jit
            def robust_rmsle(y_pred, y_true):
                epsilon = 1e-10
                y_pred_clipped = jnp.clip(y_pred, epsilon, 1e10)
                log_y_true = jnp.log(jnp.maximum(y_true, epsilon))
                log_y_pred = jnp.log(y_pred_clipped)
                diff = log_y_pred - log_y_true
                abs_diff = jnp.abs(diff)
                delta = 1.0
                squared = jnp.minimum(abs_diff, delta) ** 2 * 0.5
                linear = (abs_diff - delta) * delta
                losses = jnp.where(abs_diff <= delta, squared, linear)
                return jnp.sqrt(jnp.mean(losses))

            return robust_rmsle

        else:

            @jit
            def robust_rmsle_default(y_pred, y_true):
                epsilon = 1e-10
                y_pred_clipped = jnp.clip(y_pred, epsilon, 1e10)
                log_y_true = jnp.log(jnp.maximum(y_true, epsilon))
                log_y_pred = jnp.log(y_pred_clipped)
                diff = log_y_pred - log_y_true
                abs_diff = jnp.abs(diff)
                delta = 1.0
                squared = jnp.minimum(abs_diff, delta) ** 2 * 0.5
                linear = (abs_diff - delta) * delta
                losses = jnp.where(abs_diff <= delta, squared, linear)
                return jnp.sqrt(jnp.mean(losses))

            return robust_rmsle_default
