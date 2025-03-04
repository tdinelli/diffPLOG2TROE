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
        self.T_range = T_range
        self.P_range = P_range
        self.k_plog = k_plog
        self.loss_function = self._get_loss_function(loss_name)
        self.logger = logger

    @partial(jit, static_argnums=(0,))
    def _compute_loss(self, params: Array) -> Float64:
        """
        Compute the loss for the given parameters.
        Here params are masked we need to transform them into the full one.
        """
        full_params = self.initial_params.at[self.active_indices].set(params)

        k_fitted = self.model_builder.compute_rate_constants(full_params, self.param_names, self.T_range, self.P_range)

        # Compute loss
        return self.loss_function(k_fitted, self.k_plog)

    @partial(jit, static_argnums=(0,))
    def _project_params(self, params: Array, lower_bounds: Array, upper_bounds: Array) -> Array:
        """Project parameters to respect bounds using optax's projection_box."""
        projected_params = optax.projections.projection_box(
            params,
            lower_bounds,
            upper_bounds,
        )

        return projected_params

    @staticmethod
    def _fix_bounds(lower_bounds: Array, upper_bounds: Array) -> Tuple[Array, Array]:
        wrong_order = lower_bounds > upper_bounds
        new_lower = jnp.where(wrong_order, upper_bounds, lower_bounds)
        new_upper = jnp.where(wrong_order, lower_bounds, upper_bounds)
        return new_lower, new_upper

    def optimize(
        self,
        lower_bounds: Array,
        upper_bounds: Array,
        max_iterations: int = 100,
        tol: float = 1e-6,
    ) -> Dict[str, Any]:
        if self.logger:
            self.logger.info("\nStarting optimization with L-BFGS")
            self.logger.info(f"Maximum iterations: {max_iterations}")
            self.logger.info(f"Function tolerance: {tol}")

        lower_bounds, upper_bounds = self._fix_bounds(lower_bounds, upper_bounds)
        active_params = self.initial_params[self.active_indices]
        active_params = self._project_params(
            active_params,
            lower_bounds[self.active_indices],
            upper_bounds[self.active_indices],
        )

        optimizer = optax.lbfgs()

        opt_state = optimizer.init(active_params)

        loss_history = []
        param_history = []

        value_and_grad_fn = jax.value_and_grad(self._compute_loss)

        if self.logger:
            self.logger.info("\nOptimization progress:")
            self.logger.info("Iteration\tLoss")

        i = 0
        loss_prev = jnp.inf
        continue_opt = True

        while i < max_iterations and continue_opt:
            loss, grads = value_and_grad_fn(active_params)
            loss_history.append(float(loss))
            param_history.append(active_params)

            if self.logger and (i % 10 == 0 or i == max_iterations - 1):
                self.logger.info(f"{i}\t{loss:.6e}")

            # Check convergence
            if i > 0 and jnp.abs(loss - loss_prev) < tol:
                if self.logger:
                    self.logger.info(f"Converged at iteration {i}")
                break

            updates, opt_state = optimizer.update(
                grads, opt_state, params=active_params, value=loss, grad=grads, value_fn=self._compute_loss
            )
            active_params = optax.apply_updates(active_params, updates)

            active_params = self._project_params(
                active_params,
                lower_bounds[self.active_indices],
                upper_bounds[self.active_indices],
            )

            loss_prev = loss
            i += 1

        final_loss = self._compute_loss(active_params)

        if self.logger:
            self.logger.info("\nOptimization complete")
            self.logger.info(f"Final loss: {final_loss:.6e}")

        return {
            "initial_params": self.initial_params,
            "optimized_params": active_params,
            "loss": final_loss,
            "loss_history": jnp.array(loss_history),
            "param_history": jnp.array(param_history),
            "iterations": len(loss_history),
            "success": i < max_iterations
            or (len(loss_history) > 1 and jnp.abs(loss_history[-1] - loss_history[-2]) < tol),
            "bounds": (lower_bounds, upper_bounds),
        }

    def _get_loss_function(self, loss_name: str) -> Callable:
        """Get a loss function by name, preferring optax built-ins when available."""
        if loss_name == "mse":
            return lambda y_pred, y_true: optax.l2_loss(y_pred, y_true).mean()
        elif loss_name == "rmse":
            return lambda y_pred, y_true: jnp.sqrt(optax.l2_loss(y_pred, y_true).mean())
        elif loss_name == "rmsle":

            def rmsle(y_pred, y_true):
                log_y_true = jnp.log(jnp.maximum(y_true, 1e-30))
                log_y_pred = jnp.log(jnp.maximum(y_pred, 1e-30))
                return jnp.sqrt(jnp.mean((log_y_pred - log_y_true) ** 2))

            return rmsle
        else:
            return lambda y_pred, y_true: jnp.sqrt(optax.l2_loss(y_pred, y_true).mean())
