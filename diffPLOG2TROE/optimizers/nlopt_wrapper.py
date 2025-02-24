from typing import Any, Callable, Dict, Optional, Tuple

import jax.numpy as jnp
import nlopt
from jaxtyping import Array


class NLOptWrapper:
    """Wrapper for NLOpt optimizers."""

    def __init__(
        self,
        algorithm: int,
        bounds: Tuple[Array, Array],
        max_eval: int = 10000,
        ftol_rel: float = 1e-6,
        xtol_rel: float = 1e-6,
        log_interval: int = 100,
        logger=None,
    ) -> None:
        self.algorithm = algorithm
        self.bounds_low = bounds[0]
        self.bounds_high = bounds[1]
        self.max_eval = max_eval
        self.ftol_rel = ftol_rel
        self.xtol_rel = xtol_rel
        self.log_interval = log_interval
        self.logger = logger

        # Runtime state
        self.step = 0
        self.best_loss = float("inf")
        self.best_params = jnp.array([])

    def _log(self, message: str) -> None:
        """Helper method to log messages"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def optimize(
        self,
        loss_fn: Callable,
        base_params: Array,
        active_indices: Array,
        initial_active_params: Optional[Array] = None,
    ) -> Dict[str, Any]:
        # Reset state
        self.step = 0
        self.best_loss = float("inf")
        self.best_params = jnp.array(base_params)

        # Extract active parameters
        if initial_active_params is None:
            initial_active_params = base_params[active_indices].copy()

        # Configure NLOpt optimizer
        opt = nlopt.opt(self.algorithm, len(active_indices))
        opt.set_lower_bounds(self.bounds_low)
        opt.set_upper_bounds(self.bounds_high)
        opt.set_maxeval(self.max_eval)
        opt.set_ftol_rel(self.ftol_rel)
        opt.set_xtol_rel(self.xtol_rel)

        # Define objective function
        def objective(x, grad):
            full_params = self.best_params.copy()
            full_params = full_params.at[active_indices].set(x)
            loss_value = float(loss_fn(full_params))

            if loss_value < self.best_loss:
                self.best_loss = loss_value
                self.best_params = full_params.copy()

            if self.step % self.log_interval == 0:
                self._log(f"  Step {self.step}: loss = {loss_value:.6e}")

            self.step += 1
            return loss_value

        opt.set_min_objective(objective)

        # Run optimization
        try:
            result = opt.optimize(initial_active_params)
            status = opt.last_optimize_result()
            self._log(f"NLOpt finished with status: {status}")
            self._log(f"Found minimum at: {self.best_loss:.6e}")
            success = True
        except nlopt.RoundoffLimited:
            self._log("NLOpt stopped due to roundoff errors")
            result = self.best_params[active_indices]
            status = "RoundoffLimited"
            success = False
        except Exception as e:
            self._log(f"NLOpt stopped with error: {str(e)}")
            result = self.best_params[active_indices]
            status = str(e)
            success = False

        # Prepare final results
        full_result = self.best_params.copy()
        full_result = full_result.at[active_indices].set(result)

        return {
            "params": full_result,
            "active_params": result,
            "loss": self.best_loss,
            "status": status,
            "success": success,
            "iterations": self.step,
        }
