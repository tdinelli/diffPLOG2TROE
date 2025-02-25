import logging
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import nlopt
import numpy as np
from jax import jit, value_and_grad
from jaxtyping import Array


class NLOptWrapper:
    """Wrapper for NLOpt optimizers with JAX optimizations."""

    def __init__(
        self, nlopt_options: Dict[str, Any], bounds: Tuple[Array, Array], logger: Optional[logging.Logger] = None
    ) -> None:
        self.algorithm = self._get_algorithm(nlopt_options["algorithm"])
        self.is_gradient_based = self.algorithm in [
            nlopt.LD_LBFGS,
            nlopt.LD_TNEWTON,
            nlopt.LD_TNEWTON_PRECOND_RESTART,
            nlopt.LD_SLSQP,
            nlopt.LD_MMA,
            nlopt.LD_CCSAQ,
            nlopt.LD_VAR1,
            nlopt.LD_VAR2,
        ]

        # Convert bounds to NumPy arrays for NLOpt compatibility
        self.bounds_low = np.asarray(bounds[0])
        self.bounds_high = np.asarray(bounds[1])
        self.max_eval = nlopt_options["max_eval"]
        self.ftol_rel = nlopt_options["ftol_rel"]
        self.xtol_rel = nlopt_options["xtol_rel"]
        self.log_interval = nlopt_options["steps_file"]
        self.logger = logger

        # Runtime state
        self.step = 0
        self.best_loss = float("inf")
        self.best_params = None

        # Cache for JIT-compiled functions
        self._jitted_fns = {}

    def optimize(self, loss_fn: Callable, base_params: Array, active_indices: Array) -> Dict[str, Any]:
        """Optimize using NLOpt with JAX acceleration."""
        active_indices = jnp.asarray(active_indices)
        cache_key = (len(base_params), len(active_indices))
        if cache_key not in self._jitted_fns:
            update_fn = self._create_update_fn(active_indices)

            @jit
            def full_loss_fn(x):
                full_params = update_fn(base_params, x)
                return loss_fn(full_params)

            if self.is_gradient_based:
                grad_fn = jit(value_and_grad(full_loss_fn))
            else:
                grad_fn = None

            self._jitted_fns[cache_key] = (update_fn, full_loss_fn, grad_fn)
        else:
            update_fn, full_loss_fn, grad_fn = self._jitted_fns[cache_key]

        active_params = np.array(base_params[active_indices], dtype=np.float64)  # NLOpt requires np.float64

        self.best_params = jnp.array(base_params)

        self.step = 0

        opt = nlopt.opt(self.algorithm, len(active_params))
        opt.set_lower_bounds(self.bounds_low[active_indices])
        opt.set_upper_bounds(self.bounds_high[active_indices])
        opt.set_maxeval(self.max_eval)
        opt.set_ftol_rel(self.ftol_rel)
        opt.set_xtol_rel(self.xtol_rel)

        jax.device_put(base_params)

        def objective(x, grad):
            """Wrapper for the objective function as required by NLOpt"""
            x_jax = jnp.array(x)

            if self.is_gradient_based and grad.size > 0:
                loss_value, gradient = grad_fn(x_jax)
                np.copyto(grad, np.array(gradient, dtype=np.float64))
                loss_value = float(loss_value)
            else:
                loss_value = float(full_loss_fn(x_jax))

            if loss_value < self.best_loss:
                self.best_loss = loss_value
                self.best_params = update_fn(self.best_params, x_jax)

            if self.step % self.log_interval == 0:
                log_str = f"  Step {self.step}: loss = {loss_value:.6e}"

                if self.is_gradient_based and grad.size > 0:
                    grad_norm = float(jnp.linalg.norm(jnp.array(grad)))
                    log_str += f", Gradient norm: {grad_norm:.6e}"
                self._log(log_str)

            self.step += 1
            return loss_value

        opt.set_min_objective(objective)

        # Run optimization
        try:
            _ = opt.optimize(active_params)
            result = self.best_params
            status_code = opt.last_optimize_result()
            status_message = self._get_status_message(status_code)
            self._log(f"NLOpt finished with status: {status_code} - {status_message}")
            self._log(f"Found minimum at: {self.best_loss:.6e}")
            success = status_code > 0  # Positive values indicate success
        except nlopt.RoundoffLimited:
            status_code = -4  # NLOpt code for roundoff errors
            status_message = self._get_status_message(status_code)
            self._log(f"NLOpt stopped due to roundoff errors (status: {status_code} - {status_message})")
            result = self.best_params
            success = False
        except Exception as e:
            self._log(f"NLOpt stopped with error: {str(e)}")
            status_code = "Exception"
            status_message = "zzz"
            result = self.best_params
            success = False

        return {
            "params": result,
            "loss": self.best_loss,
            "status_message": status_message,
            "success": success,
            "iterations": self.step,
        }

    def _log(self, message: str) -> None:
        """Helper method to log messages"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    @staticmethod
    def _get_algorithm(algorithm_name: str) -> int:
        """Get NLOpt algorithm enum from string name."""
        algorithm_map = {
            "ISRES": nlopt.GN_ISRES,
            "DIRECT": nlopt.GN_DIRECT,
            "DIRECT_L": nlopt.GN_DIRECT_L,
            "DIRECT_L_NOSCAL": nlopt.GN_DIRECT_L_NOSCAL,
            "DIRECT_L_RAND": nlopt.GN_DIRECT_L_RAND,
            "DIRECT_L_RAND_NOSCAL": nlopt.GN_DIRECT_L_RAND_NOSCAL,
            "DIRECT_NOSCAL": nlopt.GN_DIRECT_NOSCAL,  # Fixed typo in algorithm name
            "CRSLM": nlopt.GN_CRS2_LM,
            "ESCH": nlopt.GN_ESCH,
            "TNEWTON_PRECOND_RESTART": nlopt.LD_TNEWTON_PRECOND_RESTART,
            "TNEWTON_RESTART": nlopt.LD_TNEWTON_RESTART,
            "LBFGS": nlopt.LD_LBFGS,
        }

        if algorithm_name not in algorithm_map:
            raise ValueError(f"Unknown algorithm {algorithm_name}")

        return algorithm_map[algorithm_name]

    def _create_update_fn(self, active_indices: Array) -> Callable:
        """Create a JIT-compiled function to update full params with active params."""

        @jit
        def update_params(base_params: Array, active_params: Array) -> Array:
            return base_params.at[active_indices].set(active_params)

        return update_params

    @staticmethod
    def _get_status_message(status_code: int) -> str:
        """Get human-readable message for NLOpt status code."""
        status_messages = {
            1: "Success: Generic success return value",
            2: "Success: Optimization stopped because stopval was reached",
            3: "Success: Optimization stopped because ftol_rel or ftol_abs was reached",
            4: "Success: Optimization stopped because xtol_rel or xtol_abs was reached",
            5: "Success: Optimization stopped because maxeval was reached",
            6: "Success: Optimization stopped because maxtime was reached",
            -1: "Failure: Generic failure code",
            -2: "Failure: Invalid arguments (e.g., lower bounds > upper bounds, unknown algorithm, etc.)",
            -3: "Failure: Ran out of memory",
            -4: "Failure: Roundoff errors led to failure",
            -5: "Failure: Forced termination",
        }
        return status_messages.get(status_code, f"Unknown status code: {status_code}")
