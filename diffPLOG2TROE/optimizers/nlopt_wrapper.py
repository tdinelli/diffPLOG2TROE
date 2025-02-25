import logging
from typing import Any, Callable, Dict, Optional, Tuple

import jax.numpy as jnp
import nlopt
from jax import value_and_grad
from jaxtyping import Array


class NLOptWrapper:
    """Wrapper for NLOpt optimizers."""

    def __init__(
        self,
        nlopt_options: Dict[str, Any],
        bounds: Tuple[Array, Array],
        logger: Optional[logging.Logger] = None,
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
        self.bounds_low = bounds[0]
        self.bounds_high = bounds[1]
        self.max_eval = nlopt_options["max_eval"]
        self.ftol_rel = nlopt_options["ftol_rel"]
        self.xtol_rel = nlopt_options["xtol_rel"]
        self.log_interval = nlopt_options["steps_file"]
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

    @staticmethod
    def _get_algorithm(algorithm_name: str):
        if algorithm_name == "ISRES":
            return nlopt.GN_ISRES
        elif algorithm_name == "DIRECT":
            return nlopt.GN_DIRECT
        elif algorithm_name == "DIRECT_L":
            return nlopt.GN_DIRECT_L
        elif algorithm_name == "DIRECT_L_NOSCAL":
            return nlopt.GN_DIRECT_L_NOSCAL
        elif algorithm_name == "DIRECT_L_RAND":
            return nlopt.GN_DIRECT_L_RAND
        elif algorithm_name == "DIRECT_L_RAND_NOSCAL":
            return nlopt.GN_DIRECT_L_RAND_NOSCAL
        elif algorithm_name == "DIRECT_L_NOSCAL":
            return nlopt.GN_DIRECT_NOSCAL
        elif algorithm_name == "CRSLM":
            return nlopt.GN_CRS2_LM
        elif algorithm_name == "ESCH":
            return nlopt.GN_ESCH
        elif algorithm_name == "NLOPT_LD_TNEWTON_PRECOND_RESTART":
            return nlopt.LD_TNEWTON_PRECOND_RESTART
        elif algorithm_name == "NLOPT_LD_LBFGS":
            return nlopt.LD_LBFGS
        else:
            raise ValueError(f"Unknown algorithm {algorithm_name}")

    def optimize(
        self,
        loss_fn: Callable,
        base_params: Array,
        active_indices: Array,
    ) -> Dict[str, Any]:
        active_params = base_params[active_indices].copy()

        # Configure NLOpt optimizer
        opt = nlopt.opt(self.algorithm, len(active_params))
        opt.set_lower_bounds(self.bounds_low[active_indices])
        opt.set_upper_bounds(self.bounds_high[active_indices])
        opt.set_maxeval(self.max_eval)
        opt.set_ftol_rel(self.ftol_rel)
        opt.set_xtol_rel(self.xtol_rel)

        if self.is_gradient_based:

            def full_loss_fn(x):
                full_params = base_params.at[active_indices].set(x)
                return loss_fn(full_params)

            grad_fn = value_and_grad(full_loss_fn)

        def objective(x, grad):
            """Wrapper for the objective function as required by NLOpt"""
            if self.is_gradient_based and grad.size > 0:
                # Compute both loss and gradient using JAX
                loss_value, gradient = grad_fn(jnp.array(x))
                loss_value = float(loss_value)

                # Copy gradient values into the grad parameter
                # (NLOpt expects the grad array to be modified in-place)
                for i in range(len(gradient)):
                    grad[i] = float(gradient[i])
            else:
                # If not using gradients or grad array not provided, just compute loss
                full_params = base_params.at[active_indices].set(jnp.array(x))
                loss_value = float(loss_fn(full_params))

            # Track best result
            if loss_value < self.best_loss:
                self.best_loss = loss_value
                self.best_params = base_params.at[active_indices].set(jnp.array(x)).copy()

            # Log progress
            if self.step % self.log_interval == 0:
                self._log(f"  Step {self.step}: loss = {loss_value:.6e}")
                if self.is_gradient_based and grad.size > 0:
                    grad_norm = float(jnp.linalg.norm(jnp.array([grad[i] for i in range(len(grad))])))
                    self._log(f"  Gradient norm: {grad_norm:.6e}")

            self.step += 1
            return loss_value

        opt.set_min_objective(objective)

        # Run optimization
        try:
            _ = opt.optimize(active_params)
            result = self.best_params
            status = opt.last_optimize_result()
            self._log(f"NLOpt finished with status: {status}")
            self._log(f"Found minimum at: {self.best_loss:.6e}")
            success = True
        except nlopt.RoundoffLimited:
            self._log("NLOpt stopped due to roundoff errors")
            result = self.best_params
            status = "RoundoffLimited"
            success = False
        except Exception as e:
            self._log(f"NLOpt stopped with error: {str(e)}")
            result = self.best_params
            status = str(e)
            success = False

        return {
            "params": result,
            "loss": self.best_loss,
            "status": status,
            "success": success,
            "iterations": self.step,
        }
