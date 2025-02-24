import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
import nlopt
import numpy as np
from jaxtyping import Array, Float64

from .rate_constants import FallOff, Plog


class PlogRefitter(eqx.Module):
    plog: Plog
    param_mask: Array
    param_names: List[str]
    T_range: Array
    P_range: Array
    k_plog: Array
    logger: logging.Logger
    initial_values: Any
    R_gas = jnp.float64(1.987)

    def __init__(
        self,
        plog_dict: Dict[str, Any],
        T_range: Tuple[Float64, Float64],
        P_range: Tuple[Float64, Float64],
        n_T: int = 100,
        n_P: int = 100,
        param_config: Optional[Dict[str, Union[bool, float, Dict[str, Any]]]] = None,
        log_dir: Optional[str] = None,
    ) -> None:
        self.logger = self._setup_logging(log_dir)
        self.logger.info("=" * 89)
        self.logger.info("Plog 2 TROE refitter")
        self.logger.info(f" Temperature range [K]: {T_range}")
        self.logger.info(f" Pressure range [atm]: {P_range}")
        self.plog = Plog(plog_dict)

        # Generate training data
        self.T_range = jnp.linspace(T_range[0], T_range[1], n_T)
        self.P_range = jnp.logspace(jnp.log10(P_range[0]), jnp.log10(P_range[1]), n_P)
        self.k_plog = self.plog.kinetic_constant(self.T_range, self.P_range)

        # Process parameter configuration
        self.param_names = ["A_low", "n_low", "E_low", "A_high", "n_high", "E_high", "A", "T3", "T1", "T2"]
        if param_config is None:
            self.param_mask = jnp.ones(len(self.param_names), dtype=bool)
            self.initial_values = [None] * len(self.param_names)
        else:
            mask_list = []
            initial_values = []

            for name in self.param_names:
                config = param_config.get(name, {"optimize": True})

                if isinstance(config, dict):
                    optimize = config.get("optimize", True)
                    mask_list.append(optimize)
                    initial_values.append(float(config["value"]) if "value" in config else None)
                else:
                    raise ValueError(f"Invalid configuration for parameter {name}")

            self.param_mask = jnp.array(mask_list, dtype=bool)
            self.initial_values = initial_values

        self.logger.info("\nParameters configuration:")
        for i, (name, mask) in enumerate(zip(self.param_names, self.param_mask)):
            if mask and self.initial_values[i] is not None:
                self.logger.info(f"  {name}: optimizing, starting from {self.initial_values[i]:.5e}")
            elif mask:
                self.logger.info(f"  {name}: optimizing from default initial value")
            else:
                self.logger.info(f"  {name}: fixed at {self.initial_values[i]:.5e}")

    def fit_arrhenius(self, k_values):
        log_k = jnp.log(k_values)
        log_T = jnp.log(self.T_range)
        inv_T = 1.0 / self.T_range
        X = jnp.vstack([jnp.ones_like(log_T), log_T, -inv_T]).T
        beta = jnp.linalg.lstsq(X, log_k, rcond=None)[0]
        return beta[0], beta[1], beta[2]  # lnA, n, Ea/R

    def estimate_initial_params(self) -> Array:
        """Estimate initial parameters with consideration for fixed values."""
        self.logger.info("\nFirst guess estimate of the parameters:")
        params_dict = {}

        # Low pressure limit estimation
        if any(not self.initial_values[i] for i in range(3)):  # If any low-pressure params need estimation
            lnA_low, n_low, EaR_low = self.fit_arrhenius(self.k_plog[0])
            params_dict.update(
                {
                    "lnA_low": lnA_low if self.param_mask[0] else jnp.log(self.initial_values[0]),
                    "n_low": n_low if self.param_mask[1] else self.initial_values[1],
                    "EaR_low": EaR_low if self.param_mask[2] else self.initial_values[2] / self.R_gas,
                }
            )
        else:  # All low-pressure params are fixed
            params_dict.update(
                {
                    "lnA_low": jnp.log(self.initial_values[0]),
                    "n_low": self.initial_values[1],
                    "EaR_low": self.initial_values[2] / self.R_gas,
                }
            )

        # High pressure limit estimation
        if any(not self.initial_values[i] for i in range(3, 6)):  # If any high-pressure params need estimation
            lnA_high, n_high, EaR_high = self.fit_arrhenius(self.k_plog[-1])
            params_dict.update(
                {
                    "lnA_high": lnA_high if self.param_mask[3] else jnp.log(self.initial_values[3]),
                    "n_high": n_high if self.param_mask[4] else self.initial_values[4],
                    "EaR_high": EaR_high if self.param_mask[5] else self.initial_values[5] / self.R_gas,
                }
            )
        else:  # All high-pressure params are fixed
            params_dict.update(
                {
                    "lnA_high": jnp.log(self.initial_values[3]),
                    "n_high": self.initial_values[4],
                    "EaR_high": self.initial_values[5] / self.R_gas,
                }
            )

        # Troe parameters estimation
        T_mean = jnp.mean(self.T_range)
        default_troe = {"A": 0.5, "T3": T_mean * 0.7, "T1": T_mean * 0.2, "T2": T_mean * 1.5}

        # Update Troe parameters based on fixed values or defaults
        for i, param in enumerate(["A", "T3", "T1", "T2"], start=6):
            if self.initial_values and self.initial_values[i] is not None:
                params_dict[param] = self.initial_values[i]
            else:
                params_dict[param] = default_troe[param]

        self.logger.info(
            "  Low pressure limit (A, n, Ea): {:.3e}, {:.3f}, {:.3e}".format(
                jnp.exp(params_dict["lnA_low"]), params_dict["n_low"], params_dict["EaR_low"] * self.R_gas
            )
        )
        self.logger.info(
            "  High pressure limit (A, n, Ea): {:.3e}, {:.3f}, {:.3e}".format(
                jnp.exp(params_dict["lnA_high"]), params_dict["n_high"], params_dict["EaR_high"] * self.R_gas
            )
        )
        self.logger.info(
            "  Troe parameters (A, T3, T1, T2): {:.3f}, {:.3e}, {:.3e}, {:.3e}\n".format(
                params_dict["A"], params_dict["T3"], params_dict["T1"], params_dict["T2"]
            )
        )

        params = jnp.array([params_dict[name] for name in params_dict.keys()], dtype=jnp.float64)

        return params

    @eqx.filter_jit
    def loss(self, params: Array) -> Float64:
        """Compute the loss between PLOG and fitted Troe rates."""
        falloff_dict = self._create_falloff_dict(self.plog.name, params)
        falloff = FallOff(falloff_dict)
        k_troe = falloff.kinetic_constant(self.T_range, self.P_range)

        # Use relative error in log space to handle large magnitude differences
        # log_k_troe = jnp.log(k_troe + 1e-30)
        # log_k_plog = jnp.log(self.k_plog + 1e-30)
        # loss = jnp.mean((log_k_troe - log_k_plog) ** 2)

        squared_errors = jnp.sum((1 - (k_troe / self.k_plog)) ** 2)
        loss = jnp.sqrt(squared_errors)

        return loss

    def fit(self, nlopt_options: Dict) -> Dict[str, Any]:
        """Fit Troe parameters using hybrid NLOpt + gradient-based optimization."""
        self.logger.info("Starting hybrid optimization")

        active_indices = jnp.where(self.param_mask)[0]
        base_params = self.estimate_initial_params()
        bounds_low = np.array(
            [
                base_params[0] - 10,
                base_params[1] - 5,
                base_params[2] / self.R_gas - 30000,
                base_params[3] - 10,
                base_params[4] - 5,
                base_params[5] / self.R_gas - 30000,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )[active_indices]
        bounds_high = np.array(
            [
                base_params[0] + 10,
                base_params[1] + 5,
                base_params[2] / self.R_gas + 30000,
                base_params[3] + 10,
                base_params[4] + 5,
                base_params[5] / self.R_gas + 30000,
                1.0,
                1e5,
                1e30,
                1e30,
            ]
        )[active_indices]

        # Phase 1: Global search with NLOpt
        self.logger.info("\nPhase 1: Global optimization with NLOpt")

        # Create NLOpt optimizer
        steps_file = nlopt_options["steps_file"]
        opt = nlopt.opt(nlopt.GN_ISRES, len(active_indices))
        opt.set_lower_bounds(bounds_low)
        opt.set_upper_bounds(bounds_high)
        opt.set_maxeval(nlopt_options["max_eval"])
        opt.set_ftol_rel(nlopt_options["ftol_rel"])
        opt.set_xtol_rel(nlopt_options["xtol_rel"])

        # Define objective function for NLOpt
        best_loss = float("inf")
        best_params = base_params[active_indices].copy()
        nlopt_step = 0

        def objective(x, grad):
            nonlocal best_loss, best_params, nlopt_step, steps_file

            full_params = base_params.copy()
            full_params = full_params.at[active_indices].set(x)
            loss_value = float(self.loss(jnp.array(full_params)))

            if loss_value < best_loss:
                best_loss = loss_value
                best_params = x.copy()

            if nlopt_step % steps_file == 0:
                self.logger.info(f"  Step {nlopt_step}: loss = {loss_value:.6e}")

            nlopt_step += 1

            return loss_value

        opt.set_min_objective(objective)

        # Run global optimization
        try:
            x = opt.optimize(best_params)
            self.logger.info(f"NLOpt finished with status: {opt.last_optimize_result()}")
            self.logger.info(f"Found minimum at: {best_loss:.6e}")
        except nlopt.RoundoffLimited:
            self.logger.info("NLOpt stopped due to roundoff errors")
        except Exception as e:
            self.logger.info(f"NLOpt stopped with error: {str(e)}")

        return self._create_falloff_dict(self.plog.name, jnp.array(best_params, dtype=jnp.float64))

    @staticmethod
    def _create_falloff_dict(name: str, params: Array) -> Dict[str, Any]:
        R_gas = jnp.float64(1.987)
        return {
            "name": name,
            "type": "falloff",
            "falloff-type": "troe",
            "rate-constant": {
                "lpl-coefficients": [jnp.exp(params[0]), params[1], params[2] * R_gas],
                "hpl-coefficients": [jnp.exp(params[3]), params[4], params[5] * R_gas],
                "falloff-coefficients": [params[6], params[7], params[8], params[9]],
            },
        }

    @staticmethod
    def _setup_logging(log_dir: Optional[str] = None) -> logging.Logger:
        logger = logging.getLogger("PlogRefitter")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter("%(message)s")

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        log_path = Path(log_dir) if log_dir else Path.cwd()
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / "plog_refitter.log", mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger
