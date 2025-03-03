import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64
from optax import losses

from .optimization import NLOptWrapper, OptaxWrapper
from .physical_constants import PhysicalConstants as constants
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
    loss_name: str
    falloff_type: str

    def __init__(
        self,
        plog_dict: Dict[str, Any],
        T_range: Tuple[Float64, Float64],
        P_range: Tuple[Float64, Float64],
        n_T: int = 100,
        n_P: int = 100,
        param_config: Optional[Dict[str, Union[bool, float, Dict[str, Any]]]] = None,
        loss_name: str = "log",
        log_name: str = "refitter.log",
        falloff_type: str = "troe",
    ) -> None:
        self.logger = self._setup_logging(log_name)
        self.logger.info("=" * 89)
        self.logger.info(f"Plog 2 {falloff_type.upper()} refitter")
        self.logger.info(f" Temperature range [K]: {T_range}")
        self.logger.info(f" Pressure range [atm]: {P_range}")

        self.falloff_type = falloff_type.lower()
        if self.falloff_type not in ["troe", "sri"]:
            raise ValueError(f"Unsupported falloff type: {falloff_type}. Must be 'troe' or 'sri'.")

        self.loss_name = loss_name
        self.logger.info(f" Loss function: {self.loss_name}")

        # Generate training data
        self.plog = Plog(plog_dict)
        self.T_range = jnp.linspace(T_range[0], T_range[1], n_T)
        self.P_range = jnp.logspace(jnp.log10(P_range[0]), jnp.log10(P_range[1]), n_P)
        self.k_plog = self.plog.kinetic_constant(self.T_range, self.P_range)

        self.param_names = ["A_low", "n_low", "E_low", "A_high", "n_high", "E_high"]
        if self.falloff_type == "troe":
            self.param_names.extend(["A", "T3", "T1", "T2"])
        else:  # sri
            self.param_names.extend(["a", "b", "c", "d", "e"])

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

    def _fit_arrhenius(self, k_values: Array) -> Tuple[Float64, Float64, Float64]:
        log_k = jnp.log(k_values)
        log_T = jnp.log(self.T_range)
        inv_T = 1.0 / self.T_range
        X = jnp.vstack([jnp.ones_like(log_T), log_T, -inv_T]).T
        beta = jnp.linalg.lstsq(X, log_k, rcond=None)[0]
        return beta[0], beta[1], beta[2]  # lnA, n, Ea/R

    def estimate_initial_params(self) -> Array:
        """Estimate initial parameters with consideration for fixed values."""
        self.logger.info("First guess estimate of the parameters:")
        params_dict = {}

        # ====================================================================
        # Low pressure limit estimation
        if any(not self.initial_values[i] for i in range(3)):  # If any low-pressure params need estimation
            lnA_low, n_low, EaR_low = self._fit_arrhenius(
                self.k_plog[0]
                / ((self.plog.p_levels[0] / (constants.R_L_atm_K_mol * self.T_range)) * jnp.float64(0.001))
            )
            params_dict.update(
                {
                    "lnA_low": lnA_low if self.param_mask[0] else jnp.log(self.initial_values[0]),
                    "n_low": n_low if self.param_mask[1] else self.initial_values[1],
                    "EaR_low": EaR_low if self.param_mask[2] else self.initial_values[2] / constants.R_cal_mol,
                }
            )
        else:  # All low-pressure params are fixed
            params_dict.update(
                {
                    "lnA_low": jnp.log(self.initial_values[0]),
                    "n_low": self.initial_values[1],
                    "EaR_low": self.initial_values[2] / constants.R_cal_mol,
                }
            )

        # ====================================================================
        # High pressure limit estimation
        if any(not self.initial_values[i] for i in range(3, 6)):  # If any high-pressure params need estimation
            lnA_high, n_high, EaR_high = self._fit_arrhenius(self.k_plog[-1])
            params_dict.update(
                {
                    "lnA_high": lnA_high if self.param_mask[3] else jnp.log(self.initial_values[3]),
                    "n_high": n_high if self.param_mask[4] else self.initial_values[4],
                    "EaR_high": EaR_high if self.param_mask[5] else self.initial_values[5] / constants.R_cal_mol,
                }
            )
        else:  # All high-pressure params are fixed
            params_dict.update(
                {
                    "lnA_high": jnp.log(self.initial_values[3]),
                    "n_high": self.initial_values[4],
                    "EaR_high": self.initial_values[5] / constants.R_cal_mol,
                }
            )

        T_mean = jnp.mean(self.T_range)
        if self.falloff_type == "troe":
            # ================================================================
            # TROE parameters estimation
            default_troe = {"A": 0.5, "T3": T_mean * 0.7, "T1": T_mean * 0.2, "T2": T_mean * 1.5}

            for i, param in enumerate(["A", "T3", "T1", "T2"], start=6):
                if self.initial_values and self.initial_values[i] is not None:
                    params_dict[param] = self.initial_values[i]
                else:
                    params_dict[param] = default_troe[param]

            self.logger.info(
                "  Troe parameters (A, T3, T1, T2): {:.3f}, {:.3e}, {:.3e}, {:.3e}\n".format(
                    params_dict["A"], params_dict["T3"], params_dict["T1"], params_dict["T2"]
                )
            )
        else:  # sri
            # ================================================================
            # SRI parameters estimation
            default_sri = {"a": 1.0, "b": 0.5 * T_mean, "c": T_mean, "d": 1.0, "e": 0.0}

            for i, param in enumerate(["a", "b", "c", "d", "e"], start=6):
                if self.initial_values and self.initial_values[i] is not None:
                    params_dict[param] = self.initial_values[i]
                else:
                    params_dict[param] = default_sri[param]

            self.logger.info(
                "  SRI parameters (a, b, c, d, e): {:.3f}, {:.3e}, {:.3e}, {:.3f}, {:.3f}\n".format(
                    params_dict["a"], params_dict["b"], params_dict["c"], params_dict["d"], params_dict["e"]
                )
            )

        self.logger.info(
            "  High pressure limit (A, n, Ea): {:.3e}, {:.3f}, {:.3e}".format(
                jnp.exp(params_dict["lnA_high"]),
                params_dict["n_high"],
                params_dict["EaR_high"] * constants.R_cal_mol,
            )
        )
        self.logger.info(
            "  Low pressure limit (A, n, Ea): {:.3e}, {:.3f}, {:.3e}".format(
                jnp.exp(params_dict["lnA_low"]),
                params_dict["n_low"],
                params_dict["EaR_low"] * constants.R_cal_mol,
            )
        )

        params = jnp.array([params_dict[name] for name in params_dict.keys()], dtype=jnp.float64)

        return params

    @eqx.filter_jit
    def loss(self, params: Array) -> Float64:
        """Compute the loss between PLOG and fitted Troe/SRI rates."""
        falloff_dict = self._create_falloff_dict(self.plog.name, params)
        falloff = FallOff(falloff_dict)
        k_troe = falloff.kinetic_constant(self.T_range, self.P_range)

        if self.loss_name == "squared-error":
            residuals = losses.squared_error(k_troe, self.k_plog)
            loss = jnp.mean(residuals)
        elif self.loss_name == "log-squared-error":
            residuals = losses.squared_error(jnp.log(k_troe), jnp.log(self.k_plog))
            loss = jnp.mean(residuals)
        elif self.loss_name == "l2":
            l2_norm = losses.l2_loss(k_troe, self.k_plog)
            loss = jnp.mean(l2_norm)
        elif self.loss_name == "log-l2":
            l2_norm = losses.l2_loss(jnp.log(k_troe), jnp.log(self.k_plog))
            loss = jnp.mean(l2_norm)
        elif self.loss_name == "logcosh":
            logcosh = losses.log_cosh(k_troe, self.k_plog)
            loss = jnp.mean(logcosh)
        elif self.loss_name == "log-logcosh":
            logcosh = losses.log_cosh(jnp.log(k_troe), jnp.log(self.k_plog))
            loss = jnp.mean(logcosh)

        # if self.loss_name == "relative-ratio":
        #     squared_errors = jnp.sum((1 - (k_troe / self.k_plog)) ** 2)
        #     loss = jnp.sqrt(squared_errors)
        # elif self.loss_name == "log":
        #     log_k_troe = jnp.log(k_troe + 1)
        #     log_k_plog = jnp.log(self.k_plog + 1)
        #     loss = jnp.mean((log_k_troe - log_k_plog) ** 2)
        # elif self.loss_name == "max":
        #     squared_errors = (1 - (k_troe / self.k_plog)) ** 2
        #     loss = jnp.max(squared_errors)
        # elif self.loss_name == "rel-abs":
        #     rel_error = jnp.abs(1 - (k_troe / self.k_plog))
        #     loss = jnp.mean(rel_error**2)
        # elif self.loss_name == "huber":
        #     log_k_troe = jnp.log(k_troe + 1)
        #     log_k_plog = jnp.log(self.k_plog + 1)
        #     rel_error = jnp.mean((log_k_troe - log_k_plog) ** 2)
        #     delta = 0.1  # This can be adjusted as needed
        #
        #     # Apply Huber loss
        #     # For small errors (|x| <= delta): 0.5 * x^2
        #     # For large errors (|x| > delta): delta * (|x| - 0.5 * delta)
        #     abs_rel_error = jnp.abs(rel_error)
        #     loss = jnp.mean(
        #         jnp.where(
        #             abs_rel_error <= delta,
        #             0.5 * rel_error**2,
        #             delta * (abs_rel_error - 0.5 * delta),
        #         )
        #     )
        # elif self.loss_name == "ciao":
        #     log_k_troe = jnp.log(k_troe + 1)
        #     log_k_plog = jnp.log(self.k_plog + 1)
        #     residuals = log_k_troe - log_k_plog
        #     loss = jnp.mean(jnp.abs(residuals / log_k_plog))

        return loss

    def fit(self, nlopt_options: Optional[Dict[str, Any]] = None, optax_options: Optional[Dict[str, Any]] = None):
        self.logger.info("")
        self.logger.info("=" * 89)
        self.logger.info("Starting optimization")

        active_indices = jnp.where(self.param_mask)[0]
        base_params = self.estimate_initial_params()

        # Define bounds based on falloff type
        common_bounds_low = [
            base_params[0] - 20,  # lnA_low
            base_params[1] - 5,  # n_low
            base_params[2] / constants.R_cal_mol - 70000,  # EaR_low
            base_params[3] - 20,  # lnA_high
            base_params[4] - 5,  # n_high
            base_params[5] / constants.R_cal_mol - 70000,  # EaR_high
        ]

        common_bounds_high = [
            base_params[0] + 20,  # lnA_low
            base_params[1] + 5,  # n_low
            base_params[2] / constants.R_cal_mol + 70000,  # EaR_low
            base_params[3] + 20,  # lnA_high
            base_params[4] + 5,  # n_high
            base_params[5] / constants.R_cal_mol + 70000,  # EaR_high
        ]

        if self.falloff_type == "troe":
            bounds_low = jnp.array(common_bounds_low + [0, 0.0, 0.0, 0.0])
            bounds_high = jnp.array(common_bounds_high + [1.0, 1e5, 1e30, 1e30])
        else:  # sri
            bounds_low = jnp.array(common_bounds_low + [0.0, 0.0, 0.0, 0.0, -1.0])
            bounds_high = jnp.array(common_bounds_high + [1e2, 1e5, 1e5, 2.0, 2.0])

        if nlopt_options is not None:
            nlopt_optimizer = NLOptWrapper(nlopt_options, (bounds_low, bounds_high), self.logger)
            results = nlopt_optimizer.optimize(self.loss, base_params, active_indices)
            return self._create_falloff_dict(self.plog.name, results["params"])

        if optax_options is not None:
            optax_optimizer = OptaxWrapper(optax_options, (bounds_low, bounds_high), self.logger)
            results = optax_optimizer.optimize(self.loss, base_params, active_indices)
            return self._create_falloff_dict(self.plog.name, results["params"])

    def _create_falloff_dict(self, name: str, params: Array) -> Dict[str, Any]:
        """Create a falloff dictionary with the appropriate falloff type."""
        result = {
            "name": name,
            "type": "falloff",
            "falloff-type": self.falloff_type,
            "rate-constant": {
                "lpl-coefficients": [jnp.exp(params[0]), params[1], params[2] * constants.R_cal_mol],
                "hpl-coefficients": [jnp.exp(params[3]), params[4], params[5] * constants.R_cal_mol],
            },
        }

        if self.falloff_type == "troe":
            result["rate-constant"]["falloff-coefficients"] = [params[6], params[7], params[8], params[9]]
        else:  # sri
            result["rate-constant"]["falloff-coefficients"] = [params[6], params[7], params[8], params[9], params[10]]

        return result

    @staticmethod
    def _setup_logging(log_name: str) -> logging.Logger:
        logger = logging.getLogger("PlogRefitter")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter("%(message)s")

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        log_path = Path.cwd()
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / log_name, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger
