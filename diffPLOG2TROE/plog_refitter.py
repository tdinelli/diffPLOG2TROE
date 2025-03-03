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
    fitting_mode: str

    def __init__(
        self,
        plog_dict: Dict[str, Any],
        T_range: Tuple[Float64, Float64],
        P_range: Tuple[Float64, Float64],
        n_T: int = 100,
        n_P: int = 100,
        param_config: Optional[Dict[str, Union[bool, float, Dict[str, Any]]]] = None,
        fitting_mode: str = "single",
        falloff_type: str = "troe",
        loss_name: str = "log",
        log_name: str = "refitter.log",
    ) -> None:
        if fitting_mode not in ["single", "duplicate"]:
            raise ValueError(f"Invalid fitting_mode: {fitting_mode}. Valid types are: single | duplicate")

        if falloff_type not in ["lindemann", "troe", "sri"]:
            raise ValueError(f"Unsupported falloff type: {falloff_type}. Valid types are: lindemann | troe | sri.")

        self.fitting_mode = fitting_mode.lower()
        self.falloff_type = falloff_type.lower()

        self.loss_name = loss_name

        # Start logging
        self.logger = self._setup_logging(log_name)
        self.logger.info("=" * 89)
        if fitting_mode == "single":
            self.logger.info(f"Plog to {falloff_type.upper()} refitter")
        else:
            self.logger.info(f"Plog to Duplicate Reactions ({falloff_type} + Lindemann) refitter")

        self.logger.info(f" Temperature range [K]: {T_range}")
        self.logger.info(f" Pressure range [atm]: {P_range}")
        self.logger.info(f" Loss function: {self.loss_name}")

        # Generate training data
        self.plog = Plog(plog_dict)
        self.T_range = jnp.linspace(T_range[0], T_range[1], n_T)
        self.P_range = jnp.logspace(jnp.log10(P_range[0]), jnp.log10(P_range[1]), n_P)
        self.k_plog = self.plog.kinetic_constant(self.T_range, self.P_range)

        # Define parameters based on fitting mode
        if falloff_type == "troe":
            falloff_param_names = ["A_low", "n_low", "E_low", "A_high", "n_high", "E_high", "A", "T3", "T1", "T2"]
        elif falloff_type == "sri":
            falloff_param_names = ["A_low", "n_low", "E_low", "A_high", "n_high", "E_high", "a", "b", "c", "d", "e"]
        else:  # lindemann
            falloff_param_names = ["A_low", "n_low", "E_low", "A_high", "n_high", "E_high"]

        if fitting_mode == "duplicate":
            duplicate_param_names = [
                "A_low_lind",
                "n_low_lind",
                "E_low_lind",
                "A_high_lind",
                "n_high_lind",
                "E_high_lind",
            ]
        else:  # single
            duplicate_param_names = []

        self.param_names = falloff_param_names + duplicate_param_names

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

    @eqx.filter_jit
    def loss(self, params: Array) -> Float64:
        """Compute the loss between PLOG and estimate rate constant based on the fitting type selected."""
        falloff_dict = self._create_falloff_dict(self.plog.name, params)
        falloff = FallOff(falloff_dict)
        k_falloff = falloff.kinetic_constant(self.T_range, self.P_range)

        if self.fitting_mode == "duplicate":
            lindemann_dict = self._create_lindemann_dict(self.plog.name, params)
            lindemann = FallOff(lindemann_dict)
            k_lindemann = lindemann.kinetic_constant(self.T_range, self.P_range)
        else:
            k_lindemann = jnp.zeros_like(k_falloff)

        k_estimated = k_falloff + k_lindemann

        if self.loss_name == "squared-error":
            residuals = losses.squared_error(k_estimated, self.k_plog)
            loss = jnp.mean(residuals)
        elif self.loss_name == "log-squared-error":
            residuals = losses.squared_error(jnp.log(k_estimated), jnp.log(self.k_plog))
            loss = jnp.mean(residuals)
        elif self.loss_name == "l2":
            l2_norm = losses.l2_loss(k_estimated, self.k_plog)
            loss = jnp.mean(l2_norm)
        elif self.loss_name == "log-l2":
            l2_norm = losses.l2_loss(jnp.log(k_estimated), jnp.log(self.k_plog))
            loss = jnp.mean(l2_norm)
        elif self.loss_name == "logcosh":
            logcosh = losses.log_cosh(k_estimated, self.k_plog)
            loss = jnp.mean(logcosh)
        elif self.loss_name == "log-logcosh":
            logcosh = losses.log_cosh(jnp.log(k_estimated), jnp.log(self.k_plog))
            loss = jnp.mean(logcosh)
        elif self.loss_name == "huber":
            pass
        elif self.loss_name == "log-huber":
            pass
        else:
            raise ValueError(f"Unknown loss type {self.loss_name}")

        return loss

    def fit(self, nlopt_options: Optional[Dict[str, Any]] = None, optax_options: Optional[Dict[str, Any]] = None):
        self.logger.info("")
        self.logger.info("=" * 89)
        self.logger.info("Starting optimization")

        active_indices = jnp.where(self.param_mask)[0]
        base_params = self._estimate_initial_params()
        bounds_low, bounds_high = self._create_boundaries(base_params)

        if nlopt_options is not None:
            nlopt_optimizer = NLOptWrapper(nlopt_options, (bounds_low, bounds_high), self.logger)
            results = nlopt_optimizer.optimize(self.loss, base_params, active_indices)
            optimized_params = results["params"]
        elif optax_options is not None:
            optax_optimizer = OptaxWrapper(optax_options, (bounds_low, bounds_high), self.logger)
            results = optax_optimizer.optimize(self.loss, base_params, active_indices)
            optimized_params = results["params"]
        else:
            self.logger.info("No optimizer options provided, using initial parameter estimates")
            optimized_params = base_params

        # Return results based on fitting mode
        if self.fitting_mode == "single":
            if self.falloff_type == "troe":
                return self._create_falloff_dict(self.plog.name, optimized_params)
            elif self.falloff_type == "sri":
                return self._create_falloff_dict(self.plog.name, optimized_params)
            else:  # lindemann
                return self._create_lindemann_dict(self.plog.name, optimized_params)
        else:  # duplicate mode
            troe_dict = self._create_falloff_dict(self.plog.name, optimized_params)
            lind_dict = self._create_lindemann_dict(self.plog.name, optimized_params)
            return troe_dict, lind_dict

    def _estimate_initial_params(self) -> Array:
        """Estimate initial parameters based on the selected fitting mode. This function is bad but I don't have time now!"""
        self.logger.info("First guess estimate of the parameters:")
        params_dict = {}

        T_mean = jnp.mean(self.T_range)

        # Estimate high pressure limit from highest pressure PLOG
        lnA_high, n_high, EaR_high = self._fit_arrhenius(self.k_plog[-1])

        # Estimate low pressure limit from lowest pressure PLOG
        lnA_low, n_low, EaR_low = self._fit_arrhenius(
            self.k_plog[0] / ((self.plog.p_levels[0] / (constants.R_L_atm_K_mol * self.T_range)) * jnp.float64(0.001))
        )

        # Default values for different falloff parameters
        default_troe = {"A": 0.5, "T3": T_mean * 0.7, "T1": T_mean * 0.2, "T2": T_mean * 1.5}
        default_sri = {"a": 1.0, "b": 0.5 * T_mean, "c": T_mean, "d": 1.0, "e": 0.0}

        if self.fitting_mode == "single":
            for i, (name, value) in enumerate(
                [
                    ("lnA_low", lnA_low),
                    ("n_low", n_low),
                    ("EaR_low", EaR_low),
                    ("lnA_high", lnA_high),
                    ("n_high", n_high),
                    ("EaR_high", EaR_high),
                ]
            ):
                if self.initial_values[i] is not None:
                    if name.startswith("lnA"):
                        params_dict[name] = jnp.log(self.initial_values[i])
                    elif name.startswith("EaR"):
                        params_dict[name] = self.initial_values[i] / constants.R_cal_mol
                    else:
                        params_dict[name] = self.initial_values[i]
                else:
                    params_dict[name] = value

            # Store falloff parameters based on type
            if self.falloff_type == "troe":
                for i, param in enumerate(["A", "T3", "T1", "T2"], start=6):
                    if i < len(self.initial_values) and self.initial_values[i] is not None:
                        params_dict[param] = self.initial_values[i]
                    else:
                        params_dict[param] = default_troe[param]

                # Log the parameters
                self.logger.info("\nTroe parameters:")
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
                self.logger.info(
                    "  Troe parameters (A, T3, T1, T2): {:.3f}, {:.3e}, {:.3e}, {:.3e}\n".format(
                        params_dict["A"], params_dict["T3"], params_dict["T1"], params_dict["T2"]
                    )
                )

                params = jnp.array(
                    [
                        params_dict["lnA_low"],
                        params_dict["n_low"],
                        params_dict["EaR_low"],
                        params_dict["lnA_high"],
                        params_dict["n_high"],
                        params_dict["EaR_high"],
                        params_dict["A"],
                        params_dict["T3"],
                        params_dict["T1"],
                        params_dict["T2"],
                    ],
                    dtype=jnp.float64,
                )

            elif self.falloff_type == "sri":
                for i, param in enumerate(["a", "b", "c", "d", "e"], start=6):
                    if i < len(self.initial_values) and self.initial_values[i] is not None:
                        params_dict[param] = self.initial_values[i]
                    else:
                        params_dict[param] = default_sri[param]

                # Log the parameters
                self.logger.info("\nSRI parameters:")
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
                self.logger.info(
                    "  SRI parameters (a, b, c, d, e): {:.3f}, {:.3e}, {:.3e}, {:.3f}, {:.3f}\n".format(
                        params_dict["a"], params_dict["b"], params_dict["c"], params_dict["d"], params_dict["e"]
                    )
                )

                params = jnp.array(
                    [
                        params_dict["lnA_low"],
                        params_dict["n_low"],
                        params_dict["EaR_low"],
                        params_dict["lnA_high"],
                        params_dict["n_high"],
                        params_dict["EaR_high"],
                        params_dict["a"],
                        params_dict["b"],
                        params_dict["c"],
                        params_dict["d"],
                        params_dict["e"],
                    ],
                    dtype=jnp.float64,
                )

            else:  # lindemann
                # Log the parameters
                self.logger.info("\nLindemann parameters:")
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

                params = jnp.array(
                    [
                        params_dict["lnA_low"],
                        params_dict["n_low"],
                        params_dict["EaR_low"],
                        params_dict["lnA_high"],
                        params_dict["n_high"],
                        params_dict["EaR_high"],
                    ],
                    dtype=jnp.float64,
                )

        else:  # duplicate mode
            # ---------------------------------------------------------
            # Estimate Troe reaction parameters (first reaction)
            # ---------------------------------------------------------
            # Store Troe parameters
            for i, (name, value) in enumerate(
                [
                    ("lnA_low_troe", lnA_low),
                    ("n_low_troe", n_low),
                    ("EaR_low_troe", EaR_low),
                    ("lnA_high_troe", lnA_high),
                    ("n_high_troe", n_high),
                    ("EaR_high_troe", EaR_high),
                ]
            ):
                if self.initial_values[i] is not None:
                    if name.startswith("lnA"):
                        params_dict[name] = jnp.log(self.initial_values[i])
                    elif name.startswith("EaR"):
                        params_dict[name] = self.initial_values[i] / constants.R_cal_mol
                    else:
                        params_dict[name] = self.initial_values[i]
                else:
                    params_dict[name] = value

            # Store Troe falloff parameters
            for i, param in enumerate(["A_troe", "T3_troe", "T1_troe", "T2_troe"], start=6):
                if i < len(self.initial_values) and self.initial_values[i] is not None:
                    params_dict[param] = self.initial_values[i]
                else:
                    base_param = param.split("_")[0]
                    params_dict[param] = default_troe[base_param]

            # ---------------------------------------------------------
            # Estimate Lindemann reaction parameters (second reaction)
            # ---------------------------------------------------------

            # For the Lindemann reaction, we'll use different initial values
            lnA_high_lind = lnA_high - 0.3  # Slightly lower A factor
            n_high_lind = n_high + 0.1  # Slightly higher temperature dependence
            EaR_high_lind = EaR_high * 0.95  # Slightly lower activation energy

            lnA_low_lind = lnA_low - 0.3
            n_low_lind = n_low + 0.1
            EaR_low_lind = EaR_low * 0.95

            # Store Lindemann parameters
            for i, (name, value) in enumerate(
                [
                    ("lnA_low_lind", lnA_low_lind),
                    ("n_low_lind", n_low_lind),
                    ("EaR_low_lind", EaR_low_lind),
                    ("lnA_high_lind", lnA_high_lind),
                    ("n_high_lind", n_high_lind),
                    ("EaR_high_lind", EaR_high_lind),
                ],
                start=10,
            ):
                if i < len(self.initial_values) and self.initial_values[i] is not None:
                    if name.startswith("lnA"):
                        params_dict[name] = jnp.log(self.initial_values[i])
                    elif name.startswith("EaR"):
                        params_dict[name] = self.initial_values[i] / constants.R_cal_mol
                    else:
                        params_dict[name] = self.initial_values[i]
                else:
                    params_dict[name] = value

            # Log the initial parameters
            self.logger.info("\nTroe reaction parameters:")
            self.logger.info(
                "  High pressure limit (A, n, Ea): {:.3e}, {:.3f}, {:.3e}".format(
                    jnp.exp(params_dict["lnA_high_troe"]),
                    params_dict["n_high_troe"],
                    params_dict["EaR_high_troe"] * constants.R_cal_mol,
                )
            )
            self.logger.info(
                "  Low pressure limit (A, n, Ea): {:.3e}, {:.3f}, {:.3e}".format(
                    jnp.exp(params_dict["lnA_low_troe"]),
                    params_dict["n_low_troe"],
                    params_dict["EaR_low_troe"] * constants.R_cal_mol,
                )
            )
            self.logger.info(
                "  Troe parameters (A, T3, T1, T2): {:.3f}, {:.3e}, {:.3e}, {:.3e}\n".format(
                    params_dict["A_troe"], params_dict["T3_troe"], params_dict["T1_troe"], params_dict["T2_troe"]
                )
            )

            self.logger.info("\nLindemann reaction parameters:")
            self.logger.info(
                "  High pressure limit (A, n, Ea): {:.3e}, {:.3f}, {:.3e}".format(
                    jnp.exp(params_dict["lnA_high_lind"]),
                    params_dict["n_high_lind"],
                    params_dict["EaR_high_lind"] * constants.R_cal_mol,
                )
            )
            self.logger.info(
                "  Low pressure limit (A, n, Ea): {:.3e}, {:.3f}, {:.3e}".format(
                    jnp.exp(params_dict["lnA_low_lind"]),
                    params_dict["n_low_lind"],
                    params_dict["EaR_low_lind"] * constants.R_cal_mol,
                )
            )

            # Create array of parameters in the order of param_names
            params = jnp.array(
                [
                    params_dict["lnA_low_troe"],
                    params_dict["n_low_troe"],
                    params_dict["EaR_low_troe"],
                    params_dict["lnA_high_troe"],
                    params_dict["n_high_troe"],
                    params_dict["EaR_high_troe"],
                    params_dict["A_troe"],
                    params_dict["T3_troe"],
                    params_dict["T1_troe"],
                    params_dict["T2_troe"],
                    params_dict["lnA_low_lind"],
                    params_dict["n_low_lind"],
                    params_dict["EaR_low_lind"],
                    params_dict["lnA_high_lind"],
                    params_dict["n_high_lind"],
                    params_dict["EaR_high_lind"],
                ],
                dtype=jnp.float64,
            )

        return params

    def _create_boundaries(self, params: Array) -> Tuple[Array, Array]:
        if self.fitting_mode == "single":
            common_bounds_low = [
                params[0] - 20,  # lnA_low
                params[1] - 5,  # n_low
                params[2] / constants.R_cal_mol - 70000,  # EaR_low
                params[3] - 20,  # lnA_high
                params[4] - 5,  # n_high
                params[5] / constants.R_cal_mol - 70000,  # EaR_high
            ]

            common_bounds_high = [
                params[0] + 20,  # lnA_low
                params[1] + 5,  # n_low
                params[2] / constants.R_cal_mol + 70000,  # EaR_low
                params[3] + 20,  # lnA_high
                params[4] + 5,  # n_high
                params[5] / constants.R_cal_mol + 70000,  # EaR_high
            ]

            if self.falloff_type == "troe":
                bounds_low = jnp.array(common_bounds_low + [0, 0.0, 0.0, 0.0])
                bounds_high = jnp.array(common_bounds_high + [1.0, 1e5, 1e30, 1e30])
            elif self.falloff_type == "sri":
                bounds_low = jnp.array(common_bounds_low + [0.0, 0.0, 0.0, 0.0, -1.0])
                bounds_high = jnp.array(common_bounds_high + [1e2, 1e5, 1e5, 2.0, 2.0])
            else:  # lindemann
                bounds_low = jnp.array(common_bounds_low)
                bounds_high = jnp.array(common_bounds_high)
        else:  # duplicate mode
            # Bounds for Troe reaction
            troe_bounds_low = [
                params[0] - 20,  # lnA_low_troe
                params[1] - 5,  # n_low_troe
                params[2] - 70000,  # EaR_low_troe
                params[3] - 20,  # lnA_high_troe
                params[4] - 5,  # n_high_troe
                params[5] - 70000,  # EaR_high_troe
                0.0,  # A_troe
                0.0,  # T3_troe
                0.0,  # T1_troe
                0.0,  # T2_troe
            ]

            troe_bounds_high = [
                params[0] + 20,  # lnA_low_troe
                params[1] + 5,  # n_low_troe
                params[2] + 70000,  # EaR_low_troe
                params[3] + 20,  # lnA_high_troe
                params[4] + 5,  # n_high_troe
                params[5] + 70000,  # EaR_high_troe
                1.0,  # A_troe
                1e5,  # T3_troe
                1e30,  # T1_troe
                1e30,  # T2_troe
            ]

            # Bounds for Lindemann reaction
            lind_bounds_low = [
                params[10] - 20,  # lnA_low_lind
                params[11] - 5,  # n_low_lind
                params[12] - 70000,  # EaR_low_lind
                params[13] - 20,  # lnA_high_lind
                params[14] - 5,  # n_high_lind
                params[15] - 70000,  # EaR_high_lind
            ]

            lind_bounds_high = [
                params[10] + 20,  # lnA_low_lind
                params[11] + 5,  # n_low_lind
                params[12] + 70000,  # EaR_low_lind
                params[13] + 20,  # lnA_high_lind
                params[14] + 5,  # n_high_lind
                params[15] + 70000,  # EaR_high_lind
            ]

            bounds_low = jnp.array(troe_bounds_low + lind_bounds_low)
            bounds_high = jnp.array(troe_bounds_high + lind_bounds_high)

        return bounds_low, bounds_high

    def _fit_arrhenius(self, k_values: Array) -> Tuple[Float64, Float64, Float64]:
        log_k = jnp.log(k_values)
        log_T = jnp.log(self.T_range)
        inv_T = 1.0 / self.T_range
        X = jnp.vstack([jnp.ones_like(log_T), log_T, -inv_T]).T
        beta = jnp.linalg.lstsq(X, log_k, rcond=None)[0]
        return beta[0], beta[1], beta[2]  # lnA, n, Ea/R

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

    def _create_lindemann_dict(self, name: str, params: Array) -> Dict[str, Any]:
        """Create a Lindemann falloff dictionary from parameters."""
        return {
            "name": name,
            "type": "falloff",
            "falloff-type": "lindemann",
            "rate-constant": {
                "lpl-coefficients": [jnp.exp(params[10]), params[11], params[12] * constants.R_cal_mol],
                "hpl-coefficients": [jnp.exp(params[13]), params[14], params[15] * constants.R_cal_mol],
            },
        }

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
