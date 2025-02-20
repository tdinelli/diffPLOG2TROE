from typing import Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float64

from .rate_constants import FallOff, Plog
from .rate_constants.rate_interpreter import parse_rate_constant


class PlogRefitter(eqx.Module):
    """Fits a FallOff Troe function to PLOG rate data."""

    plog: Plog
    param_mask: Array
    param_names: List[str]
    T_range: Array
    P_range: Array
    k_plog: Array

    def __init__(
        self,
        plog_dict: Dict,
        T_range: Tuple[Float64, Float64],
        P_range: Tuple[Float64, Float64],
        n_T: int = 50,
        n_P: int = 40,
    ) -> None:
        self.plog = Plog(plog_dict)

        # Generate training data
        self.T_range = jnp.logspace(jnp.log10(T_range[0]), jnp.log10(T_range[1]), n_T)
        self.P_range = jnp.logspace(jnp.log10(P_range[0]), jnp.log10(P_range[1]), n_P)

        # Compute the reference plog rate values
        self.k_plog = self.plog.kinetic_constant(self.T_range, self.P_range)

        # Define parameter names for reference
        self.param_names = ["A_low", "n_low", "E_low", "A_high", "n_high", "E_high", "A", "T3", "T1", "T2"]

        # By default all the TROE parameters are going under the optimization
        self.param_mask = jnp.ones(len(self.param_names))

    def estimate_initial_params(self):
        """Estimate initial parameters from the computed PLOG data."""
        # 1. Extract rate constants at lowest and highest pressures
        k_low_p = self.k_plog[0]
        k_high_p = self.k_plog[-1]

        # 2. Estimate Arrhenius parameters for low and high pressure limits using log-linear regression
        log_k_low = jnp.log(k_low_p)
        log_k_high = jnp.log(k_high_p)
        log_T = jnp.log(self.T_range)
        inv_T = 1.0 / self.T_range

        # 3. For low pressure limit: ln(k) = ln(A) + n*ln(T) - E/RT
        X_low = jnp.vstack([jnp.ones_like(log_T), log_T, -inv_T]).T
        beta_low = jnp.linalg.lstsq(X_low, log_k_low)[0]
        A_low = jnp.exp(beta_low[0])
        n_low = beta_low[1]
        E_low = beta_low[2] * jnp.float64(1.987)

        # 4. For high pressure limit: ln(k) = ln(A) + n*ln(T) - E/RT
        X_high = jnp.vstack([jnp.ones_like(log_T), log_T, -inv_T]).T
        beta_high = jnp.linalg.lstsq(X_high, log_k_high)[0]
        A_high = jnp.exp(beta_high[0])
        n_high = beta_high[1]
        E_high = beta_high[2] * jnp.float64(1.987)

        # 5. Initial Troe parameters - reasonable defaults
        T_mean = jnp.mean(self.T_range)
        alpha = 0.5
        T3 = T_mean * 0.7
        T1 = T_mean * 0.2
        T2 = T_mean * 1.5

        return jnp.array([A_low, n_low, E_low, A_high, n_high, E_high, alpha, T3, T1, T2], dtype=jnp.float64)

    def set_fixed_params(self, fixed_params: Dict[str, float]) -> None:
        """Set which parameters to hold fixed during optimization."""
        # Reset mask to optimize all parameters
        self.param_mask = jnp.ones(len(self.param_names))

        # Set mask to 0 for fixed parameters
        for param_name, _ in fixed_params.items():
            if param_name in self.param_names:
                idx = self.param_names.index(param_name)
                self.param_mask = self.param_mask.at[idx].set(0)

    @eqx.filter_jit
    def loss_fn(self, params: Array, fixed_params: Dict[str, float]) -> Float64:
        """Compute loss between PLOG and fitted Troe rates with parameter masking."""
        # Combine optimized and fixed parameters
        full_params = params.copy()
        for param_name, value in fixed_params.items():
            if param_name in self.param_names:
                idx = self.param_names.index(param_name)
                full_params = full_params.at[idx].set(value)

        k_pred = self._troe_rate(full_params, self.T_range, self.P_range)
        # Use relative error in log space
        log_diff = jnp.log(k_pred + 1e-30) - jnp.log(self.k_plog + 1e-30)
        return jnp.mean(log_diff**2)

    def fit(self, fixed_params: Dict[str, float] = {}, n_steps: int = 1000, learning_rate: float = 1e-3):
        """Fit Troe parameters to match PLOG data."""

        # Set the fixed parameters
        self.set_fixed_params(fixed_params)

        # Get initial parameter estimates
        init_params = self.estimate_initial_params()

        # Setup the TROE object

        # Setup optimizer
        optimizer = optax.adam(learning_rate=learning_rate)
        opt_state = optimizer.init(init_params)

        # Optimization loop
        losses = []
        params = init_params

        for i in range(n_steps):
            loss_val, grads = jax.value_and_grad(self.loss_fn)(params, fixed_params)

            # Apply mask to gradients
            grads = grads * self.param_mask

            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            losses.append(float(loss_val))

            if i % 100 == 0:
                print(f"Step {i}, Loss: {loss_val:.6f}")

        # Create dictionary of final parameters
        # final_params = {}
        # for name, value in zip(self.param_names, params):
        #     final_params[name] = float(value)
        #
        # return final_params, losses

    def create_falloff_dict(self, fitted_params: Dict[str, float]) -> Dict:
        """Helper function to create FallOff dictionary from the first guesses of the parameters."""
        return {
            "name": self.plog.name,
            "type": "falloff",
            "falloff-type": "troe",
            "rate-constant": {
                "lpl-coefficients": [fitted_params["A_low"], fitted_params["n_low"], fitted_params["E_low"]],
                "hpl-coefficients": [fitted_params["A_high"], fitted_params["n_high"], fitted_params["E_high"]],
                "falloff-coefficients": [
                    fitted_params["alpha"],
                    fitted_params["T3"],
                    fitted_params["T1"],
                    fitted_params["T2"],
                ],
            },
        }
