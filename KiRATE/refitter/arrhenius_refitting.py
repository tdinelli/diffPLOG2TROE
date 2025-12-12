"""
Copyright (c) 2025 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float64
from optimistix import RESULTS, LevenbergMarquardt, least_squares

from KiRATE.kinetics import Arrhenius
from KiRATE.refitter.utils import RefittingResult, compute_statistics
from KiRATE.utilities import constants


def compute_initial_guess(
    temperature: Float64[Array, "n"],
    log_rate_constant: Float64[Array, "n"],
    params: Optional[dict[str, float]] = None,
) -> Float64[Array, "n_params"]:
    """
    Compute initial parameter guess using linear least squares.

    Solves the linearized Arrhenius equation:
        ln(k) = ln(A) + n*ln(T) - Ea/(R*T)

    by constructing a design matrix and using least squares regression.

    Parameters
    ----------
    temperature : Float64[Array, "n"]
        Temperature values (K)
    log_rate_constant : Float64[Array, "n"]
        Natural logarithm of rate constants
    params : dict[str, float], optional
        Known parameters to fix during fitting. If provided, only unknown
        parameters will be included in the initial guess.

    Returns
    -------
    Float64[Array, "n_params"]
        Initial guess for free parameters. Order depends on which parameters
        are being fitted:
        - If params is None: [A, n, Ea]
        - Otherwise: subset of free parameters in order [A, n, Ea]

    Notes
    -----
    The linear least squares solution provides an excellent initial guess
    because the logarithm of the Arrhenius equation is linear in the
    transformed parameters [ln(A), n, Ea/R].

    For many well-conditioned problems, this initial guess is already
    very close to the optimal solution, leading to rapid convergence.
    """
    # ==================================================================================
    # CASE 1: Fit all three parameters
    if params is None:
        # Design matrix: [1, ln(T), -1/T]
        # Corresponds to: ln(k) = ln(A) + n*ln(T) - Ea/(R*T)
        X = jnp.column_stack([jnp.ones_like(temperature), jnp.log(temperature), -1.0 / temperature])
        coeffs, *_ = jnp.linalg.lstsq(X, log_rate_constant, rcond=None)

        # Return [A, n, Ea]
        # Note: coeffs[0] = ln(A), coeffs[2] = Ea/R
        return jnp.array([jnp.exp(coeffs[0]), coeffs[1], coeffs[2] * constants.R_cal_mol])

    # ==================================================================================
    # CASE 2: Fit subset of parameters
    else:
        known = {k: v for k, v in params.items() if v is not None}
        param_names = ["A", "n", "Ea"]

        # Build design matrix and adjust residual for known parameters
        # Start with full log_k
        residual = log_rate_constant.copy()
        design_cols = []

        # For each parameter, either subtract its contribution (if known)
        # or add to design matrix (if unknown)
        # Order: A, n, Ea

        # Handle A
        if "A" in known:
            # Known A: subtract ln(A) from residual
            residual = residual - jnp.log(known["A"])
        else:
            # Unknown A: add intercept column
            design_cols.append(jnp.ones_like(temperature))

        # Handle n
        if "n" in known:
            # Known n: subtract n*ln(T) from residual
            residual = residual - known["n"] * jnp.log(temperature)
        else:
            # Unknown n: add ln(T) column
            design_cols.append(jnp.log(temperature))

        # Handle Ea
        if "Ea" in known:
            # Known Ea: subtract -Ea/(R*T) from residual (which is adding Ea/(R*T))
            residual = residual + known["Ea"] / (constants.R_cal_mol * temperature)
        else:
            # Unknown Ea: add -1/T column
            design_cols.append(-1.0 / temperature)

        # Solve for unknowns
        if not design_cols:
            raise ValueError("No free parameters to fit (all are fixed)")

        X = jnp.column_stack(design_cols)
        coeffs, *_ = jnp.linalg.lstsq(X, residual, rcond=None)

        # Map coeffs back to parameters (A needs exp transform, Ea needs R scaling)
        p0 = []
        coeff_idx = 0
        for name in param_names:
            if name not in known:
                if name == "A":
                    p0.append(jnp.exp(coeffs[coeff_idx]))
                elif name == "Ea":
                    p0.append(coeffs[coeff_idx] * constants.R_cal_mol)
                else:  # n
                    p0.append(coeffs[coeff_idx])
                coeff_idx += 1

        return jnp.array(p0)


def refitter(
    temperature: ArrayLike,
    rate_constant: ArrayLike,
    params: Optional[dict[str, float]] = None,
    initial_guess: Optional[ArrayLike] = None,
    max_steps: int = 1000,
    atol: float = 1e-8,
    rtol: float = 1e-8,
) -> "RefittingResult":
    """
    Fit Arrhenius parameters (A, n, Ea) to temperature-rate constant data.

    Parameters
    ----------
    temperature : ArrayLike
        Temperature values (K) - accepts list, tuple, numpy array, or JAX array
    rate_constant : ArrayLike
        Rate constant values - accepts list, tuple, numpy array, or JAX array
    params : dict[str, float], optional
        Optional dict with known parameters {'A': val, 'n': val, 'Ea': val}.
        Any subset can be provided; unknown params will be fitted.
        If None, all three parameters will be fitted.
    initial_guess : ArrayLike, optional
        Initial guess for free parameters. If None, computed automatically
        using linear least squares. Order must match free parameters:
        - If params is None: [A, n, Ea]
        - Otherwise: values for free parameters in order [A, n, Ea]
    max_steps : int, optional
        Maximum number of steps for the least sqaure solver, by default 1000
    atol : float, optional
        Absolute convergence tolerance for optimization, by default 1e-8
    rtol : float, optional
        Relative convergence tolerance for optimization, by default 1e-8

    Returns
    -------
    RefittingResult
        Dataclass containing fitted parameters and statistics:

        - arrhenius: Fitted Arrhenius object
        - A, n, Ea: Parameter properties
        - R2, SSE, RMSE, MAE: Fit quality metrics
        - optimality: Gradient norm at solution
        - converged: Boolean convergence status
        - n_steps: Number of optimization steps
        - fun: Final objective value

    Notes
    -----
    **Fitting Methodology:**

    1. **Log-space fitting**: Minimizes sum of squared errors in log(k) space:

       .. math::
           \\mathrm{SSE} = \\sum_i [\\ln k_i^{\\mathrm{exp}} - \\ln k_i^{\\mathrm{calc}}(T_i)]^2

    2. **Initial guess**: Uses linear least squares on the linearized form:

       .. math::
           \\ln k = \\ln A + n \\ln T - \\frac{E_a}{R T}

    **Parameter Masking:**

    When `params` is provided, only unspecified parameters are fitted. For example:

    - `params={'n': 0.0}` -> fits only A and Ea (standard Arrhenius)
    - `params={'A': 1e14, 'n': 0.0}` -> fits only Ea
    """
    # ==================================================================================
    # Input validation and conversion to JAX arrays
    T = jnp.asarray(temperature, dtype=jnp.float64)
    k = jnp.asarray(rate_constant, dtype=jnp.float64)

    if T.shape != k.shape:
        raise ValueError(f"Temperature and rate_constant must have same shape, got {T.shape} vs {k.shape}")

    if jnp.any(k <= 0):
        raise ValueError("All rate constants must be positive (k > 0) for log-space fitting")

    if jnp.any(T <= 0):
        raise ValueError("All temperatures must be positive (T > 0 K)")

    # Convert to log space for fitting
    log_k = jnp.log(k)

    # ==================================================================================
    # Compute or use provided initial guess
    if initial_guess is None:
        p0 = compute_initial_guess(T, log_k, params)
    else:
        p0 = jnp.asarray(initial_guess, dtype=jnp.float64)

    # ==================================================================================
    # CASE 1: Fit all three parameters (A, n, Ea)
    if params is None:
        # Create Arrhenius object once with initial guess
        arrh_base = Arrhenius(
            parameters={
                "A": float(p0[0]),
                "n": float(p0[1]),
                "Ea": float(p0[2]),
            }
        )

        # Define residual function for least_squares
        # Note: least_squares expects residuals (vector), not sum of squares (scalar)
        @jax.jit
        def _residuals(params_vec: Float64[Array, "3"], _) -> Float64[Array, "n"]:
            # Update Arrhenius object parameters efficiently using eqx.tree_at
            arrh_updated = eqx.tree_at(
                lambda arr: (arr._A, arr._n, arr._Ea),
                arrh_base,
                (params_vec[0], params_vec[1], params_vec[2]),
            )
            k_pred = arrh_updated.rate_constant(T)
            log_k_pred = jnp.log(k_pred)
            return log_k - log_k_pred

        # Set up solver
        solver = LevenbergMarquardt(rtol=rtol, atol=atol)

        # Solve
        solution = least_squares(
            fn=_residuals,
            solver=solver,
            y0=p0,
            max_steps=max_steps,
            throw=False,
        )

        if solution.result != RESULTS.successful:
            print(f"Warning: Optimization did not fully converge: {solution.result}")

        # Extract optimized parameters from Optimistix result
        A_opt, n_opt, Ea_opt = float(solution.value[0]), float(solution.value[1]), float(solution.value[2])

        # Calculate statistics using utility function
        arrh_final = Arrhenius(parameters={"A": A_opt, "n": n_opt, "Ea": Ea_opt})
        k_pred = arrh_final.rate_constant(T)
        log_k_pred = jnp.log(k_pred)
        stats = compute_statistics(log_k_pred, log_k)

        # Compute objective value (sum of squared residuals)
        residuals_final = _residuals(solution.value, None)
        sse_final = float(jnp.sum(residuals_final**2))

        return RefittingResult(
            arrhenius=arrh_final,
            **stats,
            optimality=float(jnp.linalg.norm(residuals_final)),
            converged=bool(solution.result == RESULTS.successful),
            n_steps=int(solution.stats["num_steps"]),
            fun=sse_final,
            std_errors=None,  # TODO: Compute from Jacobian
            cov_matrix=None,  # TODO: Compute from Hessian
            corr_matrix=None,  # TODO: Compute from covariance
        )

    # ==================================================================================
    # CASE 2: Fit subset of parameters (parameter masking)
    else:
        known = {k: v for k, v in params.items() if v is not None}
        param_names = ["A", "n", "Ea"]

        # Determine which parameters need fitting
        free_params = [name for name in param_names if name not in known]

        if not free_params:
            raise ValueError("All parameters are fixed; nothing to fit!")

        # ==================================================================================
        # Create base Arrhenius object with initial guess
        initial_params = {}
        free_idx = 0
        for name in param_names:
            if name in known:
                initial_params[name] = known[name]
            else:
                initial_params[name] = float(p0[free_idx])
                free_idx += 1

        arrh_base = Arrhenius(parameters=initial_params)

        # ==================================================================================
        # Define masked residual function for least_squares
        @jax.jit
        def _masked_residuals(free_params_vec: Float64[Array, "n_free"], _) -> Float64[Array, "n"]:
            """Residual function with some parameters fixed."""
            # Reconstruct full parameter array [A, n, Ea]
            full_params_array = jnp.array(
                [
                    free_params_vec[free_params.index("A")] if "A" in free_params else known["A"],
                    free_params_vec[free_params.index("n")] if "n" in free_params else known["n"],
                    free_params_vec[free_params.index("Ea")] if "Ea" in free_params else known["Ea"],
                ]
            )

            # Update Arrhenius object parameters efficiently using eqx.tree_at
            arrh_updated = eqx.tree_at(
                lambda arr: (arr._A, arr._n, arr._Ea),
                arrh_base,
                (full_params_array[0], full_params_array[1], full_params_array[2]),
            )
            k_pred = arrh_updated.rate_constant(T)
            log_k_pred = jnp.log(k_pred)
            return log_k - log_k_pred

        # ==================================================================================
        # Set up solver
        solver = LevenbergMarquardt(rtol=rtol, atol=atol)

        # Solve
        solution = least_squares(
            fn=_masked_residuals,
            solver=solver,
            y0=p0,
            max_steps=max_steps,
            throw=False,
        )

        if solution.result != RESULTS.successful:
            print(f"Warning: Optimization did not fully converge: {solution.result}")

        # ==================================================================================
        # Reconstruct full parameter set from Optimistix result
        result_params = known.copy()
        free_idx = 0
        for name in param_names:
            if name not in known:
                result_params[name] = float(solution.value[free_idx])
                free_idx += 1

        # Calculate statistics using utility function
        arrh_final = Arrhenius(parameters=result_params)
        k_pred = arrh_final.rate_constant(T)
        log_k_pred = jnp.log(k_pred)
        stats = compute_statistics(log_k_pred, log_k)

        # Compute objective value (sum of squared residuals)
        residuals_final = _masked_residuals(solution.value, None)
        sse_final = float(jnp.sum(residuals_final**2))

        return RefittingResult(
            arrhenius=arrh_final,
            **stats,
            optimality=float(jnp.linalg.norm(residuals_final)),
            converged=bool(solution.result == RESULTS.successful),
            n_steps=int(solution.stats["num_steps"]),
            fun=sse_final,
            std_errors=None,  # TODO: Compute from Jacobian
            cov_matrix=None,  # TODO: Compute from Hessian
            corr_matrix=None,  # TODO: Compute from covariance
        )
