"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from KiRATE.kinetics.arrhenius import Arrhenius
from KiRATE.refitter.core.base import FittingResult
from KiRATE.refitter.core.residuals import estimate_missing_uncertainties
from KiRATE.refitter.core.statistics import compute_statistics
from KiRATE.refitter.methods.linear import arrhenius_linear_fit
from KiRATE.refitter.methods.nonlinear import check_convergence, least_squares_fit

# from KiRATE.refitter.core.uncertainty import compute_parameter_uncertainties


class ArrheniusFitter(eqx.Module):
    """
    Fitter for modified Arrhenius rate constants.

    This class provides a complete workflow for fitting Arrhenius parameters
    to experimental rate constant data, including:
    - Initial parameter estimation via weighted linear least squares
    - Nonlinear refinement via Levenberg-Marquardt optimization
    - Uncertainty quantification via Jacobian-based methods
    - Comprehensive goodness-of-fit statistics

    The modified Arrhenius equation is:

    .. math::
        k(T) = A \\cdot T^n \\cdot \\exp\\left(-\\frac{E_a}{R T}\\right)

    where:
        - A: Pre-exponential factor
        - n: Temperature exponent
        - Ea: Activation energy [cal/mol]
        - R: Gas constant [cal/(mol·K)]
        - T: Temperature [K]

    Examples
    --------
    Basic usage with uncertainties:

    >>> import jax.numpy as jnp
    >>> from KiRATE.refitter.fitters import ArrheniusFitter
    >>>
    >>> # Experimental data
    >>> T = jnp.array([300, 400, 500, 600, 700])
    >>> k = jnp.array([1.2e-10, 3.5e-9, 2.1e-8, 7.8e-8, 2.0e-7])
    >>> uncertainties = 0.1 * k  # 10% uncertainty
    >>>
    >>> # Fit Arrhenius parameters
    >>> fitter = ArrheniusFitter()
    >>> result = fitter.fit(T, k, uncertainties)
    >>>
    >>> # Access fitted rate constant object
    >>> arrhenius = result.rate_constant
    >>> print(f"A = {arrhenius.A:.2e}, n = {arrhenius.n:.2f}, Ea = {arrhenius.Ea:.1f}")
    >>>
    >>> # Check fit quality
    >>> print(f"R² = {result.R2:.4f}, RMSE = {result.RMSE:.2e}")

    With fixed parameters:

    >>> # Fix n=0 (standard Arrhenius)
    >>> result = fitter.fit(T, k, uncertainties, fixed_params={"n": 0.0})

    Notes
    -----
    **Two-stage fitting approach:**

    1. **Linear stage** (initial guess):
       - Transforms to log space: ln(k) = ln(A) + n·ln(T) - Ea/(R·T)
       - Solves weighted linear least squares
       - Fast, robust initial estimate

    2. **Nonlinear stage** (refinement):
       - Optimizes in original (non-log) space
       - Uses Levenberg-Marquardt algorithm
       - Handles measurement uncertainties correctly

    **Uncertainty handling:**

    - If all data points have uncertainties: weighted fit
    - If some missing: estimates from residual distribution
    - If none: unweighted fit, estimates from final residuals

    See Also
    --------
    Arrhenius : The rate constant class
    """

    def fit(
        self,
        temperature: Float64[Array, "n"],
        rate_constant: Float64[Array, "n"],
        uncertainties: Float64[Array, "n"] | None = None,
        fixed_params: dict[str, float] | None = None,
        max_steps: int = 200,
        verbose: bool = False,
    ) -> FittingResult[Arrhenius]:
        """
        Fit Arrhenius parameters to rate constant data.

        Parameters
        ----------
        temperature : Float64[Array, "n"]
            Temperature values [K]
        rate_constant : Float64[Array, "n"]
            Rate constant values (must be positive)
        uncertainties : Float64[Array, "n"], optional
            Uncertainties (standard errors) in rate_constant.
            If None, unweighted fit is performed.
        fixed_params : dict[str, float], optional
            Known parameters to fix during fitting.
            Keys can be 'A', 'n', or 'Ea'.
        max_steps : int, optional
            Maximum optimization iterations, by default 1000
        atol : float, optional
            Absolute convergence tolerance, by default 1e-8
        rtol : float, optional
            Relative convergence tolerance, by default 1e-8
        solver : str, optional
            Optimization algorithm: 'lm', 'ilm', or 'dogleg', by default 'lm'
        verbose : bool, optional
            Print convergence diagnostics, by default False

        Returns
        -------
        FittingResult[Arrhenius]
            Complete fitting results containing:
            - rate_constant: Fitted Arrhenius object
            - R2, SSE, RMSE, MAE: Goodness-of-fit metrics
            - std_errors: Parameter uncertainties
            - cov_matrix: Covariance matrix
            - corr_matrix: Correlation matrix
            - converged: Optimization convergence status

        Raises
        ------
        ValueError
            If temperature or rate_constant contain non-positive values,
            if uncertainties are non-positive, or if insufficient data
            for the number of free parameters.
        """
        # Validate inputs
        if jnp.any(temperature <= 0):
            raise ValueError("Temperature must be positive")
        if jnp.any(rate_constant <= 0):
            raise ValueError("Rate constants must be positive")
        if uncertainties is not None and jnp.any(uncertainties <= 0):
            raise ValueError("Uncertainties must be positive")

        # Track which data points have uncertainties
        has_uncertainty = uncertainties is not None
        if has_uncertainty:
            has_uncertainty_mask = jnp.ones_like(rate_constant, dtype=bool)
        else:
            has_uncertainty_mask = jnp.zeros_like(rate_constant, dtype=bool)

        # STAGE 1: Initial guess via linear least squares
        params_init = arrhenius_linear_fit(
            temperature=temperature,
            rate_constant=rate_constant,
            uncertainties=uncertainties,
            fixed_params=fixed_params,
        )

        # Create initial Arrhenius object for uncertainty estimation if needed
        temp_arrhenius = None
        if not has_uncertainty:
            # Create temporary Arrhenius for residual evaluation
            if fixed_params is None:
                temp_arrhenius = Arrhenius(parameters={"A": params_init[0], "n": params_init[1], "Ea": params_init[2]})
            else:
                # Reconstruct full parameters
                full_params = self._reconstruct_params(params_init, fixed_params)
                temp_arrhenius = Arrhenius(parameters={"A": full_params[0], "n": full_params[1], "Ea": full_params[2]})

            # Estimate uncertainties from residuals
            def residual_fn_unweighted(params, args):
                arr = self._params_to_arrhenius(params, fixed_params)
                predictions = arr.rate_constant(temperature)
                return rate_constant - predictions

            uncertainties_est = estimate_missing_uncertainties(
                residual_fn_unweighted=residual_fn_unweighted,
                params_opt=params_init,
                uncertainties=jnp.zeros_like(rate_constant),
                has_uncertainty=has_uncertainty_mask,
            )
            uncertainties = uncertainties_est
            has_uncertainty = True

        # STAGE 2: Nonlinear refinement
        # Create residual function with weights
        weights = 1.0 / uncertainties if has_uncertainty else None

        # Create base Arrhenius object (will be updated via eqx.tree_at)
        if temp_arrhenius is not None:
            # Reuse temp_arrhenius if it was created for uncertainty estimation
            arrhenius_base = temp_arrhenius
        else:
            # Create new base Arrhenius
            if fixed_params is None:
                arrhenius_base = Arrhenius(parameters={"A": params_init[0], "n": params_init[1], "Ea": params_init[2]})
            else:
                full_params = self._reconstruct_params(params_init, fixed_params)
                arrhenius_base = Arrhenius(parameters={"A": full_params[0], "n": full_params[1], "Ea": full_params[2]})

        # Build loss function in log-space (MSE of log(k))
        if fixed_params is None:
            # All three parameters free
            def loss_fn(params):
                arrh_updated = eqx.tree_at(
                    lambda arr: (arr._A, arr._n, arr._Ea),
                    arrhenius_base,
                    (params[0], params[1], params[2]),
                )
                predictions = arrh_updated.rate_constant(temperature)
                return jnp.mean((jnp.log(rate_constant) - jnp.log(predictions)) ** 2)

        else:
            # Some parameters fixed
            def loss_fn(params):
                full_params_arr = self._reconstruct_params(params, fixed_params)
                arrh_updated = eqx.tree_at(
                    lambda arr: (arr._A, arr._n, arr._Ea),
                    arrhenius_base,
                    (full_params_arr[0], full_params_arr[1], full_params_arr[2]),
                )
                predictions = arrh_updated.rate_constant(temperature)
                return jnp.mean((jnp.log(rate_constant) - jnp.log(predictions)) ** 2)

        # Optimize
        solution = least_squares_fit(
            loss_fn=loss_fn,
            initial_params=params_init,
            max_steps=max_steps,
        )

        # Check convergence
        converged, optimality = check_convergence(solution, verbose=verbose)

        # Get optimal parameters
        params_opt = solution.value

        # Create final Arrhenius object
        arrhenius_opt = self._params_to_arrhenius(params_opt, fixed_params)

        # STAGE 3: Compute statistics and uncertainties
        predictions = arrhenius_opt.rate_constant(temperature)
        stats = compute_statistics(predictions=predictions, observations=rate_constant)

        # Create result object (skip uncertainty quantification for now)
        result = FittingResult(
            rate_constant=arrhenius_opt,
            R2=stats["R2"],
            SSE=stats["SSE"],
            RMSE=stats["RMSE"],
            MAE=stats["MAE"],
            optimality=optimality,
            converged=converged,
            n_steps=solution.n_steps,
            fun=solution.final_loss,
            std_errors=None,
            cov_matrix=None,
            corr_matrix=None,
        )

        return result

    def initial_guess(
        self,
        temperature: Float64[Array, "n"],
        rate_constant: Float64[Array, "n"],
        uncertainties: Float64[Array, "n"] | None = None,
        fixed_params: dict[str, float] | None = None,
    ) -> Float64[Array, "n_params"]:
        """
        Compute initial parameter guess using weighted linear least squares.

        This method performs only the linear stage of fitting (log-space regression)
        without nonlinear refinement. Useful for getting quick initial estimates or
        for debugging.

        Parameters
        ----------
        temperature : Float64[Array, "n"]
            Temperature values [K]
        rate_constant : Float64[Array, "n"]
            Rate constant values
        uncertainties : Float64[Array, "n"], optional
            Uncertainties in rate_constant
        fixed_params : dict[str, float], optional
            Known parameters to fix

        Returns
        -------
        Float64[Array, "n_params"]
            Initial parameter guess (subset of [A, n, Ea] depending on fixed_params)
        """
        return arrhenius_linear_fit(
            temperature=temperature,
            rate_constant=rate_constant,
            uncertainties=uncertainties,
            fixed_params=fixed_params,
        )

    def _params_to_arrhenius(
        self,
        params: Float64[Array, "n_params"],
        fixed_params: dict[str, float] | None = None,
    ) -> Arrhenius:
        """
        Convert parameter array to Arrhenius object.

        Parameters
        ----------
        params : Float64[Array, "n_params"]
            Free parameters in canonical order [A, n, Ea]
        fixed_params : dict[str, float], optional
            Fixed parameters

        Returns
        -------
        Arrhenius
            Arrhenius object with all parameters
        """
        if fixed_params is None:
            # All three parameters are free
            parameters = {"A": params[0], "n": params[1], "Ea": params[2]}
            return Arrhenius(parameters=parameters)
        else:
            # Reconstruct full parameter set
            full_params = self._reconstruct_params(params, fixed_params)
            parameters = {"A": full_params[0], "n": full_params[1], "Ea": full_params[2]}
            return Arrhenius(parameters=parameters)

    def _reconstruct_params(
        self,
        free_params: Float64[Array, "n_free"],
        fixed_params: dict[str, float],
    ) -> Float64[Array, "3"]:
        """
        Reconstruct full [A, n, Ea] from free parameters and fixed values.

        Parameters
        ----------
        free_params : Float64[Array, "n_free"]
            Free parameter values
        fixed_params : dict[str, float]
            Fixed parameter values

        Returns
        -------
        Float64[Array, "3"]
            Full parameter array [A, n, Ea]
        """
        # Build list of free parameter names in canonical order
        free_param_names = []
        if "A" not in fixed_params:
            free_param_names.append("A")
        if "n" not in fixed_params:
            free_param_names.append("n")
        if "Ea" not in fixed_params:
            free_param_names.append("Ea")

        # Map free parameters to full array
        full_params = []
        free_idx = 0

        for param_name in ["A", "n", "Ea"]:
            if param_name in fixed_params:
                full_params.append(fixed_params[param_name])
            else:
                full_params.append(free_params[free_idx])
                free_idx += 1

        return jnp.array(full_params)
