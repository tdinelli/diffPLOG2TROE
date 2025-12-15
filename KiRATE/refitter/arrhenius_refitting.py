"""
Copyright (c) 2025 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float64
from optimistix import RESULTS, LevenbergMarquardt, least_squares

from KiRATE.kinetics import Arrhenius
from KiRATE.refitter.utils import RefittingResult, compute_statistics
from KiRATE.utilities import constants


def least_square_estimate(
    temperature: Float64[Array, "n"],
    log_rate_constant: Float64[Array, "n"],
    params: Optional[dict[str, float]] = None,
) -> Float64[Array, "n_params"]:
    """
    Compute initial parameter guess using linear least squares regression.

    This function solves the linearized Arrhenius equation in closed form to
    obtain an excellent initial guess for nonlinear optimization. The method
    transforms the nonlinear Arrhenius equation into a linear regression problem.

    Parameters
    ----------
    temperature : Float64[Array, "n"]
        Temperature values (K) at which rate constants are measured
    log_rate_constant : Float64[Array, "n"]
        Natural logarithm of rate constants: ln(k)
    params : dict[str, float], optional
        Known parameters to fix during estimation. Keys can be 'A', 'n', or 'Ea'.
        If provided, only unknown parameters will be estimated. For example:
        - params={'n': 0.0} → estimate only A and Ea (standard Arrhenius)
        - params={'A': 1e14} → estimate only n and Ea

    Returns
    -------
    Float64[Array, "n_params"]
        Initial guess for free parameters. Order depends on which parameters
        are being estimated:
        - If params is None: [A, n, Ea] (all three parameters)
        - Otherwise: subset of free parameters in canonical order [A, n, Ea]

    Notes
    -----
    **Mathematical Approach:**

    The modified Arrhenius equation:

    .. math::
        k(T) = A \\cdot T^n \\cdot \\exp\\left(-\\frac{E_a}{R T}\\right)

    Taking logarithms linearizes the equation:

    .. math::
        \\ln k = \\ln A + n \\ln T - \\frac{E_a}{R T}

    This is linear in the transformed parameters [ln(A), n, Ea/R], allowing
    closed-form solution via ordinary least squares:

    .. math::
        \\mathbf{y} = \\mathbf{X} \\boldsymbol{\\beta}

    where:
        - **y** = ln(k) (observations)
        - **X** = [**1**, ln(T), -1/T] (design matrix)
        - **β** = [ln(A), n, Ea/R] (coefficients)

    The solution is:

    .. math::
        \\boldsymbol{\\beta} = (\\mathbf{X}^T \\mathbf{X})^{-1} \\mathbf{X}^T \\mathbf{y}

    **Why This Works Well:**

    1. **Optimal for linear problem**: Exact solution for linearized equation
    2. **Fast**: No iterative optimization required
    3. **Robust**: No sensitivity to initial guess
    4. **Near-optimal**: For well-conditioned data, often within 1-2 steps of
       the true nonlinear optimum

    **Parameter Masking:**

    When some parameters are known (via `params`), the method subtracts their
    contribution from ln(k) and estimates only the remaining parameters.
    """
    # CASE 1: Fit all three parameters
    if params is None:
        # Design matrix: [1, ln(T), -1/T]
        # Corresponds to: ln(k) = ln(A) + n*ln(T) - Ea/(R*T)
        X = jnp.column_stack([jnp.ones_like(temperature), jnp.log(temperature), -1.0 / temperature])
        coeffs, *_ = jnp.linalg.lstsq(X, log_rate_constant, rcond=None)

        # Return [A, n, Ea]
        # Note: coeffs[0] = ln(A), coeffs[2] = Ea/R
        return jnp.array([jnp.exp(coeffs[0]), coeffs[1], coeffs[2] * constants.R_cal_mol])

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


def _make_residual_function(
    arrh_base: Arrhenius,
    T: Float64[Array, "n"],
    log_k: Float64[Array, "n"],
    weights: Optional[Float64[Array, "n"]] = None,
    free_params: Optional[list[str]] = None,
    known: Optional[dict[str, float]] = None,
) -> Callable[[Float64[Array, "n_params"], None], Float64[Array, "n"]]:
    """
    Factory function to create residual functions for least squares fitting.

    This function builds JIT-compiled residual functions that can handle:
    - Weighted vs unweighted fitting
    - All parameters free vs masked (some parameters fixed)

    The returned function computes residuals in log space:
        residual_i = (ln(k_i) - ln(k_pred(T_i))) / weight_i

    Parameters
    ----------
    arrh_base : Arrhenius
        Base Arrhenius object used as a template. Parameters will be updated
        during optimization via equinox.tree_at.
    T : Float64[Array, "n"]
        Temperature values (K) at which to evaluate rate constants
    log_k : Float64[Array, "n"]
        Natural logarithm of experimental rate constants
    weights : Float64[Array, "n"], optional
        Weights for residuals (1/relative_uncertainty). If None, returns
        unweighted residuals. For relative uncertainties of 5%, use weights=20.
    free_params : list[str], optional
        Names of parameters to fit (for masked case): subset of ["A", "n", "Ea"].
        If None, all three parameters are fitted.
    known : dict[str, float], optional
        Fixed parameter values for masked fitting (e.g., {"n": 0.0}).
        Required if free_params is provided.

    Returns
    -------
    residual_fn : Callable[[Float64[Array, "n_params"], None], Float64[Array, "n"]]
        JIT-compiled residual function with signature:
            fn(params_vec, args) -> residuals
        where:
        - params_vec: Parameter vector to optimize (length = n_free_params)
        - args: Unused (for compatibility with optimistix)
        - residuals: Weighted residuals (length = n_data_points)

    Notes
    -----
    **Weighted vs Unweighted:**

    - **Unweighted** (weights=None): Minimizes sum of squared log-residuals
        .. math::
            \\min \\sum_i [\\ln k_i - \\ln k_i^{\\text{calc}}]^2

    - **Weighted** (weights provided): Minimizes weighted sum of squares
        .. math::
            \\min \\sum_i \\left[\\frac{\\ln k_i - \\ln k_i^{\\text{calc}}}{\\sigma_i}\\right]^2

    **Parameter Masking:**

    When free_params is provided, only the specified parameters are optimized.
    Others are held fixed at values in `known`. For example:
        - free_params=["A", "Ea"], known={"n": 0.0} -> Standard Arrhenius
    """
    # CASE 1: All parameters free (no masking)
    if free_params is None:
        if weights is None:
            # Unweighted residuals, all parameters free
            @jax.jit
            def _residuals(params_vec: Float64[Array, "3"], _) -> Float64[Array, "n"]:
                arrh_updated = eqx.tree_at(
                    lambda arr: (arr._A, arr._n, arr._Ea),
                    arrh_base,
                    (params_vec[0], params_vec[1], params_vec[2]),
                )
                k_pred = arrh_updated.rate_constant(T)
                log_k_pred = jnp.log(k_pred)
                return log_k - log_k_pred

        else:
            # Weighted residuals, all parameters free
            @jax.jit
            def _residuals(params_vec: Float64[Array, "3"], _) -> Float64[Array, "n"]:
                arrh_updated = eqx.tree_at(
                    lambda arr: (arr._A, arr._n, arr._Ea),
                    arrh_base,
                    (params_vec[0], params_vec[1], params_vec[2]),
                )
                k_pred = arrh_updated.rate_constant(T)
                log_k_pred = jnp.log(k_pred)
                return (log_k - log_k_pred) / weights

    # CASE 2: Masked parameters (some fixed)
    else:
        if known is None:
            raise ValueError("Must provide 'known' dict when using 'free_params'")

        if weights is None:
            # Unweighted residuals, masked parameters
            @jax.jit
            def _residuals(params_vec: Float64[Array, "n_free"], _) -> Float64[Array, "n"]:
                # Reconstruct full parameter array [A, n, Ea]
                full_params = jnp.array(
                    [
                        params_vec[free_params.index("A")] if "A" in free_params else known["A"],
                        params_vec[free_params.index("n")] if "n" in free_params else known["n"],
                        params_vec[free_params.index("Ea")] if "Ea" in free_params else known["Ea"],
                    ]
                )
                arrh_updated = eqx.tree_at(
                    lambda arr: (arr._A, arr._n, arr._Ea),
                    arrh_base,
                    (full_params[0], full_params[1], full_params[2]),
                )
                k_pred = arrh_updated.rate_constant(T)
                log_k_pred = jnp.log(k_pred)
                return log_k - log_k_pred

        else:
            # Weighted residuals, masked parameters
            @jax.jit
            def _residuals(params_vec: Float64[Array, "n_free"], _) -> Float64[Array, "n"]:
                # Reconstruct full parameter array [A, n, Ea]
                full_params = jnp.array(
                    [
                        params_vec[free_params.index("A")] if "A" in free_params else known["A"],
                        params_vec[free_params.index("n")] if "n" in free_params else known["n"],
                        params_vec[free_params.index("Ea")] if "Ea" in free_params else known["Ea"],
                    ]
                )
                arrh_updated = eqx.tree_at(
                    lambda arr: (arr._A, arr._n, arr._Ea),
                    arrh_base,
                    (full_params[0], full_params[1], full_params[2]),
                )
                k_pred = arrh_updated.rate_constant(T)
                log_k_pred = jnp.log(k_pred)
                return (log_k - log_k_pred) / weights

    return _residuals


def _estimate_missing_uncertainties(
    residual_fn_unweighted: Callable[[Float64[Array, "n_params"], None], Float64[Array, "n"]],
    params_opt: Float64[Array, "n_params"],
    uncertainties: Float64[Array, "n"],
    has_uncertainty: Bool[Array, "n"],
) -> Float64[Array, "n"]:
    """
    Estimate missing uncertainties from fit residuals (two-stage fitting).

    This function implements the second stage of two-stage weighted least squares
    when some data points have known uncertainties and others don't. It estimates
    the missing uncertainties from the residuals of an initial fit.

    Strategy:
    1. Compute raw (unweighted) residuals at optimal parameters from first fit
    2. For points WITH known uncertainties, compute weighted residuals
    3. Calculate RMS of weighted residuals (should be almost equal to 1.0 if uncertainties correct)
    4. Estimate missing uncertainties: :math:`\\sigma_{unknown} = |raw_{residual}| / RMS_{weighted}`

    Parameters
    ----------
    residual_fn_unweighted : Callable
        Unweighted residual function with signature:
            fn(params, args) -> residuals
        Used to compute raw residuals at optimal parameters.
    params_opt : Float64[Array, "n_params"]
        Optimal parameters from the first-stage fit
    uncertainties : Float64[Array, "n"]
        Partial uncertainties (may contain NaN/Inf for missing values)
    has_uncertainty : Bool[Array, "n"]
        Boolean mask: True where uncertainty is known, False where missing

    Returns
    -------
    complete_uncertainties : Float64[Array, "n"]
        Full uncertainty array with estimated values filled in for missing entries.
        Known values are preserved exactly.

    Notes
    -----
    **Estimation Method:**

    The RMS of weighted residuals from points with known uncertainties provides
    a scale factor for estimating missing uncertainties:

    .. math::
        \\text{RMS}_{\\text{known}} = \\sqrt{\\frac{1}{N_{\\text{known}}} \\sum_{i \\in \\text{known}} \\left(\\frac{r_i}{\\sigma_i}\\right)^2}

    Then for points without uncertainties:

    .. math::
        \\hat{\\sigma}_j = \\frac{|r_j|}{\\text{RMS}_{\\text{known}}}

    This assumes that the unknown points have similar weighted residuals to the
    known points.

    **Quality Indicators:**

    - RMS ~= 1.0: Known uncertainties are well-calibrated
    - RMS >> 1.0: Known uncertainties underestimated (poor fit) or model inadequate
    - RMS << 1.0: Known uncertainties overestimated
    """
    # Compute raw residuals at optimal parameters from first fit
    raw_residuals = residual_fn_unweighted(params_opt, None)

    # Extract points with known uncertainties
    uncertainties_known = uncertainties[has_uncertainty]
    residuals_known = raw_residuals[has_uncertainty]

    # Compute weighted residuals for known points
    weighted_residuals_known = residuals_known / uncertainties_known

    # RMS of weighted residuals (should be ~1.0 for well-calibrated uncertainties)
    rms_known = jnp.sqrt(jnp.mean(weighted_residuals_known**2))

    # Avoid division by zero (should never happen unless all residuals are exactly zero)
    rms_known = jnp.maximum(rms_known, 1e-10)

    # Estimate missing uncertainties: scale raw residuals by RMS
    # Intuition: if known points have weighted residuals ~RMS, unknown points
    # should have uncertainties such that their weighted residuals are also ~RMS
    estimated_uncertainties = jnp.abs(raw_residuals) / rms_known

    # Combine known and estimated uncertainties
    complete_uncertainties = jnp.where(
        has_uncertainty,
        uncertainties,
        estimated_uncertainties,  # Keep known values  # Fill in estimates
    )

    # Print diagnostic information
    n_known = int(jnp.sum(has_uncertainty))
    n_missing = int(jnp.sum(~has_uncertainty))
    n_total = len(uncertainties)

    print(" * Two-Stage Uncertainty Estimation:")
    print(f"    Known uncertainties:            {n_known}/{n_total} points")
    print(f"    Estimated uncertainties:        {n_missing}/{n_total} points")
    print(f"    RMS weighted residuals (known): {float(rms_known):.4f}")

    if float(rms_known) > 2.0:
        print("   - Warning: RMS >> 1 suggests known uncertainties may be underestimated")
    elif float(rms_known) < 0.5:
        print("  - Warning: RMS << 1 suggests known uncertainties may be overestimated")

    estimated_min = float(jnp.min(estimated_uncertainties[~has_uncertainty]))
    estimated_max = float(jnp.max(estimated_uncertainties[~has_uncertainty]))
    print(f"  - Estimated sigma range: [{estimated_min:.4f}, {estimated_max:.4f}]")

    return complete_uncertainties


def refitter(
    temperature: ArrayLike,
    rate_constant: ArrayLike,
    uncertainties: Optional[ArrayLike] = None,
    params: Optional[dict[str, float]] = None,
    initial_guess: Optional[ArrayLike] = None,
    max_steps: int = 1000,
    atol: float = 1e-8,
    rtol: float = 1e-8,
) -> RefittingResult:
    """
    Fit Arrhenius parameters (A, n, Ea) to temperature-rate constant data.

    Supports weighted least squares fitting when experimental uncertainties are
    available, with automatic estimation of missing uncertainties via two-stage
    fitting when only partial uncertainty information is provided.

    Parameters
    ----------
    temperature : ArrayLike
        Temperature values (K) - accepts list, tuple, numpy array, or JAX array
    rate_constant : ArrayLike
        Rate constant values - accepts list, tuple, numpy array, or JAX array
    uncertainties : ArrayLike, optional
        Relative uncertainties on rate constants (dimensionless). Examples:
        - 0.05 for 5% uncertainty
        - 0.10 for 10% uncertainty

        Can be:
        - None: Unweighted fit (all data treated equally)
        - Full array: Weighted fit using provided uncertainties
        - Partial array with NaN/Inf: Two-stage fit (estimates missing values)

        **Important**: These are RELATIVE uncertainties (sig/k), not absolute.
        They are used directly in log-space fitting since d(ln k) ~= dk/k.
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
        Maximum number of steps for the least square solver, by default 1000
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
        - std_errors, cov_matrix, corr_matrix: Uncertainty statistics

    Notes
    -----
    **Fitting Methodology:**

    1. **Log-space fitting**: Minimizes sum of squared errors in log(k) space:

       .. math::
           \\mathrm{SSE} = \\sum_i \\left[\\frac{\\ln k_i^{\\mathrm{exp}} - \\ln k_i^{\\mathrm{calc}}(T_i)}{\\sigma_i}\\right]^2

       where σ_i is the relative uncertainty (= 1 for unweighted).

    2. **Initial guess**: Uses linear least squares on the linearized form:

       .. math::
           \\ln k = \\ln A + n \\ln T - \\frac{E_a}{R T}

    3. **Two-stage fitting** (when some uncertainties are missing):
       - Stage 1: Fit with placeholder weights for missing uncertainties
       - Estimate missing σ from residuals of points with known σ
       - Stage 2: Refit with complete uncertainty information

    **Parameter Masking:**

    When `params` is provided, only unspecified parameters are fitted. For example:

    - `params={'n': 0.0}` -> fits only A and Ea (standard Arrhenius)
    - `params={'A': 1e14, 'n': 0.0}` -> fits only Ea

    **Uncertainty Estimation:**

    Parameter uncertainties (std_errors, cov_matrix, corr_matrix) are computed
    from the Jacobian at the optimal point:

    - **Unweighted fit**: Uncertainties estimated from residual variance
    - **Weighted fit**: Uses provided uncertainties (assumes they are correct)

    Examples
    --------
    >>> # Basic unweighted fit
    >>> result = refitter(T, k)

    >>> # Weighted fit with 5% uncertainty on all points
    >>> uncertainties = jnp.full(len(k), 0.05)
    >>> result = refitter(T, k, uncertainties=uncertainties)

    >>> # Mixed uncertainties (some known, some unknown)
    >>> uncertainties = jnp.array([0.05, 0.10, jnp.nan, 0.05, jnp.nan])
    >>> result = refitter(T, k, uncertainties=uncertainties)  # Two-stage fit

    >>> # Fix n=0 (standard Arrhenius) with uncertainties
    >>> result = refitter(T, k, uncertainties=uncertainties, params={'n': 0.0})
    """
    # Input validation and conversion to JAX arrays
    T = jnp.asarray(temperature, dtype=jnp.float64)
    k = jnp.asarray(rate_constant, dtype=jnp.float64)

    if T.shape != k.shape:
        raise ValueError(f"Temperature and rate_constant must have same shape, got {T.shape} vs {k.shape}")

    # Is this true?
    # if jnp.any(k <= 0):
    #     raise ValueError("All rate constants must be positive (k > 0) for log-space fitting")

    if jnp.any(T <= 0):
        raise ValueError("All temperatures must be positive (T > 0 K)")

    # Convert to log space for fitting
    log_k = jnp.log(k)

    # Process uncertainties (relative uncertainties in log space)
    if uncertainties is not None:
        sigma = jnp.asarray(uncertainties, dtype=jnp.float64)

        # Validate shape
        if sigma.shape != k.shape:
            raise ValueError(f"Uncertainties must have same shape as rate_constant, got {sigma.shape} vs {k.shape}")

        # Check which points have known uncertainties
        has_uncertainty = jnp.isfinite(sigma)
        n_known = int(jnp.sum(has_uncertainty))
        n_missing = int(jnp.sum(~has_uncertainty))
        n_total = len(sigma)

        # Validate that at least some uncertainties are known
        if n_known == 0:
            raise ValueError(
                "All uncertainties are NaN/Inf. Either provide valid uncertainties "
                "or use uncertainties=None for unweighted fit."
            )

        # Check for negative or zero uncertainties
        if jnp.any(sigma[has_uncertainty] <= 0):
            raise ValueError(
                "All uncertainties must be positive (σ > 0). "
                "Relative uncertainties should be like 0.05 for 5%, 0.10 for 10%."
            )

        # Determine if two-stage fitting is needed
        needs_two_stage = n_missing > 0

        if needs_two_stage:
            # Partial uncertainties: use placeholders for missing values
            print(" * Uncertainty Information:")
            print(f"    - Known:   {n_known}/{n_total} points ({100 * n_known / n_total:.1f}%)")
            print(f"    - Missing: {n_missing}/{n_total} points ({100 * n_missing / n_total:.1f}%)")

            # Initial weights: use 1.0 (neutral weight) for missing uncertainties
            sigma_initial = jnp.where(has_uncertainty, sigma, 1.0)
            weights_initial = 1.0 / sigma_initial
        else:
            # All uncertainties known
            print(" * Uncertainty Information:")
            print(f"    - All {n_total} data points have known uncertainties")
            print(f"    - sigma range: [{float(jnp.min(sigma)):.4f}, {float(jnp.max(sigma)):.4f}]")

            weights_initial = 1.0 / sigma
            needs_two_stage = False

    else:
        # No uncertainties provided: unweighted fit
        weights_initial = None
        needs_two_stage = False
        has_uncertainty = None
        sigma = None

    # Compute or use provided initial guess
    if initial_guess is None:
        p0 = least_square_estimate(T, log_k, params)
    else:
        p0 = jnp.asarray(initial_guess, dtype=jnp.float64)

    # Determine parameter masking (if any)
    if params is None:
        # CASE 1: Fit all three parameters
        free_params = None
        known = None
        initial_params = {"A": float(p0[0]), "n": float(p0[1]), "Ea": float(p0[2])}
    else:
        # CASE 2: Masked parameters (some fixed)
        known = {k: v for k, v in params.items() if v is not None}
        param_names = ["A", "n", "Ea"]
        free_params = [name for name in param_names if name not in known]

        if not free_params:
            raise ValueError("All parameters are fixed; nothing to fit!")

        # Build initial parameters dict
        initial_params = {}
        free_idx = 0
        for name in param_names:
            if name in known:
                initial_params[name] = known[name]
            else:
                initial_params[name] = float(p0[free_idx])
                free_idx += 1

    # Create base Arrhenius object with initial guess
    arrh_base = Arrhenius(parameters=initial_params)

    # Build residual function using factory
    residual_fn = _make_residual_function(
        arrh_base=arrh_base,
        T=T,
        log_k=log_k,
        weights=weights_initial,
        free_params=free_params,
        known=known,
    )

    # Set up solver and perform first fit
    solver = LevenbergMarquardt(rtol=rtol, atol=atol)

    solution = least_squares(
        fn=residual_fn,
        solver=solver,
        y0=p0,
        max_steps=max_steps,
        throw=False,
    )

    # Two-stage fitting: estimate missing uncertainties and refit
    if needs_two_stage:
        print(" * Stage 1 complete: Initial fit with placeholder weights")

        # Build unweighted residual function for estimation
        residual_fn_unweighted = _make_residual_function(
            arrh_base=arrh_base,
            T=T,
            log_k=log_k,
            weights=None,  # Unweighted
            free_params=free_params,
            known=known,
        )

        # Estimate missing uncertainties
        sigma_complete = _estimate_missing_uncertainties(
            residual_fn_unweighted=residual_fn_unweighted,
            params_opt=solution.value,
            uncertainties=sigma,
            has_uncertainty=has_uncertainty,
        )

        # Rebuild residual function with complete weights
        weights_final = 1.0 / sigma_complete
        residual_fn = _make_residual_function(
            arrh_base=arrh_base,
            T=T,
            log_k=log_k,
            weights=weights_final,
            free_params=free_params,
            known=known,
        )

        # Refit with complete uncertainties
        print(" * Stage 2: Refitting with estimated uncertainties...")
        solution = least_squares(
            fn=residual_fn,
            solver=solver,
            y0=solution.value,  # Start from previous solution
            max_steps=max_steps,
            throw=False,
        )

        # Use final weights for uncertainty computation
        weights_for_stats = weights_final
    else:
        # Single-stage fit: use initial weights (may be None)
        weights_for_stats = weights_initial

    # Check convergence
    if solution.result != RESULTS.successful:
        print(f" *** Warning: Optimization did not fully converge: {solution.result}")

    # Extract optimized parameters and construct final Arrhenius object
    if params is None:
        # All parameters fitted
        A_opt = float(solution.value[0])
        n_opt = float(solution.value[1])
        Ea_opt = float(solution.value[2])
        result_params = {"A": A_opt, "n": n_opt, "Ea": Ea_opt}
    else:
        # Masked parameters: reconstruct full parameter set
        result_params = known.copy()
        free_idx = 0
        for name in ["A", "n", "Ea"]:
            if name not in known:
                result_params[name] = float(solution.value[free_idx])
                free_idx += 1

    arrh_final = Arrhenius(parameters=result_params)

    # Compute fit statistics
    k_pred = arrh_final.rate_constant(T)
    log_k_pred = jnp.log(k_pred)
    stats = compute_statistics(log_k_pred, log_k)

    # Compute final residuals and objective
    residuals_final = residual_fn(solution.value, None)
    sse_final = float(jnp.sum(residuals_final**2))

    # Compute parameter uncertainties from Jacobian
    std_errors, cov_matrix, corr_matrix = jacobian_related_statistics(
        residual_fn=residual_fn,
        solution=solution,
        weights=weights_for_stats,
        result_params=result_params,
        free_params=free_params,
    )

    # Return results
    return RefittingResult(
        arrhenius=arrh_final,
        **stats,
        optimality=float(jnp.linalg.norm(residuals_final)),
        converged=bool(solution.result == RESULTS.successful),
        n_steps=int(solution.stats["num_steps"]),
        fun=sse_final,
        std_errors=std_errors,
        cov_matrix=cov_matrix,
        corr_matrix=corr_matrix,
    )


def jacobian_related_statistics(
    residual_fn: Callable[[Float64[Array, "n_params"], None], Float64[Array, "n"]],
    solution: "optimistix.Solution",
    weights: Optional[Float64[Array, "n"]] = None,
    result_params: Optional[dict[str, float]] = None,
    free_params: Optional[list[str]] = None,
) -> tuple[
    Optional[Float64[Array, "n_params"]],
    Optional[Float64[Array, "n_params n_params"]],
    Optional[Float64[Array, "n_params n_params"]],
]:
    """
    Compute parameter uncertainties from least-squares solution using the Jacobian.

    This function computes standard errors, covariance matrix, and correlation matrix
    for the fitted parameters using the Jacobian of the residual function at the
    optimal point. Handles both weighted and unweighted cases correctly.

    Parameters
    ----------
    residual_fn : Callable
        Residual function with signature:
            fn(params, args) -> residuals
        This should be the SAME function passed to least_squares.
    solution : optimistix.Solution
        Solution object returned by least_squares containing optimal parameters
        and convergence information.
    weights : Float64[Array, "n"], optional
        Weights used in the fit (1/relative_uncertainty). If None, assumes
        unweighted fit and estimates uncertainty from residuals. If provided,
        assumes uncertainties are KNOWN and uses weighted covariance formula.

    Returns
    -------
    std_errors : Float64[Array, "n_params"] or None
        Standard errors of parameters. None if computation fails.
    cov_matrix : Float64[Array, "n_params n_params"] or None
        Covariance matrix of parameters. None if computation fails.
    corr_matrix : Float64[Array, "n_params n_params"] or None
        Correlation matrix of parameters. None if computation fails.

    Notes
    -----
    **Covariance Matrix Formulas:**

    1. **Unweighted case** (weights=None):
        Uncertainties are UNKNOWN, estimate from residuals:

        .. math::
            \\mathrm{Cov}(\\theta) = \\hat{\\sigma}^2 (J^T J)^{-1}

        where $\\hat{\\sigma}^2 = \\sum r_i^2 / (n - p)$ is the residual variance.

    2. **Weighted case** (weights provided):
        Uncertainties are KNOWN, use weighted formula:

        .. math::
            \\mathrm{Cov}(\\theta) = (J^T W J)^{-1}

        where $W = \\mathrm{diag}(w_1^2, ..., w_n^2)$ is the weight matrix.

    **Goodness of Fit:**

    For weighted fits, computes reduced chi-squared:

    .. math::
        \\chi^2_{\\text{red}} = \\frac{1}{n-p} \\sum_i r_i^2

    where residuals are already weighted. Ideally $\\chi^2_{\\text{red}} \\approx 1$.

    **Numerical Stability:**

    - Checks condition number before inversion
    - Falls back to pseudo-inverse if matrix is singular
    - Warns about ill-conditioning or non-identifiable parameters
    """
    try:
        # Compute Jacobian at optimal parameters using JAX autodiff
        jac_fn = jax.jacobian(residual_fn, argnums=0)
        J = jac_fn(solution.value, None)

        # Ensure J is 2D (n_data, n_params)
        if J.ndim == 1:
            J = J.reshape(-1, 1)

        n_data, n_params = J.shape

        # Get residuals at optimal point
        residuals = residual_fn(solution.value, None)

        # Compute degrees of freedom
        dof = n_data - n_params
        if dof <= 0:
            print(f" *** Warning: Not enough data points ({n_data}) for {n_params} parameters")
            print(f"              Cannot compute reliable uncertainties (DOF = {dof})")
            return None, None, None

        # Compute covariance matrix
        if weights is None:
            # UNWEIGHTED CASE: Estimate uncertainty from residuals
            residual_variance = float(jnp.sum(residuals**2) / dof)
            JtJ = J.T @ J

            # Check condition number
            cond_number = float(jnp.linalg.cond(JtJ))
            if cond_number > 1e12:
                print(f" *** Warning: Ill-conditioned Hessian (κ = {cond_number:.2e})")
                print("              Parameters may be poorly identified or highly correlated")

            # Invert with fallback to pseudo-inverse
            try:
                cov_matrix = residual_variance * jnp.linalg.inv(JtJ)
            except (jnp.linalg.LinAlgError, ValueError):
                print(" *** Warning: Singular Hessian - using pseudo-inverse")
                cov_matrix = residual_variance * jnp.linalg.pinv(JtJ)

        else:
            # WEIGHTED CASE: Uncertainties are known
            # Formula: Cov = (J^T W J)^{-1}
            W = jnp.diag(weights**2)  # Weight matrix
            JtWJ = J.T @ W @ J

            # Check condition number
            cond_number = float(jnp.linalg.cond(JtWJ))
            if cond_number > 1e12:
                print(f" *** Warning: Ill-conditioned weighted Hessian (κ = {cond_number:.2e})")
                print("              Parameters may be poorly identified or highly correlated")

            # Invert with fallback to pseudo-inverse
            try:
                cov_matrix = jnp.linalg.inv(JtWJ)
            except (jnp.linalg.LinAlgError, ValueError):
                print(" *** Warning: Singular weighted Hessian - using pseudo-inverse")
                cov_matrix = jnp.linalg.pinv(JtWJ)

            # Compute reduced chi-squared for goodness of fit
            chi2 = float(jnp.sum(residuals**2))  # Residuals already weighted
            reduced_chi2 = chi2 / dof

            print(" * Goodness of Fit:")
            print(f"    χ²         = {chi2:.4f}")
            print(f"    Reduced χ² = {reduced_chi2:.4f} (expect ~1.0 for good fit)")

            if reduced_chi2 > 2.0:
                print(" *** Warning: χ² >> 1 suggests poor fit or underestimated uncertainties")
            elif reduced_chi2 < 0.5:
                print(" *** Warning: χ² << 1 suggests overestimated uncertainties")

        # Transform covariance from [A, n, Ea] to [ln(A), n, Ea/R] space (Turányi-Nagy)
        if result_params is not None:
            # Build transformation Jacobian: d[ln(A), n, Ea/R] / d[A, n, Ea]
            # J_transform = diag([1/A, 1, 1/R_cal])
            transform_diag = []
            param_order = free_params if free_params else ["A", "n", "Ea"]

            for name in param_order:
                if name == "A":
                    transform_diag.append(1.0 / result_params["A"])  # d(ln A)/dA = 1/A
                elif name == "Ea":
                    transform_diag.append(1.0 / constants.R_cal_mol)  # d(Ea/R)/dEa = 1/R
                else:  # n
                    transform_diag.append(1.0)  # dn/dn = 1

            J_transform = jnp.diag(jnp.array(transform_diag))

            # Transform covariance: Cov' = J @ Cov @ J^T
            cov_matrix = J_transform @ cov_matrix @ J_transform.T

        # Compute standard errors (diagonal of covariance matrix)
        std_errors = jnp.sqrt(jnp.diag(cov_matrix))

        # Compute correlation matrix
        std_matrix = jnp.outer(std_errors, std_errors)

        # Avoid division by zero
        corr_matrix = jnp.where(std_matrix > 0, cov_matrix / std_matrix, 0.0)

        return std_errors, cov_matrix, corr_matrix

    except Exception as e:
        print(f" *** Error computing parameter uncertainties: {e}")
        print("     Returning None for all uncertainty statistics")
        return None, None, None
