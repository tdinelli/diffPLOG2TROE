"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float64
from optimistix import (
    RESULTS,
    Dogleg,
    IndirectLevenbergMarquardt,
    LevenbergMarquardt,
    Solution,
    least_squares,
)


def least_squares_fit(
    residual_fn: Callable[[Float64[Array, "n_params"], None], Float64[Array, "n"]],
    initial_params: Float64[Array, "n_params"],
    max_steps: int = 1000,
    atol: float = 1e-8,
    rtol: float = 1e-8,
    solver: str = "lm",
) -> Solution:
    """
    Solve nonlinear least squares problem using optimistix.

    Parameters
    ----------
    residual_fn : Callable
        Residual function with signature:
            fn(params, args) -> residuals
        where params is the parameter vector to optimize and args is unused
        (for compatibility with optimistix interface).
    initial_params : Float64[Array, "n_params"]
        Initial parameter guess
    max_steps : int, optional
        Maximum number of optimization steps, by default 1000
    atol : float, optional
        Absolute convergence tolerance, by default 1e-8
    rtol : float, optional
        Relative convergence tolerance, by default 1e-8
    solver : str, optional
        Solver algorithm to use. Options:
        - "lm": Levenberg-Marquardt (default, robust)
        - "ilm": Indirect Levenberg-Marquardt (memory efficient)
        - "dogleg": Dogleg trust region (alternative)
        By default "lm"

    Returns
    -------
    Solution
        Optimistix Solution object containing:
        - value: Optimal parameter values
        - result: Convergence status code
        - stats: Optimization statistics

    Notes
    -----
    **Levenberg-Marquardt Algorithm:**

    The LM algorithm interpolates between Gauss-Newton and gradient descent:

    .. math::
        (J^T J + \\lambda I) \\Delta \\theta = -J^T r

    where:
        - J is the Jacobian of residuals
        - r is the residual vector
        - λ is the damping parameter (adjusted adaptively)

    **Convergence Criteria:**

    The solver stops when:
        - ||gradient|| < atol (first-order optimality)
        - ||step|| / ||params|| < rtol (relative change small)
        - max_steps reached

    **Solver Selection:**

    - **LM**: Best for small-medium problems (< 100 parameters)
    - **ILM**: Better for large problems (memory efficient)
    - **Dogleg**: Alternative trust-region method, good for ill-conditioned problems
    """
    # Select solver algorithm
    solver_obj: LevenbergMarquardt | IndirectLevenbergMarquardt | Dogleg
    if solver == "lm":
        solver_obj = LevenbergMarquardt(rtol=rtol, atol=atol)
    elif solver == "ilm":
        solver_obj = IndirectLevenbergMarquardt(rtol=rtol, atol=atol)
    elif solver == "dogleg":
        solver_obj = Dogleg(rtol=rtol, atol=atol)
    else:
        raise ValueError(f"Unknown solver '{solver}'. Choose from: 'lm', 'ilm', 'dogleg'")

    # Solve least squares problem
    solution: Solution = least_squares(
        fn=residual_fn,
        solver=solver_obj,
        y0=initial_params,
        args=None,
        max_steps=max_steps,
        throw=False,  # Don't raise on non-convergence, return Solution with status
    )

    return solution


def check_convergence(solution: Solution, verbose: bool = False) -> tuple[bool, float]:
    """
    Check convergence status of optimization solution.

    Parameters
    ----------
    solution : Solution
        Optimistix Solution object from least_squares
    verbose : bool, optional
        If True, print convergence diagnostics, by default False

    Returns
    -------
    converged : bool
        True if optimization converged successfully
    optimality : float
        First-order optimality measure (gradient norm at solution)

    Notes
    -----
    Checks the RESULTS enum from optimistix to determine convergence:
        - successful: Converged within tolerances
        - max_steps_reached: Hit max iterations (may still be good)
        - other: Solver failed (singular matrix, NaN, etc.)
    """

    # Check result code
    if solution.result == RESULTS.successful:
        converged = True
        if verbose:
            print("Optimization converged successfully")
    elif solution.result == RESULTS.max_steps_reached:
        # May have converged "close enough"
        converged = False
        if verbose:
            print(f"Max steps reached ({solution.stats['num_steps']} steps)")
    else:
        converged = False
        if verbose:
            print(f"Optimization failed: {solution.result}")

    # Extract optimality measure
    # Note: optimistix doesn't always populate stats, so we need to be careful
    optimality = float(solution.stats.get("grad_norm", jnp.nan)) if hasattr(solution.stats, "get") else jnp.nan

    if verbose:
        print(f"  Gradient norm: {optimality:.3e}")
        print(f"  Steps taken: {solution.stats.get('num_steps', 'unknown')}")

    return converged, optimality
