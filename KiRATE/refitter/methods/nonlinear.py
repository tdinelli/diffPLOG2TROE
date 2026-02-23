"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import optax
from jaxtyping import Array, Float64


@dataclass
class Solution:
    """Solution from Optax L-BFGS optimization."""

    value: Float64[Array, "n_params"]
    converged: bool
    n_steps: int
    final_loss: float
    grad_norm: float


def least_squares_fit(
    loss_fn: Callable[[Float64[Array, "n_params"]], float],
    initial_params: Float64[Array, "n_params"],
    max_steps: int = 200,
    grad_tol: float = 1e-10,
) -> Solution:
    """
    Solve optimization problem using Optax L-BFGS.

    Parameters
    ----------
    loss_fn : Callable
        Loss function to minimize: fn(params) -> scalar
    initial_params : Float64[Array, "n_params"]
        Initial parameter guess
    max_steps : int, optional
        Maximum optimization iterations, by default 200
    grad_tol : float, optional
        Gradient norm tolerance for convergence, by default 1e-10

    Returns
    -------
    Solution
        Solution object with optimized parameters and convergence info
    """
    optimizer = optax.lbfgs()
    opt_state = optimizer.init(initial_params)
    value_and_grad_fn = optax.value_and_grad_from_state(loss_fn)

    params = initial_params
    converged = False

    for step in range(max_steps):
        value, grad = value_and_grad_fn(params, state=opt_state)
        updates, opt_state = optimizer.update(grad, opt_state, params, value=value, grad=grad, value_fn=loss_fn)
        params = optax.apply_updates(params, updates)

        grad_norm = float(jnp.linalg.norm(grad))
        if grad_norm < grad_tol:
            converged = True
            n_steps = step + 1
            break
    else:
        n_steps = max_steps
        grad_norm = float(jnp.linalg.norm(grad))

    return Solution(
        value=params,
        converged=converged,
        n_steps=n_steps,
        final_loss=float(loss_fn(params)),
        grad_norm=grad_norm,
    )


def check_convergence(solution: Solution, verbose: bool = False) -> tuple[bool, float]:
    """Check convergence status."""
    if verbose:
        status = "converged" if solution.converged else "max steps reached"
        print(f"Optimization {status}")
        print(f"  Gradient norm: {solution.grad_norm:.3e}")
        print(f"  Steps taken: {solution.n_steps}")

    return solution.converged, solution.grad_norm
