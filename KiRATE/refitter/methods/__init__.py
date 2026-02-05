"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details

Optimization methods for fitting.
"""

from .linear import arrhenius_linear_fit
from .nonlinear import check_convergence, least_squares_fit

__all__ = [
    "arrhenius_linear_fit",
    "least_squares_fit",
    "check_convergence",
]
