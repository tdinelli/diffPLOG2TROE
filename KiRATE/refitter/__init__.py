"""
Copyright (c) 2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

from .core import (
    Fitter,
    FittingResult,
    compute_parameter_uncertainties,
    compute_statistics,
    estimate_missing_uncertainties,
)
from .fitters import ArrheniusFitter
from .methods import (
    arrhenius_linear_fit,
    check_convergence,
    least_squares_fit,
)

__all__ = [
    # New API - Fitters
    "ArrheniusFitter",
    # Core infrastructure
    "Fitter",
    "FittingResult",
    "compute_statistics",
    "compute_parameter_uncertainties",
    "estimate_missing_uncertainties",
    # Methods
    "arrhenius_linear_fit",
    "least_squares_fit",
    "check_convergence",
]
