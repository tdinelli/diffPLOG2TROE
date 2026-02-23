"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

from .nasa7_polynomial import build_temperature_powers
from .nasa7_smooth_fit import fit_smooth_nasa7_coefficients
from .species import Species

# from .transport_properties import compute_thermal_conductivity, compute_viscosity

__all__ = [
    "Species",
    "fit_smooth_nasa7_coefficients",
    "build_temperature_powers",
    # "compute_viscosity",
    # "compute_thermal_conductivity",
]
