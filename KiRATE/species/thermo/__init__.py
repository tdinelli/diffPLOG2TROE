"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

from .nasa7_polynomial import cp_r, h_rt, s_r, temperature_powers

# NOTE: fit_smooth_nasa7_coefficients is not imported here to avoid circular import
# (it depends on Species class). Import directly from nasa7_smooth_fit module when needed:
# from KiRATE.species.thermo.nasa7_smooth_fit import fit_smooth_nasa7_coefficients

__all__ = [
    # nasa7_polynomial
    "temperature_powers",
    "cp_r",
    "s_r",
    "h_rt",
]
