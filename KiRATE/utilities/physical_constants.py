"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details

Physical constants for chemical kinetics calculations.

All constants are defined as class attributes with JAX arrays for consistent
numerical precision. Since physical constants are truly static and never
participate in gradients or JAX transformations, a plain class is most appropriate.

References
----------
.. [1] CODATA 2018 recommended values
       https://physics.nist.gov/cuu/Constants/
"""

import jax.numpy as jnp
from jaxtyping import Array, Float64


class PhysicalConstants:
    """
    Physical constants for chemical kinetics calculations.

    This class provides fundamental physical constants and the ideal gas constant
    in various unit systems commonly used in chemical kinetics. All constants are
    defined as class attributes since they are truly static values.

    Attributes
    ----------
    R_J_mol_K : Float64[Array, ""]
        Ideal gas constant [J/(mol·K)]
    R_cal_mol_K : Float64[Array, ""]
        Ideal gas constant [cal/(mol·K)]
    R_kcal_mol_K : Float64[Array, ""]
        Ideal gas constant [kcal/(mol·K)]
    R_eV_mol_K : Float64[Array, ""]
        Ideal gas constant [eV/(mol·K)]
    R_J_kmol_K : Float64[Array, ""]
        Ideal gas constant [J/(kmol·K)]
    R_cal_kmol_K : Float64[Array, ""]
        Ideal gas constant [cal/(kmol·K)]
    R_L_atm_K_mol : Float64[Array, ""]
        Ideal gas constant [L·atm/(K·mol)]
    R_cm3_atm_mol_K : Float64[Array, ""]
        Ideal gas constant [(cm³·atm)/(K·mol)]
    N_A : Float64[Array, ""]
        Avogadro constant [1/mol]
    KB : Float64[Array, ""]
        Boltzmann constant [J/K]
    KB_CGS : Float64[Array, ""]
        Boltzmann constant [erg/K]
    EPSILON_ZERO : Float64[Array, ""]
        Vacuum permittivity [F/m]

    Notes
    -----
    All constants are pre-computed and stored as JAX arrays for consistent
    numerical precision. The gas constant R is provided in multiple unit
    systems to avoid repeated conversions in performance-critical code.
    """

    # Fundamental constants (CODATA 2018)
    R_J_mol_K: Float64[Array, ""] = jnp.float64(8.31446261815324)  # [J/(mol·K)]
    N_A: Float64[Array, ""] = jnp.float64(6.02214076e23)  # [1/mol]
    KB: Float64[Array, ""] = jnp.float64(1.380649e-23)  # [J/K]
    KB_CGS: Float64[Array, ""] = jnp.float64(1.380649e-16)  # [erg/K]
    EPSILON_ZERO: Float64[Array, ""] = jnp.float64(8.8541878128e-12)  # [F/m]

    # Derived gas constants in various unit systems
    R_cal_mol_K: Float64[Array, ""] = jnp.float64(1.98720425864083)  # [cal/(mol·K)]
    R_kcal_mol_K: Float64[Array, ""] = jnp.float64(0.00198720425864083)  # [kcal/(mol·K)]
    R_eV_mol_K: Float64[Array, ""] = jnp.float64(8.617333262e-5 * 6.02214076e23)  # [eV/(mol·K)]
    R_J_kmol_K: Float64[Array, ""] = jnp.float64(8314.46261815324)  # [J/(kmol·K)]
    R_cal_kmol_K: Float64[Array, ""] = jnp.float64(1987.20425864083)  # [cal/(kmol·K)]
    R_L_atm_K_mol: Float64[Array, ""] = jnp.float64(0.082057366080960)  # [L·atm/(K·mol)]
    R_cm3_atm_mol_K: Float64[Array, ""] = jnp.float64(82.05736608096)  # [(cm³·atm)/(K·mol)]


# Singleton instance for convenient access throughout the codebase
constants = PhysicalConstants()
