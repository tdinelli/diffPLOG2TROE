from typing import Dict

import jax.numpy as jnp
from jaxtyping import Float64


class PhysicalConstants:
    """
    Physical constants for chemical kinetics calculations.

    All constants are stored as jnp.float64 to ensure consistent numerical precision
    in JAX operations. The gas constant is defined in multiple unit systems
    for convenience.

    References:
    - CODATA 2018 recommended values
    - https://physics.nist.gov/cuu/Constants/
    """

    _J_TO_CAL: Float64 = jnp.float64(4.184)  # Conversion from joules to calories
    _MOL_TO_KMOL: Float64 = jnp.float64(1000.0)  # Conversion from mol to kmol
    R_J_mol: Float64 = jnp.float64(8.31446261815324)  # Ideal gas constant [J/mol/K]

    @property
    def R_cal_mol(self) -> Float64:
        """Ideal gas constant [cal/mol/K]"""
        return self.R_J_mol / jnp.float64(self._J_TO_CAL)

    @property
    def R_kcal_mol(self) -> Float64:
        """Ideal gas constant [kcal/mol/K]"""
        return self.R_J_mol / jnp.float64(self._J_TO_CAL * 1000.0)

    @property
    def R_eV_mol(self) -> Float64:
        """Ideal gas constant [eV/mol/K]"""
        return jnp.float64(8.617333262e-5 * 6.02214076e23)  # kB * NA

    @property
    def R_J_kmol(self) -> Float64:
        """Ideal gas constant [J/kmol/K]"""
        return self.R_J_mol * jnp.float64(self._MOL_TO_KMOL)

    @property
    def R_cal_kmol(self) -> Float64:
        """Ideal gas constant [cal/kmol/K]"""
        return self.R_J_kmol / jnp.float64(self._J_TO_CAL)

    @property
    def R_L_atm_K_mol(self) -> Float64:
        """Ideal gas constant [L·atm/K/mol]"""
        return jnp.float64(0.082057366080960)  # CODATA 2018 value

    def as_dict(self) -> Dict[str, Float64]:
        """Return all constants as a dictionary."""
        return {
            "R_J_mol": self.R_J_mol,
            "R_cal_mol": self.R_cal_mol,
            "R_kcal_mol": self.R_kcal_mol,
            "R_eV_mol": self.R_eV_mol,
            "R_J_kmol": self.R_J_kmol,
            "R_cal_kmol": self.R_cal_kmol,
            "R_L_atm_K_mol": self.R_L_atm_K_mol,
        }


constants = PhysicalConstants()
