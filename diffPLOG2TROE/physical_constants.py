from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class PhysicalConstants:
    R_J_mol = jnp.float64(8.3144621)  # Ideal gas constant [J/mol/K]
    R_cal_mol = jnp.float64(8.3144621 / 4.18443)  # Ideal gas constant [cal/mol/K]
    R_kcal_mol = jnp.float64(8.3144621 / 4184.43)  # Ideal gas constant [kcal/mol/K]
    R_eV_mol = jnp.float64(5.189479288e19)  # Ideal gas constant [eV/mol/K]
    R_J_kmol = jnp.float64(8314.4621)  # Ideal gas constant [J/kmol/K]
    R_cal_kmol = jnp.float64(8314.4621 / 4.18443)  # Ideal gas constant [cal/kmol/K]
    R_L_atm_K_mol = jnp.float64(0.08205746)  # Ideal gas constant [L atm/K/mol]
