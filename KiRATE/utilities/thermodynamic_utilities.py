from typing import Dict, Optional, Union

import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Float64

from .physical_constants import constants


@jit
def calculate_effective_concentration(
    T: Union[Float64[Array, ""], Float64[Array, "nt"]],
    P: Union[Float64[Array, ""], Float64[Array, "np"]],
    composition: Optional[Dict[str, Float64[Array, ""]]] = None,
    efficiencies: Optional[Dict[str, Float64[Array, ""]]] = None,
) -> Union[Float64[Array, ""], Float64[Array, "nt"], Float64[Array, "np"], Float64[Array, "nt np"]]:
    """Calculate concentration with collision efficiencies applied (if provided)."""
    M = calculate_concentration(T, P)  # [mol/cm3]

    if efficiencies is None or composition is None:
        return M

    species_list = list(composition.keys())
    mole_fractions = jnp.array([composition[s] for s in species_list])

    default_eff = jnp.float64(1.0)
    eff_values = jnp.array([efficiencies.get(species, default_eff) for species in species_list])

    total_accounted_fraction = jnp.sum(mole_fractions)
    weighted_efficiency = jnp.sum(eff_values * mole_fractions)

    remaining_fraction = jnp.maximum(0.0, 1.0 - total_accounted_fraction)

    eff_M = M * (weighted_efficiency + remaining_fraction)

    return eff_M


def calculate_concentration(
    T: Union[Float64[Array, ""], Float64[Array, "np"]],
    P: Union[Float64[Array, ""], Float64[Array, "nt"]],
) -> Union[Float64[Array, ""], Float64[Array, "nt"], Float64[Array, "np"], Float64[Array, "np nt"]]:
    # TODO: Update the documentation
    """
    Calculate molar concentration from pressure and temperature using the ideal gas law.

    Computes total concentration Ctot [mol/cm³] = P/(R*T), where:
    - P is pressure in atmospheres [atm]
    - T is temperature in Kelvin [K]
    - R is the gas constant, units must be consistent with T and P

    Parameters
    ----------
    T : float or ndarray
        Temperature in Kelvin [K]. Can be a scalar or array.
    P : float or ndarray
        Pressure in atmospheres [atm]. Can be a scalar or array.

    Returns
    -------
    float or ndarray
        Concentration in mol/cm³ with shape depending on inputs:
        - If both T and P are scalars: returns a scalar
        - If one is scalar and one is array: returns an array matching the non-scalar input
        - If both are arrays: returns a 2D meshgrid where result[i, j] corresponds to T[i], P[j]
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    P = jnp.asarray(P, dtype=jnp.float64)
    P = P * jnp.float64(101325.0)  # [Pa] which is [J/m3]

    R = constants.R_J_mol_K  # [J/mol/K]
    conversion_factor = jnp.float64(1e6)  # from [m3] to [cm3]

    if not (jnp.isscalar(T) or T.ndim == 0) and not (jnp.isscalar(P) or P.ndim == 0):
        T_grid, P_grid = jnp.meshgrid(T, P, indexing="ij")
        return (P_grid / (R * T_grid)) / conversion_factor
    else:
        return (P / (R * T)) / conversion_factor
