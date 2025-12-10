from typing import Optional

import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Float64

from KiRATE.utilities.physical_constants import constants


@jit
def calculate_effective_concentration(
    T: Float64[Array, ""] | Float64[Array, "nt"],
    P: Float64[Array, ""] | Float64[Array, "np"],
    composition: Optional[dict[str, Float64[Array, ""]]] = None,
    efficiencies: Optional[dict[str, Float64[Array, ""]]] = None,
) -> Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]:
    """
    TODO: Documentation much needed here

    T is in kelvin
    P is bar
    composition is mole fraction
    """
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
    T: Float64[Array, ""] | Float64[Array, "nt"],
    P: Float64[Array, ""] | Float64[Array, "np"],
) -> Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "np nt"]:
    """
    Function that computes the concentration given the ideal gas law.
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
