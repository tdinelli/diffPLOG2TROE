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
    Calculate effective third-body concentration for pressure-dependent reactions.

    This function computes the concentration of collision partners (third bodies)
    in pressure-dependent reactions, accounting for species-specific collision
    efficiencies. The effective concentration is given by:

    .. math::
        [M]_{eff} = [M] \\cdot \\left( \\sum_i \\alpha_i x_i + (1 - \\sum_i x_i) \\right)

    where:
        - :math:`[M]_{eff}` is the effective third-body concentration [mol/cm3]
        - [M] is the total molar concentration [mol/cm3]
        - :math:`\\alpha_i` is the collision efficiency of species i [dimensionless]
        - :math:`x_i` is the mole fraction of species i [dimensionless]

    The sum runs over all species specified in the composition dictionary. Species
    not explicitly listed in the efficiencies dictionary are assigned a default
    efficiency of 1.0. The remaining mole fraction accounts for any unspecified
    species with unity efficiency.

    Parameters
    ----------
    T : Float64[Array, ""] | Float64[Array, "nt"]
        Temperature(s) in Kelvin.

    P : Float64[Array, ""] | Float64[Array, "np"]
        Pressure(s) in bar.

    composition : dict[str, Float64[Array, ""]], optional
        Dictionary mapping species names to their mole fractions [dimensionless].
        Mole fractions should sum to =< 1.0. If None, returns total concentration [M].

    efficiencies : dict[str, Float64[Array, ""]], optional
        Dictionary mapping species names to collision efficiency factors [dimensionless].
        Species not in this dict are assigned default efficiency of 1.0.
        If None, returns total concentration [M].

    Returns
    -------
    Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]
        Effective third-body concentration [M]_eff in mol/cm3.

        - Scalar output for scalar T and P
        - 1D array (nt,) for array T and scalar P
        - 1D array (np,) for scalar T and array P
        - 2D array (nt, np) for both array T and P (meshgrid)

    Notes
    -----
    When both composition and efficiencies are None, this function returns the
    total molar concentration [M] = P/(RT) calculated using the ideal gas law.
    see `calculate_concentration()`.
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
    Calculate total molar concentration using the ideal gas law.

    This function computes the total molar concentration from temperature and
    pressure using the ideal gas law:

    .. math::
        [M] = \\frac{P}{RT}

    where:
        - [M] is the total molar concentration [mol/cm3]
        - P is the pressure [bar]
        - R is the universal gas constant [J/mol/K]
        - T is the absolute temperature [K]

    The function handles both scalar and array inputs. When both T and P are
    arrays, it creates a 2D meshgrid to compute concentrations at all (T, P)
    combinations.

    Parameters
    ----------
    T : Float64[Array, ""] | Float64[Array, "nt"]
        Temperature(s) in Kelvin.

    P : Float64[Array, ""] | Float64[Array, "np"]
        Pressure(s) in bar.

        Note: Input pressure in bar is internally converted to Pascal
        by multiplying by 100000 (1 bar = 10^5 Pa exactly).
        Standard atmospheric pressure is approximately 1.01325 bar.

    Returns
    -------
    Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]
        Total molar concentration [M] in mol/cm3.

        - Scalar output for scalar T and P
        - 1D array (nt,) for array T and scalar P
        - 1D array (np,) for scalar T and array P
        - 2D array (nt, np) for both array T and P (meshgrid with indexing='ij')

    Notes
    -----
    **Unit conversions:**

    - Pressure: bar -> Pa (multiply by 100000, since 1 bar = 10^5 Pa exactly)
    - Volume: m3 -> cm3 (divide by 1e6)
    - Gas constant: R = 8.31446261815324 J/mol/K (CODATA 2018)

    The final concentration is computed as:

    .. math::
        [M] = \\frac{P \\times 10^5}{R \\times T \\times 10^6} \\quad [\\text{mol/cm}^3]
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    P = jnp.asarray(P, dtype=jnp.float64)
    P = P * jnp.float64(100000.0)  # bar -> Pa (1 bar = 10^5 Pa)

    R = constants.R_J_mol_K  # [J/mol/K]
    conversion_factor = jnp.float64(1e6)  # from [m3] to [cm3]

    if not (jnp.isscalar(T) or T.ndim == 0) and not (jnp.isscalar(P) or P.ndim == 0):
        T_grid, P_grid = jnp.meshgrid(T, P, indexing="ij")
        return (P_grid / (R * T_grid)) / conversion_factor
    else:
        return (P / (R * T)) / conversion_factor
