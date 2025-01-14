from typing import Tuple

from jax import jit, vmap
from jaxtyping import Array, Float64

from .arrhenius_base import kinetic_constant_base


# Define constants
R_GAS_CONSTANT = 0.08206  # Gas constant in L⋅atm/(K⋅mol)
CONVERSION_FACTOR = 1 / 1000  # Convert from L to cm³


@jit
def kinetic_constant_threebody(
    threebody_constant: Tuple, T: Float64, P: Float64
) -> Tuple[Float64, Float64]:
    """Calculate the three-body reaction kinetic constant.

    Args:
        threebody_constant: Tuple of Arrhenius parameters
        T: Temperature in Kelvin
        P: Pressure in atmospheres

    Returns:
        Tuple containing:
            - k_threebody: Three-body reaction rate constant [mol/cm³/s]
            - M: Third body concentration [mol/cm³]
    """
    M = P / (R_GAS_CONSTANT * T) * CONVERSION_FACTOR  # [mol/cm³]
    k_threebody = kinetic_constant_base(threebody_constant, T) * M
    return (k_threebody, M)


@jit
def compute_threebody(
    threebody_constant: Tuple, T_range: Array, P_range: Array
) -> Array:
    """Compute three-body reaction rates for ranges of temperature and pressure.

    Args:
        threebody_constant: Tuple of Arrhenius parameters
        T_range: Array of temperatures in Kelvin
        P_range: Array of pressures in atmospheres

    Returns:
        Array of three-body reaction rate constants [mol/cm³/s]
    """

    def compute_single(t: Float64, p: Float64) -> Float64:
        k_threebody, _ = kinetic_constant_threebody(threebody_constant, t, p)
        return k_threebody

    # Vectorize computation over temperature and pressure ranges
    compute_single_t_fixed = vmap(
        lambda p: vmap(lambda t: compute_single(t, p))(T_range)
    )
    k_threebody = compute_single_t_fixed(P_range)

    return k_threebody
