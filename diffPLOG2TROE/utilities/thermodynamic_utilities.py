from typing import Union

import jax.numpy as jnp
from jaxtyping import Array, Float64

from .physical_constants import constants

def calculate_concentration(T: Union[Float64, Array], P: Union[Float64, Array]) -> Float64:
    """Calculate concentration [mol/cm3] from pressure [atm] and temperature [K]."""
    return (P / (constants.R_L_atm_K_mol * T)) * jnp.float64(0.001)
