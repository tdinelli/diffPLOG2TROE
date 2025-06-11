import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float64

from ..utilities.custom_types import Array64f, ScalarOrVector
from ..utilities.physical_constants import constants

class ReparameterizedArrhenius(eqx.Module):
    """
    Reparameterized Arrhenius equation following Schwaab & Pinto (2007).

    Uses the form: k = k_ref * exp(-E/R * (1/T - 1/T_ref))

    This reduces parameter correlation compared to the traditional form.
    """

    ln_k_ref: Float64
    n: Float64
    E_over_R: Float64
    T_ref: Float64
    name: str

    def __init__(self, k_ref: Float64, n: Float64, Ea: Float64, T_ref: Float64, name: str = ""):
        """
        Initialize with reparameterized form.

        Parameters:
        -----------
        ln_k_ref : Float64
            Natural log of rate constant at reference temperature
        n : Float64
            Temperature exponent
        E : Float64
            Activation energy in cal/mol
        T_ref : Float64
            Reference temperature in K
        """
        self.name = name
        # self._validate_parameters(parameters)

        # ==============================================================================
        # Pre-exponential factor
        self.ln_k_ref = jnp.log(k_ref)

        # ==============================================================================
        # Temperature exponent
        self.n = n

        # ==============================================================================
        # Activation energy
        self.EaR = Ea / constants.R_cal_mol

    @eqx.filter_jit
    def kinetic_constant(self, T: Array64f) -> Array64f:
        """Calculate rate constant at given temperature(s)."""
        return jnp.exp(self.ln_k_ref + self.n * jnp.log(T) - self.E_over_R * (1 / T - 1 / self.T_ref))

    def get_traditional_params(self) -> Tuple[Float64, Float64, Float64]:
        """Convert to traditional Arrhenius parameters [A, n, E]."""
        ln_A = self.ln_k_ref + self.n * jnp.log(self.T_ref) + self.E_over_R / self.T_ref
        A = jnp.exp(ln_A)
        E = self.E_over_R * 1.987
        return A, self.n, E
