from typing import Dict, Optional, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from jax import jit, lax
from jaxtyping import Array, Float64

from ..physical_constants import constants
from .rate_interpreter import parse_rate_constant


class Arrhenius(eqx.Module):
    lnA: Float64
    n: Float64
    EaR: Float64
    name: str

    def __init__(
        self,
        rate_dict: Optional[Dict] = None,
        params: Optional[Array] = None,
        name: Optional[str] = "unknown :(",
    ) -> None:
        if isinstance(rate_dict, dict):
            self._init_from_dict(rate_dict)
        elif params is not None:
            self._init_from_array(name, params)
        else:
            raise ValueError("Either rate_dict or params must be provided")

    def _init_from_dict(self, rate_const: Dict) -> None:
        """Initialize from a dictionary containing rate constant information."""
        self.name = rate_const["name"]
        parameters = parse_rate_constant(rate_const)
        self.lnA = jnp.log(parameters[0])
        self.n = parameters[1]
        self.EaR = parameters[2] / constants.R_cal_mol

    def _init_from_array(self, name: str, params: Array) -> None:
        """Initialize from an array of [A, n, Ea] values."""
        self.name = name
        self.lnA = jnp.log(params[0])
        self.n = params[1]
        self.EaR = params[2] / constants.R_cal_mol

    @classmethod
    def from_data(
        cls, rates: Array, temps: Array, three_params: bool = True, name: Optional[str] = "unknown :("
    ) -> "Arrhenius":
        """Create an Arrhenius instance by fitting to experimental data."""
        lnA, n, EaR = refit_arrhenius(rates, temps, three_params)
        params = jnp.array([jnp.exp(lnA), n, EaR * constants.R_cal_mol])
        return cls(params=params, name=name)

    @eqx.filter_jit
    def kinetic_constant(self, T: Union[Float64, Array]) -> Union[Float64, Array]:
        """Calculate rate constant at given temperature(s)."""
        return jnp.exp(self.lnA + self.n * jnp.log(T) - self.EaR / T)

    def get_parameters(self) -> Tuple[Float64, Float64, Float64]:
        """Return the Arrhenius parameters (A, n, Ea)."""
        return jnp.exp(self.lnA), self.n, self.EaR * constants.R_cal_mol

    def __str__(self) -> str:
        """Return a string representation in CHEMKIN format."""
        return "{}\t\t{:.5e} {:.5e} {:.5e}".format(self.name, jnp.exp(self.lnA), self.n, self.EaR * constants.R_cal_mol)


@jit
def refit_arrhenius(
    rate_constant: Array,
    temperature: Array,
    three_params: bool = False,
) -> Tuple[Float64, Float64, Float64]:
    """Refit Arrhenius parameters from rate constant data using least squares regression."""
    log_k = jnp.log(rate_constant)
    inv_T = 1.0 / temperature

    X = lax.cond(
        three_params,
        lambda _: jnp.vstack([jnp.ones_like(inv_T), jnp.log(temperature), -inv_T]).T,
        lambda _: jnp.vstack([jnp.ones_like(inv_T), jnp.zeros_like(inv_T), -inv_T]).T,
        None,
    )

    beta = jnp.linalg.lstsq(X, log_k, rcond=None)[0]

    # Return ln(A), n, Ea/R with n=0 for two-parameter model
    return beta[0], beta[1], beta[-1]
