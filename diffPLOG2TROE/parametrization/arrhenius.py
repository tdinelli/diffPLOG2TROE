from typing import Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from jax import jit, lax
from jaxtyping import Array, Float64

from ..utilities.physical_constants import constants


class Arrhenius(eqx.Module):
    lnA: Float64
    n: Float64
    EaR: Float64
    name: str

    def __init__(self, parameters: Array, name: str = "") -> None:
        self.name = name

        self._validate_parameters(parameters)

        # ==============================================================================
        # Pre-exponential factor
        self.lnA = jnp.log(parameters[0])

        # ==============================================================================
        # Temperature exponent
        self.n = parameters[1]

        # ==============================================================================
        # Activation energy
        self.EaR = parameters[2] / constants.R_cal_mol

    @classmethod
    def from_data(cls, rates: Array, temps: Array, three_params: bool = True, name: str = "") -> "Arrhenius":
        """Create an Arrhenius instance by fitting to experimental data."""
        lnA, n, EaR = refit_arrhenius(rates, temps, three_params)
        params = jnp.array([jnp.exp(lnA), n, EaR * constants.R_cal_mol])
        return cls(parameters=params, name=name)

    @eqx.filter_jit
    def kinetic_constant(self, T: Union[Float64, Array]) -> Union[Float64, Array]:
        """Calculate rate constant at given temperature(s)."""
        return jnp.exp(self.lnA + self.n * jnp.log(T) - self.EaR / T)

    def __str__(self) -> str:
        """Return a string representation in CHEMKIN format."""
        return "{}\t\t{:.5e} {:.5e} {:.5e}".format(self.name, jnp.exp(self.lnA), self.n, self.EaR * constants.R_cal_mol)

    @staticmethod
    def _validate_parameters(parameters: Array) -> None:
        """
        Validation method in order to ensure consistency in the calculations.

        Note: Maybe in the future add consistent validation of the unit of measurements
        """
        A, n, Ea = parameters
        if A == 0:
            raise ValueError("Pre-exponential factor cannot be equal to 0")

    @staticmethod
    def save_kinetic_constants_table(rate_constant: Array, temperatures: Array, output_file: str) -> None:
        """
        Note: Still to be implemented and tested
        """
        with open(output_file, "w") as f:
            f.write("T;k\n")
            for T, k in zip(temperatures, rate_constant):
                f.write(f"{T:.3f};{k:10e}\n")


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
