from typing import Dict, Optional, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from .cabr import CABR
from .falloff import FallOff
from .rate_constant import AnyRate, forward_rate_constant


class MixtureRule(eqx.Module):
    default_rate_constant: AnyRate
    explicit_rate_constants: Dict[str, AnyRate]
    linear: bool
    reduced_pressure: bool
    name: str

    def __init__(
        self,
        default_rate_constant: AnyRate,
        explicit_rate_constants: Dict[str, AnyRate],
        linear: bool = True,
        reduced_pressure: bool = False,
        name: str = "",
    ) -> None:
        self.default_rate_constant = default_rate_constant

        for rate_constant in explicit_rate_constants:
            if isinstance(rate_constant, FallOff) or isinstance(rate_constant, CABR):
                if rate_constant.efficiencies is not None:
                    raise ValueError(
                        "Explicit rate constant in the mixture rules formalism should not have collision efficiencies defined!"
                    )

        self.explicit_rate_constants = explicit_rate_constants

        if linear is False:
            raise ValueError("Non-Linear mixture rules are not implemented yet!")

        self.linear = True if linear is True else False
        self.reduced_pressure = True if reduced_pressure is True else False
        self.name = name

    @eqx.filter_jit
    def rate_constant(
        self,
        T: Union[Float64, Float64[Array, "dim"]],
        P: Union[Float64, Float64[Array, "dim"]],
        composition: Optional[Dict[str, Float64]] = None,
    ) -> Union[Float64, Float64[Array, "dim"]]:
        if self.reduced_pressure and self.linear:
            return self._lmr_r(T, P, composition)
        else:  # self.reduced_pressure False and self.linear True
            return self._lmr_p(T, P, composition)

    def _lmr_p(
        self,
        T: Union[Float64, Float64[Array, "dim"]],
        P: Union[Float64, Float64[Array, "dim"]],
        composition: Optional[Dict[str, Float64]] = None,
    ) -> Union[Float64, Float64[Array, "dim"]]:
        k_default = forward_rate_constant(self.default_rate_constant, T, P, composition)

        if composition is None:
            return k_default

        # Mole fraction weighted average (like Cantera does)
        weighted_sum = 0.0
        total_explicit_fraction = 0.0

        # Calculate contributions from explicit species
        for species_name, rate_constant in self.explicit_rate_constants.items():
            if species_name in composition:
                x_i = composition[species_name]  # Mole fraction
                k_i = forward_rate_constant(rate_constant, T, P, composition)
                weighted_sum += x_i * k_i
                total_explicit_fraction += x_i

        # Remaining mole fraction gets the default rate constant
        x_remaining = jnp.maximum(0.0, 1.0 - total_explicit_fraction)

        # Final weighted average: Σ(x_i * k_i) + x_remaining * k_default
        kinetic_constant = weighted_sum + x_remaining * k_default

        return kinetic_constant

    def _lmr_r(
        self,
        T: Union[Float64, Float64[Array, "dim"]],
        P: Union[Float64, Float64[Array, "dim"]],
        composition: Optional[Dict[str, Float64]] = None,
    ) -> Union[Float64, Float64[Array, "dim"]]:
        k_default = forward_rate_constant(self.default_rate_constant, T, P, composition)

        if composition is None:
            return k_default

    def __str__(self) -> str:
        """Return string representation in CHEMKIN format."""
        return ""

    # Equivalent implementation for the LMR_P rate constant or as they are called OpenSMOKE ExtendedFallOff
    # def _lmr_p(
    #     self,
    #     T: Union[Float64, Float64[Array, "dim"]],
    #     P: Union[Float64, Float64[Array, "dim"]],
    #     composition: Optional[Dict[str, Float64]] = None,
    # ) -> Union[Float64, Float64[Array, "dim"]]:
    #     c_tot = calculate_concentration(T, P)
    #
    #     k_default = forward_rate_constant(self.default_rate_constant, T, P, composition)
    #
    #     if composition is None:
    #         return k_default
    #     else:
    #         # Linear mixture rule with concentration weighting
    #         weighted_sum = 0.0
    #         total_explicit_concentration = 0.0
    #
    #         for species_name, rate_constant in self.explicit_rate_constants.items():
    #             if species_name in composition:
    #                 x_i = composition[species_name]
    #                 c_i = x_i * c_tot
    #
    #                 k_i = forward_rate_constant(rate_constant, T, P, composition)
    #
    #                 weighted_sum += c_i * k_i
    #                 total_explicit_concentration += c_i
    #
    #         # Remaining concentration
    #         c_remaining = jnp.maximum(0.0, c_tot - total_explicit_concentration)
    #
    #         # Apply linear mixture rule: (Σ(c[i] * k[i]) + c_remaining * k_default) / c_tot
    #         kinetic_constant = (weighted_sum + c_remaining * k_default) / c_tot
    #
    #         return kinetic_constant
