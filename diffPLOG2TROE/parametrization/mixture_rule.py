from typing import Dict, List, Optional, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from ..utilities.thermodynamic_utilities import calculate_concentration
from .collision_efficiency import CollisionEfficiency, serialize_collision_efficiencies
from .rate_constant import AnyRate, forward_rate_constant


class MixtureRule(eqx.Module):
    default_rate_constant: AnyRate
    explicit_rate_constants: Dict[str, AnyRate]
    linear: bool
    reduced_pressure: bool
    name: str
    efficiencies: Optional[Dict[str, Dict]] = None

    def __init__(
        self,
        default_rate_constant: AnyRate,
        explicit_rate_constants: Dict[str, AnyRate],
        efficiencies: Optional[List[CollisionEfficiency]] = None,
        linear: bool = True,
        reduced_pressure: bool = False,
        name: str = "",
    ) -> None:
        self.default_rate_constant = default_rate_constant
        self.explicit_rate_constants = explicit_rate_constants
        self.efficiencies = None if efficiencies is None else serialize_collision_efficiencies(efficiencies)

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
        if self.reduced_pressure:
            pass
        else:
            return self._miller_like_mixture_rules(T, P, composition)

    def _miller_like_mixture_rules(
        self,
        T: Union[Float64, Float64[Array, "dim"]],
        P: Union[Float64, Float64[Array, "dim"]],
        composition: Optional[Dict[str, Float64]] = None,
    ) -> Union[Float64, Float64[Array, "dim"]]:
        c_tot = calculate_concentration(T, P)

        k_default = forward_rate_constant(self.default_rate_constant, T, P, composition)

        if composition is None:
            return k_default
        else:
            # Linear mixture rule with concentration weighting
            weighted_sum = 0.0
            total_explicit_concentration = 0.0

            for species_name, rate_constant in self.explicit_rate_constants.items():
                if species_name in composition:
                    x_i = composition[species_name]
                    c_i = x_i * c_tot

                    k_i = forward_rate_constant(rate_constant, T, P, composition)

                    weighted_sum += c_i * k_i
                    total_explicit_concentration += c_i

            # Remaining concentration
            c_remaining = jnp.maximum(0.0, c_tot - total_explicit_concentration)

            # Apply linear mixture rule: (Σ(c[i] * k[i]) + c_remaining * k_default) / c_tot
            kinetic_constant = (weighted_sum + c_remaining * k_default) / c_tot

            return kinetic_constant

    def __str__(self) -> str:
        """Return string representation in CHEMKIN format."""
        return ""
