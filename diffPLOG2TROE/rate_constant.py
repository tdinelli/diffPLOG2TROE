from typing import Dict, Optional, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from .parametrization.arrhenius import Arrhenius
from .parametrization.cabr import CABR
from .parametrization.falloff import FallOff
from .parametrization.plog import Plog


# RateConstantParametrization = Union[Arrhenius, Plog, FallOff, CABR]
RateConstantParametrization = Union[Arrhenius, Plog]


class RateConstant(eqx.Module):
    reaction: RateConstantParametrization

    def __init__(self, rate_constant: RateConstantParametrization) -> None:
        self.reaction = rate_constant

    @eqx.filter_jit
    def forward_rate_constant(
        self,
        T: Union[Float64, Array],
        P: Optional[Union[Float64, Array]] = None,
        concentrations: Optional[Dict[str, float]] = None,
    ) -> Union[Float64, Array, Tuple[Union[Float64, Array], Union[Float64, Array]]]:
        # ==============================================================================
        # Temperature dependency only
        if isinstance(self.reaction, Arrhenius):
            return self.reaction.kinetic_constant(T)

        # ==============================================================================
        # Pressure dependency only
        if P is None:
            raise ValueError(f"{type(self.reaction).__name__} requires pressure")

        if isinstance(self.reaction, Plog):
            return self.reaction.kinetic_constant(T, P)
        # ==============================================================================
        # Possible mixture dependency
        elif isinstance(self.reaction, FallOff):
            if concentrations is None:
                return self.reaction.kinetic_constant(T, P)
            else:
                if self.reaction.explicit_efficiencies is True:
                    raise ValueError("Concentration dep not implemented yet!")
                else:
                    return self.reaction.kinetic_constant(T, P)
        elif isinstance(self.reaction, CABR):
            if concentrations is None:
                return self.reaction.kinetic_constant(T, P)
            else:
                if self.reaction.explicit_efficiencies is True:
                    raise ValueError("Concentration dep not implemented yet!")
                else:
                    return self.reaction.kinetic_constant(T, P)

        # if self.type in ["falloff", "cabr"]:
        #     # Calculate base M value (total concentration)
        #     M = self._calculate_concentration(P, T)
        #
        #     # Calculate effective M with efficiencies
        #     eff_M = jnp.zeros_like(M) if jnp.isscalar(M) else jnp.zeros_like(M)[0]
        #
        #     # Apply efficiencies for species in composition
        #     for species, mole_frac in composition.items():
        #         efficiency = self.rate_constant.efficiencies.get(species, 1.0)
        #         eff_M = eff_M + efficiency * mole_frac * M
        #
        #     # For any remaining fraction, use default efficiency of 1.0
        #     remaining_fraction = 1.0 - sum(composition.values())
        #     if remaining_fraction > 0:
        #         eff_M = eff_M + remaining_fraction * M
        #
        #     # Calculate kinetic constant with effective M
        #     if self.type == "falloff":
        #         k_hpl = self.rate_constant.hpl.kinetic_constant(T)
        #         k_lpl = self.rate_constant.lpl.kinetic_constant(T)
        #         Pr = (k_lpl * eff_M) / k_hpl
        #         F = self.rate_constant._compute_falloff_factor(T, Pr)
        #         k_eff = k_hpl * (Pr / (1 + Pr)) * F
        #     else:  # CABR
        #         k_hpl = self.rate_constant.hpl.kinetic_constant(T)
        #         k_lpl = self.rate_constant.lpl.kinetic_constant(T)
        #         Pr = (k_lpl * eff_M) / k_hpl
        #         F = self.rate_constant._compute_blending_function(T, Pr)
        #         k_eff = k_lpl * (1 / (1 + Pr)) * F
        #
        #     return k_eff, eff_M
        #
        # # For other pressure-dependent types without efficiencies
        # return self.rate_constant.kinetic_constant(T, P)
