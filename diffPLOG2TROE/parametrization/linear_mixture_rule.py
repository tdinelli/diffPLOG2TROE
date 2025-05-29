from typing import Dict, Optional, TypeAlias, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float64

from ..utilities.custom_types import ScalarOrVector
from ..utilities.thermodynamic_utilities import calculate_concentration
from .arrhenius import Arrhenius
from .cabr import CABR
from .chebyshev import Chebyshev
from .falloff import FallOff
from .falloff_functions import validate_efficiencies
from .plog import Plog


# ============================================================================
# Chemical Kinetics Type Aliases
# ============================================================================
#: Union of all supported base reaction rate parametrization classes
AnyRate: TypeAlias = Union[Plog, FallOff, CABR, Chebyshev]


class LinearMixtureRule(eqx.Module):
    default_rate_constant: AnyRate
    explicit_rate_constants: Dict[str, AnyRate]
    efficiencies: Dict[str, Float64]
    explicit_efficiencies: bool
    name: str

    def __init__(
        self,
        default_rate_constant: AnyRate,
        explicit_rate_constants: Dict[str, AnyRate],
        efficiencies: Optional[Dict] = None,
        name: str = "",
    ) -> None:
        self.default_rate_constant = default_rate_constant
        self.explicit_rate_constants = explicit_rate_constants
        self.name = name
        if efficiencies is None:
            efficiencies = {}
        else:
            validate_efficiencies(efficiencies)

        self.efficiencies = efficiencies
        self.explicit_efficiencies = False if self.efficiencies is {} else True

    @eqx.filter_jit
    def kinetic_constant(
        self,
        T: ScalarOrVector,
        P: ScalarOrVector,
        composition: Optional[Dict[str, Float64]] = None,
    ) -> ScalarOrVector:
        if isinstance(self.default_rate_constant, Plog):
            k_default = self.default_rate_constant.kinetic_constant(T, P)
        elif isinstance(self.default_rate_constant, FallOff) or isinstance(self.default_rate_constant, CABR):
            k_default = self.default_rate_constant.kinetic_constant(T, P, composition)
        elif isinstance(self.default_rate_constant, Chebyshev):
            k_default = self.default_rate_constant.kinetic_constant(T, P, True)
        else:
            raise ValueError(f"Unknown default reaction type!")

        if composition is None:
            return k_default

        c_tot = calculate_concentration(T, P)

        # Linear mixture rule with concentration weighting
        weighted_sum = 0.0
        total_explicit_concentration = 0.0
        for species_name, rate_constant in self.explicit_rate_constants.items():
            if species_name in composition:
                x_i = composition[species_name]
                c_i = x_i * c_tot

                if isinstance(rate_constant, Plog):
                    k_i = rate_constant.kinetic_constant(T, P)
                elif isinstance(rate_constant, FallOff) or isinstance(rate_constant, CABR):
                    k_i = rate_constant.kinetic_constant(T, P, composition)
                elif isinstance(rate_constant, Chebyshev):
                    k_i = rate_constant.kinetic_constant(T, P, True)
                else:
                    raise ValueError(f"Unknown specific reaction type!")

                weighted_sum += c_i * k_i
                total_explicit_concentration += c_i

        # Remaining concentration
        c_remaining = jnp.maximum(0.0, c_tot - total_explicit_concentration)

        # Apply linear mixture rule: (\sum(c[i] * k[i]) + c_remaining * k_default) / c_tot
        kinetic_constant = (weighted_sum + c_remaining * k_default) / c_tot

        return kinetic_constant

    def __str__(self) -> str:
        """Return string representation in CHEMKIN format."""
        return ""
