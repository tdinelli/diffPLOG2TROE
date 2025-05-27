from typing import Dict, Optional

import equinox as eqx
from jaxtyping import Float64

from ..utilities.custom_types import AnyRate, ScalarOrVector
from .falloff_functions import validate_efficiencies


class LinearMixtureRule(eqx.Module):
    default_rate_constant: AnyRate
    explicit_rate_constant: Dict[str, AnyRate]
    efficiencies: Dict[str, Float64]
    explicit_efficiencies: bool
    name: str

    def __init__(
        self,
        default_rate_constant: AnyRate,
        explicit_rate_constant: Dict[str, AnyRate],
        efficiencies: Optional[Dict] = None,
        name: str = "",
    ) -> None:
        self.default_rate_constant = default_rate_constant
        self.explicit_rate_constant = explicit_rate_constant
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
        pass

    def __str__(self) -> str:
        """Return string representation in CHEMKIN format."""
        return ""
