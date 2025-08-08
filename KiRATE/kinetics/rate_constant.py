from typing import Optional, TypeAlias, Union

from ..types.common import RateType, ScalarOrVector, ParamsDict
from .arrhenius import Arrhenius
from .cabr import CABR
from .chebyshev import Chebyshev
from .falloff import FallOff
from .plog import Plog


AnyRate: TypeAlias = Union[Arrhenius, Plog, FallOff, CABR, Chebyshev]


def forward_rate_constant(
    reaction: Union[Arrhenius, Plog, FallOff, CABR, Chebyshev],
    T: ScalarOrVector,
    P: Optional[ScalarOrVector] = None,
    composition: Optional[ParamsDict] = None,
) -> RateType:
    if isinstance(reaction, Arrhenius):
        # ==============================================================================
        # Temperature dependency only
        return reaction.rate_constant(T)
    elif isinstance(reaction, Plog) or isinstance(reaction, Chebyshev):
        # ==============================================================================
        # Additional pressure dependency
        if P is None:
            raise ValueError(f"{type(reaction).__name__} requires pressure")
        return reaction.rate_constant(T, P)
    elif isinstance(reaction, FallOff) or isinstance(reaction, CABR):
        if P is None:
            raise ValueError(f"{type(reaction).__name__} requires pressure")
        # ==============================================================================
        # Possible mixture dependency (additional)
        if composition is None:
            return reaction.rate_constant(T, P)
        else:
            return reaction.rate_constant(T, P, composition)
    else:  # Maybe this will be redundant when the runtime type check will be enforced not at the moment
        raise ValueError(f"Unknown reaction type {reaction.__name__}")
