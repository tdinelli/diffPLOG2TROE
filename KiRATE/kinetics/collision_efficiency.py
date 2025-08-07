from typing import Optional, Union

import equinox as eqx

from ..types.common import Either, ParamsDict, Real
from .arrhenius import Arrhenius


class CollisionEfficiency(eqx.Module):
    _efficiency: Union[Real, Arrhenius]
    _name: str = eqx.field(static=True, default="M")
    _is_constant: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        name: str = "M",
        *,  # Force keyword-only arguments
        value: Optional[float] = None,
        parameters: Optional[ParamsDict] = None,
    ) -> None:
        if value is not None and parameters is None:
            if value <= 0:
                raise ValueError(f"Collision efficiency must be positive, got {value}")

            self._efficiency = value
            self._is_constant = True
        elif value is None and parameters is not None:
            self._efficiency = Arrhenius(parameters=parameters)
            self._is_constant = False
        elif value is not None and parameters is not None:
            raise ValueError("Specify either value or arrhenius_parameters, not both")
        else:  # value is None and arrhenius_parameters is None:
            raise ValueError("Must specify either value or arrhenius_parameters")

        self._name = name

    def value(self, T: Optional[Either] = None):
        if isinstance(self._efficiency, Arrhenius) and T is not None:
            return self._efficiency.rate_constant(T)
        else:
            return self._efficiency

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_constant(self) -> bool:
        return self._is_constant
