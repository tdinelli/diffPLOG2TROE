from typing import Dict, Optional, Union

import equinox as eqx
from jaxtyping import Array, Float64

from .arrhenius import Arrhenius


class CollisionEfficiency(eqx.Module):
    name: str = "M"
    _is_constant: bool = True
    _efficiency: Union[Float64, Arrhenius] = Float64

    def __init__(
        self,
        name: str = "M",
        *,  # Force keyword-only arguments
        value: Optional[Float64] = None,
        parameters: Optional[Dict[str, Float64]] = None,
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

        self.name = name

    def __call__(
        self,
        T: Optional[Union[Float64, Float64[Array, "dim"]]] = None,
    ) -> Union[Float64, Float64[Array, "dim"]]:
        if isinstance(self._efficiency, Arrhenius) and T is not None:
            return self._efficiency.rate_constant(T)
        else:
            return self._efficiency

    @property
    def is_constant(self) -> bool:
        return self._is_constant

    @property
    def value(self):
        if self._is_constant:
            return {"efficiency": self._efficiency}
        else:  # Arrhenius like expression
            return {"lnA": self._efficiency.lnA, "n": self._efficiency.n, "EaR": self._efficiency.EaR}
