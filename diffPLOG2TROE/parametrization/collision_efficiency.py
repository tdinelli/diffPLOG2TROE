from typing import Dict, List, Optional, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from ..utilities.physical_constants import constants
from .parametrization_utils import validate_arrhenius_parameters


class CollisionEfficiency(eqx.Module):
    lnA: Float64

    n: Optional[Float64] = None
    EaR: Optional[Float64] = None
    name: str = "M"

    def __init__(self, parameters: Union[Float64, Dict[str, Float64]], name: str = "M") -> None:
        self.name = name

        if isinstance(parameters, Dict):
            validate_arrhenius_parameters(parameters)
            self.lnA = jnp.log(parameters["A"])
            self.n = parameters["n"]
            self.EaR = parameters["Ea"] / constants.R_cal_mol
        else:
            if parameters < 0:
                raise ValueError(f"Collision efficiency cannot be negative, given {parameters}")
            self.lnA = parameters

    def __call__(
        self,
        T: Optional[Union[Float64, Float64[Array, "dim"]]] = None,
    ) -> Union[Float64, Float64[Array, "dim"]]:
        if T is None and self.n is None:
            return self.lnA
        else:
            return self.arrhenius_like(T)

    @eqx.filter_jit
    def arrhenius_like(self, T: Union[Float64, Float64[Array, "dim"]]) -> Union[Float64, Float64[Array, "dim"]]:
        return jnp.exp(self.lnA + self.n * jnp.log(T) - self.EaR / T)

    def __str__(self) -> str:
        if self.n is None:
            return f"{self.name} / {self.lnA:.5e} /"
        else:
            return f"{self.name} / {jnp.exp(self.lnA):.5e} {self.n:.5e} {self.EaR * constants.R_cal_mol:.5e} /"


def serialize_collision_efficiencies(collision_list: List[CollisionEfficiency]) -> Dict[str, Dict]:
    """Serialize list to complete state dictionary."""
    result = {}
    for obj in collision_list:
        result[obj.name] = {
            "lnA": float(obj.lnA),
            "n": float(obj.n) if obj.n is not None else None,
            "EaR": float(obj.EaR) if obj.EaR is not None else None,
        }
    return result


def deserialize_collision_efficiencies(serialized_dict: Dict[str, Dict]) -> List[CollisionEfficiency]:
    """Deserialize complete state dictionary to list."""
    collision_list = []
    for name, state in serialized_dict.items():
        if state["n"] is not None and state["EaR"] is not None:
            params = {"A": jnp.exp(state["lnA"]), "n": state["n"], "Ea": state["EaR"] * constants.R_cal_mol}
            obj = CollisionEfficiency(params, name)
        else:
            obj = CollisionEfficiency(state["lnA"], name)
        collision_list.append(obj)
    return collision_list
