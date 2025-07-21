from typing import Dict, List, Optional, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from ..utilities.physical_constants import constants
from .utils import validate_arrhenius_parameters


class CollisionEfficiency(eqx.Module):
    """
    Represents collision efficiency for a species in pressure-dependent reactions.

    Collision efficiency can be either:
    1. Constant value (temperature-independent)
    2. Temperature-dependent following Arrhenius-like form: A * T^n * exp(-Ea/(R*T))

    Args:
        name: Species name (e.g., "H2", "CO", "M")
        value: Temperature-independent collision efficiency value
        arrhenius_parameters: Dict with keys "A", "n", "Ea" for temperature dependence

    Example:
        # Constant efficiency
        eff1 = CollisionEfficiency(name="CO", value=2.0)

        # Temperature-dependent efficiency
        eff2 = CollisionEfficiency(
            name="H2",
            arrhenius_parameters={"A": 1e13, "n": 0, "Ea": 30000}
        )
    """

    # Internal storage - always store as Arrhenius parameters for consistency
    _log_A: Float64
    _n: Optional[Float64] = None
    _EaR: Optional[Float64] = None
    name: str = "M"
    _is_constant: bool = True

    def __init__(
        self,
        name: str = "M",
        *,  # Force keyword-only arguments
        value: Optional[Float64] = None,
        arrhenius_parameters: Optional[Dict[str, Float64]] = None,
    ) -> None:
        """Initialize collision efficiency with either constant value or Arrhenius parameters."""

        if value is not None and arrhenius_parameters is None:
            if value <= 0:
                raise ValueError(f"Collision efficiency must be positive, got {value}")

            self._log_A = value
            self._n = None
            self._EaR = None
            self._is_constant = True
        elif value is None and arrhenius_parameters is not None:
            validate_arrhenius_parameters(arrhenius_parameters)
            self._log_A = jnp.log(arrhenius_parameters["A"])
            self._n = arrhenius_parameters["n"]
            self._EaR = arrhenius_parameters["Ea"] / constants.R_cal_mol
            self._is_constant = False
        elif value is not None and arrhenius_parameters is not None:
            raise ValueError("Specify either value or arrhenius_parameters, not both")
        else:  # value is None and arrhenius_parameters is None:
            raise ValueError("Must specify either value or arrhenius_parameters")

        self.name = name

    def __call__(
        self,
        T: Optional[Union[Float64, Float64[Array, "dim"]]] = None,
    ) -> Union[Float64, Float64[Array, "dim"]]:
        """
        Calculate collision efficiency at given temperature(s).

        Args:
            T: Temperature in K. Not required for constant efficiencies.

        Returns:
            Collision efficiency value(s)
        """
        if self._is_constant:
            return self._log_A # Here is not the actual log but in this way we avoid some calculations
        else:
            if T is None:
                raise ValueError("Temperature T is required for temperature-dependent collision efficiency")
            return self._evaluate_arrhenius(T)

    @eqx.filter_jit
    def _evaluate_arrhenius(self, T: Union[Float64, Float64[Array, "dim"]]) -> Union[Float64, Float64[Array, "dim"]]:
        """Evaluate Arrhenius-like expression: A * T^n * exp(-Ea/(R*T))"""
        log_result = self._log_A + self._n * jnp.log(T) - self._EaR / T
        return jnp.exp(log_result)

    # TODO: Add and check this
    # @property
    # def is_constant(self) -> bool:
    #     """True if collision efficiency is temperature-independent."""
    #     return self._is_constant
    # def __str__(self) -> str:
    #     if self._is_constant:
    #         return f"{self.name}: {jnp.exp(self._log_A):.3f}"
    #     else:
    #         A = jnp.exp(self._log_A)
    #         Ea = self._EaR * constants.R_cal_mol
    #         return f"{self.name}: {A:.3e} * T^{self._n:.3f} * exp(-{Ea:.1f}/RT)"


def serialize_collision_efficiencies(collision_list: List[CollisionEfficiency]) -> Dict[str, Dict]:
    """Serialize list of collision efficiencies to dictionary."""
    # TODO: Check this float transformation
    result = {}
    for eff in collision_list:
        result[eff.name] = {
            "lnA": float(eff._log_A),
            "n": float(eff._n) if eff._n is not None else None,
            "EaR": float(eff._EaR) if eff._EaR is not None else None,
        }
    return result


def deserialize_collision_efficiencies(serialized_dict: Dict[str, Dict]) -> List[CollisionEfficiency]:
    """Deserialize dictionary to list of collision efficiencies."""
    collision_list = []

    for name, data in serialized_dict.items():
        if data["type"] == "constant":
            eff = CollisionEfficiency(name=name, value=data["value"])
        elif data["type"] == "arrhenius":
            eff = CollisionEfficiency(name=name, arrhenius_parameters=data["parameters"])
        else:
            raise ValueError(f"Unknown collision efficiency type: {data['type']}")

        collision_list.append(eff)

    return collision_list
