from typing import Dict, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from KiRATE.kinetics.utils import validate_arrhenius_parameters
from KiRATE.utilities.physical_constants import constants


class ReparametrizedArrhenius(eqx.Module):
    _k_ref: Float64[Array, ""]
    _n: Float64[Array, ""]
    _Ea: Float64[Array, ""]
    _T_ref: Float64[Array, ""]
    _name: str = eqx.field(static=True, default="")

    def __init__(self, parameters: Dict[str, float], name: str = "") -> None:
        # ====================================================================
        # Validate basic parameters
        temp_params = {
            "A": parameters["k_ref"],
            "n": parameters["n"],
            "Ea": parameters["Ea"],
        }
        validate_arrhenius_parameters(temp_params)

        # ====================================================================
        # Additional validation for T_ref
        if parameters["T_ref"] <= 0:
            raise ValueError("Reference temperature T_ref must be positive")

        self._name = name
        self._k_ref = jnp.float64(parameters["k_ref"])
        self._n = jnp.float64(parameters["n"])
        self._EaR = jnp.float64(parameters["Ea"])
        self._T_ref = jnp.float64(parameters["T_ref"])

    @eqx.filter_jit
    def rate_constant(
        self, T: Union[float, Float64[Array, ""], Float64[Array, "n"]]
    ) -> Union[Float64[Array, ""], Float64[Array, "n"]]:
        T = jnp.asarray(T, dtype=jnp.float64)
        return (
            self._k_ref
            + self._n * jnp.log(T / self._T_ref)
            - (self._Ea / constants.R_cal_mol) * (1.0 / T - 1.0 / self._T_ref)
        )

    def to_standard_form(self) -> Dict[str, float]:
        A_standard = self._k_ref * (self._T_ref ** (-self._n)) * jnp.exp(self._Ea / constants.R_cal_mol / self._T_ref)
        return {"A": float(A_standard), "n": float(self._n), "Ea": float(self._EaR * constants.R_cal_mol)}

    @classmethod
    def from_standard_form(
        cls,
        parameters: Dict[str, float],
        T_ref: Union[float, Float64[Array, ""]],
        name: str = "",
    ) -> "ReparametrizedArrhenius":
        k_ref = parameters["A"] * (T_ref ** parameters["n"]) * jnp.exp(-parameters["Ea"] / constants.R_cal_mol / T_ref)
        centered_params = {"k_ref": float(k_ref), "n": parameters["n"], "Ea": parameters["Ea"], "T_ref": T_ref}

        return cls(centered_params, name)

    def __str__(self) -> str:
        """Return string representation in centered format."""
        k_ref_original = self._k_ref
        Ea_original = self._Ea
        return f"{self._name}\t\t{k_ref_original:.5e} {self._n:.5e} {Ea_original:.5e} {self._T_ref:.1f}"

    def __repr__(self) -> str:
        return (
            f"ReparametrizedArrhenius("
            f"\n name = {self._name}"
            f"\n Tref = {self._T_ref:.5f}"
            f"\n kref = {self._k_ref:.5E}"
            f"\n n    = {self._n:.5E}"
            f"\n Ea   = {self._Ea:.5E}"
            "\n)"
        )

    @property
    def k_ref(self) -> Float64[Array, ""]:
        return self._k_ref

    @property
    def A(self) -> Float64[Array, ""]:
        return self._k_ref * (self._T_ref ** (-self._n)) * jnp.exp(self._Ea / constants.R_cal_mol / self._T_ref)

    @property
    def n(self) -> Float64[Array, ""]:
        return self._n

    @property
    def Ea(self) -> Float64[Array, ""]:
        return self._Ea

    @property
    def T_ref(self) -> Float64[Array, ""]:
        return self._T_ref

    @property
    def name(self) -> str:
        return self._name
