"""
Copyright (c) 2025 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from KiRATE.utilities.physical_constants import constants


class ReparametrizedArrhenius(eqx.Module):
    _k_ref: Float64[Array, ""]
    _n: Float64[Array, ""]
    _Ea: Float64[Array, ""]
    _T_ref: Float64[Array, ""]
    _name: str = eqx.field(static=True, default="")

    def __init__(self, parameters: dict[str, float], name: str = "") -> None:
        self._name = name
        self._k_ref = jnp.float64(parameters["k_ref"])
        self._n = jnp.float64(parameters["n"])
        self._Ea = jnp.float64(parameters["Ea"])
        self._T_ref = jnp.float64(parameters["T_ref"])

    @eqx.filter_jit
    def rate_constant(
        self, T: float | Float64[Array, ""] | Float64[Array, "n"]
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        T = jnp.asarray(T, dtype=jnp.float64)
        return (
            self._k_ref
            * jnp.pow(T / self._T_ref, self._n)
            * jnp.exp(-(self._Ea / constants.R_cal_mol) * (1.0 / T - 1.0 / self._T_ref))
        )

    def to_standard_form(self) -> dict[str, float]:
        A_standard = self._k_ref * (self._T_ref ** (-self._n)) * jnp.exp(self._Ea / constants.R_cal_mol / self._T_ref)
        return {"A": float(A_standard), "n": float(self._n), "Ea": float(self._Ea)}

    @classmethod
    def from_standard_form(
        cls,
        parameters: dict[str, float],
        T_ref: float | Float64[Array, ""],
        name: str = "",
    ) -> "ReparametrizedArrhenius":
        T_ref = jnp.float64(T_ref)
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
    def name(self) -> str:
        return self._name

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
