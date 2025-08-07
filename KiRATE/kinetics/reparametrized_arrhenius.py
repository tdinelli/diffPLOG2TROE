import equinox as eqx
import jax.numpy as jnp

from ..types.common import ArrayLike, Either, ParamsDict, Real
from ..utilities.physical_constants import constants
from .utils import validate_arrhenius_parameters


class ReparametrizedArrhenius(eqx.Module):
    _lnk_ref: Real
    _n: Real
    _EaR: Real
    _T_ref: Real
    _name: str

    def __init__(self, parameters: ParamsDict, name: str = "") -> None:
        # ====================================================================
        # TODO: This needs to be checked
        # Validate basic parameters (reuse existing validation)
        # temp_params = {"A": parameters["k_ref"], "n": parameters["n"], "Ea": parameters["Ea"]}
        # validate_arrhenius_parameters(temp_params)

        # ====================================================================
        # Additional validation for T_ref
        if parameters["T_ref"] <= 0:
            raise ValueError("Reference temperature T_ref must be positive")

        self._name = name
        self._lnk_ref = jnp.log(parameters["k_ref"])
        self._n = parameters["n"]
        self._EaR = parameters["Ea"] / constants.R_cal_mol
        self._T_ref = parameters["T_ref"]

    @eqx.filter_jit
    def rate_constant(self, T: Either) -> ArrayLike:
        return jnp.exp(self._lnk_ref + self._n * jnp.log(T / self._T_ref) - self._EaR * (1.0 / T - 1.0 / self._T_ref))

    def to_standard_form(self) -> ParamsDict:
        A_standard = float(jnp.exp(self._lnk_ref) * (self._T_ref ** (-self._n)) * jnp.exp(self._EaR / self._T_ref))
        return {"A": A_standard, "n": float(self._n), "Ea": float(self._EaR * constants.R_cal_mol)}

    @classmethod
    def from_standard_form(cls, parameters: ParamsDict, T_ref: Real, name: str = "") -> "ReparametrizedArrhenius":
        EaR = parameters["Ea"] / constants.R_cal_mol
        k_ref = parameters["A"] * (T_ref ** parameters["n"]) * jnp.exp(-EaR / T_ref)

        centered_params = {"k_ref": k_ref, "n": parameters["n"], "Ea": parameters["Ea"], "T_ref": T_ref}

        return cls(centered_params, name)

    def __str__(self) -> str:
        """Return string representation in centered format."""
        k_ref_original = jnp.exp(self._lnk_ref)
        Ea_original = self._EaR * constants.R_cal_mol
        return f"{self._name}\t\t{k_ref_original:.5e} {self._n:.5e} {Ea_original:.5e} {self._T_ref:.1f}"

    @property
    def k_ref(self) -> Real:
        return jnp.exp(self._lnk_ref)

    @property
    def A(self) -> Real:
        return jnp.exp(self._lnk_ref) * (self._T_ref ** (-self._n)) * jnp.exp(self._EaR / self._T_ref)

    @property
    def n(self) -> Real:
        return self._n

    @property
    def Ea(self) -> Real:
        return self._EaR * constants.R_cal_mol

    @property
    def T_ref(self) -> Real:
        return self._T_ref

    @property
    def name(self) -> str:
        return self._name
