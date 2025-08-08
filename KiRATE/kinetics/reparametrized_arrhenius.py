import equinox as eqx
import jax.numpy as jnp

from ..types.common import ParamsDict, RateType, Scalar, ScalarOrVector
from ..utilities.physical_constants import constants
from .utils import validate_arrhenius_parameters


class ReparametrizedArrhenius(eqx.Module):
    _lnk_ref: Scalar
    _n: Scalar
    _EaR: Scalar
    _T_ref: Scalar
    _name: str = eqx.field(static=True, default="")

    def __init__(self, parameters: ParamsDict, name: str = "") -> None:
        # ====================================================================
        # Validate basic parameters
        temp_params = {"A": parameters["k_ref"], "n": parameters["n"], "Ea": parameters["Ea"]}
        eqx.filter_pure_callback(validate_arrhenius_parameters, temp_params, result_shape_dtypes=None)

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
    def rate_constant(self, T: ScalarOrVector) -> RateType:
        return jnp.exp(self._lnk_ref + self._n * jnp.log(T / self._T_ref) - self._EaR * (1.0 / T - 1.0 / self._T_ref))

    def to_standard_form(self) -> ParamsDict:
        A_standard = float(jnp.exp(self._lnk_ref) * (self._T_ref ** (-self._n)) * jnp.exp(self._EaR / self._T_ref))
        return {"A": A_standard, "n": float(self._n), "Ea": float(self._EaR * constants.R_cal_mol)}

    @classmethod
    def from_standard_form(cls, parameters: ParamsDict, T_ref: Scalar, name: str = "") -> "ReparametrizedArrhenius":
        EaR = parameters["Ea"] / constants.R_cal_mol
        k_ref = parameters["A"] * (T_ref ** parameters["n"]) * jnp.exp(-EaR / T_ref)

        centered_params = {"k_ref": k_ref, "n": parameters["n"], "Ea": parameters["Ea"], "T_ref": T_ref}

        return cls(centered_params, name)

    def __str__(self) -> str:
        """Return string representation in centered format."""
        k_ref_original = jnp.exp(self._lnk_ref)
        Ea_original = self._EaR * constants.R_cal_mol
        return f"{self._name}\t\t{k_ref_original:.5e} {self._n:.5e} {Ea_original:.5e} {self._T_ref:.1f}"

    def __repr__(self) -> str:
        return (
            f"ReparametrizedArrhenius("
            f"\n name = {self._name}"
            f"\n kref = {jnp.exp(self._lnk_ref):.3e}"
            f"\n n    = {self._n:.3f}"
            f"\n Ea   = {self._EaR * constants.R_cal_mol:.3e}"
            f"\n Tref = {self._T_ref:.3f}"
            "\n)"
        )

    # ========================================================================
    # Getters
    @property
    def k_ref(self) -> Scalar:
        return jnp.exp(self._lnk_ref)

    @property
    def A(self) -> Scalar:
        return jnp.exp(self._lnk_ref) * (self._T_ref ** (-self._n)) * jnp.exp(self._EaR / self._T_ref)

    @property
    def n(self) -> Scalar:
        return self._n

    @property
    def Ea(self) -> Scalar:
        return self._EaR * constants.R_cal_mol

    @property
    def T_ref(self) -> Scalar:
        return self._T_ref

    @property
    def name(self) -> str:
        return self._name
