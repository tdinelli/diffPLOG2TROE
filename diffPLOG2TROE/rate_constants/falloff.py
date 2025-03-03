from typing import Dict, Optional, Union

import equinox as eqx
import jax.numpy as jnp
from jax import lax, vmap
from jaxtyping import Array, Float64

from ..physical_constants import constants
from .arrhenius import Arrhenius
from .falloff_functions import lindemann, sri, troe
from .rate_interpreter import parse_rate_constant


class FallOff(eqx.Module):
    hpl: Arrhenius  # High-pressure limit
    lpl: Arrhenius  # Low-pressure limit
    falloff_type: int  # 0: Lindemann, 1: Troe, 2: SRI
    falloff_params: Array
    name: str

    def __init__(
        self,
        rate_dict: Optional[Dict] = None,
        hpl_params: Optional[Array] = None,
        lpl_params: Optional[Array] = None,
        falloff_params: Optional[Array] = None,
        falloff_type: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize FallOff reaction from dictionary or arrays.
        """
        if isinstance(rate_dict, dict):
            self._init_from_dict(rate_dict)
        elif all(x is not None for x in [name, hpl_params, lpl_params, falloff_type]):
            self._init_from_array(name, hpl_params, lpl_params, falloff_params, falloff_type)
        else:
            raise ValueError("Either rate_dict or all required parameters must be provided")

    def _init_from_dict(self, rate_constant: Dict) -> None:
        """Initialize from a dictionary containing rate constant information."""
        self.name = rate_constant["name"]
        hpl_coeff, lpl_coeff, self.falloff_params, self.falloff_type = parse_rate_constant(rate_constant)
        self.hpl = Arrhenius(params=hpl_coeff, name=rate_constant["name"])
        self.lpl = Arrhenius(params=lpl_coeff, name=rate_constant["name"])

    def _init_from_array(
        self,
        name: str,
        hpl_params: Array,
        lpl_params: Array,
        falloff_params: Array,
        falloff_type: int,
    ) -> None:
        """Initialize directly from arrays of parameters."""
        self.name = name
        self.falloff_params = falloff_params
        self.falloff_type = falloff_type
        self.hpl = Arrhenius(params=hpl_params, name=name)
        self.lpl = Arrhenius(params=lpl_params, name=name)

    def _calculate_concentration(self, P: Float64, T: Union[Float64, Array]) -> Union[Float64, Array]:
        """Calculate concentration [mol/L] from pressure [atm] and temperature [K]."""
        return (P / (constants.R_L_atm_K_mol * T)) * jnp.float64(0.001)

    def _compute_falloff_factor(self, T: Union[Float64, Array], Pr: Union[Float64, Array]) -> Union[Float64, Array]:
        """Compute falloff factor based on falloff type and reduced pressure."""
        operand = (T, Pr, self.falloff_params)
        return lax.switch(
            self.falloff_type,
            [lambda _: lindemann(T), lambda x: troe(*x), lambda x: sri(*x)],
            operand,
        )

    @eqx.filter_jit
    def _single_P_kinetic_constant(self, T: Union[Float64, Array], P: Float64) -> Union[Float64, Array]:
        k_hpl = self.hpl.kinetic_constant(T)
        k_lpl = self.lpl.kinetic_constant(T)
        M = self._calculate_concentration(P, T)
        Pr = k_lpl * M / k_hpl
        F = self._compute_falloff_factor(T, Pr)
        return k_hpl * (Pr / (1 + Pr)) * F

    @eqx.filter_jit
    def kinetic_constant(self, T: Union[Float64, Array], P: Union[Float64, Array]) -> Union[Float64, Array]:
        if jnp.isscalar(P) or P.ndim == 0:
            return self._single_P_kinetic_constant(T, P)
        else:
            vectorized_k = vmap(lambda p: self._single_P_kinetic_constant(T, p))
            return vectorized_k(P)

    def __str__(self) -> str:
        """Return string representation in CHEMKIN format."""
        representation = "{}\t\t{:.5e} {:.5f} {:.5e}\n".format(
            self.name, jnp.exp(self.hpl.lnA), self.hpl.n, self.hpl.EaR * constants.R_cal_mol
        )
        representation += " LOW / \t\t{:.5e} {:.5f} {:.5e} /\n".format(
            jnp.exp(self.lpl.lnA), self.lpl.n, self.lpl.EaR * constants.R_cal_mol
        )
        if self.falloff_type == 1:
            representation += " TROE / {:.5e} {:.5e} {:.5e} {:.5e} /".format(
                self.falloff_params[0],
                self.falloff_params[1],
                self.falloff_params[2],
                self.falloff_params[3],
            )
        elif self.falloff_type == 2:
            representation += " SRI / {:.5e} {:.5e} {:.5e} {:.5e} {:.5e} /".format(
                self.falloff_params[0],
                self.falloff_params[1],
                self.falloff_params[2],
                self.falloff_params[3],
                self.falloff_params[4],
            )
        return representation
