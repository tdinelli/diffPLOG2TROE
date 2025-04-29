from typing import Dict, Optional, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from jax import lax, vmap
from jaxtyping import Array, Float64

from ..physical_constants import constants
from .arrhenius import Arrhenius
from .falloff_functions import lindemann, sri, troe
from .rate_interpreter import FittingType, parse_rate_constant


class CABR(eqx.Module):
    hpl: Arrhenius  # High-pressure limit
    lpl: Arrhenius  # Low-pressure limit
    cabr_type: int  # 0: Lindemann, 1: Troe, 2: SRI
    cabr_params: Array
    efficiencies: Dict
    explicit_efficiencies: bool
    name: str

    def __init__(
        self,
        rate_dict: Optional[Dict] = None,
        hpl_params: Optional[Array] = None,
        lpl_params: Optional[Array] = None,
        cabr_params: Optional[Array] = None,
        cabr_type: Optional[str] = None,
        efficiencies: Optional[Dict] = None,
        name: Optional[str] = "unknown :(",
    ) -> None:
        """Initialize FallOff reaction from dictionary or arrays."""
        if efficiencies is None:
            efficiencies = {}

        if isinstance(rate_dict, dict):
            self._init_from_dict(rate_dict)
        elif all(x is not None for x in [hpl_params, lpl_params, cabr_type]):
            self._init_from_array(name, hpl_params, lpl_params, cabr_params, cabr_type, efficiencies)
        else:
            raise ValueError("Either rate_dict or array for parameters must be provided")

    def _init_from_dict(self, rate_constant: Dict) -> None:
        """Initialize from a dictionary containing rate constant information."""
        hpl_coeff, lpl_coeff, self.cabr_params, self.cabr_type, self.efficiencies = parse_rate_constant(rate_constant)

        self.explicit_efficiencies = False if self.efficiencies is {} else True

        self.name = rate_constant["name"]
        self.hpl = Arrhenius(params=hpl_coeff, name=rate_constant["name"])
        self.lpl = Arrhenius(params=lpl_coeff, name=rate_constant["name"])

    def _init_from_array(
        self,
        name: str,
        hpl_params: Array,
        lpl_params: Array,
        cabr_params: Array,
        cabr_type: str,
        efficiencies: Dict,
    ) -> None:
        """Initialize directly from arrays of parameters."""
        self.name = name
        self.cabr_type = self._convert_to_cabr_type(cabr_type)

        if self.cabr_type == 1:  # Troe
            cabr_params = jnp.pad(
                jnp.array(cabr_params, dtype=jnp.float64), (0, 5 - len(cabr_params)), constant_values=0.0
            )
        elif self.cabr_type == 2:  # SRI
            cabr_params = jnp.pad(
                jnp.array(cabr_params, dtype=jnp.float64), (0, 5 - len(cabr_params)), constant_values=0.0
            )
            if cabr_params[3] == 0.0:
                cabr_params = cabr_params.at[3].set(1.0)
        else:  # Lindemann
            cabr_params = jnp.zeros(5, dtype=jnp.float64)
        self.cabr_params = cabr_params

        self.hpl = Arrhenius(params=hpl_params, name=name)
        self.lpl = Arrhenius(params=lpl_params, name=name)
        self.efficiencies = efficiencies
        self.explicit_efficiencies = False if self.efficiencies is {} else True

    def _compute_blending_function(self, T: Union[Float64, Array], Pr: Union[Float64, Array]) -> Union[Float64, Array]:
        """Compute blending factor based on fitting type and reduced pressure."""
        operand = (T, Pr, self.cabr_params)
        return lax.switch(
            self.cabr_type,
            [lambda x: lindemann(*x), lambda x: troe(*x), lambda x: sri(*x)],
            operand,
        )

    @eqx.filter_jit
    def _single_P_kinetic_constant(
        self, T: Union[Float64, Array], P: Float64
    ) -> Tuple[Union[Float64, Array], Union[Float64, Array]]:
        k_hpl = self.hpl.kinetic_constant(T)
        k_lpl = self.lpl.kinetic_constant(T)
        M = self._calculate_concentration(P, T)
        Pr = (k_lpl * M) / k_hpl
        F = self._compute_blending_function(T, Pr)
        return (k_lpl * (1 / (1 + Pr)) * F, M)

    @eqx.filter_jit
    def kinetic_constant(
        self, T: Union[Float64, Array], P: Union[Float64, Array]
    ) -> Tuple[Union[Float64, Array], Union[Float64, Array]]:
        """
        Note for future development in principle we could precompute the vectorized functions in the constructor of the
        class to make things even more fast.
        """
        if (jnp.isscalar(T) or T.ndim == 0) and (jnp.isscalar(P) or P.ndim == 0):  # both are scalars
            return self._single_P_kinetic_constant(T, P)
        elif not (jnp.isscalar(T) or T.ndim == 0) and (jnp.isscalar(P) or P.ndim == 0):  # T is array, P is scalar
            return vmap(lambda t: self._single_P_kinetic_constant(t, P))(T)
        elif (jnp.isscalar(T) or T.ndim == 0) and not (jnp.isscalar(P) or P.ndim == 0):  # P is array, T is scalar
            return vmap(lambda p: self._single_P_kinetic_constant(T, p))(P)
        else:  # both are arrays
            return vmap(lambda p: vmap(lambda t: self._single_P_kinetic_constant(t, p))(T))(P)

    def __str__(self) -> str:
        """Return string representation in CHEMKIN format."""
        representation = "{}\t\t{:.5e} {:.5f} {:.5e}\n".format(
            self.name, jnp.exp(self.lpl.lnA), self.lpl.n, self.lpl.EaR * constants.R_cal_mol
        )
        representation += " HIGH / \t\t{:.5e} {:.5f} {:.5e} /\n".format(
            jnp.exp(self.hpl.lnA), self.hpl.n, self.hpl.EaR * constants.R_cal_mol
        )
        if self.cabr_type == 1:
            representation += " TROE / {:.5e} {:.5e} {:.5e} {:.5e} /".format(
                self.cabr_params[0],
                self.cabr_params[1],
                self.cabr_params[2],
                self.cabr_params[3],
            )
        elif self.cabr_type == 2:
            representation += " SRI / {:.5e} {:.5e} {:.5e} {:.5e} {:.5e} /".format(
                self.cabr_params[0],
                self.cabr_params[1],
                self.cabr_params[2],
                self.cabr_params[3],
                self.cabr_params[4],
            )
        if self.explicit_efficiencies:
            representation += "\n"
            for key, value in self.efficiencies.items():
                representation += " {} / {:.5f} /".format(key, value)
        return representation

    @staticmethod
    def _calculate_concentration(P: Float64, T: Union[Float64, Array]) -> Union[Float64, Array]:
        """Calculate concentration [mol/cm3] from pressure [atm] and temperature [K]."""
        return (P / (constants.R_L_atm_K_mol * T)) * jnp.float64(0.001)

    @staticmethod
    def _convert_to_cabr_type(cabr_type: str) -> int:
        """Convert string representation to FalloffType enum."""
        try:
            return {
                "lindemann": FittingType.lindemann,
                "troe": FittingType.troe,
                "sri": FittingType.sri,
            }[cabr_type.lower()]
        except KeyError:
            available = ", ".join(f"'{k}'" for k in ["lindemann", "troe", "sri"])
            raise ValueError(f"Unknown blending function type '{cabr_type}'. Available types: {available}")
