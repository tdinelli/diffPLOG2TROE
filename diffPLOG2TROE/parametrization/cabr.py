from typing import Dict, Optional, Union

import equinox as eqx
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float64

from ..utilities.physical_constants import constants
from ..utilities.thermodynamic_utilities import calculate_effective_concentration
from .arrhenius import Arrhenius
from .broadening_functions import compute_broadening_factor
from .parametrization_utils import validate_broadening_parameters  # , validate_efficiencies


class CABR(eqx.Module):
    hpl: Arrhenius
    lpl: Arrhenius
    cabr_type: str
    cabr_parameters: Dict[str, Float64]
    efficiencies: Dict[str, Float64]
    explicit_efficiencies: bool
    name: str

    def __init__(
        self,
        hpl_parameters: Dict[str, Float64],
        lpl_parameters: Dict[str, Float64],
        cabr_type: str,
        cabr_parameters: Optional[Dict[str, Float64]] = None,
        efficiencies: Optional[Dict[str, Float64]] = None,
        name: str = "",
    ) -> None:
        self.hpl = Arrhenius(parameters=hpl_parameters, name=name)
        self.lpl = Arrhenius(parameters=lpl_parameters, name=name)

        if efficiencies is None:
            self.efficiencies = {}
            self.explicit_efficiencies = False
        else:
            # validate_efficiencies(efficiencies)
            self.efficiencies = efficiencies
            self.explicit_efficiencies = True

        self.name = name
        self.cabr_parameters = validate_broadening_parameters(cabr_type, cabr_parameters)
        self.cabr_type = cabr_type

    @eqx.filter_jit
    def rate_constant(
        self,
        T: Union[Float64, Float64[Array, "dim"]],
        P: Union[Float64, Float64[Array, "dim"]],
        composition: Optional[Dict[str, Float64]] = None,
    ) -> Union[Float64, Float64[Array, "dim"]]:
        k_hpl = self.hpl.rate_constant(T)  # [cm3/mol/s]
        k_lpl = self.lpl.rate_constant(T)  # [cm3/mol/s]

        if jnp.isscalar(P) or P.ndim == 0:  # P is scalar
            return self._single_P_rate_constant(T, P, k_lpl, k_hpl, composition)
        else:  # P is array
            vec_func = vmap(lambda p: self._single_P_rate_constant(T, p, k_lpl, k_hpl, composition))
            return vec_func(P)

    def _single_P_rate_constant(
        self,
        T: Union[Float64, Float64[Array, "dim"]],
        P: Float64,
        lpl: Union[Float64, Float64[Array, "dim"]],
        hpl: Union[Float64, Float64[Array, "dim"]],
        composition: Optional[Dict[str, Float64]] = None,
    ) -> Union[Float64, Float64[Array, "dim"]]:
        M = calculate_effective_concentration(T, P, composition, self.efficiencies)  # [mol/cm3]
        Pr = (lpl * M) / hpl
        F = compute_broadening_factor(self.cabr_type, T, Pr, self.cabr_parameters)

        return lpl * (1 / (1 + Pr)) * F

    def __str__(self) -> str:
        """Return string representation in CHEMKIN format."""
        representation = "{}\t\t{:.5e} {:.5f} {:.5e}\n".format(
            self.name, jnp.exp(self.lpl.lnA), self.lpl.n, self.lpl.EaR * constants.R_cal_mol
        )
        representation += " HIGH / \t\t{:.5e} {:.5f} {:.5e} /\n".format(
            jnp.exp(self.hpl.lnA), self.hpl.n, self.hpl.EaR * constants.R_cal_mol
        )
        if self.cabr_type == 1 and self.cabr_parameters is not None:
            representation += " TROE / {:.5e} {:.5e} {:.5e} {:.5e} /".format(
                self.cabr_parameters["A"],
                self.cabr_parameters["T3"],
                self.cabr_parameters["T1"],
                self.cabr_parameters["T2"],
            )
        elif self.cabr_type == 2:
            representation += " SRI / {:.5e} {:.5e} {:.5e} {:.5e} {:.5e} /".format(
                self.cabr_parameters["a"],
                self.cabr_parameters["b"],
                self.cabr_parameters["c"],
                self.cabr_parameters["d"],
                self.cabr_parameters["e"],
            )
        elif self.cabr_type == 3:
            representation += " TSANG / {:.5e} {:.5e} /".format(
                self.cabr_parameters["A"],
                self.cabr_parameters["B"],
            )
        if self.explicit_efficiencies:
            representation += "\n"
            for key, value in self.efficiencies.items():
                representation += " {} / {:.5f} /".format(key, value)
        return representation
