from typing import Dict, List, Optional, Union

import equinox as eqx
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float64

from ..utilities.physical_constants import constants
from ..utilities.thermodynamic_utilities import calculate_effective_concentration
from .arrhenius import Arrhenius
from .broadening_functions import compute_broadening_factor
from .collision_efficiency import CollisionEfficiency, serialize_collision_efficiencies
from .utils import validate_broadening_parameters


class FallOff(eqx.Module):
    hpl: Arrhenius
    lpl: Arrhenius
    falloff_type: str
    name: str
    falloff_parameters: Optional[Dict[str, Float64]] = None
    efficiencies: Optional[Dict[str, Dict]] = None

    def __init__(
        self,
        hpl_parameters: Dict[str, Float64],
        lpl_parameters: Dict[str, Float64],
        falloff_type: str,
        falloff_parameters: Optional[Dict[str, Float64]] = None,
        efficiencies: Optional[List[CollisionEfficiency]] = None,
        name: str = "",
    ) -> None:
        self.hpl = Arrhenius(parameters=hpl_parameters, name=name)
        self.lpl = Arrhenius(parameters=lpl_parameters, name=name)
        self.falloff_type = falloff_type
        self.falloff_parameters = validate_broadening_parameters(falloff_type, falloff_parameters)
        self.efficiencies = None if efficiencies is None else serialize_collision_efficiencies(efficiencies)

        self.name = name

    @eqx.filter_jit
    def rate_constant(
        self,
        T: Union[Float64, Float64[Array, "dim"]],
        P: Union[Float64, Float64[Array, "dim"]],
        composition: Optional[Dict[str, Float64]] = None,
    ) -> Union[Float64, Float64[Array, "dim"]]:
        k_hpl = self.hpl.rate_constant(T)  # [cm3/mol/s]
        k_lpl = self.lpl.rate_constant(T)  # [cm3/mol/s] TODO: Check this unit just for the comment and doc

        if jnp.isscalar(P) or P.ndim == 0:  # P is scalar
            return self._single_P_rate_constant(T, P, k_lpl, k_hpl, composition)
        else:  # P is array
            vec_func = vmap(lambda p: self._single_P_rate_constant(T, p, k_lpl, k_hpl, composition))
            return vec_func(P)

    @eqx.filter_jit
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
        F = compute_broadening_factor(self.falloff_type, T, Pr, self.falloff_parameters)

        return hpl * (Pr / (1 + Pr)) * F

    def __str__(self) -> str:
        """Return string representation in CHEMKIN format."""
        representation = "{}\t\t{:.5e} {:.5f} {:.5e}\n".format(
            self.name, jnp.exp(self.hpl.lnA), self.hpl.n, self.hpl.EaR * constants.R_cal_mol
        )
        representation += " LOW / \t\t{:.5e} {:.5f} {:.5e} /\n".format(
            jnp.exp(self.lpl.lnA), self.lpl.n, self.lpl.EaR * constants.R_cal_mol
        )
        if self.falloff_type == 1 and self.falloff_parameters is not None:
            representation += " TROE / {:.5e} {:.5e} {:.5e} {:.5e} /".format(
                self.falloff_parameters["A"],
                self.falloff_parameters["T3"],
                self.falloff_parameters["T1"],
                self.falloff_parameters["T2"],
            )
        elif self.falloff_type == 2 and self.falloff_parameters is not None:
            representation += " SRI / {:.5e} {:.5e} {:.5e} {:.5e} {:.5e} /".format(
                self.falloff_parameters["a"],
                self.falloff_parameters["b"],
                self.falloff_parameters["c"],
                self.falloff_parameters["d"],
                self.falloff_parameters["e"],
            )
        elif self.falloff_type == 3 and self.falloff_parameters is not None:
            representation += " TSANG / {:.5e} {:.5e} /".format(
                self.falloff_parameters["A"],
                self.falloff_parameters["B"],
            )
        if self.efficiencies is not None:
            # TODO:
            # representation += "\n"
            # for collision_efficiency in self.efficiencies:
            #     representation += " {} / {:.5f} /".format(collision_efficiency.name, collision_efficiency.lnA)
            pass
        return representation
