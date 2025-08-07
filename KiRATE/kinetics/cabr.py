from typing import Dict, Optional

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jax import vmap
from jaxtyping import jaxtyped

from ..types.common import ArrayLike, Either, ParamsDict, Real
from ..utilities.thermodynamic_utilities import calculate_effective_concentration
from .arrhenius import Arrhenius
from .broadening_functions import compute_broadening_factor
from .collision_efficiency import CollisionEfficiency
from .utils import validate_broadening_parameters


class CABR(eqx.Module):
    _hpl: Arrhenius
    _lpl: Arrhenius
    _cabr_type: str
    _name: str
    _cabr_parameters: Optional[ParamsDict] = None
    _efficiencies: Optional[Dict[str, CollisionEfficiency]] = None

    def __init__(
        self,
        hpl_parameters: ParamsDict,
        lpl_parameters: ParamsDict,
        cabr_type: str,
        cabr_parameters: Optional[ParamsDict] = None,
        efficiencies: Optional[Dict[str, CollisionEfficiency]] = None,
        name: str = "",
    ) -> None:
        self._hpl = Arrhenius(parameters=hpl_parameters, name=f"HPL: {name}")
        self._lpl = Arrhenius(parameters=lpl_parameters, name=f"LPL: {name}")
        self._cabr_type = cabr_type
        self._cabr_parameters = validate_broadening_parameters(cabr_type, cabr_parameters)
        self._efficiencies = None if efficiencies is None else efficiencies

        self._name = name

    @eqx.filter_jit
    def rate_constant(self, T: Either, P: Either, composition: Optional[ParamsDict] = None) -> ArrayLike:
        k_hpl = self._hpl.rate_constant(T)  # [cm3/mol/s]
        k_lpl = self._lpl.rate_constant(T)  # [cm3/mol/s] TODO: Check this unit just for the comment and doc

        if jnp.isscalar(P) or P.ndim == 0:  # P is scalar
            return self._single_P_rate_constant(T, P, k_lpl, k_hpl, composition)
        else:  # P is array
            vec_func = vmap(lambda p: self._single_P_rate_constant(T, p, k_lpl, k_hpl, composition))
            return vec_func(P)

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def _single_P_rate_constant(
        self,
        T: Either,
        P: Real,
        lpl: Either,
        hpl: Either,
        composition: Optional[ParamsDict] = None,
    ) -> Either:
        M = calculate_effective_concentration(T, P, composition, self._efficiencies)  # [mol/cm3]
        Pr = (lpl * M) / hpl
        F = compute_broadening_factor(self._cabr_type, T, Pr, self._cabr_parameters)

        return lpl * (1 / (1 + Pr)) * F

    def __str__(self) -> str:
        """Return string representation in CHEMKIN format."""
        representation = "{}\t\t{:.5e} {:.5f} {:.5e}\n".format(self._name, self._lpl.A, self._lpl.n, self._lpl.Ea)
        representation += " HIGH / \t\t{:.5e} {:.5f} {:.5e} /\n".format(self._hpl.A, self._hpl.n, self._hpl.Ea)
        if self._cabr_type == 1 and self._cabr_parameters is not None:
            representation += " TROE / {:.5e} {:.5e} {:.5e} {:.5e} /".format(
                self._cabr_parameters["A"],
                self._cabr_parameters["T3"],
                self._cabr_parameters["T1"],
                self._cabr_parameters["T2"],
            )
        elif self._cabr_type == 2 and self._cabr_parameters is not None:
            representation += " SRI / {:.5e} {:.5e} {:.5e} {:.5e} {:.5e} /".format(
                self._cabr_parameters["a"],
                self._cabr_parameters["b"],
                self._cabr_parameters["c"],
                self._cabr_parameters["d"],
                self._cabr_parameters["e"],
            )
        elif self._cabr_type == 3 and self._cabr_parameters is not None:
            representation += " TSANG / {:.5e} {:.5e} /".format(
                self._cabr_parameters["A"],
                self._cabr_parameters["B"],
            )
        # TODO: add the print of the efficiencies here
        # if self.efficiencies is not None:
        # representation += "\n"
        # for collision_efficiency in self.efficiencies:
        #     representation += " {} / {:.5f} /".format(collision_efficiency.name, collision_efficiency.lnA)
        return representation

    @property
    def hpl(self) -> Arrhenius:
        return self._hpl

    @property
    def lpl(self) -> Arrhenius:
        return self._lpl

    @property
    def cabr_parameters(self) -> Optional[ParamsDict]:
        if self._cabr_parameters is not None:
            return self._cabr_parameters
        else:
            return None

    @property
    def efficiencies(self) -> Optional[Dict[str, CollisionEfficiency]]:
        if self._efficiencies is not None:
            return self._efficiencies
        else:
            return None

    @property
    def cabr_type(self) -> str:
        return self._cabr_type

    @property
    def name(self) -> str:
        return self._name
