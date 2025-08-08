from typing import Dict, Optional

import equinox as eqx
import jax.numpy as jnp
from jax import vmap

from ..types.common import ParamsDict, RateType, Scalar, ScalarOrVector
from ..utilities.thermodynamic_utilities import calculate_effective_concentration
from .arrhenius import Arrhenius
from .broadening_functions import compute_broadening_factor
from .collision_efficiency import CollisionEfficiency
from .utils import validate_broadening_parameters, validate_efficiencies


class FallOff(eqx.Module):
    _hpl: Arrhenius
    _lpl: Arrhenius
    _falloff_type: str = eqx.field(static=True)
    _falloff_parameters: Optional[ParamsDict] = None
    _efficiencies: Optional[Dict[str, CollisionEfficiency]] = None
    _name: str = eqx.field(static=True, default="")

    def __init__(
        self,
        hpl_parameters: ParamsDict,
        lpl_parameters: ParamsDict,
        falloff_type: str,
        falloff_parameters: Optional[ParamsDict] = None,
        efficiencies: Optional[Dict[str, CollisionEfficiency]] = None,
        name: str = "",
    ) -> None:
        self._hpl = Arrhenius(parameters=hpl_parameters, name=f"HPL: {name}")
        self._lpl = Arrhenius(parameters=lpl_parameters, name=f"LPL: {name}")
        self._falloff_type = falloff_type

        eqx.filter_pure_callback(
            lambda bt, params: validate_broadening_parameters(bt, params),
            falloff_type,
            falloff_parameters,
            result_shape_dtypes=None,
        )
        self._falloff_parameters = falloff_parameters

        if efficiencies is None:
            self._efficiencies = None
        else:
            eqx.filter_pure_callback(validate_efficiencies, efficiencies, result_shape_dtypes=None)
            self._efficiencies = efficiencies

        self._name = name

    @eqx.filter_jit
    def rate_constant(
        self,
        T: ScalarOrVector,
        P: ScalarOrVector,
        composition: Optional[ParamsDict] = None,
    ) -> RateType:
        k_hpl = self._hpl.rate_constant(T)  # [cm3/mol/s]
        k_lpl = self._lpl.rate_constant(T)  # [cm3/mol/s] TODO: Check this unit just for the comment and doc

        if jnp.isscalar(P):  # P is scalar
            return self._single_P_rate_constant(T, P, k_lpl, k_hpl, composition)
        else:  # P is array
            vec_func = vmap(lambda p: self._single_P_rate_constant(T, p, k_lpl, k_hpl, composition))
            return vec_func(P)

    @eqx.filter_jit
    def _single_P_rate_constant(
        self,
        T: ScalarOrVector,
        P: Scalar,
        lpl: ScalarOrVector,
        hpl: ScalarOrVector,
        composition: Optional[ParamsDict] = None,
    ) -> ScalarOrVector:
        M = calculate_effective_concentration(T, P, composition, self._efficiencies)  # [mol/cm3]
        Pr = (lpl * M) / hpl
        F = compute_broadening_factor(self._falloff_type, T, Pr, self._falloff_parameters)

        return hpl * (Pr / (1 + Pr)) * F

    def __str__(self) -> str:
        """Return string representation in CHEMKIN format."""
        representation = "{}\t\t{:.5e} {:.5f} {:.5e}\n".format(self._name, self._hpl.A, self._hpl.n, self._hpl.Ea)
        representation += " LOW / \t\t{:.5e} {:.5f} {:.5e} /\n".format(self._lpl.A, self._lpl.n, self._lpl.Ea)
        if self._falloff_type == 1 and self._falloff_parameters is not None:
            representation += " TROE / {:.5e} {:.5e} {:.5e} {:.5e} /".format(
                self._falloff_parameters["A"],
                self._falloff_parameters["T3"],
                self._falloff_parameters["T1"],
                self._falloff_parameters["T2"],
            )
        elif self._falloff_type == 2 and self._falloff_parameters is not None:
            representation += " SRI / {:.5e} {:.5e} {:.5e} {:.5e} {:.5e} /".format(
                self._falloff_parameters["a"],
                self._falloff_parameters["b"],
                self._falloff_parameters["c"],
                self._falloff_parameters["d"],
                self._falloff_parameters["e"],
            )
        elif self._falloff_type == 3 and self._falloff_parameters is not None:
            representation += " TSANG / {:.5e} {:.5e} /".format(
                self._falloff_parameters["A"],
                self._falloff_parameters["B"],
            )
        # TODO: add the print of the efficiencies here
        # if self.efficiencies is not None:
        # representation += "\n"
        # for collision_efficiency in self.efficiencies:
        #     representation += " {} / {:.5f} /".format(collision_efficiency.name, collision_efficiency.lnA)
        return representation

    # ==================================================================================
    # Getters
    @property
    def hpl(self) -> Arrhenius:
        return self._hpl

    @property
    def lpl(self) -> Arrhenius:
        return self._lpl

    @property
    def falloff_parameters(self) -> Optional[ParamsDict]:
        if self._falloff_parameters is not None:
            return self._falloff_parameters
        else:
            return None

    @property
    def efficiencies(self) -> Optional[Dict[str, CollisionEfficiency]]:
        if self._efficiencies is not None:
            return self._efficiencies
        else:
            return None

    @property
    def falloff_type(self) -> str:
        return self._falloff_type

    @property
    def name(self) -> str:
        return self._name
