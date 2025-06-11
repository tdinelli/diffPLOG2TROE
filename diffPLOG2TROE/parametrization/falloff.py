from typing import Dict, Optional

import equinox as eqx
import jax.numpy as jnp
from jax import lax, vmap
from jaxtyping import Float64

from ..utilities.custom_types import Array64f, Array64f_3, Array64f_5, ScalarOrVector
from ..utilities.physical_constants import constants
from ..utilities.thermodynamic_utilities import calculate_effective_concentration
from .arrhenius import Arrhenius
from .broadening_functions import lindemann, sri, troe, tsang
from .parametrization_utils import (
    convert_to_fitting_type,
    validate_efficiencies,
    validate_sri_parameters,
    validate_troe_parameters,
    validate_tsang_parameters,
)


class FallOff(eqx.Module):
    hpl: Arrhenius     # High-pressure limit
    lpl: Arrhenius     # Low-pressure limit
    falloff_type: int  # 0: Lindemann, 1: Troe, 2: SRI, 3: Tsang
    falloff_parameters: Array64f_5
    efficiencies: Dict[str, Float64]
    explicit_efficiencies: bool
    name: str

    def __init__(
        self,
        hpl_parameters: Array64f_3,
        lpl_parameters: Array64f_3,
        falloff_type: str,
        falloff_parameters: Optional[Array64f] = None,
        efficiencies: Optional[Dict[str, Float64]] = None,
        name: str = "",
    ) -> None:
        self.hpl = Arrhenius(parameters=hpl_parameters, name=name)
        self.lpl = Arrhenius(parameters=lpl_parameters, name=name)

        if efficiencies is None:
            self.efficiencies = {}
            self.explicit_efficiencies = False
        else:
            validate_efficiencies(efficiencies)
            self.efficiencies = efficiencies
            self.explicit_efficiencies = True

        self.name = name
        self.falloff_type = convert_to_fitting_type(falloff_type)

        if self.falloff_type == 0:  # Lindemann
            falloff_parameters = jnp.empty(5, dtype=jnp.float64)
        elif self.falloff_type == 1 and falloff_parameters is not None:  # Troe
            falloff_parameters = validate_troe_parameters(falloff_parameters)
        elif self.falloff_type == 2 and falloff_parameters is not None:  # SRI
            falloff_parameters = validate_sri_parameters(falloff_parameters)
        elif self.falloff_type == 3 and falloff_parameters is not None:  # Tsang
            falloff_parameters = validate_tsang_parameters(falloff_parameters)
        else:
            raise ValueError(f"Unknown falloff type {falloff_type} or incorrect falloff parameters.")
        self.falloff_parameters = falloff_parameters

    @eqx.filter_jit
    def kinetic_constant(
        self,
        T: ScalarOrVector,
        P: ScalarOrVector,
        composition: Optional[Dict[str, Float64]] = None,
    ) -> ScalarOrVector:
        k_hpl = self.hpl.kinetic_constant(T)  # [cm3/mol/s]
        k_lpl = self.lpl.kinetic_constant(T)  # [cm3/mol/s]

        if jnp.isscalar(P) or P.ndim == 0:  # P is scalar
            return self._single_P_kinetic_constant(T, P, k_lpl, k_hpl, composition)
        else:  # P is array
            vec_func = vmap(lambda p: self._single_P_kinetic_constant(T, p, k_lpl, k_hpl, composition))
            return vec_func(P)

    @eqx.filter_jit
    def _single_P_kinetic_constant(
        self,
        T: ScalarOrVector,
        P: Float64,
        lpl: ScalarOrVector,
        hpl: ScalarOrVector,
        composition: Optional[Dict[str, Float64]] = None,
    ) -> ScalarOrVector:
        M = calculate_effective_concentration(T, P, composition, self.efficiencies)  # [mol/cm3]
        Pr = (lpl * M) / hpl
        F = self._compute_falloff_factor(T, Pr)

        return hpl * (Pr / (1 + Pr)) * F

    def _compute_falloff_factor(self, T: ScalarOrVector, Pr: ScalarOrVector) -> ScalarOrVector:
        """Compute falloff factor (F) based on falloff type and reduced pressure."""
        operand = (T, Pr, self.falloff_parameters)
        return lax.switch(
            self.falloff_type,
            [
                lambda x: lindemann(*x),
                lambda x: troe(*x),
                lambda x: sri(*x),
                lambda x: tsang(*x),
            ],
            operand,
        )

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
                self.falloff_parameters[0],
                self.falloff_parameters[1],
                self.falloff_parameters[2],
                self.falloff_parameters[3],
            )
        elif self.falloff_type == 2:
            representation += " SRI / {:.5e} {:.5e} {:.5e} {:.5e} {:.5e} /".format(
                self.falloff_parameters[0],
                self.falloff_parameters[1],
                self.falloff_parameters[2],
                self.falloff_parameters[3],
                self.falloff_parameters[4],
            )
        elif self.falloff_type == 3:
            representation += " TSANG / {:.5e} {:.5e} /".format(
                self.falloff_parameters[0],
                self.falloff_parameters[1],
            )
        if self.explicit_efficiencies:
            representation += "\n"
            for key, value in self.efficiencies.items():
                representation += " {} / {:.5f} /".format(key, value)
        return representation
