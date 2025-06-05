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


class CABR(eqx.Module):
    hpl: Arrhenius  # High-pressure limit
    lpl: Arrhenius  # Low-pressure limit
    cabr_type: int  # 0: Lindemann, 1: Troe, 2: SRI, 3: Tsang
    cabr_params: Array64f_5
    efficiencies: Dict[str, Float64]
    explicit_efficiencies: bool
    name: str

    def __init__(
        self,
        hpl_parameters: Array64f_3,
        lpl_parameters: Array64f_3,
        cabr_type: str,
        cabr_parameters: Optional[Array64f] = None,
        efficiencies: Optional[Dict[str, Float64]] = None,
        name: str = "",
    ) -> None:
        if efficiencies is None:
            self.efficiencies = {}
            self.explicit_efficiencies = False
        else:
            validate_efficiencies(efficiencies)
            self.efficiencies = efficiencies
            self.explicit_efficiencies = True

        self.name = name
        self.cabr_type = convert_to_fitting_type(cabr_type)

        if self.cabr_type == 0:  # Lindemann
            cabr_parameters = jnp.empty(5, dtype=jnp.float64)
        elif self.cabr_type == 1 and cabr_parameters is not None:  # Troe
            cabr_parameters = validate_troe_parameters(cabr_parameters)
        elif self.cabr_type == 2 and cabr_parameters is not None:  # SRI
            cabr_parameters = validate_sri_parameters(cabr_parameters)
        elif self.cabr_type == 3 and cabr_parameters is not None:  # Tsang
            cabr_parameters = validate_tsang_parameters(cabr_parameters)
        else:
            raise ValueError(f"Unknown CABR type {cabr_type} or incorrect CABR parameters.")
        self.cabr_parameters = cabr_parameters

        self.hpl = Arrhenius(parameters=hpl_parameters, name=name)
        self.lpl = Arrhenius(parameters=lpl_parameters, name=name)

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
        F = self._compute_blending_function(T, Pr)

        return lpl * (1 / (1 + Pr)) * F

    def _compute_blending_function(self, T: ScalarOrVector, Pr: ScalarOrVector) -> ScalarOrVector:
        """Compute blending factor based on fitting type and reduced pressure."""
        operand = (T, Pr, self.cabr_params)
        return lax.switch(
            self.cabr_type,
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
        elif self.cabr_type == 3:
            representation += " TSANG / {:.5e} {:.5e} /".format(
                self.cabr_params[0],
                self.cabr_params[1],
            )
        if self.explicit_efficiencies:
            representation += "\n"
            for key, value in self.efficiencies.items():
                representation += " {} / {:.5f} /".format(key, value)
        return representation
