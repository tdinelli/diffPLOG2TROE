from typing import Dict, Optional, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from jax import lax, vmap
from jaxtyping import Array, Float64

from ..utilities.physical_constants import constants
from ..utilities.thermodynamic_utilities import calculate_concentration
from .arrhenius import Arrhenius
from .falloff_functions import (
    convert_to_fitting_type,
    lindemann,
    sri,
    troe,
    validate_sri_parameters,
    validate_troe_parameters,
)


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
        hpl_params: Array,
        lpl_params: Array,
        cabr_params: Array,
        cabr_type: str,
        efficiencies: Optional[Dict] = None,
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

        if self.cabr_type == 1:  # Troe
            cabr_params = validate_troe_parameters(cabr_params)
        elif self.cabr_type == 2:  # SRI
            cabr_params = validate_sri_parameters(cabr_params)
        else:  # Lindemann
            cabr_params = jnp.zeros(5, dtype=jnp.float64)
        self.cabr_params = cabr_params

        self.hpl = Arrhenius(parameters=hpl_params, name=name)
        self.lpl = Arrhenius(parameters=lpl_params, name=name)

    def _compute_blending_function(self, T: Union[Float64, Array], Pr: Union[Float64, Array]) -> Union[Float64, Array]:
        """Compute blending factor based on fitting type and reduced pressure."""
        operand = (T, Pr, self.cabr_params)
        return lax.switch(
            self.cabr_type,
            [
                lambda x: lindemann(*x),
                lambda x: troe(*x),
                lambda x: sri(*x),
            ],
            operand,
        )

    @eqx.filter_jit
    def _single_P_kinetic_constant(
        self, T: Union[Float64, Array], P: Float64
    ) -> Tuple[Union[Float64, Array], Union[Float64, Array]]:
        k_hpl = self.hpl.kinetic_constant(T)
        k_lpl = self.lpl.kinetic_constant(T)
        M = calculate_concentration(T, P)
        Pr = (k_lpl * M) / k_hpl
        F = self._compute_blending_function(T, Pr)
        return (k_lpl * (1 / (1 + Pr)) * F, M)

    @eqx.filter_jit
    def kinetic_constant(
        self,
        T: Union[Float64, Array],
        P: Union[Float64, Array],
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
