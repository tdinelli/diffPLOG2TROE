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


class FallOff(eqx.Module):
    hpl: Arrhenius  # High-pressure limit
    lpl: Arrhenius  # Low-pressure limit
    falloff_type: int  # 0: Lindemann, 1: Troe, 2: SRI
    falloff_parameters: Array
    efficiencies: Dict
    explicit_efficiencies: bool
    name: str

    def __init__(
        self,
        hpl_params: Array,
        lpl_params: Array,
        falloff_parameters: Array,
        falloff_type: str,
        efficiencies: Optional[Dict] = None,
        name: str = "",
    ) -> None:
        if efficiencies is None:
            efficiencies = {}

        self.name = name
        self.falloff_type = convert_to_fitting_type(falloff_type)

        if self.falloff_type == 1:  # Troe
            falloff_parameters = validate_troe_parameters(falloff_parameters)
        elif self.falloff_type == 2:  # SRI
            falloff_parameters = validate_sri_parameters(falloff_parameters)
        else:  # Lindemann
            falloff_parameters = jnp.zeros(5, dtype=jnp.float64)

        self.falloff_parameters = falloff_parameters

        self.hpl = Arrhenius(parameters=hpl_params, name=name)
        self.lpl = Arrhenius(parameters=lpl_params, name=name)
        self.efficiencies = efficiencies
        self.explicit_efficiencies = False if self.efficiencies is {} else True

    def _compute_falloff_factor(self, T: Union[Float64, Array], Pr: Union[Float64, Array]) -> Union[Float64, Array]:
        """Compute falloff factor based on falloff type and reduced pressure."""
        operand = (T, Pr, self.falloff_parameters)
        return lax.switch(
            self.falloff_type,
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
        F = self._compute_falloff_factor(T, Pr)
        return (k_hpl * (Pr / (1 + Pr)) * F, M)

    @eqx.filter_jit
    def kinetic_constant(
        self, T: Union[Float64, Array], P: Union[Float64, Array]
    ) -> Tuple[Union[Float64, Array], Union[Float64, Array]]:
        """
        Note for future development in principle we could precompute the vectorized functions in the constructor of the
        class to make things even more fast.
        """
        if (jnp.isscalar(T) or T.ndim == 0) and (jnp.isscalar(P) or P.ndim == 0):  # Both are scalars
            return self._single_P_kinetic_constant(T, P)
        elif not (jnp.isscalar(T) or T.ndim == 0) and (jnp.isscalar(P) or P.ndim == 0):  # T is array, P is scalar
            return vmap(lambda t: self._single_P_kinetic_constant(t, P))(T)
        elif (jnp.isscalar(T) or T.ndim == 0) and not (jnp.isscalar(P) or P.ndim == 0):  # P is array, T is scalar
            return vmap(lambda p: self._single_P_kinetic_constant(T, p))(P)
        else:  # Both are arrays
            return vmap(lambda p: vmap(lambda t: self._single_P_kinetic_constant(t, p))(T))(P)

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
        if self.explicit_efficiencies:
            representation += "\n"
            for key, value in self.efficiencies.items():
                representation += " {} / {:.5f} /".format(key, value)
        return representation
