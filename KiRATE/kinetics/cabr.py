"""
Copyright (c) 2025 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

from typing import Dict, Optional, Union

import equinox as eqx
import jax.numpy as jnp
from jax import lax, vmap
from jaxtyping import Array, Float64

from KiRATE.kinetics.arrhenius import Arrhenius
from KiRATE.kinetics.broadening_functions import compute_broadening_factor
from KiRATE.kinetics.utils import validate_broadening_parameters, validate_efficiencies
from KiRATE.utilities.thermodynamic_utilities import calculate_effective_concentration


class CABR(eqx.Module):
    _hpl: Arrhenius
    _lpl: Arrhenius
    _cabr_type: str = eqx.field(static=True)
    _cabr_parameters: Optional[Dict[str, Float64[Array, ""]]] = None
    _efficiencies: Optional[Dict[str, Float64[Array, ""]]] = None
    _name: str = eqx.field(static=True, default="")

    def __init__(
        self,
        hpl_parameters: Dict[str, float],
        lpl_parameters: Dict[str, float],
        cabr_type: str,
        cabr_parameters: Optional[Dict[str, float]] = None,
        efficiencies: Optional[Dict[str, float]] = None,
        name: str = "",
    ) -> None:
        # Validate parameters at initialization (static validation)
        validate_broadening_parameters(cabr_type, cabr_parameters)
        if efficiencies is not None:
            validate_efficiencies(efficiencies)

        self._hpl = Arrhenius(parameters=hpl_parameters, name=f"{name} (HPL)")
        self._lpl = Arrhenius(parameters=lpl_parameters, name=f"{name} (LPL)")
        self._falloff_type = cabr_type
        self._name = name

        self._falloff_parameters = (
            {key: jnp.array(value, dtype=jnp.float64) for key, value in cabr_parameters.items()}
            if cabr_parameters is not None
            else None
        )
        self._efficiencies = (
            {key: jnp.array(value, dtype=jnp.float64) for key, value in efficiencies.items()}
            if efficiencies is not None
            else None
        )

    @eqx.filter_jit
    def rate_constant(
        self,
        T: Union[float, Float64[Array, ""], Float64[Array, "nt"]],
        P: Union[float, Float64[Array, ""], Float64[Array, "np"]],
        composition: Optional[Dict[str, float]] = None,
    ) -> Union[Float64[Array, ""], Float64[Array, "nt"], Float64[Array, "np"], Float64[Array, "nt np"]]:
        T = jnp.asarray(T, dtype=jnp.float64)
        P = jnp.asarray(P, dtype=jnp.float64)
        jax_composition = (
            {key: jnp.array(value, dtype=jnp.float64) for key, value in composition.items()}
            if composition is not None
            else None
        )

        k_hpl = self._hpl.rate_constant(T)  # [cm3/mol/s]
        k_lpl = self._lpl.rate_constant(T)  # [cm6/mol2/s]

        if jnp.isscalar(P):  # P is scalar
            return self._single_P_rate_constant(
                T,
                P,
                k_lpl,
                k_hpl,
                jax_composition,
            )
        else:  # P is array
            vec_func = vmap(
                lambda p: self._single_P_rate_constant(
                    T,
                    p,
                    k_lpl,
                    k_hpl,
                    jax_composition,
                )
            )
            return vec_func(P)

    @eqx.filter_jit
    def _single_P_rate_constant(
        self,
        T: Union[Float64[Array, ""], Float64[Array, "nt"]],
        P: Float64[Array, ""],
        lpl: Union[Float64[Array, ""], Float64[Array, "nt"]],
        hpl: Union[Float64[Array, ""], Float64[Array, "nt"]],
        composition: Optional[Dict[str, Float64[Array, ""]]] = None,
    ) -> Union[Float64[Array, ""], Float64[Array, "nt"]]:
        M = calculate_effective_concentration(T, P, composition, self._efficiencies)
        Pr = (lpl * M) / hpl
        F = compute_broadening_factor(self._cabr_type, T, Pr, self._cabr_parameters)

        return lpl * (1 / (1 + Pr)) * F

    # ==================================================================================
    # CHEMKIN string parser
    @staticmethod
    def parse_chemkin_entry(input_string: str):
        pass

    # ==================================================================================
    # String Representations and Debugging
    def __str__(self) -> str:
        """Return string representation in CHEMKIN format."""
        representation = "{}\t\t{:.5E} {:.5E} {:.5E}\n".format(self._name, self._lpl.A, self._lpl.n, self._lpl.Ea)
        representation += " HIGH / \t\t{:.5E} {:.5E} {:.5E} /\n".format(self._hpl.A, self._hpl.n, self._hpl.Ea)
        if self._cabr_type == 1 and self._cabr_parameters is not None:
            representation += " TROE / {:.5E} {:.5E} {:.5E} {:.5E} /".format(
                self._cabr_parameters["A"],
                self._cabr_parameters["T3"],
                self._cabr_parameters["T1"],
                self._cabr_parameters["T2"],
            )
        elif self._cabr_type == 2 and self._cabr_parameters is not None:
            representation += " SRI / {:.5E} {:.5E} {:.5E} {:.5E} {:.5E} /".format(
                self._cabr_parameters["a"],
                self._cabr_parameters["b"],
                self._cabr_parameters["c"],
                self._cabr_parameters["d"],
                self._cabr_parameters["e"],
            )
        elif self._cabr_type == 3 and self._cabr_parameters is not None:
            representation += " TSANG / {:.5E} {:.5E} /".format(
                self._cabr_parameters["A"],
                self._cabr_parameters["B"],
            )

        if self._efficiencies is not None:
            representation += "\n"
            for species, efficiency in self._efficiencies.items():
                representation += " {} / {:.5E} /".format(species, efficiency)

        return representation

    @property
    def hpl(self) -> Arrhenius:
        return self._hpl

    @property
    def lpl(self) -> Arrhenius:
        return self._lpl

    @property
    def cabr_parameters(self) -> Optional[Dict[str, Float64[Array, ""]]]:
        return lax.cond(
            self._cabr_parameters is not None,
            lambda _: self._cabr_parameters,
            lambda _: None,
            None,
        )

    @property
    def efficiencies(self) -> Optional[Dict[str, Float64[Array, ""]]]:
        return lax.cond(
            self._efficiencies is not None,
            lambda _: self._efficiencies,
            lambda _: None,
            None,
        )

    @property
    def cabr_type(self) -> str:
        return self._cabr_type

    @property
    def name(self) -> str:
        return self._name
