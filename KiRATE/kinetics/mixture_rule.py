from typing import Dict, Optional, TypeAlias, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from KiRATE.kinetics.arrhenius import Arrhenius
from KiRATE.kinetics.cabr import CABR
from KiRATE.kinetics.chebyshev import Chebyshev
from KiRATE.kinetics.falloff import FallOff
from KiRATE.kinetics.plog import Plog


AnyRate: TypeAlias = Union[Arrhenius, Plog, FallOff, CABR, Chebyshev]
PressureDepRate: TypeAlias = Union[Plog, FallOff, CABR, Chebyshev]


class MixtureRule(eqx.Module):
    _default_rate_constant: PressureDepRate
    _explicit_rate_constants: Dict[str, PressureDepRate]
    _eigenvalue_ratios: Optional[Dict[str, Dict]]
    _efficiencies: Optional[Dict[str, Float64[Array, ""]]]

    _linear: bool = eqx.field(static=True, default=False)
    _reduced_pressure: bool = eqx.field(static=True, default=False)
    _name: str = eqx.field(static=True, default="")

    def __init__(
        self,
        default_rate_constant: PressureDepRate,
        explicit_rate_constants: Dict[str, PressureDepRate],
        linear: bool = True,
        reduced_pressure: bool = False,
        efficiencies: Optional[Dict[str, float]] = None,
        name: str = "",
    ) -> None:
        self._default_rate_constant = default_rate_constant

        for rate_constant in explicit_rate_constants.values():
            if isinstance(rate_constant, FallOff) or isinstance(rate_constant, CABR):
                if rate_constant.efficiencies is not None:
                    raise ValueError(
                        "Explicit rate constant in the mixture rules formalism should not have collision efficiencies defined!"
                    )

        self._explicit_rate_constants = explicit_rate_constants
        self._efficiencies = (
            {key: jnp.float64(value) for key, value in efficiencies.items()} if efficiencies is not None else None
        )

        self._linear = True if linear is True else False
        self._reduced_pressure = True if reduced_pressure is True else False
        self._name = name

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
            {key: jnp.float64(value) for key, value in composition.items()} if composition is not None else None
        )

        if not self._reduced_pressure and self._linear:  # This is what is called in the standard literature LMR-P
            return self._lmr_p(T, P, jax_composition)
        elif self._reduced_pressure and self._linear:  # This is what is called in the standard literature LMR-R
            return self._lmr_r(T, P, jax_composition)
        else:  # This is what is called in the standard literature LMR-R
            raise NotImplementedError("LMR-R not implemented yet!")

    def _lmr_p(
        self,
        T: Union[Float64[Array, ""], Float64[Array, "nt"]],
        P: Union[Float64[Array, ""], Float64[Array, "np"]],
        composition: Optional[Dict[str, Float64[Array, ""]]] = None,
    ):
        k_default = self._default_rate_constant.rate_constant(T, P)

        if composition is None:
            return k_default

        weighted_sum = jnp.float64(0.0)
        total_explicit_fraction = jnp.float64(0.0)

        # Calculate contributions from explicit species
        for species_name, rate_expression in self._explicit_rate_constants.items():
            if species_name in composition:
                x_i = composition[species_name]
                k_i = rate_expression.rate_constant(T, P)
                weighted_sum += x_i * k_i
                total_explicit_fraction += x_i

        # Remaining mole fraction gets the default rate constant
        x_remaining = jnp.maximum(0.0, 1.0 - total_explicit_fraction)

        return weighted_sum + x_remaining * k_default

    def _lmr_r(
        self,
        T: Union[Float64[Array, ""], Float64[Array, "nt"]],
        P: Union[Float64[Array, ""], Float64[Array, "np"]],
        composition: Optional[Dict[str, Float64[Array, ""]]] = None,
    ):
        k_default = self._default_rate_constant.rate_constant(T, P)

        if composition is None:
            return k_default

        # Separate explicit vs default species
        species_list = list(composition.keys())
        explicit_species = set(self._explicit_rate_constants.keys())
        explicit_in_composition = [s for s in species_list if s in explicit_species]
        default_species = [s for s in species_list if s not in explicit_species]

        return None

    @staticmethod
    def _calculate_eigenvalue_ratio(species, T, eigenvalue_ratios):
        if eigenvalue_ratios is None or species not in eigenvalue_ratios:
            return jnp.float64(1.0)  # Default to reference value

        ratio_data = eigenvalue_ratios[species]
        if ratio_data["n"] is None:  # Constant eigenvalue ratio
            return ratio_data["lnA"]  # For constant, this is the actual value
        else:  # Temperature-dependent eigenvalue ratio: A * T^n * exp(-Ea/RT)
            return jnp.exp(ratio_data["lnA"] + ratio_data["n"] * jnp.log(T) - ratio_data["EaR"] / T)

    def __str__(self) -> str:
        # lines = [f"MixtureRule: {self.name}"]
        # lines.append(f"Linear: {self.linear}, Reduced Pressure: {self.reduced_pressure}")
        # lines.append("Default rate constant:")
        # lines.append(f"  {self.default_rate_constant}")
        # lines.append("Explicit rate constants:")
        # for species, rate_const in self.explicit_rate_constants.items():
        #     lines.append(f"  {species}: {rate_const}")
        # if self.eigenvalue_ratios is not None:
        #     lines.append("Eigenvalue ratios ε₀,ᵢ(T) = Λ₀,ᵢ(T)/Λ₀,M(T):")
        #     for species, ratio_data in self.eigenvalue_ratios.items():
        #         if ratio_data["n"] is None:
        #             lines.append(f"  {species}: {ratio_data['lnA']:.3f} (constant)")
        #         else:
        #             lines.append(
        #                 f"  {species}: {jnp.exp(ratio_data['lnA']):.3e} * T^{ratio_data['n']:.3f} * exp(-{ratio_data['EaR'] * 1.987:.1f}/RT)"
        #             )
        # return "\n".join(lines)
        return ""
