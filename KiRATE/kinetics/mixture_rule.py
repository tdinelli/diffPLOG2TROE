from typing import Dict, List, Optional, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from .cabr import CABR
from .collision_efficiency import CollisionEfficiency
from .falloff import FallOff
from .rate_constant import AnyRate, forward_rate_constant


class MixtureRule(eqx.Module):
    _default_rate_constant: AnyRate
    _explicit_rate_constants: Dict[str, AnyRate]
    _linear: bool
    _reduced_pressure: bool
    _eigenvalue_ratios: Optional[Dict[str, Dict]]
    _name: str

    def __init__(
        self,
        default_rate_constant: AnyRate,
        explicit_rate_constants: Dict[str, AnyRate],
        linear: bool = True,
        reduced_pressure: bool = False,
        eigenvalue_ratios: Optional[List[CollisionEfficiency]] = None,
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

        if linear is False:
            raise ValueError("Non-Linear mixture rules are not implemented yet!")

        self._linear = True if linear is True else False
        self._reduced_pressure = True if reduced_pressure is True else False
        self.name = name

    @eqx.filter_jit
    def rate_constant(self, T, P, composition):
        if self._reduced_pressure and self._linear:
            return self._lmr_r(T, P, composition)
        else:  # self.reduced_pressure False and self.linear True
            return self._lmr_p(T, P, composition)

    def _lmr_p(self, T, P, composition):
        k_default = forward_rate_constant(self._default_rate_constant, T, P, composition)

        if composition is None:
            return k_default

        weighted_sum = jnp.float64(0.0)
        total_explicit_fraction = jnp.float64(0.0)

        # Calculate contributions from explicit species
        for species_name, rate_constant in self._explicit_rate_constants.items():
            if species_name in composition:
                x_i = composition[species_name]
                k_i = forward_rate_constant(rate_constant, T, P, composition)
                weighted_sum += x_i * k_i
                total_explicit_fraction += x_i

        # Remaining mole fraction gets the default rate constant
        x_remaining = jnp.maximum(0.0, 1.0 - total_explicit_fraction)

        return weighted_sum + x_remaining * k_default

    def _lmr_r(self, T, P, composition):
        k_default = forward_rate_constant(self._default_rate_constant, T, P, composition)

        if composition is None:
            return k_default

        # Separate explicit vs default species
        species_list = list(composition.keys())
        explicit_species = set(self._explicit_rate_constants.keys())
        explicit_in_composition = [s for s in species_list if s in explicit_species]
        default_species = [s for s in species_list if s not in explicit_species]

        # Calculate Σⱼ ε₀,ⱼ(T) * Xⱼ
        sum_eigenvalue_ratios_X = self._calculate_sum_eigenvalue_ratios_X(composition, T, explicit_species)

        if sum_eigenvalue_ratios_X <= 0:
            raise ValueError("Sum of eigenvalue ratios must be positive")

        # Process explicit species contributions
        k_explicit = self._process_explicit_species_lmr_r(
            explicit_in_composition, composition, T, P, sum_eigenvalue_ratios_X
        )

        # Process default species contributions
        k_default_contrib = self._process_default_species_lmr_r(
            default_species, composition, T, P, sum_eigenvalue_ratios_X
        )

        return k_explicit + k_default_contrib

    @staticmethod
    def _calculate_eigenvalue_ratio(species, T, eigenvalue_ratios):
        if eigenvalue_ratios is None or species not in eigenvalue_ratios:
            return jnp.float64(1.0)  # Default to reference value

        ratio_data = eigenvalue_ratios[species]
        if ratio_data["n"] is None:  # Constant eigenvalue ratio
            return ratio_data["lnA"]  # For constant, this is the actual value
        else:  # Temperature-dependent eigenvalue ratio: A * T^n * exp(-Ea/RT)
            return jnp.exp(ratio_data["lnA"] + ratio_data["n"] * jnp.log(T) - ratio_data["EaR"] / T)

    def _calculate_sum_eigenvalue_ratios_X(self, composition, T, explicit_species):
        total = jnp.float64(0.0)

        for species, mole_fraction in composition.items():
            if species in explicit_species:
                eigenvalue_ratio = self._calculate_eigenvalue_ratio(species, T, self._eigenvalue_ratios)
            else:
                eigenvalue_ratio = jnp.float64(1.0)  # ε₀,M = 1.0 for default species

            total += eigenvalue_ratio * mole_fraction

        return total

    def _process_explicit_species_lmr_r(
        self,
        explicit_in_composition,
        composition,
        T,
        P,
        sum_eigenvalue_ratios_X,
    ):
        k_explicit = jnp.float64(0.0)

        for species in explicit_in_composition:
            mole_fraction = composition[species]
            rate_const = self._explicit_rate_constants[species]

            # Get eigenvalue ratio ε₀,ᵢ(T) for this species
            eigenvalue_ratio = self._calculate_eigenvalue_ratio(species, T, self._eigenvalue_ratios)

            # Calculate effective pressure: P_eff_i = (Σⱼ ε₀,ⱼ(T) * Xⱼ) / ε₀,ᵢ(T) * P
            P_eff = (sum_eigenvalue_ratios_X / eigenvalue_ratio) * P

            # Evaluate rate constant at effective pressure
            k_i = forward_rate_constant(rate_const, T, P_eff, composition)

            # Calculate fractional contribution: X̃ᵢ = ε₀,ᵢ(T) * Xᵢ / (Σⱼ ε₀,ⱼ(T) * Xⱼ)
            fractional_contribution = (eigenvalue_ratio * mole_fraction) / sum_eigenvalue_ratios_X

            # Add weighted contribution
            k_explicit += k_i * fractional_contribution

        return k_explicit

    def _process_default_species_lmr_r(
        self,
        default_species,
        composition,
        T,
        P,
        sum_eigenvalue_ratios_X,
    ):
        if not default_species:
            return jnp.float64(0.0)

        # Sum mole fractions of all default species
        total_default_fraction = sum(composition[s] for s in default_species)

        # For default species: ε₀,M(T) = 1.0
        eigenvalue_ratio_M = jnp.float64(1.0)

        # Calculate effective pressure for default species
        P_eff = (sum_eigenvalue_ratios_X / eigenvalue_ratio_M) * P

        # Evaluate default rate constant at effective pressure
        k_M = forward_rate_constant(self._default_rate_constant, T, P_eff, composition)

        # Calculate fractional contribution for all default species combined
        fractional_contribution = (eigenvalue_ratio_M * total_default_fraction) / sum_eigenvalue_ratios_X

        return k_M * fractional_contribution

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
