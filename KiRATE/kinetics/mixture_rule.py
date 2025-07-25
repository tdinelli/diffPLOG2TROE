from typing import Dict, List, Optional, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from .cabr import CABR
from .collision_efficiency import CollisionEfficiency, serialize_collision_efficiencies
from .falloff import FallOff
from .rate_constant import AnyRate, forward_rate_constant


class MixtureRule(eqx.Module):
    """
    Implementation of mixture rules for pressure-dependent reactions.

    This class provides implementations of both Linear Mixture Rule in Pressure (LMR-P)
    and Linear Mixture Rule in Reduced pressure (LMR-R) for evaluating rate constants
    in multi-component gas mixtures.

    The LMR-R approach, based on Lei & Burke's work, offers significant improvements
    over traditional mixture rules by mapping rate constants to a common reduced
    pressure before mixture averaging.

    Parameters
    ----------
    default_rate_constant : AnyRate
        Default rate constant used for species not explicitly specified.
        Acts as the reference "M" collider with ε₀,M(T) = 1.0
    explicit_rate_constants : Dict[str, AnyRate]
        Dictionary mapping species names to their specific rate constants.
        These species will use their own pressure-dependent expressions
    linear : bool, optional
        Whether to use linear mixture rule, by default True.
        Non-linear rules are not yet implemented
    reduced_pressure : bool, optional
        Whether to use reduced pressure formulation (LMR-R) or absolute
        pressure formulation (LMR-P), by default False
    eigenvalue_ratios : Optional[List[CollisionEfficiency]], optional
        List of eigenvalue ratios ε₀,ᵢ(T) = Λ₀,ᵢ(T)/Λ₀,M(T) for LMR-R calculations.
        These represent ratios of low-pressure eigenvalues and are different from
        traditional collision efficiencies, by default None
    name : str, optional
        Optional identifier for the mixture rule, by default ""

    Attributes
    ----------
    default_rate_constant : AnyRate
        Reference rate constant
    explicit_rate_constants : Dict[str, AnyRate]
        Species-specific rate constants
    linear : bool
        Linear mixture rule flag
    reduced_pressure : bool
        Reduced pressure formulation flag
    eigenvalue_ratios : Optional[Dict[str, Dict]]
        Serialized eigenvalue ratio data
    name : str
        Mixture rule identifier

    Raises
    ------
    ValueError
        If explicit rate constants contain collision efficiencies
    ValueError
        If non-linear mixture rules are requested (not implemented)

    Notes
    -----
    The class supports two mixture rule formulations:

    **LMR-P (Classic Linear Mixture Rule in Pressure)**:

    .. math::
        k_{LMR-P}(T, P, X) = \\sum_i k_i(T, P) X_i

    **LMR-R (Linear Mixture Rule in Reduced pressure)**:

    .. math::
        k_{LMR-R}(T, P, X) = \\sum_i k_i(T, P_{eff,i}) \\tilde{X}_i

    where eigenvalue ratios ε₀,ᵢ(T) can be either constant or temperature-dependent
    following Arrhenius form.

    Examples
    --------
    >>> from diffPLOG2TROE.kinetics import Plog, MixtureRule, CollisionEfficiency
    >>>
    >>> # Define rate constants
    >>> ar_rate = Plog(parameters={1.0: {"A": 1e13, "n": 0, "Ea": 0}})
    >>> default_rate = Plog(parameters={1.0: {"A": 1e12, "n": 0, "Ea": 0}})
    >>>
    >>> # Define eigenvalue ratios for LMR-R
    >>> eigenvalue_ratios = [
    ...     CollisionEfficiency(name="Ar", value=0.5),
    ...     CollisionEfficiency(name="H2O", value=12.0)
    ... ]
    >>>
    >>> # Create LMR-R mixture rule
    >>> mixture_rule = MixtureRule(
    ...     default_rate_constant=default_rate,
    ...     explicit_rate_constants={"Ar": ar_rate},
    ...     reduced_pressure=True,
    ...     eigenvalue_ratios=eigenvalue_ratios
    ... )
    >>>
    >>> # Calculate mixture rate constant
    >>> k_mix = mixture_rule.rate_constant(1500.0, 1.0, {"Ar": 0.5, "H2O": 0.5})

    References
    ----------
    .. [1] Lei, L., & Burke, M. P. (2019). Bath gas mixture effects on multichannel
           reactions: Insights and representations for systems beyond single-channel
           reactions. J. Phys. Chem. A, 123(3), 631-649.
    .. [2] Burke, M. P., & Song, R. (2017). Evaluating mixture rules for multi-component
           pressure dependence: H + O₂ (+M) = HO₂ (+M). Proc. Combust. Inst., 36(1), 245-253.
    """

    default_rate_constant: AnyRate
    explicit_rate_constants: Dict[str, AnyRate]
    linear: bool
    reduced_pressure: bool
    eigenvalue_ratios: Optional[Dict[str, Dict]]  # ε₀,ᵢ(T) = Λ₀,ᵢ(T)/Λ₀,M(T)
    name: str

    def __init__(
        self,
        default_rate_constant: AnyRate,
        explicit_rate_constants: Dict[str, AnyRate],
        linear: bool = True,
        reduced_pressure: bool = False,
        eigenvalue_ratios: Optional[List[CollisionEfficiency]] = None,
        name: str = "",
    ) -> None:
        """
        Initialize MixtureRule for pressure-dependent reactions.

        Parameters
        ----------
        default_rate_constant : AnyRate
            Default rate constant used for species not explicitly specified (acts as "M")
        explicit_rate_constants : Dict[str, AnyRate]
            Dictionary mapping species names to their specific rate constants
        linear : bool, optional
            Whether to use linear mixture rule, by default True
        reduced_pressure : bool, optional
            Whether to use reduced pressure (LMR-R) or absolute pressure (LMR-P), by default False
        eigenvalue_ratios : Optional[List[CollisionEfficiency]], optional
            List of eigenvalue ratios ε₀,ᵢ(T) = Λ₀,ᵢ(T)/Λ₀,M(T) for LMR-R calculations.
            These are different from traditional collision efficiencies, by default None
        name : str, optional
            Optional identifier for the mixture rule, by default ""
        """
        self.default_rate_constant = default_rate_constant

        for rate_constant in explicit_rate_constants.values():
            if isinstance(rate_constant, FallOff) or isinstance(rate_constant, CABR):
                if rate_constant.efficiencies is not None:
                    raise ValueError(
                        "Explicit rate constant in the mixture rules formalism should not have collision efficiencies defined!"
                    )

        self.explicit_rate_constants = explicit_rate_constants

        if linear is False:
            raise ValueError("Non-Linear mixture rules are not implemented yet!")

        self.linear = True if linear is True else False
        self.reduced_pressure = True if reduced_pressure is True else False
        self.eigenvalue_ratios = (
            None if eigenvalue_ratios is None else serialize_collision_efficiencies(eigenvalue_ratios)
        )
        self.name = name

    @eqx.filter_jit
    def rate_constant(
        self,
        T: Union[Float64, Float64[Array, "dim"]],
        P: Union[Float64, Float64[Array, "dim"]],
        composition: Optional[Dict[str, Float64]] = None,
    ) -> Union[Float64, Float64[Array, "dim"]]:
        """
        Calculate mixture-averaged rate constant using the specified mixture rule.

        This is the main interface for calculating rate constants in gas mixtures.
        The method automatically selects between LMR-P and LMR-R based on the
        `reduced_pressure` flag set during initialization.

        Parameters
        ----------
        T : Union[Float64, Float64[Array, "dim"]]
            Temperature in Kelvin. Can be scalar or array for vectorized calculations
        P : Union[Float64, Float64[Array, "dim"]]
            Pressure in atm. Can be scalar or array for vectorized calculations
        composition : Optional[Dict[str, Float64]], optional
            Dictionary of species mole fractions {species_name: mole_fraction}.
            If None, returns the default rate constant, by default None

        Returns
        -------
        Union[Float64, Float64[Array, "dim"]]
            Mixture-averaged rate constant with units depending on reaction order:
            - 1st order: [1/s]
            - 2nd order: [cm³/mol/s] or [L/mol/s]
            - 3rd order: [cm⁶/mol²/s] or [L²/mol²/s]

        Notes
        -----
        The method is JIT-compiled for optimal performance and supports both
        scalar and vectorized inputs for temperature and pressure.

        **Composition Handling:**
        - Species present in `explicit_rate_constants` use their specific expressions
        - Remaining species use the `default_rate_constant`
        - Mole fractions should sum to 1.0 for physical consistency

        **Algorithm Selection:**
        - If `reduced_pressure=False`: Uses LMR-P algorithm
        - If `reduced_pressure=True`: Uses LMR-R algorithm

        Examples
        --------
        >>> # Scalar calculation
        >>> k = mixture_rule.rate_constant(1500.0, 1.0, {"Ar": 0.3, "N2": 0.7})
        >>>
        >>> # Vectorized temperature
        >>> T_array = jnp.array([1000.0, 1500.0, 2000.0])
        >>> k_array = mixture_rule.rate_constant(T_array, 1.0, {"Ar": 0.5, "N2": 0.5})
        """
        if self.reduced_pressure and self.linear:
            return self._lmr_r(T, P, composition)
        else:  # self.reduced_pressure False and self.linear True
            return self._lmr_p(T, P, composition)

    def _lmr_p(
        self,
        T: Union[Float64, Float64[Array, "dim"]],
        P: Union[Float64, Float64[Array, "dim"]],
        composition: Optional[Dict[str, Float64]] = None,
    ) -> Union[Float64, Float64[Array, "dim"]]:
        """
        Classic Linear Mixture Rule in Pressure (LMR-P) implementation.

        This method implements the traditional mixture rule where rate constants
        are evaluated at the actual mixture pressure and then linearly averaged
        by mole fraction.

        Parameters
        ----------
        T : Union[Float64, Float64[Array, "dim"]]
            Temperature in Kelvin
        P : Union[Float64, Float64[Array, "dim"]]
            Pressure in atm
        composition : Optional[Dict[str, Float64]], optional
            Dictionary of species mole fractions, by default None

        Returns
        -------
        Union[Float64, Float64[Array, "dim"]]
            Mixture-averaged rate constant using LMR-P

        Notes
        -----
        The LMR-P algorithm follows the simple mole-fraction-weighted average:

        .. math::
            k_{LMR-P}(T, P, X) = \\sum_i k_i(T, P) X_i

        where:
        - :math:`k_i(T, P)` is the rate constant for pure collider i at pressure P
        - :math:`X_i` is the mole fraction of species i

        **Limitations:**
        LMR-P has been shown to produce significant errors (factors of 2-10) for
        many important combustion reactions, particularly in the intermediate
        falloff regime and for multi-well, multi-channel reactions [1]_.

        **Algorithm:**
        1. Calculate rate constants for explicit species at pressure P
        2. Weight by their mole fractions
        3. Add contribution from remaining species using default rate constant

        References
        ----------
        .. [1] Lei, L., & Burke, M. P. (2019). J. Phys. Chem. A, 123(3), 631-649.
        """
        k_default = forward_rate_constant(self.default_rate_constant, T, P, composition)

        if composition is None:
            return k_default

        weighted_sum = jnp.float64(0.0)
        total_explicit_fraction = jnp.float64(0.0)

        # Calculate contributions from explicit species
        for species_name, rate_constant in self.explicit_rate_constants.items():
            if species_name in composition:
                x_i = composition[species_name]
                k_i = forward_rate_constant(rate_constant, T, P, composition)
                weighted_sum += x_i * k_i
                total_explicit_fraction += x_i

        # Remaining mole fraction gets the default rate constant
        x_remaining = jnp.maximum(0.0, 1.0 - total_explicit_fraction)

        return weighted_sum + x_remaining * k_default

    def _lmr_r(
        self,
        T: Union[Float64, Float64[Array, "dim"]],
        P: Union[Float64, Float64[Array, "dim"]],
        composition: Optional[Dict[str, Float64]] = None,
    ) -> Union[Float64, Float64[Array, "dim"]]:
        """
        Linear Mixture Rule in Reduced pressure (LMR-R) implementation.

        This method implements the LMR-R algorithm following Lei & Burke's formulation,
        which maps rate constants to a common reduced pressure before mixture averaging.

        Parameters
        ----------
        T : Union[Float64, Float64[Array, "dim"]]
            Temperature in Kelvin
        P : Union[Float64, Float64[Array, "dim"]]
            Pressure in atm
        composition : Optional[Dict[str, Float64]], optional
            Dictionary of species mole fractions {species_name: mole_fraction},
            by default None

        Returns
        -------
        Union[Float64, Float64[Array, "dim"]]
            Mixture-averaged rate constant using LMR-R algorithm

        Notes
        -----
        The LMR-R algorithm follows Eqs. (5), (6), (7) from Lei & Burke [1]_:

        .. math::
            k_{LMR-R}(T, P, X) = \\sum_i k_i(T, P_{eff,i}) \\cdot \\tilde{X}_i

        Where the effective pressure for each collider is:

        .. math::
            P_{eff,i} = \\frac{\\sum_j \\varepsilon_{0,j}(T) X_j}{\\varepsilon_{0,i}(T)} \\cdot P

        And the fractional contribution of each component is:

        .. math::
            \\tilde{X}_i = \\frac{\\varepsilon_{0,i}(T) X_i}{\\sum_j \\varepsilon_{0,j}(T) X_j}

        The eigenvalue ratios are defined as:

        .. math::
            \\varepsilon_{0,i}(T) = \\frac{\\Lambda_{0,i}(T)}{\\Lambda_{0,M}(T)}

        With the reference collider normalized: :math:`\\varepsilon_{0,M}(T) = 1.0`

        References
        ----------
        .. [1] Lei, L., & Burke, M. P. (2019). Bath gas mixture effects on multichannel
               reactions: Insights and representations for systems beyond single-channel
               reactions. J. Phys. Chem. A, 123(3), 631-649.
        """
        k_default = forward_rate_constant(self.default_rate_constant, T, P, composition)

        if composition is None:
            return k_default

        # Separate explicit vs default species
        species_list = list(composition.keys())
        explicit_species = set(self.explicit_rate_constants.keys())
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
    def _calculate_eigenvalue_ratio(
        species: str, T: Union[Float64, Float64[Array, "dim"]], eigenvalue_ratios: Optional[Dict[str, Dict]]
    ) -> Union[Float64, Float64[Array, "dim"]]:
        """
        Calculate eigenvalue ratio ε₀,ᵢ(T) for a species at temperature T.

        Parameters
        ----------
        species : str
            Name of the species
        T : Union[Float64, Float64[Array, "dim"]]
            Temperature in Kelvin
        eigenvalue_ratios : Optional[Dict[str, Dict]]
            Dictionary containing eigenvalue ratio data for each species

        Returns
        -------
        Union[Float64, Float64[Array, "dim"]]
            Eigenvalue ratio ε₀,ᵢ(T) = Λ₀,ᵢ(T)/Λ₀,M(T)

        Notes
        -----
        The eigenvalue ratio can be either constant or temperature-dependent:

        - **Constant**: :math:`\\varepsilon_{0,i} = \\text{constant}`
        - **Temperature-dependent**: :math:`\\varepsilon_{0,i}(T) = A \\cdot T^n \\cdot \\exp(-E_a/RT)`

        If no specific ratio is provided for a species, it defaults to 1.0 (reference value).
        """
        if eigenvalue_ratios is None or species not in eigenvalue_ratios:
            return jnp.float64(1.0)  # Default to reference value

        ratio_data = eigenvalue_ratios[species]
        if ratio_data["n"] is None:  # Constant eigenvalue ratio
            return ratio_data["lnA"]  # For constant, this is the actual value
        else:  # Temperature-dependent eigenvalue ratio: A * T^n * exp(-Ea/RT)
            return jnp.exp(ratio_data["lnA"] + ratio_data["n"] * jnp.log(T) - ratio_data["EaR"] / T)

    def _calculate_sum_eigenvalue_ratios_X(
        self, composition: Dict[str, Float64], T: Union[Float64, Float64[Array, "dim"]], explicit_species: set
    ) -> Union[Float64, Float64[Array, "dim"]]:
        """
        Calculate the sum Σⱼ ε₀,ⱼ(T) * Xⱼ for all species in the mixture.

        Parameters
        ----------
        composition : Dict[str, Float64]
            Dictionary of species mole fractions {species_name: mole_fraction}
        T : Union[Float64, Float64[Array, "dim"]]
            Temperature in Kelvin
        explicit_species : set
            Set of species names that have explicit rate constants defined

        Returns
        -------
        Union[Float64, Float64[Array, "dim"]]
            Sum of eigenvalue ratio weighted mole fractions

        Notes
        -----
        This calculates the denominator term used in both effective pressure
        and fractional contribution calculations:

        .. math::
            \\sum_j \\varepsilon_{0,j}(T) X_j

        Species not in the explicit list use the reference value ε₀,M = 1.0.
        """
        total = jnp.float64(0.0)

        for species, mole_fraction in composition.items():
            if species in explicit_species:
                eigenvalue_ratio = self._calculate_eigenvalue_ratio(species, T, self.eigenvalue_ratios)
            else:
                eigenvalue_ratio = jnp.float64(1.0)  # ε₀,M = 1.0 for default species

            total += eigenvalue_ratio * mole_fraction

        return total

    def _process_explicit_species_lmr_r(
        self,
        explicit_in_composition: List[str],
        composition: Dict[str, Float64],
        T: Union[Float64, Float64[Array, "dim"]],
        P: Union[Float64, Float64[Array, "dim"]],
        sum_eigenvalue_ratios_X: Union[Float64, Float64[Array, "dim"]],
    ) -> Union[Float64, Float64[Array, "dim"]]:
        """
        Process explicit species contributions for LMR-R calculation.

        Parameters
        ----------
        explicit_in_composition : List[str]
            List of explicit species present in the current composition
        composition : Dict[str, Float64]
            Dictionary of species mole fractions
        T : Union[Float64, Float64[Array, "dim"]]
            Temperature in Kelvin
        P : Union[Float64, Float64[Array, "dim"]]
            Pressure in atm
        sum_eigenvalue_ratios_X : Union[Float64, Float64[Array, "dim"]]
            Pre-calculated sum of eigenvalue ratio weighted mole fractions

        Returns
        -------
        Union[Float64, Float64[Array, "dim"]]
            Summed contributions from all explicit species

        Notes
        -----
        For each explicit species, this method:

        1. Calculates effective pressure: :math:`P_{eff,i} = \\frac{\\sum_j \\varepsilon_{0,j}(T) X_j}{\\varepsilon_{0,i}(T)} \\cdot P`
        2. Evaluates rate constant at effective pressure: :math:`k_i(T, P_{eff,i})`
        3. Calculates fractional contribution: :math:`\\tilde{X}_i = \\frac{\\varepsilon_{0,i}(T) X_i}{\\sum_j \\varepsilon_{0,j}(T) X_j}`
        4. Adds weighted contribution: :math:`k_i(T, P_{eff,i}) \\cdot \\tilde{X}_i`
        """
        k_explicit = jnp.float64(0.0)

        for species in explicit_in_composition:
            mole_fraction = composition[species]
            rate_const = self.explicit_rate_constants[species]

            # Get eigenvalue ratio ε₀,ᵢ(T) for this species
            eigenvalue_ratio = self._calculate_eigenvalue_ratio(species, T, self.eigenvalue_ratios)

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
        default_species: List[str],
        composition: Dict[str, Float64],
        T: Union[Float64, Float64[Array, "dim"]],
        P: Union[Float64, Float64[Array, "dim"]],
        sum_eigenvalue_ratios_X: Union[Float64, Float64[Array, "dim"]],
    ) -> Union[Float64, Float64[Array, "dim"]]:
        """
        Process default species contributions for LMR-R calculation.

        Parameters
        ----------
        default_species : List[str]
            List of species that use the default rate constant
        composition : Dict[str, Float64]
            Dictionary of species mole fractions
        T : Union[Float64, Float64[Array, "dim"]]
            Temperature in Kelvin
        P : Union[Float64, Float64[Array, "dim"]]
            Pressure in atm
        sum_eigenvalue_ratios_X : Union[Float64, Float64[Array, "dim"]]
            Pre-calculated sum of eigenvalue ratio weighted mole fractions

        Returns
        -------
        Union[Float64, Float64[Array, "dim"]]
            Combined contribution from all default species

        Notes
        -----
        Default species are those not explicitly specified in the rate constants dictionary.
        They use the reference eigenvalue ratio ε₀,M = 1.0 and are evaluated using the
        default rate constant at the effective pressure:

        .. math::
            P_{eff,M} = \\sum_j \\varepsilon_{0,j}(T) X_j \\cdot P

        since :math:`\\varepsilon_{0,M} = 1.0`.
        """
        if not default_species:
            return jnp.float64(0.0)

        # Sum mole fractions of all default species
        total_default_fraction = sum(composition[s] for s in default_species)

        # For default species: ε₀,M(T) = 1.0
        eigenvalue_ratio_M = jnp.float64(1.0)

        # Calculate effective pressure for default species
        P_eff = (sum_eigenvalue_ratios_X / eigenvalue_ratio_M) * P

        # Evaluate default rate constant at effective pressure
        k_M = forward_rate_constant(self.default_rate_constant, T, P_eff, composition)

        # Calculate fractional contribution for all default species combined
        fractional_contribution = (eigenvalue_ratio_M * total_default_fraction) / sum_eigenvalue_ratios_X

        return k_M * fractional_contribution

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the MixtureRule.

        Returns
        -------
        str
            Formatted string containing mixture rule configuration and parameters

        Notes
        -----
        The string representation includes:
        - Mixture rule name and type (Linear/Reduced Pressure flags)
        - Default rate constant information
        - List of explicit rate constants by species
        - Eigenvalue ratios with their functional forms (if defined)

        For temperature-dependent eigenvalue ratios, the activation energy is
        converted from K to cal/mol using R = 1.987 cal/mol/K for clarity.
        """
        lines = [f"MixtureRule: {self.name}"]
        lines.append(f"Linear: {self.linear}, Reduced Pressure: {self.reduced_pressure}")
        lines.append("Default rate constant:")
        lines.append(f"  {self.default_rate_constant}")
        lines.append("Explicit rate constants:")
        for species, rate_const in self.explicit_rate_constants.items():
            lines.append(f"  {species}: {rate_const}")
        if self.eigenvalue_ratios is not None:
            lines.append("Eigenvalue ratios ε₀,ᵢ(T) = Λ₀,ᵢ(T)/Λ₀,M(T):")
            for species, ratio_data in self.eigenvalue_ratios.items():
                if ratio_data["n"] is None:
                    lines.append(f"  {species}: {ratio_data['lnA']:.3f} (constant)")
                else:
                    lines.append(
                        f"  {species}: {jnp.exp(ratio_data['lnA']):.3e} * T^{ratio_data['n']:.3f} * exp(-{ratio_data['EaR'] * 1.987:.1f}/RT)"
                    )
        return "\n".join(lines)
