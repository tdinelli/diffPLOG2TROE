from typing import Optional, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float64

from KiRATE.kinetics.arrhenius import Arrhenius
from KiRATE.kinetics.cabr import CABR
from KiRATE.kinetics.chebyshev import Chebyshev
from KiRATE.kinetics.falloff import FallOff
from KiRATE.kinetics.plog import Plog

AnyRate: TypeAlias = Arrhenius | Plog | FallOff | CABR | Chebyshev
PressureDepRate: TypeAlias = Plog | FallOff | CABR | Chebyshev


class MixtureRule(eqx.Module):
    _default_rate_constant: PressureDepRate
    _explicit_rate_constants: dict[str, PressureDepRate] | None
    _efficiencies: dict[str, Arrhenius]

    _explicit_species: tuple[str, ...] = eqx.field(static=True, default=())
    _linear: bool = eqx.field(static=True, default=False)
    _reduced_pressure: bool = eqx.field(static=True, default=False)
    _name: str = eqx.field(static=True, default="")

    def __init__(
        self,
        default_rate_constant: PressureDepRate,
        explicit_rate_constants: Optional[dict[str, PressureDepRate]] = None,
        efficiencies: Optional[dict[str, Arrhenius]] = None,
        name: str = "",
        linear: bool = True,
        reduced_pressure: bool = False,
    ) -> None:
        self._name = name

        self._linear = True if linear is True else False
        self._reduced_pressure = True if reduced_pressure is True else False
        if self._linear is False and self._reduced_pressure is True:
            raise NotImplementedError("Non-Linear Mixture Rules in the reduced pressure space are not implemented!")

        # rate constant for the default collider
        self._default_rate_constant = default_rate_constant

        # Parsing the colliders informations in principle we have a couple of options here:
        # 1. A collider might have a its own rate constant with an explicit pressure dependency
        #    here we input them in `explicit_rate_constants`
        # 2. A collider might not have an explicit pressure dependency but its pressure dependency
        #    is the same one as the default one and just have correction given a collision
        #    efficiency that could be temperature dependent or not passed here as an `Arrhenius`
        #    expression to ease out the following calculations.

        # Check that both explicit_rate_constants and efficiencies are not None
        if explicit_rate_constants is None and efficiencies is None:
            raise ValueError("Not defyining any rate might be useless since you do not need any mixing rule.")

        # Initialize the efficiencies dictionary - all efficiencies will be stored here
        efficiency_dict: dict[str, Arrhenius] = {}

        # Process explicit rate constants and convert them to efficiency Arrhenius objects
        if explicit_rate_constants is not None:
            # Validate explicit rate constants
            for species_name, rate_constant in explicit_rate_constants.items():
                if not isinstance(rate_constant, (Plog, FallOff, CABR, Chebyshev)):
                    raise ValueError(
                        f"Explicit rate constant for species '{species_name}' must be a pressure-dependent "
                        f"rate constant (Plog, FallOff, CABR, or Chebyshev). Got {type(rate_constant).__name__}"
                    )

                if isinstance(rate_constant, FallOff) or isinstance(rate_constant, CABR):
                    # they dont need any efficiency within their actual definition
                    if rate_constant.efficiencies is not None:
                        raise ValueError(
                            "Explicit rate constant in the mixture rules formalism should "
                            "not have collision efficiencies defined!"
                        )

            # Extract k0 from default rate constant once
            k0_default = self._extract_k0_arrhenius(default_rate_constant, "default")

            # Compute efficiency Arrhenius for each explicit species
            for species_name, rate_constant in explicit_rate_constants.items():
                k0_species = self._extract_k0_arrhenius(rate_constant, species_name)
                efficiency_dict[species_name] = self._compute_efficiency_arrhenius(k0_species, k0_default, species_name)

            # Store the explicit rate constants - needed for LMR-R evaluation
            self._explicit_rate_constants = explicit_rate_constants
        else:
            self._explicit_rate_constants = None

        # Add provided efficiencies (these are already Arrhenius objects)
        if efficiencies is not None:
            for species_name, efficiency_arrhenius in efficiencies.items():
                if species_name in efficiency_dict:
                    raise ValueError(
                        f"Species '{species_name}' has both an explicit rate constant and an efficiency defined. "
                        "This is redundant - use only one."
                    )
                efficiency_dict[species_name] = efficiency_arrhenius

        # Store all efficiencies and explicit species list
        self._efficiencies = efficiency_dict
        self._explicit_species = tuple(sorted(efficiency_dict.keys()))

    # @eqx.filter_jit
    def rate_constant(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "nt"],
        P: float | Float64[Array, ""] | Float64[Array, "np"],
        composition: dict[str, float],
    ) -> Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]:
        """
        Calculate the mixture rate constant at given temperature(s) and pressure(s).

        This method implements mixture rules for handling reactions with multiple colliders
        that can have different pressure dependencies. Two formulations are available:

        - **LMR-P** (Linear Mixture Rule in Pressure space): Direct weighted average
        - **LMR-R** (Linear Mixture Rule in Reduced pressure space): Theoretically rigorous

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "nt"]
            Temperature(s) in Kelvin
        P : float | Float64[Array, ""] | Float64[Array, "np"]
            Pressure(s) in bar or atm (depending on rate constant definition)
        composition : dict[str, float]
            Mole fractions for each species in the mixture

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]
            Mixed rate constant with shape matching input broadcasting

        Raises
        ------
        NotImplementedError
            If non-linear mixture rule is requested (not yet implemented)
        """
        T = jnp.asarray(T, dtype=jnp.float64)
        P = jnp.asarray(P, dtype=jnp.float64)
        jax_composition = {key: jnp.float64(value) for key, value in composition.items()}

        if self._reduced_pressure and self._linear:  # LMR-R
            # Handle broadcasting similar to Plog:
            # - Scalar T, Scalar P -> directly evaluate
            # - Vector T, Scalar P -> directly evaluate (T vectorization handled internally)
            # - Scalar T, Vector P -> vmap over P
            # - Vector T, Vector P -> vmap over P creates grid
            if jnp.isscalar(P) or P.ndim == 0:  # Scalar pressure
                return self._lmr_r_single_P(T, P, jax_composition)
            else:  # Vector pressure - vectorize over pressure dimension
                vec_func = vmap(lambda p: self._lmr_r_single_P(T, p, jax_composition))
                return vec_func(P)
        elif not self._reduced_pressure and self._linear:  # LMR-P
            if jnp.isscalar(P) or P.ndim == 0:  # Scalar pressure
                return self._lmr_p_single_P(T, P, jax_composition)
            else:  # Vector pressure - vectorize over pressure dimension
                vec_func = vmap(lambda p: self._lmr_p_single_P(T, p, jax_composition))
                return vec_func(P)
        else:  # Non-linear mixture rules
            raise NotImplementedError("Non-linear mixture rules are not implemented yet!")

    def _lmr_r_single_P(
        self,
        T: Float64[Array, ""] | Float64[Array, "nt"],
        P: Float64[Array, ""],
        composition: dict[str, Float64[Array, ""]],
    ) -> Float64[Array, ""] | Float64[Array, "nt"]:
        """
        Linear Mixture Rule in Reduced pressure space (LMR-R) for a single pressure.

        This method implements the LMR-R formalism as per the equation:
        k_LMR-R(T,P,x) = sum_i k_i(T, P_i^eff) * X_tilde_i

        where:
        - k_i(T, P_i^eff) is the rate constant for collider i at effective pressure
        - X_tilde_i is the fractional contribution in reduced pressure space
        - P_i^eff is computed such that all colliders are evaluated at the same reduced pressure

        Parameters
        ----------
        T : Float64[Array, ""] | Float64[Array, "nt"]
            Temperature(s) in Kelvin
        P : Float64[Array, ""]
            Pressure (scalar) in bar or atm (depending on rate constant definition)
        composition : dict[str, Float64[Array, ""]]
            Mole fractions for each species

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "nt"]
            Mixed rate constant with shape matching T
        """
        species_list = list(composition.keys())

        # Identify which species in the composition have explicit rate constants
        explicit_rate_in_composition = []
        if self._explicit_rate_constants is not None:
            explicit_rate_in_composition = [s for s in species_list if s in self._explicit_rate_constants]

        # Step 1: Compute relative efficiencies
        # By definition they are epsilon_i = k0_i / k0_default
        # but here they are stored effectively as Arrhenius object have a look at the __init__
        epsilon_dict = self.compute_relative_efficiencies(T, species_list, self._efficiencies)

        # Step 2: Compute fractional contributions X_tilde_i and weighted efficiency sum
        X_tilde_dict, epsilon_weighted_sum = self.compute_fractional_contributions(composition, epsilon_dict)

        # Step 3: Compute effective pressures for each collider
        P_eff_dict = self.compute_effective_pressures(
            P, epsilon_weighted_sum, epsilon_dict, explicit_rate_in_composition
        )

        # Step 4: Evaluate rate constants at effective pressures and sum with fractional contributions
        # Cantera formulation:
        #  k_LMR-R = sum_n k_n(T, P_n^eff) * X_tilde_n + k_M(T, P_M^eff) * [1 - sum_n X_tilde_n]
        # This is equivalent to:
        #  k_LMR-R = sum_i k_i(T, P_i^eff) * X_tilde_i where we use k_M for species without explicit rates

        # Handle vectorization: when T is vector, P_eff values are also vectors
        # We need element-wise evaluation (T[i], P_eff[i]) not grid (T[:], P_eff[:])
        is_T_vector = T.ndim > 0 and T.shape[0] > 1
        k_mixture = jnp.float64(0.0)
        sum_explicit_X_tilde = jnp.float64(0.0)

        # Contribution from species with explicit rate constants
        if self._explicit_rate_constants is not None:
            for species in explicit_rate_in_composition:
                rate_expr = self._explicit_rate_constants[species]

                if is_T_vector:  # Element-wise evaluation using vmap
                    k_i = vmap(lambda t, p: rate_expr.rate_constant(t, p))(T, P_eff_dict[species])
                else:
                    k_i = rate_expr.rate_constant(T, P_eff_dict[species])

                X_tilde_i = X_tilde_dict[species]
                k_mixture += k_i * X_tilde_i
                sum_explicit_X_tilde += X_tilde_i

        # Contribution from remaining species (use default rate at effective pressure)
        # Following Cantera: k_M(T, P_M^eff) * [1 - sum_n X_tilde_n]
        remaining_X_tilde = 1.0 - sum_explicit_X_tilde

        # Always compute the default contribution; if remaining_X_tilde is 0, the contribution is 0
        if is_T_vector:
            k_default = vmap(lambda t, p: self._default_rate_constant.rate_constant(t, p))(T, P_eff_dict["default"])
        else:
            k_default = self._default_rate_constant.rate_constant(T, P_eff_dict["default"])

        k_mixture += k_default * remaining_X_tilde

        return k_mixture

    def _lmr_p_single_P(
        self,
        T: Float64[Array, ""] | Float64[Array, "nt"],
        P: Float64[Array, ""],
        composition: dict[str, Float64[Array, ""]],
    ) -> Float64[Array, ""] | Float64[Array, "nt"]:
        """
        Linear Mixture Rule in Pressure space (LMR-P).

        This method implements the classical linear mixture rule:
        k_LMR-P(T,P,x) = sum_i x_i * k_i(T,P)

        where each species with an explicit rate constant contributes its own k_i,
        and species without explicit rates use the default rate constant.

        Parameters
        ----------
        T : Float64[Array, ""] | Float64[Array, "nt"]
            Temperature(s) in Kelvin
        P : Float64[Array, ""]
            Pressure (scalar) in bar or atm
        composition : dict[str, Float64[Array, ""]]
            Mole fractions for each species

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "nt"]
            Mixed rate constant
        """
        # Handle vectorization similar to LMR-R
        is_T_vector = T.ndim > 0 and T.shape[0] > 1

        k_mixture = jnp.float64(0.0)
        total_explicit_fraction = jnp.float64(0.0)

        # Contribution from species with explicit rate constants
        if self._explicit_rate_constants is not None:
            for species_name, rate_expression in self._explicit_rate_constants.items():
                if species_name in composition:
                    x_i = composition[species_name]

                    if is_T_vector:
                        # Element-wise evaluation for vector T
                        k_i = vmap(lambda t: rate_expression.rate_constant(t, P))(T)
                    else:
                        k_i = rate_expression.rate_constant(T, P)

                    k_mixture += x_i * k_i
                    total_explicit_fraction += x_i

        # Remaining mole fraction uses the default rate constant
        x_remaining = 1.0 - total_explicit_fraction

        # Always compute default contribution; if x_remaining is 0, contribution is 0
        if is_T_vector:
            k_default = vmap(lambda t: self._default_rate_constant.rate_constant(t, P))(T)
        else:
            k_default = self._default_rate_constant.rate_constant(T, P)

        k_mixture += x_remaining * k_default

        return k_mixture

    # ==================================================================================
    # Helper methods for the computations
    @staticmethod
    def compute_relative_efficiencies(
        T: Float64[Array, ""] | Float64[Array, "nt"],
        species_list: list[str],
        efficiencies: dict[str, Arrhenius],
    ) -> dict[str, Float64[Array, ""] | Float64[Array, "nt"]]:
        """
        Compute relative collision efficiencies epsilon_i = k0_i / k0_default.

        All efficiencies are pre-computed as Arrhenius objects during construction,
        so this method simply evaluates them at the given temperature(s) for the
        requested species.

        Parameters
        ----------
        T : Float64[Array, ""] | Float64[Array, "nt"]
            Temperature(s) in Kelvin
        species_list : list[str]
            List of species for which to compute efficiencies
        efficiencies : dict[str, Arrhenius]
            Dictionary of pre-computed efficiency Arrhenius objects

        Returns
        -------
        dict[str, Float64[Array, ""] | Float64[Array, "nt"]]
            Dictionary mapping species names to their dimensionless efficiencies
        """
        epsilon_dict = {}

        for species in species_list:
            if species in efficiencies:
                epsilon_dict[species] = efficiencies[species].rate_constant(T)

        return epsilon_dict

    @staticmethod
    def compute_fractional_contributions(
        composition: dict[str, Float64[Array, ""]],
        epsilon_dict: dict[str, Float64[Array, ""] | Float64[Array, "nt"]],
    ) -> tuple[dict[str, Float64[Array, ""] | Float64[Array, "nt"]], Float64[Array, ""] | Float64[Array, "nt"]]:
        """
        Compute fractional contributions X_tilde_i in reduced pressure space.

        X_tilde_i = (epsilon_i * x_i) / sum_j(epsilon_j * x_j)

        Parameters
        ----------
        composition : dict[str, Float64[Array, ""]]
            Mole fractions for each species
        epsilon_dict : dict[str, Float64[Array, ""] | Float64[Array, "nt"]]
            Collision efficiencies for explicit species

        Returns
        -------
        tuple[dict, Float64[Array, ""]]
            - Dictionary mapping species to their fractional contributions X_tilde_i
            - Weighted sum of efficiencies: sum_j(epsilon_j * x_j)
        """
        # Compute weighted sum of efficiencies: sum_j epsilon_j(T) * x_j
        # For default collider, epsilon = 1.0 by definition
        epsilon_weighted_sum = jnp.float64(0.0)

        for species, x_i in composition.items():
            if species in epsilon_dict:
                epsilon_weighted_sum += epsilon_dict[species] * x_i
            else:
                # Species not in explicit list uses default (epsilon = 1.0)
                epsilon_weighted_sum += 1.0 * x_i

        # Compute fractional contributions X_tilde_i for each collider
        X_tilde_dict = {}
        for species, x_i in composition.items():
            if species in epsilon_dict:
                X_tilde_dict[species] = (epsilon_dict[species] * x_i) / epsilon_weighted_sum
            else:
                X_tilde_dict[species] = (1.0 * x_i) / epsilon_weighted_sum

        return X_tilde_dict, epsilon_weighted_sum

    @staticmethod
    def compute_effective_pressures(
        P: Float64[Array, ""] | Float64[Array, "np"],
        epsilon_weighted_sum: Float64[Array, ""] | Float64[Array, "nt"],
        epsilon_dict: dict[str, Float64[Array, ""] | Float64[Array, "nt"]],
        explicit_in_composition: list[str],
    ) -> dict[str, Float64[Array, ""] | Float64[Array, "np"] | Float64[Array, "nt np"]]:
        """
        Compute effective pressures for each collider in LMR-R formalism.

        P_i^eff = (sum_j epsilon_j * x_j) / epsilon_i * P

        Parameters
        ----------
        P : Float64[Array, ""] | Float64[Array, "np"]
            Pressure(s) in bar or atm
        epsilon_weighted_sum : Float64[Array, ""] | Float64[Array, "nt"]
            Weighted sum of efficiencies: sum_j(epsilon_j * x_j)
        epsilon_dict : dict[str, Float64[Array, ""] | Float64[Array, "nt"]]
            Collision efficiencies for explicit species
        explicit_in_composition : list[str]
            List of species with explicit rate constants present in composition

        Returns
        -------
        dict[str, Float64[Array, ""]]
            Dictionary mapping species (and "default") to their effective pressures
        """
        P_eff_dict = {}

        # Effective pressures for species with explicit rate constants
        for species in explicit_in_composition:
            P_eff_dict[species] = (epsilon_weighted_sum / epsilon_dict[species]) * P

        # Default collider effective pressure (epsilon_default = 1.0)
        P_eff_dict["default"] = epsilon_weighted_sum * P

        return P_eff_dict

    @staticmethod
    def _extract_k0_arrhenius(
        rate_constant: PressureDepRate,
        species_name: str,
    ) -> Arrhenius:
        """
        Extract the k0 Arrhenius object from a pressure-dependent rate constant.

        Parameters
        ----------
        rate_constant : PressureDepRate
            The rate constant object (Plog, FallOff, CABR, or Chebyshev)
        species_name : str
            Name of the species (for error messages)

        Returns
        -------
        Arrhenius
            The low-pressure limit Arrhenius expression k0

        Raises
        ------
        ValueError
            If the rate constant type is not supported or k0 cannot be extracted
        """
        if isinstance(rate_constant, (Plog, Chebyshev)):
            if rate_constant.k0 is not None:
                return rate_constant.k0
            else:
                raise ValueError(
                    f"Rate constant {type(rate_constant).__name__} for species "
                    f"'{species_name}' does not have the low pressure limit defined."
                )
        elif isinstance(rate_constant, (FallOff, CABR)):
            return rate_constant.lpl
        else:
            raise ValueError(
                f"Unsupported rate constant type {type(rate_constant).__name__} for species '{species_name}'"
            )

    @staticmethod
    def _compute_efficiency_arrhenius(
        k0_species: Arrhenius,
        k0_default: Arrhenius,
        species_name: str,
    ) -> Arrhenius:
        """
        Compute the efficiency as an Arrhenius object from two k0 expressions.

        The efficiency epsilon_i(T) = k0_i(T) / k0_default(T) can be expressed as
        a new Arrhenius expression:

        epsilon(T) = (A_i/A_default) * T^(n_i - n_default) * exp(-(Ea_i - Ea_default)/RT)

        Parameters
        ----------
        k0_species : Arrhenius
            The k0 Arrhenius expression for the specific species
        k0_default : Arrhenius
            The k0 Arrhenius expression for the default collider
        species_name : str
            Name of the species (for naming the efficiency)

        Returns
        -------
        Arrhenius
            Arrhenius object representing epsilon_i(T)
        """
        return Arrhenius(
            parameters={
                "A": float(k0_species.A / k0_default.A),
                "n": float(k0_species.n - k0_default.n),
                "Ea": float(k0_species.Ea - k0_default.Ea),
            },
            name=f"efficiency_{species_name}",
        )

    # ==================================================================================
    # Properties for parameters access
    @property
    def name(self) -> str:
        """
        Human-readable reaction name.

        Returns
        -------
        str
            The reaction name string.
        """
        return self._name

    @property
    def linear(self) -> bool:
        """ """
        return self._linear

    @property
    def reduced_pressure(self) -> bool:
        """ """
        return self._reduced_pressure

    @property
    def explicit_species(self) -> tuple[str, ...]:
        """ """
        return self._explicit_species

    # _default_rate_constant: PressureDepRate
    @property
    def efficiencies(self) -> dict[str, Arrhenius]:
        """ """
        return self._efficiencies

    @property
    def explicit_rate_constants(self) -> dict[str, PressureDepRate] | None:
        """ """
        return self._explicit_rate_constants

    @property
    def default_rate_constant(self) -> PressureDepRate:
        """ """
        return self._default_rate_constant
