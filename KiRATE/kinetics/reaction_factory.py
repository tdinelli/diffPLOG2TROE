"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from KiRATE.kinetics.arrhenius import Arrhenius
from KiRATE.kinetics.cabr import CABR
from KiRATE.kinetics.chebyshev import Chebyshev
from KiRATE.kinetics.falloff import FallOff
from KiRATE.kinetics.plog import Plog
from KiRATE.kinetics.three_body import Threebody
from KiRATE.species import Species
from KiRATE.utilities import constants, parse_stoichiometry

AnyRate: TypeAlias = Arrhenius | CABR | Chebyshev | FallOff | Plog | Threebody


class Reaction(eqx.Module):
    """
    Reaction with forward/reverse rate constants and equilibrium thermodynamics.

    This class composes a rate constant model (Arrhenius, Plog, FallOff, etc.)
    with _species thermodynamic data to provide a complete reaction representation
    that can compute forward rates, reverse rates, and equilibrium constants.

    The class follows the detailed balance principle:
        k_reverse = k_forward / K_c

    where K_c is computed from _species thermodynamic properties.

    Parameters
    ----------
    _reaction_rate_constant : AnyRate
        Rate constant model (Arrhenius, Plog, FallOff, CABR, or Chebyshev)
    _species : dict[str, Species]
        Species database with thermodynamic data for all _species in the reaction
    reactants : dict[str, float]
        Reactant stoichiometric coefficients (positive values)
        Example: H + O2 <=> O + OH → {"H": 1.0, "O2": 1.0}
    products : dict[str, float]
        Product stoichiometric coefficients (positive values)
        Example: H + O2 <=> O + OH → {"O": 1.0, "OH": 1.0}
    name : str, optional
        Reaction identifier string (default: "")

    Attributes
    ----------
    _reaction_rate_constant : AnyRate
        The underlying rate constant model (differentiable pytree node)
    _species : dict[str, Species]
        Species thermodynamic database (static field)
    reactants : dict[str, float]
        Reactant stoichiometric coefficients (static field)
    products : dict[str, float]
        Product stoichiometric coefficients (static field)
    _delta_nu : float
        Change in number of moles: sum(products) - sum(reactants) (static field)
    _name : str
        Reaction identifier (static field)

    Notes
    -----
    **Unit Conventions (CHEMKIN standard):**
    - Rate constants: cm³, mol, s, cal
    - Concentrations: mol/cm³
    - Pressure: Pa (for input), atm (for standard state)
    - Temperature: K
    - Energy: cal/mol

    **Standard State:**
    - P° = 1 atm
    - R = 82.05736153 cm³·atm/(mol·K)

    **Third-Body Reactions:**
    The collision partner 'M' should NOT be included in stoichiometry,
    as it does not participate in the thermodynamic balance.

    **Reversibility:**
    No validation is performed for reaction reversibility. Users are
    responsible for only calling reverse_rate_constant() on truly
    reversible reactions.
    """

    # Differentiable pytree node (can be optimized)
    _reaction_rate_constant: AnyRate
    _species: dict[str, Species]
    _delta_nu: Float64[Array, ""]

    # Pre-computed arrays for efficient equilibrium constant calculation
    _net_stoich_coeffs: Float64[Array, "n_species"]  # net stoichiometry (products - reactants)
    _species_for_equilibrium: tuple[Species, ...]  # ordered species list (pytree node)

    # Static fields (fixed during optimization)
    _reactants: dict[str, float] = eqx.field(static=True)
    _products: dict[str, float] = eqx.field(static=True)
    _is_reversible: bool = eqx.field(static=True, default=True)
    _name: str = eqx.field(static=True, default="")

    def __init__(
        self,
        reaction_rate_constant: AnyRate,
        species: dict[str, Species],
        reactants: dict[str, float] | None = None,
        products: dict[str, float] | None = None,
        is_reversible: bool | None = None,
        name: str = "",
    ) -> None:
        """
        Initialize a Reaction.

        Parameters
        ----------
        reaction_rate_constant : AnyRate
            Rate constant model for the forward reaction
        species : dict[str, Species]
            Species database with thermodynamic data
        reactants : dict[str, float]
            Reactant stoichiometric coefficients (positive values)
        products : dict[str, float]
            Product stoichiometric coefficients (positive values)
        name : str, optional
            Reaction identifier (default: "")

        Raises
        ------
        ValueError
            If any species in reactants or products is not found in the species database

        Notes
        -----
        The change in moles (delta_nu) is pre-computed and stored as a static field
        for efficiency in equilibrium constant calculations.

        All species referenced in reactants and products must exist in the
        species database. This is validated during initialization.
        """
        if reactants is None and products is None:
            # Parse stoichiometry from reaction name string
            if not name:
                raise ValueError("Either provide reactants/products or a valid reaction name string")
            parsing_result = parse_stoichiometry(name)
            reactants = parsing_result["reactants"]
            products = parsing_result["products"]
            is_reversible = parsing_result["reversible"]
        elif reactants is None or products is None:
            raise ValueError("Both reactants and products must be provided together, or neither (to parse from name)")
        else:
            # Both reactants and products provided - use is_reversible parameter or default to True
            if is_reversible is None:
                is_reversible = True

        # Validate that all reactant species exist in database
        for species_name in reactants:
            if species_name not in species:
                raise ValueError(
                    f"Reactant species '{species_name}' not found in _species database. "
                    f"Available species: {list(species.keys())}"
                )

        # Validate that all product species exist in database
        for species_name in products:
            if species_name not in species:
                raise ValueError(
                    f"Product species '{species_name}' not found in _species database. "
                    f"Available species: {list(species.keys())}"
                )

        # Validate elemental balance
        self._check_elemental_balance(species, reactants, products, name)

        self._reaction_rate_constant = reaction_rate_constant
        self._species = species
        self._reactants = reactants
        self._products = products
        self._is_reversible = is_reversible
        self._name = name

        # Pre-compute change in moles: delta_nu = sum(nu_products) - sum(nu_reactants)
        self._delta_nu = jnp.float64(sum(products.values()) - sum(reactants.values()))

        # Pre-compute net stoichiometric coefficients and species list for efficient
        # equilibrium calculation
        # Collect all unique species involved in the reaction
        all_species_names = set(reactants.keys()) | set(products.keys())

        # Create ordered list of species and their net stoichiometric coefficients
        species_list = []
        net_coeffs = []
        for name in sorted(all_species_names):  # Sort for deterministic ordering
            species_list.append(species[name])
            # Net coefficient = (product coeff) - (reactant coeff)
            net_coeff = products.get(name, 0.0) - reactants.get(name, 0.0)
            net_coeffs.append(net_coeff)

        self._species_for_equilibrium = tuple(species_list)
        self._net_stoich_coeffs = jnp.array(net_coeffs, dtype=jnp.float64)

    # =======================================================================
    # Rate constant methods
    @eqx.filter_jit
    def forward_rate_constant(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
        P: float | Float64[Array, ""] | Float64[Array, "n"] | None = None,
        composition: dict[str, float] | None = None,
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Compute forward rate constant k_f(T, P).

        This method delegates to the underlying rate constant model's
        rate_constant() method, handling the appropriate signature based
        on the rate constant type.

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature [K]. Can be scalar or array.
        P : float | Float64[Array, ""] | Float64[Array, "n"] | None, optional
            Pressure [Pa]. Required for pressure-dependent rate constants
            (Plog, FallOff, CABR, Chebyshev). Optional for Arrhenius.
        composition : dict[str, float] | None, optional
            Mole fractions for mixture-dependent rates. Required for FallOff
            and CABR if third-body efficiencies are defined.

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Forward rate constant in CHEMKIN units. Units depend on reaction order:
            - Unimolecular: s⁻¹
            - Bimolecular: cm³/(mol·s)
            - Termolecular: cm⁶/(mol²·s)

        Notes
        -----
        The method automatically handles different rate constant types:
        - **Arrhenius**: Only requires T (P and composition are ignored)
        - **Plog**: Requires T and P (composition is ignored)
        - **FallOff**: Requires T and P, optionally composition for third-body effects
        - **CABR**: Requires T and P, optionally composition for third-body effects
        - **Chebyshev**: Requires T and P (composition is ignored)
        """
        # Handle different rate constant types with appropriate signatures
        if isinstance(self._reaction_rate_constant, Arrhenius):
            # Arrhenius only needs temperature
            return self._reaction_rate_constant.rate_constant(T)
        elif isinstance(self._reaction_rate_constant, Threebody):
            # This is explained in the Threebody class
            return self._reaction_rate_constant.k0.rate_constant(T)
        elif isinstance(self._reaction_rate_constant, (FallOff, CABR, Plog, Chebyshev)) and P is not None:
            # FallOff and CABR require T, P, and optionally composition
            # but for the moment we are not passing that we have that
            # implemented in the case when we are going to use the TMD
            # eq for Mixture Rule like thing
            return self._reaction_rate_constant.rate_constant(T, P)
        else:
            # Fallback for any other type (shouldn't happen with current AnyRate)
            return self._reaction_rate_constant.rate_constant(T, P, composition)

    @eqx.filter_jit
    def equilibrium_constant(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Compute concentration-based equilibrium constant K_c(T).

        The equilibrium constant is computed from _species thermodynamic
        properties using:

        1. DG/(RT) = sum(nu_i * g_RT_i) - dimensionless Gibbs change
        2. K_p = exp(-DG/(RT)) - pressure-based equilibrium constant
        3. K_c = K_p * (P_std/RT)^delta_nu - concentration-based equilibrium constant

        where:
        - nu_i are stoichiometric coefficients
        - g_RT_i = G_i/(RT) from Species.g_RT(T)
        - delta_nu = sum(nu_i) (change in moles)
        - P_std = 1 atm (standard pressure)
        - R = 82.05736153 cm³·atm/(mol·K) (CHEMKIN convention)

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature [K]. Can be scalar or array.

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Concentration-based equilibrium constant K_c in units of (mol/cm³)^delta_nu

        Notes
        -----
        The standard state convention follows CHEMKIN:
        - Standard pressure: P_std = 1 atm
        - Gas constant: R = 82.05736153 cm³·atm/(mol·K)
        - Concentration units: mol/cm³

        For reactions with delta_nu = 0 (equal moles on both sides), K_c = K_p
        since (P_std/RT)^0 = 1.

        The method is fully differentiable with respect to temperature and
        compatible with JAX transformations (vmap, grad, jit).
        """
        T_array = jnp.asarray(T, dtype=jnp.float64)

        # Compute dimensionless Gibbs free energy change: DG/(RT)
        # Using pre-computed net stoichiometry: DG/(RT) = sum(net_nu_i * g_RT_i)
        # where net_nu_i = (product coeff) - (reactant coeff)
        # This is more efficient as we:
        # 1. Compute each species' g_RT only once (even if in both reactants and products)
        # 2. Use vectorized operations for the sum
        g_RT_values = jnp.array([species.g_RT(T_array) for species in self._species_for_equilibrium])
        DG_RT = jnp.dot(self._net_stoich_coeffs, g_RT_values)

        # Pressure-based equilibrium constant: K_p = exp(-DG/(RT))
        K_p = jnp.exp(-DG_RT)

        # Convert to concentration-based equilibrium constant: K_c = K_p * (P_std/RT)^delta_nu
        # CHEMKIN standard state: P_std = 1 atm, R = 82.05736153 cm³·atm/(mol·K)
        # For delta_nu = 0, this automatically equals 1.0 (no conditional needed)
        concentration_factor = jnp.pow(1.0 / (constants.R_cm3_atm_mol_K * T_array), self._delta_nu)

        return K_p * concentration_factor

    @eqx.filter_jit
    def reverse_rate_constant(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
        P: float | Float64[Array, ""] | Float64[Array, "n"] | None = None,
        composition: dict[str, float] | None = None,
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Compute reverse rate constant k_r(T, P).

        The reverse rate constant is computed using the principle of
        detailed balance:

            k_r = k_f / K_c

        where k_f is the forward rate constant and K_c is the concentration-based
        equilibrium constant.

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature [K]. Can be scalar or array.
        P : float | Float64[Array, ""] | Float64[Array, "n"] | None, optional
            Pressure [Pa]. Required for pressure-dependent rate constants
            (Plog, FallOff, CABR, Chebyshev). Optional for Arrhenius.
        composition : dict[str, float] | None, optional
            Mole fractions for mixture-dependent rates. Required for FallOff
            and CABR if third-body efficiencies are defined.

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Reverse rate constant in CHEMKIN units. Units depend on reaction order
            (same as forward rate constant divided by K_c units).

        Notes
        -----
        **No Validation for Reversibility:**
        This method does not check whether the reaction is thermodynamically
        reversible. Users are responsible for only calling this method on
        truly reversible reactions.

        **Pressure Dependence:**
        - For pressure-independent reactions (Arrhenius): k_r depends on T only
          through k_f(T) and K_c(T)
        - For pressure-dependent reactions (Plog, FallOff, etc.): k_r depends
          on both T and P through k_f(T,P), but K_c(T) depends only on T

        **Detailed Balance:**
        At equilibrium, the forward and reverse rates are equal:
            k_f * [reactants] = k_r * [products]

        This ensures thermodynamic consistency.

        **Automatic Type Handling:**
        The method automatically handles different rate constant types by
        delegating to forward_rate_constant(), which uses isinstance checks
        to call the appropriate rate_constant() signature:
        - **Arrhenius**: Only uses T
        - **Plog**: Uses T and P
        - **FallOff/CABR**: Uses T, P, and composition
        - **Chebyshev**: Uses T and P
        """
        k_f = self.forward_rate_constant(T, P, composition)
        K_c = self.equilibrium_constant(T)
        return k_f / K_c

    @staticmethod
    def _check_elemental_balance(
        species: dict[str, Species],
        reactants: dict[str, float],
        products: dict[str, float],
        name: str = "",
    ) -> None:
        """
        Check if a reaction satisfies elemental balance.

        For a valid chemical reaction, the number of atoms of each element
        must be conserved (equal on both sides of the reaction).

        Parameters
        ----------
        species : dict[str, Species]
            Species database with elemental composition data
        reactants : dict[str, float]
            Reactant stoichiometric coefficients (positive values)
        products : dict[str, float]
            Product stoichiometric coefficients (positive values)
        name : str, optional
            Reaction name for error messages (default: "")

        Raises
        ------
        ValueError
            If the reaction does not satisfy elemental balance

        Notes
        -----
        The function computes the net change in atoms for each element:
            net_atoms[element] = sum(products) - sum(reactants)

        A reaction is balanced when all net_atoms are zero (within numerical tolerance).
        """
        element_balance: dict[str, float] = {}

        # Count atoms in reactants (subtract from balance)
        for species_name, coeff in reactants.items():
            for element, count in species[species_name].elemental_composition.items():
                if element not in element_balance:
                    element_balance[element] = 0.0
                element_balance[element] -= coeff * count

        # Count atoms in products (add to balance)
        for species_name, coeff in products.items():
            for element, count in species[species_name].elemental_composition.items():
                if element not in element_balance:
                    element_balance[element] = 0.0
                element_balance[element] += coeff * count

        # Check if all elements are balanced (should be zero within tolerance)
        unbalanced_elements = {elem: balance for elem, balance in element_balance.items() if abs(balance) > 1e-10}
        if unbalanced_elements:
            reaction_str = name if name else "Reaction"
            raise ValueError(
                f"{reaction_str} does not satisfy elemental balance. Unbalanced elements: {unbalanced_elements}"
            )

    # =======================================================================
    # Properties for accessing reaction information
    @property
    def name(self) -> str:
        """
        Reaction identifier string.

        Returns
        -------
        str
            The reaction name or identifier (e.g., "H+O2<=>O+OH")
        """
        return self._name

    @property
    def delta_nu(self) -> Float64[Array, ""]:
        """
        Change in number of moles (delta_nu).

        Computed as delta_nu = sum(nu_products) - sum(nu_reactants).

        Returns
        -------
        float
            Change in moles. Positive if products have more moles,
            negative if reactants have more moles, zero if equal.
        """
        return self._delta_nu

    @property
    def reaction_rate_constant(self) -> AnyRate:
        """
        The underlying rate constant model.

        Returns
        -------
        AnyRate
            Rate constant object (Arrhenius, Plog, FallOff, CABR, or Chebyshev)

        Notes
        -----
        This property provides access to the composed rate constant object,
        allowing inspection of its parameters or direct method calls.
        """
        return self._reaction_rate_constant

    # =======================================================================
    # String representations for display and debugging
    def __str__(self) -> str:
        """
        Return concise string representation.

        Returns
        -------
        str
            Reaction equation string in human-readable format

        Raises
        ------
        NotImplementedError
            This method is not yet implemented
        """
        raise NotImplementedError("__str__ method not yet implemented")

    def __repr__(self) -> str:
        """
        Return detailed string representation for debugging.

        Returns
        -------
        str
            Multi-line formatted string with all reaction information

        Raises
        ------
        NotImplementedError
            This method is not yet implemented
        """
        raise NotImplementedError("__repr__ method not yet implemented")
