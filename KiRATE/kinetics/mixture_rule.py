"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

from typing import TypeAlias

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
    """
    Mixture rule calculator for gas-phase reactions with multiple colliders.

    This class implements linear mixture rules for computing rate constants in gas mixtures
    where different collision partners (third bodies) have distinct collision efficiencies
    and potentially different pressure dependencies. Two formulations are supported:

    - **LMR-P** (Linear Mixture Rule in Pressure space): Direct weighted average
    - **LMR-R** (Linear Mixture Rule in Reduced pressure space): Theoretically rigorous

    **LMR-P Formulation:**

    .. math::
        k_{\\text{LMR-P}}(T, P, \\mathbf{x}) = \\sum_i k_i(T, P) \\cdot x_i

    **LMR-R Formulation:**

    .. math::
        k_{\\text{LMR-R}}(T, P, \\mathbf{x}) = \\sum_i k_i(T, P_i^{\\text{eff}}) \\cdot \\tilde{X}_i

    where:
        - :math:`k_i(T, P)` is the rate constant for collider i
        - :math:`x_i` is the mole fraction of species i
        - :math:`P_i^{\\text{eff}} = P \\cdot \\varepsilon_{\\text{mix}} / \\varepsilon_i` is the effective pressure
        - :math:`\\varepsilon_i = k_{0,i}(T) / k_{0,\\text{default}}(T)` is the relative efficiency
        - :math:`\\tilde{X}_i = (\\varepsilon_i \\cdot x_i) / \\sum_j (\\varepsilon_j \\cdot x_j)` is the fractional contribution

    **Collider Types:**

    1. **Explicit colliders**: Species with their own pressure-dependent rate expressions
       (Plog, FallOff, CABR, or Chebyshev)
    2. **Efficiency-only colliders**: Species that use the default rate with a collision
       efficiency correction

    Key Features:
        - Supports mixtures of PLOG, FallOff, CABR, and Chebyshev rate expressions
        - Temperature-dependent collision efficiencies via Arrhenius expressions
        - Automatic efficiency computation from low-pressure limits (k0)
        - Optional explicit efficiency override for explicit colliders
        - Fully differentiable for gradient-based optimization
        - Vectorized evaluation over temperature and pressure arrays

    Parameters
    ----------
    default_rate_constant : Plog | FallOff | CABR | Chebyshev
        Rate constant for the default collider (M). This is used for species
        without explicit rate expressions.
    explicit_rate_constants : dict[str, PressureDepRate], optional
        Dictionary mapping species names to their pressure-dependent rate constants.
        These colliders have distinct pressure dependencies from the default.
    efficiencies : dict[str, Arrhenius], optional
        Dictionary mapping species names to their collision efficiency Arrhenius expressions.

        - For efficiency-only colliders: Required, defines epsilon_i(T)
        - For explicit colliders: Optional override of auto-computed epsilon from k0

        Efficiency Arrhenius parameters represent epsilon_i(T) directly (not k0_i).
    name : str, optional
        Human-readable name for the reaction, by default ""
    linear : bool, optional
        If True, use linear mixture rules (LMR). If False, non-linear rules (not implemented),
        by default True
    reduced_pressure : bool, optional
        If True, use LMR-R (reduced pressure space). If False, use LMR-P (pressure space),
        by default False

    Attributes
    ----------
    _default_rate_constant : PressureDepRate
        Default collider rate constant
    _explicit_rate_constants : dict[str, PressureDepRate] | None
        Explicit collider rate constants
    _efficiencies : dict[str, Arrhenius]
        All collision efficiencies (auto-computed and/or explicit)
    _explicit_species : tuple[str, ...]
        Sorted tuple of all species with efficiencies (static field)
    _linear : bool
        Linear vs non-linear mixture rule flag (static field)
    _reduced_pressure : bool
        LMR-P vs LMR-R formulation flag (static field)
    _name : str
        Reaction name (static field)

    Raises
    ------
    ValueError
        - If both explicit_rate_constants and efficiencies are None
        - If explicit rate constant is not a pressure-dependent type
        - If FallOff/CABR explicit colliders have internal efficiencies defined
    NotImplementedError
        If non-linear mixture rules in reduced pressure space are requested

    Notes
    -----
    **Efficiency Handling:**

    For LMR-R with explicit colliders, efficiencies can be:

    1. **Auto-computed from k0** (default): :math:`\\varepsilon_i = k_{0,i} / k_{0,\\text{default}}`
    2. **Explicitly provided**: Override auto-computation by including species in `efficiencies` dict

    This allows matching CHEMKIN/Cantera behavior where efficiency parameters may differ (slightly)
    from the k0 ratio.

    **Composition Requirements:**

    - Mole fractions should sum to 1.0 (not enforced, but expected)
    - Species not in explicit_rate_constants or efficiencies use default behavior
    - For LMR-R, remaining species use default rate at default effective pressure

    References
    ----------
    .. [1] M.P. Burke, R. Song. "Evaluating mixture rules for multi-component pressure
        dependence: H+O2(+M)=HO2(+M)." Proc. Combust. Inst., vol. 36, no. 1,
        pp. 245–253, 2017. https://doi.org/10.1016/j.proci.2016.06.022
    .. [2] L. Lei, M.P. Burke. "Bath gas mixture effects on multichannel reactions:
        Insights and representations for systems beyond single-channel reactions."
        J. Phys. Chem. A, vol. 123, no. 3, pp. 631–649, 2018.
        https://doi.org/10.1021/acs.jpca.8b11272
    .. [3] L. Lei, M.P. Burke. "Evaluating mixture rules and combustion implications
        for multi-component pressure dependence of allyl+HO2 reactions."
        Proc. Combust. Inst., vol. 37, no. 1, pp. 355–362, 2019.
        https://doi.org/10.1016/j.proci.2018.07.075
    .. [4] L. Lei, M.P. Burke. "Mixture rules and falloff are now major uncertainties
        in experimentally derived rate parameters for H+O2(+M)=HO2(+M)."
        Combust. Flame, vol. 213, pp. 467–474, 2020.
        https://doi.org/10.1016/j.combustflame.2020.01.002
    .. [5] P.J. Singal, J. Lee, L. Lei, R.L. Speth, M.P. Burke. "Implementation of new
        mixture rules has a substantial impact on combustion predictions for H2 and NH3."
        Proc. Combust. Inst., vol. 40, no. 1-4, p. 105779, 2024.
        https://doi.org/10.1016/j.proci.2024.105779
    .. [6] Cantera development team. "LinearBurkeRate implementation." GitHub repository,
        Cantera/cantera. Accessed 2024.
        https://github.com/Cantera/cantera/blob/main/src/kinetics/LinearBurkeRate.cpp
    .. [7] A. Stagni, T. Dinelli. "Reduced-pressure linear mixture rules for
        pressure-dependent reaction kinetics." Chem. Eng. J., vol. 498, p. 170737, 2025.
        https://doi.org/10.1016/j.cej.2025.170737
    """

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
        explicit_rate_constants: dict[str, PressureDepRate] | None = None,
        efficiencies: dict[str, Arrhenius] | None = None,
        name: str = "",
        linear: bool = True,
        reduced_pressure: bool = False,
    ) -> None:
        """
        Initialize a Mixture Rule rate constant expression.

        The constructor processes explicit rate constants and efficiencies, automatically
        computing collision efficiencies where needed for the LMR-R formalism.

        Parameters
        ----------
        default_rate_constant : PressureDepRate
            The default/reference rate constant used for species not explicitly listed.
            Must be one of: Plog, FallOff, CABR, or Chebyshev.
        explicit_rate_constants : dict[str, PressureDepRate] | None, optional
            Dictionary mapping species names to their pressure-dependent rate expressions.
            Each value must be a Plog, FallOff, CABR, or Chebyshev object.
            For FallOff and CABR, these must NOT have collision efficiencies defined
            within them (efficiencies should be handled at the mixture rule level).
            For LMR-R mode, collision efficiencies will be automatically computed
            from k0 ratios unless explicitly overridden in the `efficiencies` parameter.
        efficiencies : dict[str, Arrhenius] | None, optional
            Dictionary mapping species names to their collision efficiency expressions.

            **Two types of efficiencies are supported:**

            1. **Efficiency-only species**: Species without explicit rate constants.
               Their efficiencies modify the default rate evaluation.

            2. **Explicit efficiency override**: Species WITH explicit rate constants
               can also have explicit efficiencies. This overrides the automatic k0-based
               computation and matches CHEMKIN/Cantera behavior where the YAML efficiency
               field takes precedence.

            Each efficiency is an Arrhenius object representing epsilon_i(T).
            For constant efficiencies, use: Arrhenius(parameters={"A": value, "n": 0, "Ea": 0})
            For temperature-dependent efficiencies, use appropriate n and Ea values.
        name : str, optional
            Human-readable reaction name (e.g., "H+O2(+M)=HO2(+M)"). Default is "".
        linear : bool, optional
            Whether to use linear mixture rule (True) or non-linear (False).
            Default is True. Non-linear rules are not yet implemented.
        reduced_pressure : bool, optional
            Whether to use reduced pressure formalism (LMR-R, True) or direct pressure
            space (LMR-P, False). Default is False (LMR-P).

        Raises
        ------
        ValueError
            If neither explicit_rate_constants nor efficiencies are provided
        ValueError
            If explicit rate constants contain unsupported types (not Plog, FallOff, CABR, Chebyshev)
        ValueError
            If FallOff or CABR explicit rate constants have collision efficiencies defined within them
        ValueError
            If k0 cannot be extracted from a rate constant when needed for LMR-R
        ValueError
            If unsupported rate constant type is encountered
        NotImplementedError
            If non-linear mixture rule in reduced pressure space (NLMR-R) is requested

        Notes
        -----
        **Efficiency Handling Logic:**

        For LMR-R mode (`reduced_pressure=True`), collision efficiencies are processed as follows:

        1. For species with explicit rate constants:
           - If efficiency is explicitly provided → use the explicit value
           - Otherwise -> automatically compute from k0 ratio: :math:`\\epsilon_i = k0_i / k0_{default}`

        2. For species without explicit rate constants:
           - Must have efficiency specified in `efficiencies` dict
           - These species use default_rate_constant evaluated at modified effective pressure

        This two-stage approach ensures compatibility with CHEMKIN/Cantera YAML files that
        specify both explicit rate expressions AND explicit collision efficiencies. The
        explicit efficiency always takes precedence, matching reference implementation behavior.

        **CHEMKIN/Cantera Compatibility:**

        When reading CHEMKIN YAML files with entries like:

        .. code-block:: yaml

            - equation: H + O2 (+M) <=> HO2 (+M)
              type: falloff
              ...
            - equation: H + O2 (+Ar) <=> HO2 (+Ar)
              type: pressure-dependent-Arrhenius
              ...
              efficiency: {A: 0.1706, n: 0.2090, Ea: 191.9}

        The `efficiency` field overrides automatic k0-based computation. Pass the YAML
        efficiency values to the `efficiencies` parameter to match Cantera behavior.

        **Automatic Efficiency Computation:**

        For species with explicit rate constants in LMR-R mode, the efficiency is computed
        as epsilon_i(T) = k0_i(T) / k0_default(T), which can be expressed as an Arrhenius form:

        .. math::

            \\epsilon_i(T) = \\frac{A_i}{A_{\\text{default}}} T^{n_i - n_{\\text{default}}}
            \\exp\\left(-\\frac{Ea_i - Ea_{\\text{default}}}{RT}\\right)
        """
        self._name = name

        self._linear = bool(linear)
        self._reduced_pressure = bool(reduced_pressure)
        if self._linear is False and self._reduced_pressure is True:
            raise NotImplementedError(
                "Non-Linear Mixture Rules in the Reduced Pressure (NLMR-R) space are not implemented!"
            )

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

                if isinstance(rate_constant, (FallOff, CABR)) and rate_constant.efficiencies is not None:
                    # they dont need any efficiency within their actual definition
                    raise ValueError(
                        "Explicit rate constant in the mixture rules formalism should "
                        "not have collision efficiencies defined!"
                    )

            if self.reduced_pressure is True:
                # Extract k0 from default rate constant once
                k0_default = self._extract_k0_arrhenius(default_rate_constant, "default")

                # Compute efficiency Arrhenius for each explicit species
                # Only compute if not already provided in efficiencies dict
                for species_name, rate_constant in explicit_rate_constants.items():
                    # Check if efficiency is explicitly provided for this species
                    if efficiencies is not None and species_name in efficiencies:
                        # Skip automatic computation - will use provided efficiency
                        continue
                    k0_species = self._extract_k0_arrhenius(rate_constant, species_name)
                    efficiency_dict[species_name] = self._compute_efficiency_arrhenius(
                        k0_species, k0_default, species_name
                    )

            # Store the explicit rate constants - needed for LMR-R evaluation
            self._explicit_rate_constants = explicit_rate_constants
        else:
            self._explicit_rate_constants = None

        # Add provided efficiencies (these are already Arrhenius objects)
        # These override any automatically computed efficiencies
        if efficiencies is not None:
            for species_name, efficiency_arrhenius in efficiencies.items():
                efficiency_dict[species_name] = efficiency_arrhenius

        # Store all efficiencies and explicit species list
        self._efficiencies = efficiency_dict
        self._explicit_species = tuple(sorted(efficiency_dict.keys()))

    @eqx.filter_jit
    def rate_constant(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "nt"],
        P: float | Float64[Array, ""] | Float64[Array, "np"],
        composition: dict[str, float],
    ) -> Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]:
        """
        Calculate the mixture rate constant at given temperature(s) and pressure(s).

        This method implements mixture rules for handling reactions with multiple colliders
        that can have different pressure dependencies. The calculation automatically selects
        between LMR-P and LMR-R based on the `reduced_pressure` attribute set during initialization.

        **LMR-P (Pressure space) formulation:**

        .. math::

            k_{\\text{LMR-P}}(T, P, \\mathbf{x}) = \\sum_i x_i \\cdot k_i(T, P)

        This is a simple weighted average where each collider contributes its rate constant
        weighted by its mole fraction. All colliders are evaluated at the same pressure P.

        **LMR-R (Reduced pressure space) formulation:**

        .. math::

            k_{\\text{LMR-R}}(T, P, \\mathbf{x}) = \\sum_i \\tilde{X}_i \\cdot k_i(T, P_i^{\\text{eff}})

        where:
            - :math:`\\tilde{X}_i = \\frac{\\varepsilon_i \\cdot x_i}{\\sum_j \\varepsilon_j \\cdot x_j}` is the fractional contribution
            - :math:`P_i^{\\text{eff}} = P \\cdot \\frac{\\varepsilon_{\\text{mix}}}{\\varepsilon_i}` is the effective pressure
            - :math:`\\varepsilon_{\\text{mix}} = \\sum_j \\varepsilon_j \\cdot x_j` is the mixture efficiency
            - :math:`\\varepsilon_i(T)` is the relative collision efficiency

        For species without explicit rate constants, the default rate constant is used
        at the appropriate effective pressure (LMR-R) or actual pressure (LMR-P).

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "nt"]
            Temperature(s) in Kelvin. Can be:
            - Scalar: single temperature evaluation
            - 1D array: multiple temperatures (vectorized evaluation)
        P : float | Float64[Array, ""] | Float64[Array, "np"]
            Pressure(s) in atmospheres. Can be:
            - Scalar: single pressure evaluation
            - 1D array: multiple pressures (creates T-P grid if T is also array)
        composition : dict[str, float]
            Gas mixture composition as mole fractions. Keys are species names,
            values are mole fractions. Should sum to 1.0 (not enforced).
            Species not listed are assumed to have zero mole fraction.

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]
            Mixed rate constant in appropriate units (typically cm³/mol/s or s⁻¹).
            Output shape depends on input broadcasting:
            - Scalar T, Scalar P → Scalar output
            - Vector T, Scalar P → Shape (nt,)
            - Scalar T, Vector P → Shape (np,)
            - Vector T, Vector P → Shape (nt, np) grid

        Raises
        ------
        NotImplementedError
            If non-linear mixture rule is requested (linear=False with reduced_pressure=True)

        Notes
        -----
        **Broadcasting behavior:**

        The method handles various input combinations efficiently:

        1. Scalar T, Scalar P: Direct evaluation
        2. Vector T, Scalar P: Vectorized over T
        3. Scalar T, Vector P: Vectorized over P using vmap
        4. Vector T, Vector P: Creates T-P grid using nested vmap

        **Composition handling:**

        - Species in composition but not in explicit_rate_constants/efficiencies:
          Use default rate constant (efficiency = 1.0 by definition)
        - Species in explicit_rate_constants but not in composition:
          Do not contribute (zero mole fraction)
        - Empty composition: Raises error (at least one species required)

        **Performance considerations:**

        - JIT compiled for optimal performance
        - Automatic differentiation compatible (use jax.grad)
        - Vectorization handled internally via vmap

        See Also
        --------
        compute_relative_efficiencies : Evaluate collision efficiencies at temperature
        compute_fractional_contributions : Calculate X_tilde_i for LMR-R
        compute_effective_pressures : Calculate P_i^eff for each collider
        """
        T = jnp.asarray(T, dtype=jnp.float64)
        P = jnp.asarray(P, dtype=jnp.float64)
        jax_composition = {key: jnp.float64(value) for key, value in composition.items()}

        if not self._reduced_pressure and self._linear:  # LMR-P
            if jnp.isscalar(P) or P.ndim == 0:  # Scalar pressure
                return self._lmr_p_single_P(T, P, jax_composition)
            else:  # Vector pressure - vectorize over pressure dimension
                vec_func = vmap(lambda p: self._lmr_p_single_P(T, p, jax_composition))
                return vec_func(P)
        elif self._reduced_pressure and self._linear:  # LMR-R
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
        else:  # Non-linear mixture rules
            raise NotImplementedError(
                "Non-Linear Mixture Rules in the Reduced Pressure (NLMR-R) space are not implemented!"
            )

    @eqx.filter_jit
    def _lmr_r_single_P(
        self,
        T: Float64[Array, ""] | Float64[Array, "nt"],
        P: Float64[Array, ""],
        composition: dict[str, Float64[Array, ""]],
    ) -> Float64[Array, ""] | Float64[Array, "nt"]:
        """
        Linear Mixture Rule in Reduced pressure space (LMR-R) for a single pressure.

        This is the core implementation of the LMR-R algorithm, called by `rate_constant()`
        for each pressure point. The method evaluates each collider at its effective pressure
        to ensure all are at the same reduced pressure, then combines using fractional contributions.

        **Algorithm Steps:**

        1. **Compute relative efficiencies** :math:`\\varepsilon_i(T)` for all species in composition
        2. **Compute fractional contributions** :math:`\\tilde{X}_i` and mixture efficiency :math:`\\varepsilon_{\\text{mix}}`
        3. **Compute effective pressures** :math:`P_i^{\\text{eff}} = P \\cdot \\varepsilon_{\\text{mix}} / \\varepsilon_i`
        4. **Evaluate and sum**: :math:`k = \\sum_i k_i(T, P_i^{\\text{eff}}) \\cdot \\tilde{X}_i`

        **Mathematical Formulation:**

        .. math::

            k_{\\text{LMR-R}}(T,P,\\mathbf{x}) = \\sum_{i \\in \\text{explicit}} k_i(T, P_i^{\\text{eff}}) \\cdot \\tilde{X}_i
            + k_{\\text{default}}(T, P_{\\text{default}}^{\\text{eff}}) \\cdot \\left(1 - \\sum_{i \\in \\text{explicit}} \\tilde{X}_i\\right)

        where:
            - :math:`\\tilde{X}_i = \\frac{\\varepsilon_i(T) \\cdot x_i}{\\varepsilon_{\\text{mix}}}` is the fractional contribution
            - :math:`\\varepsilon_{\\text{mix}} = \\sum_j \\varepsilon_j(T) \\cdot x_j` is the mixture efficiency
            - :math:`P_i^{\\text{eff}} = P \\cdot \\varepsilon_{\\text{mix}} / \\varepsilon_i(T)` is the effective pressure
            - :math:`P_{\\text{default}}^{\\text{eff}} = P \\cdot \\varepsilon_{\\text{mix}}` (since :math:`\\varepsilon_{\\text{default}} = 1.0`)
            - :math:`\\varepsilon_i(T) = k_{0,i}(T) / k_{0,\\text{default}}(T)` or explicitly provided

        **Cantera Equivalence:**

        This implementation follows the Cantera LinearBurkeRate formulation exactly:

        .. code-block:: cpp

            // From Cantera's LinearBurkeRate.cpp
            for (size_t i = 0; i < nColliders; i++) {
                double Peff_i = Pr_mix / efficiency_i;
                double k_i = collider_rates[i].eval(T, Peff_i);
                k_mix += k_i * X_tilde[i];
            }
            k_mix += k_default.eval(T, Pr_mix) * (1.0 - sum_X_tilde);

        Parameters
        ----------
        T : Float64[Array, ""] | Float64[Array, "nt"]
            Temperature(s) in Kelvin. Can be scalar or 1D array.
            If array, all calculations are vectorized element-wise.
        P : Float64[Array, ""]
            Pressure (scalar only) in atmospheres.
            This method handles ONE pressure at a time.
        composition : dict[str, Float64[Array, ""]]
            Gas mixture composition as mole fractions (JAX arrays).
            Keys are species names, values are scalar mole fractions.

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "nt"]
            Mixed rate constant with shape matching T input:
            - If T is scalar → returns scalar
            - If T is (nt,) array → returns (nt,) array

        Notes
        -----
        **Effective Pressure Calculation:**

        The effective pressure ensures all colliders are evaluated at the same reduced
        pressure :math:`P_r = P \\cdot [M] / k_0`, where [M] is the total collision partner
        concentration weighted by efficiencies.

        For a collider with high efficiency (strong collider), :math:`P_i^{\\text{eff}}` is lower
        than the actual pressure because the effective collision rate is already high.

        **Vectorization Strategy:**

        When T is a vector, effective pressures P_eff_dict also become vectors (same shape as T).
        Element-wise evaluation (T[i], P_eff[i]) is achieved using `vmap(lambda t, p: ...)(T, P_eff)`.
        This is NOT a grid evaluation - it's paired element-wise evaluation.

        **Remaining Species Handling:**

        Species in the composition without explicit rate constants contribute through the
        default rate constant weighted by :math:`(1 - \\sum_{i \\in \\text{explicit}} \\tilde{X}_i)`.
        This ensures conservation: the sum of all fractional contributions equals 1.0.

        See Also
        --------
        _lmr_p_single_P : LMR-P implementation for comparison
        compute_relative_efficiencies : Step 1 of the algorithm
        compute_fractional_contributions : Step 2 of the algorithm
        compute_effective_pressures : Step 3 of the algorithm
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
                    k_i = vmap(lambda t, p, rate=rate_expr: rate.rate_constant(t, p))(T, P_eff_dict[species])
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

    @eqx.filter_jit
    def _lmr_p_single_P(
        self,
        T: Float64[Array, ""] | Float64[Array, "nt"],
        P: Float64[Array, ""],
        composition: dict[str, Float64[Array, ""]],
    ) -> Float64[Array, ""] | Float64[Array, "nt"]:
        """
        Linear Mixture Rule in Pressure space (LMR-P) for a single pressure.

        This is the core implementation of the classical LMR-P algorithm, called by
        `rate_constant()` for each pressure point. The method computes a simple
        weighted average of individual collider rate constants.

        **Mathematical Formulation:**

        .. math::

            k_{\\text{LMR-P}}(T, P, \\mathbf{x}) = \\sum_{i \\in \\text{explicit}} x_i \\cdot k_i(T, P)
            + \\left(\\sum_{j \\notin \\text{explicit}} x_j\\right) \\cdot k_{\\text{default}}(T, P)

        This simplifies to:

        .. math::

            k_{\\text{LMR-P}} = \\sum_{i \\in \\text{explicit}} x_i \\cdot k_i(T, P)
            + (1 - x_{\\text{explicit\\_total}}) \\cdot k_{\\text{default}}(T, P)

        where:
            - :math:`x_i` is the mole fraction of species i
            - :math:`k_i(T, P)` is the rate constant for species i at pressure P
            - :math:`k_{\\text{default}}(T, P)` is the default collider rate constant
            - All colliders are evaluated at the same physical pressure P

        **Key Difference from LMR-R:**

        Unlike LMR-R, this method does NOT use effective pressures or collision efficiencies.
        All rate constants are evaluated at the actual mixture pressure P. This is simpler
        but less theoretically rigorous for mixtures with vastly different collision efficiencies.

        Parameters
        ----------
        T : Float64[Array, ""] | Float64[Array, "nt"]
            Temperature(s) in Kelvin. Can be scalar or 1D array.
            If array, all calculations are vectorized element-wise.
        P : Float64[Array, ""]
            Pressure (scalar only) in atmospheres.
            This method handles ONE pressure at a time.
        composition : dict[str, Float64[Array, ""]]
            Gas mixture composition as mole fractions (JAX arrays).
            Keys are species names, values are scalar mole fractions.

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "nt"]
            Mixed rate constant with shape matching T input:
            - If T is scalar → returns scalar
            - If T is (nt,) array → returns (nt,) array

        Notes
        -----
        **Simplicity vs Accuracy Trade-off:**

        LMR-P is computationally simpler than LMR-R because:
            - No efficiency calculations needed
            - No effective pressure transformations
            - Straightforward mole fraction weighting

        However, LMR-R is more accurate for mixtures where collision efficiencies vary
        significantly with temperature or between species.

        **Vectorization:**

        When T is a vector, individual rate constants are evaluated using:
        `vmap(lambda t: k_i.rate_constant(t, P))(T)`

        This creates element-wise evaluation across the temperature array while
        keeping pressure fixed.

        **Remaining Mole Fraction:**

        The term :math:`(1 - x_{\\text{explicit\\_total}})` represents all species
        without explicit rate constants. These all use the default rate constant,
        so they can be combined into a single term.

        See Also
        --------
        _lmr_r_single_P : LMR-R implementation with effective pressures
        rate_constant : Public interface that calls this method
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
                        k_i = vmap(lambda t, rate=rate_expression: rate.rate_constant(t, P))(T)
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
        Compute temperature-dependent collision efficiencies for species in the mixture.

        This is **Step 1** of the LMR-R algorithm. Collision efficiencies quantify how
        effective each species is as a collision partner (third body) relative to the
        default collider. An efficiency > 1 means the species is more effective than
        the default, while efficiency < 1 means less effective.

        **Mathematical Definition:**

        .. math::

            \\varepsilon_i(T) = \\frac{k_{0,i}(T)}{k_{0,\\text{default}}(T)}

        This can be expressed as an Arrhenius form (computed during initialization):

        .. math::

            \\varepsilon_i(T) = \\frac{A_i}{A_{\\text{default}}} T^{n_i - n_{\\text{default}}}
            \\exp\\left(-\\frac{Ea_i - Ea_{\\text{default}}}{RT}\\right)

        By definition, :math:`\\varepsilon_{\\text{default}} = 1.0` for species not in the
        efficiencies dictionary.

        Parameters
        ----------
        T : Float64[Array, ""] | Float64[Array, "nt"]
            Temperature(s) in Kelvin. Can be:
            - Scalar: single temperature
            - 1D array: temperature vector for vectorized evaluation
        species_list : list[str]
            List of species names present in the gas mixture composition.
            Only species with explicit efficiencies will be included in output.
        efficiencies : dict[str, Arrhenius]
            Pre-computed efficiency Arrhenius objects created during initialization.
            Each Arrhenius object represents epsilon_i(T) directly (NOT k0_i).

        Returns
        -------
        dict[str, Float64[Array, ""] | Float64[Array, "nt"]]
            Dictionary mapping species names to their dimensionless collision efficiencies.
            Output shape matches T input (scalar or vector).
            Species in species_list but NOT in efficiencies are omitted (implicitly epsilon=1.0).

        Notes
        -----
        **Efficiency Storage:**

        Efficiencies are stored as Arrhenius objects during initialization in two ways:

        1. **Auto-computed from k0 ratios** (for explicit rate constants in LMR-R):
           Created by `_compute_efficiency_arrhenius()` using k0_i / k0_default

        2. **Explicitly provided** (CHEMKIN/Cantera compatibility):
           Passed directly in the efficiencies parameter, overriding k0-based computation

        **Default Collider:**

        Species not in the efficiencies dict are treated as having epsilon = 1.0 by definition.
        This is handled implicitly in `compute_fractional_contributions()` and
        `compute_effective_pressures()` methods.

        **Temperature Dependence:**

        Most CHEMKIN mechanisms use constant efficiencies (n=0, Ea=0), but temperature-dependent
        efficiencies are fully supported. This is critical for some systems (e.g., H2O as collider
        in H+O2 reactions).

        See Also
        --------
        compute_fractional_contributions : Step 2 - uses epsilon_dict to compute X_tilde_i
        compute_effective_pressures : Step 3 - uses epsilon_dict to compute P_i^eff
        _compute_efficiency_arrhenius : Creates efficiency Arrhenius from k0 ratio
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
        Compute efficiency-weighted fractional contributions in reduced pressure space.

        This is **Step 2** of the LMR-R algorithm. Fractional contributions represent
        how much each collider contributes to the total rate, weighted by both mole
        fraction AND collision efficiency. This differs from simple mole fractions because
        more effective colliders contribute more even at lower concentrations.

        **Mathematical Definition:**

        .. math::

            \\tilde{X}_i = \\frac{\\varepsilon_i(T) \\cdot x_i}{\\sum_j \\varepsilon_j(T) \\cdot x_j}

        where:
            - :math:`x_i` is the mole fraction of species i
            - :math:`\\varepsilon_i(T)` is the collision efficiency (epsilon = 1.0 if not in epsilon_dict)
            - :math:`\\sum_j \\tilde{X}_j = 1.0` (conservation - sum of all contributions equals 1)

        The denominator :math:`\\varepsilon_{\\text{mix}} = \\sum_j \\varepsilon_j \\cdot x_j` is the
        **mixture efficiency**, representing the effective collision efficiency of the entire mixture.

        Parameters
        ----------
        composition : dict[str, Float64[Array, ""]]
            Gas mixture composition as mole fractions (JAX arrays).
            Keys are species names, values are scalar mole fractions.
            Should sum to 1.0 (not enforced).
        epsilon_dict : dict[str, Float64[Array, ""] | Float64[Array, "nt"]]
            Temperature-dependent collision efficiencies from `compute_relative_efficiencies()`.
            Species NOT in this dict are treated as having epsilon = 1.0 (default collider).

        Returns
        -------
        tuple[dict[str, Float64[Array, ""] | Float64[Array, "nt"]], Float64[Array, ""] | Float64[Array, "nt"]]
            Two-element tuple containing:

            1. **X_tilde_dict**: Dictionary mapping ALL species in composition to their
               fractional contributions. Shape matches epsilon_dict values (scalar or vector).

            2. **epsilon_weighted_sum**: The mixture efficiency :math:`\\varepsilon_{\\text{mix}}`.
               Used in Step 3 to compute effective pressures. Shape matches epsilon_dict values.

        Notes
        -----
        **Physical Interpretation:**

        For a binary mixture of 50% Ar (epsilon=0.5) and 50% H2O (epsilon=17.6):

        .. math::

            \\varepsilon_{\\text{mix}} = 0.5 \\times 0.5 + 17.6 \\times 0.5 = 9.05

            \\tilde{X}_{\\text{Ar}} = \\frac{0.5 \\times 0.5}{9.05} = 0.028

            \\tilde{X}_{\\text{H2O}} = \\frac{17.6 \\times 0.5}{9.05} = 0.972

        Despite equal mole fractions, H2O contributes 97.2% to the rate because it's
        a much more effective collision partner!

        **Default Collider Handling:**

        Species without explicit efficiencies use epsilon = 1.0:

        .. code-block:: python

            if species in epsilon_dict:
                contribution = epsilon_dict[species] * x_i
            else:
                contribution = 1.0 * x_i  # Default collider

        **Conservation Property:**

        The fractional contributions always sum to 1.0:

        .. math::

            \\sum_{i \\in \\text{all species}} \\tilde{X}_i = \\frac{\\sum_i (\\varepsilon_i \\cdot x_i)}{\\sum_j (\\varepsilon_j \\cdot x_j)} = 1.0

        This is verified in debugging by checking :math:`1 - \\sum_{i \\in \\text{explicit}} \\tilde{X}_i \\approx 0`
        when all species have explicit efficiencies.
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
            Pressure(s) in atmospheres
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
        """
        Whether the mixture rule uses linear blending.

        Returns
        -------
        bool
            True for linear mixture rule (LMR), False for non-linear (NLMR).
            Non-linear rules are not yet implemented.
        """
        return self._linear

    @property
    def reduced_pressure(self) -> bool:
        """
        Whether the mixture rule uses reduced pressure formalism.

        Returns
        -------
        bool
            True for reduced pressure space (LMR-R), False for direct pressure space (LMR-P).
            LMR-R is more theoretically rigorous and recommended for complex mixtures.
        """
        return self._reduced_pressure

    @property
    def explicit_species(self) -> tuple[str, ...]:
        """
        List of species with explicit collision efficiencies.

        This includes both species with explicit rate constants AND efficiency-only species.
        The tuple is sorted alphabetically for reproducibility.

        Returns
        -------
        tuple[str, ...]
            Sorted tuple of species names that have explicit collision efficiencies defined.
        """
        return self._explicit_species

    @property
    def efficiencies(self) -> dict[str, Arrhenius]:
        """
        Collision efficiency expressions for all explicit species.

        Each efficiency is stored as an Arrhenius object representing epsilon_i(T).
        For species with explicit rate constants in LMR-R mode, these may be either:
        - Automatically computed from k0 ratios (epsilon_i = k0_i / k0_default), or
        - Explicitly provided to override automatic computation (CHEMKIN/Cantera compatibility)

        Returns
        -------
        dict[str, Arrhenius]
            Dictionary mapping species names to their collision efficiency Arrhenius objects.
        """
        return self._efficiencies

    @property
    def explicit_rate_constants(self) -> dict[str, PressureDepRate] | None:
        """
        Pressure-dependent rate expressions for species with explicit pressure dependencies.

        These are species that have their own distinct pressure-dependent rate constants
        (different from the default collider). Each must be Plog, FallOff, CABR, or Chebyshev.

        Returns
        -------
        dict[str, PressureDepRate] | None
            Dictionary mapping species names to their rate constant objects, or None if
            no explicit rate constants were provided.
        """
        return self._explicit_rate_constants

    @property
    def default_rate_constant(self) -> PressureDepRate:
        """
        The default/reference rate constant for the mixture.

        This rate expression is used for:
        - Species without explicit rate constants
        - Computing k0_default for efficiency calculations in LMR-R mode

        Returns
        -------
        PressureDepRate
            The default pressure-dependent rate constant (Plog, FallOff, CABR, or Chebyshev).
        """
        return self._default_rate_constant
