import jax.numpy as jnp
from jaxtyping import Array, Float64

from KiRATE.species.species import Species
from KiRATE.species.thermo.nasa7_polynomial import h_rt, s_r, temperature_powers


def fit_smooth_nasa7_coefficients(species: Species, T_mid: float | None = None) -> Species:
    """
    Fit smooth NASA7 polynomial coefficients using spline basis for continuous thermodynamic properties.

    Creates new NASA7 coefficients using least-squares fitting to ensure smooth
    thermodynamic properties (Cp, H, S) across the intermediate temperature boundary.
    This eliminates discontinuities that can occur with standard NASA7 coefficients.

    Implementation follows OpenSMOKE++ ReformulationOfThermodynamics algorithm.

    Algorithm Overview
    -------------------
    1. **Fit smooth polynomial**: Fit Cp/R data using spline basis [1, T, T^2, T^3, T^4, (T-Tknot)^4]
       - The (T-Tknot)^4 spline ensures all derivatives (up to 3rd) are continuous at Tknot
       - Least-squares fitting solves: (X^T X) coeffs = X^T y (normal equations, not QR-based)

    2. **Extract NASA7 coefficients**:
       - Low-T coefficients: Use fitted [a0, a1, a2, a3, a4] directly
       - High-T coefficients: Apply spline transformation to get [AHT, BHT, CHT, DHT, EHT]
       - Integration constants: Compute a6 and a7 for both ranges using original species' H/(RT) and S/R

    3. **Temperature grid**: Split at ORIGINAL species' Tmid (not at candidate Tknot)
       - First n points: Tmin to original_Tmid (low-T region)
       - Next n points: original_Tmid to Tmax (high-T region)
       - This ensures consistent evaluation of Cp/R across candidate Tknot values

    Key Mathematical Details
    -------------------------
    For a candidate intermediate temperature :math:`T_{\\text{knot}}`, the fitted polynomial is:

    .. math::

        \\frac{C_p}{R} = a_0 + a_1 T + a_2 T^2 + a_3 T^3 + a_4 T^4 + \\alpha(T-T_{\\text{knot}})^4

    The high-T coefficients are computed by expanding :math:`(T-T_{\\text{knot}})^4` in the :math:`T` basis:

    .. math::

        (T-T_{\\text{knot}})^4 = T^4 - 4T_{\\text{knot}}T^3 + 6T_{\\text{knot}}^2T^2 - 4T_{\\text{knot}}^3T + T_{\\text{knot}}^4

    This gives:

    .. math::

        a_1^{\\text{high}} &= a_0 + \\alpha T_{\\text{knot}}^4 \\\\
        a_2^{\\text{high}} &= a_1 - 4\\alpha T_{\\text{knot}}^3 \\\\
        a_3^{\\text{high}} &= a_2 + 6\\alpha T_{\\text{knot}}^2 \\\\
        a_4^{\\text{high}} &= a_3 - 4\\alpha T_{\\text{knot}} \\\\
        a_5^{\\text{high}} &= a_4 + \\alpha

    Integration constants :math:`a_6` and :math:`a_7` are set to match the original species'
    :math:`H/(RT)` and :math:`S/R` at :math:`T_{\\text{knot}}`:

    .. math::

        a_6^{\\text{low}} &= (h_{RT}^{\\text{orig}} - h_{RT}^{\\text{low}}) T_{\\text{knot}} \\\\
        a_6^{\\text{high}} &= (h_{RT}^{\\text{orig}} - h_{RT}^{\\text{high}}) T_{\\text{knot}} \\\\
        a_7^{\\text{low}} &= s_R^{\\text{orig}} - s_R^{\\text{low}} \\\\
        a_7^{\\text{high}} &= s_R^{\\text{orig}} - s_R^{\\text{high}}

    Tmid Search Strategy
    --------------------
    If Tmid is not specified, automatically searches for optimal value:

    - **Search range**: :math:`[T_{\\text{min}} + 0.2(T_{\\text{max}} - T_{\\text{min}}), T_{\\text{max}} - 0.2(T_{\\text{max}} - T_{\\text{min}})]`
      (excludes 20% at each boundary to avoid edge effects)
    - **Stepping**: Adaptive - divides search range into ~15 candidates
      (scales naturally with species temperature range)
    - **Error metric**: Cumulative relative error :math:`= \\sum_i \\frac{|C_p^{\\text{fitted}} - C_p^{\\text{orig}}|}{C_p^{\\text{orig}}}`
    - **Best Tmid**: Minimizes the error metric across the temperature range

    Parameters
    ----------
    species : Species
        Original species to refit with smooth coefficients
    T_mid : float, optional
        Fixed intermediate temperature [K]. If None, automatically searches for optimal T_mid
        that minimizes the fitting error.

    Returns
    -------
    Species
        New Species instance with reformulated NASA7 coefficients that have:
        - Continuous Cp/R, H/(RT), S/R across Tmid
        - Continuous derivatives (dCp/dT, dH/dT, dS/dT) through spline basis
        - Original H/(RT) and S/R values at Tmid

    Notes
    -----
    The reformulated species will NOT match the original Cp/R away from Tmid.
    This is intentional - the reformulation trades off point-wise accuracy for
    smooth transitions, which is beneficial for numerical integration and stability.

    The implementation uses normal equations (X^T X coeffs = X^T y) rather than
    QR-based least-squares, matching OpenSMOKE++ for numerical stability with
    polynomial fits at high powers.
    """

    if T_mid is not None:
        T_mid = jnp.float64(T_mid)
        temperatures = _build_temperature_grid(species.Tmin, species.Tmax, T_mid, 30)

        # Fixed intermediate temperature case
        knot_temperature = T_mid
        fitted_coeffs = _fit_spline_to_heat_capacity(species, temperatures, knot_temperature)
        new_low_coeffs, new_high_coeffs = _extract_nasa7_coefficients(species, fitted_coeffs, knot_temperature)

        return Species(
            name=species.name,
            elemental_composition=species.elemental_composition,
            phase=species.phase,
            Tmin=float(species.Tmin),
            Tmax=float(species.Tmax),
            Tmid=float(knot_temperature),
            low_coeffs=new_low_coeffs,
            high_coeffs=new_high_coeffs,
        )
    else:
        temperatures = _build_temperature_grid(species.Tmin, species.Tmax, species.Tmid, 30)
        # Auto-search for optimal intermediate temperature
        # Search range: centered around the actual species temperature bounds
        # Exclude 20% at each boundary to avoid edge effects
        range_width = species.Tmax - species.Tmin
        T_min_search = species.Tmin + 0.2 * range_width
        T_max_search = species.Tmax - 0.2 * range_width

        best_T_mid = None
        best_error = jnp.inf

        # Adaptive stepping: divide search range into consistent number of candidates
        # This scales naturally with the species' temperature range
        n_candidates = 15
        step = (T_max_search - T_min_search) / n_candidates

        knot_candidate = T_min_search
        while knot_candidate <= T_max_search:
            fitted_coeffs = _fit_spline_to_heat_capacity(species, temperatures, knot_candidate)
            error = _compute_relative_fit_error(species, temperatures, knot_candidate, fitted_coeffs)

            if error < best_error:
                best_error = error
                best_T_mid = knot_candidate

            knot_candidate += step

        # Final fit with optimal T_mid
        fitted_coeffs = _fit_spline_to_heat_capacity(species, temperatures, best_T_mid)
        new_low_coeffs, new_high_coeffs = _extract_nasa7_coefficients(species, fitted_coeffs, best_T_mid)

        return Species(
            name=species.name,
            elemental_composition=species.elemental_composition,
            phase=species.phase,
            Tmin=float(species.Tmin),
            Tmax=float(species.Tmax),
            Tmid=float(best_T_mid),
            low_coeffs=new_low_coeffs,
            high_coeffs=new_high_coeffs,
        )


def _build_temperature_grid(
    Tmin: Float64[Array, ""],
    Tmax: Float64[Array, ""],
    Tmid_original: Float64[Array, ""],
    n: int = 15,
) -> Float64[Array, "2n-1"]:
    """
    Build temperature grid split at original Tmid.

    Helper method to construct consistent temperature grid for fitting.
    Temperature points are split at the original species' Tmid to ensure
    consistent evaluation across all candidate Tknot values.

    Parameters
    ----------
    Tmin : Float64[Array, ""]
        Minimum temperature [K]
    Tmax : Float64[Array, ""]
        Maximum temperature [K]
    Tmid_original : Float64[Array, ""]
        Original species' intermediate temperature [K]
    n : int
        Number of points in each temperature range (default: 15)

    Returns
    -------
    Float64[Array, "2n-1"]
        Combined temperature grid
    """
    return jnp.concatenate(
        [
            jnp.linspace(Tmin, Tmid_original, n),
            jnp.linspace(Tmid_original, Tmax, n)[1:],
        ]
    )


def _fit_spline_to_heat_capacity(
    species: Species,
    temperatures: Float64[Array, "n"],
    knot_temperature: Float64[Array, ""],
) -> Float64[Array, "6"]:
    """
    Fit spline basis to heat capacity data using least-squares normal equations.

    Internal helper method that performs the core fitting operation:
    builds spline design matrix, solves normal equations to get fitted coefficients.

    Parameters
    ----------
    species : Species
        Original species to fit
    temperatures : Float64[Array, "n"]
        Temperature points for fitting [K]
    knot_temperature : Float64[Array, ""]
        Intermediate temperature for spline correction [K]

    Returns
    -------
    Float64[Array, "6"]
        Fitted coefficients [a0, a1, a2, a3, a4, alpha]
    """
    y = species.cp_over_r(temperatures)
    X = _build_spline_basis_matrix(temperatures, knot_temperature)

    # Assemble and solve the normal equations like OpenSMOKE: solve (X^T X) * params = X^T * y
    XTX = X.T @ X
    XTy = X.T @ y
    return jnp.linalg.solve(XTX, XTy)


def _compute_relative_fit_error(
    species: Species,
    temperatures: Float64[Array, "n"],
    knot_temperature: Float64[Array, ""],
    fitted_coeffs: Float64[Array, "6"],
) -> float:
    """
    Compute cumulative relative fitting error for spline coefficients.

    Calculates error metric: :math:`\\sum | Cp_{fitted} - Cp_{original}| / Cp_{original}`

    Parameters
    ----------
    species : Species
        Original species for comparison
    temperatures : Float64[Array, "n"]
        Temperature points [K]
    knot_temperature : Float64[Array, ""]
        Intermediate temperature [K]
    fitted_coeffs : Float64[Array, "6"]
        Fitted coefficients [a0, a1, a2, a3, a4, alpha]

    Returns
    -------
    float
        Cumulative relative error
    """
    X = _build_spline_basis_matrix(temperatures, knot_temperature)
    y = species.cp_over_r(temperatures)
    cp_fitted = X @ fitted_coeffs

    return float(jnp.sum(jnp.abs(cp_fitted - y) / jnp.abs(y)))


def _extract_nasa7_coefficients(
    species: Species,
    fitted_coeffs: Float64[Array, "6"],
    knot_temperature: Float64[Array, ""],
) -> tuple[list[float], list[float]]:
    """
    Compute NASA7 coefficients ensuring continuity with original H/(RT) and S/R.

    Following OpenSMOKE++ approach, extracts and transforms the fitted spline
    coefficients into NASA7 polynomial sets, then computes a6 and a7 integration
    constants to match the original species' H/(RT) and S/R at Tknot.

    Coefficient Transformation
    ---------------------------
    Input fitted coefficients represent:

    .. math::

        \\frac{C_p}{R} = a_0 + a_1 T + a_2 T^2 + a_3 T^3 + a_4 T^4 + \\alpha(T-T_{\\text{knot}})^4

    For :math:`T \\leq T_{\\text{knot}}` (low-T region):

    .. math::

        a_1^{\\text{low}} = a_0, \\quad a_2^{\\text{low}} = a_1, \\quad a_3^{\\text{low}} = a_2,
        \\quad a_4^{\\text{low}} = a_3, \\quad a_5^{\\text{low}} = a_4

    (Direct assignment since spline term is zero)

    For :math:`T > T_{\\text{knot}}` (high-T region), apply transformation based on
    :math:`(T-T_{\\text{knot}})^4` expansion:

    .. math::

        a_1^{\\text{high}} &= a_0 + \\alpha T_{\\text{knot}}^4 \\\\
        a_2^{\\text{high}} &= a_1 - 4\\alpha T_{\\text{knot}}^3 \\\\
        a_3^{\\text{high}} &= a_2 + 6\\alpha T_{\\text{knot}}^2 \\\\
        a_4^{\\text{high}} &= a_3 - 4\\alpha T_{\\text{knot}} \\\\
        a_5^{\\text{high}} &= a_4 + \\alpha

    This transformation ensures that both polynomial sets, when evaluated with
    :math:`(T-T_{\\text{knot}})^4 = 0` at :math:`T_{\\text{knot}}`, give the same
    :math:`C_p/R` value (continuity).

    Integration Constants
    ---------------------
    For :math:`H/(RT)`:

    .. math::

        \\frac{H}{RT} = \\int \\frac{C_p/R}{T} \\, dT + \\frac{a_6}{T}

    Compute both low and high range integration constants to match original:

    .. math::

        a_6^{\\text{low}} &= \\left(\\frac{H}{RT}\\bigg|_{\\text{orig}} - \\frac{H}{RT}\\bigg|_{\\text{low}}\\right) T_{\\text{knot}} \\\\
        a_6^{\\text{high}} &= \\left(\\frac{H}{RT}\\bigg|_{\\text{orig}} - \\frac{H}{RT}\\bigg|_{\\text{high}}\\right) T_{\\text{knot}}

    For :math:`S/R`:

    .. math::

        \\frac{S}{R} = \\int \\frac{C_p/R}{T} \\, dT + a_7

    Compute both low and high range constants:

    .. math::

        a_7^{\\text{low}} &= \\frac{S}{R}\\bigg|_{\\text{orig}} - \\frac{S}{R}\\bigg|_{\\text{low}} \\\\
        a_7^{\\text{high}} &= \\frac{S}{R}\\bigg|_{\\text{orig}} - \\frac{S}{R}\\bigg|_{\\text{high}}

    This ensures :math:`H/(RT)` and :math:`S/R` are continuous and match the original at
    :math:`T_{\\text{knot}}`, while :math:`dH/dT = C_p/R/T` and :math:`dS/dT = C_p/R/T`
    are also continuous (through :math:`C_p/R` continuity).

    Parameters
    ----------
    species : Species
        Original species for thermodynamic property matching
    fitted_coeffs : Float64[Array, "6"]
        Fitted coefficients [a0, a1, a2, a3, a4, alpha] from least-squares fit
    knot_temperature : Float64[Array, ""]
        Intermediate temperature [K] where spline basis changes

    Returns
    -------
    tuple[list[float], list[float]]
        (low_coeffs, high_coeffs) - NASA7 coefficient lists for low and high T ranges
        Each list contains [a1, a2, a3, a4, a5, a6, a7]
    """
    a0, a1, a2, a3, a4, alpha = fitted_coeffs
    T_powers = temperature_powers(knot_temperature)

    # For T <= Tknot: use base polynomial coefficients (no spline correction)
    a1_low = a0
    a2_low = a1
    a3_low = a2
    a4_low = a3
    a5_low = a4

    # For T > Tknot: apply spline correction to ensure smooth derivatives
    # Derivative of (T-Tknot)^4 is 4*(T-Tknot)^3, 12*(T-Tknot)^2, 12*(T-Tknot), 4
    # So: dCp/R|_high = da1_high + 2*da2_high*T + 3*da3_high*T^2 + 4*da4_high*T^3
    #                  = d(alpha*(T-Tknot)^4)/dT = 4*alpha*(T-Tknot)^3
    # This gives: da1_high = 4*alpha*(T-Tknot)^3 at Tknot, but we need continuous coefficients
    # Using Taylor expansion: (T-Tknot)^4 coefficients in T-basis are:
    a1_high = a0 + alpha * T_powers[4]
    a2_high = a1 - 4.0 * alpha * T_powers[3]
    a3_high = a2 + 6.0 * alpha * T_powers[2]
    a4_high = a3 - 4.0 * alpha * knot_temperature
    a5_high = a4 + alpha

    # Get original species' H/(RT) and S/R at knot_temperature
    h_RT_original = species.h_over_rt(knot_temperature)
    s_R_original = species.s_over_r(knot_temperature)

    # Build coefficient arrays for evaluation using class methods
    coeffs_low = jnp.array([a1_low, a2_low, a3_low, a4_low, a5_low, 0.0, 0.0])
    coeffs_high = jnp.array([a1_high, a2_high, a3_high, a4_high, a5_high, 0.0, 0.0])

    # Compute H/(RT) at Tknot from low-T and high-T coefficients using class method
    h_RT_low = h_rt(coeffs_low, T_powers)
    h_RT_high = h_rt(coeffs_high, T_powers)

    # Compute a6 to ensure H/(RT) continuity with original species
    a6_low = (h_RT_original - h_RT_low) * knot_temperature
    a6_high = (h_RT_original - h_RT_high) * knot_temperature

    # Compute S/R at Tknot from low-T and high-T coefficients using class method
    s_R_low = s_r(coeffs_low, T_powers)
    s_R_high = s_r(coeffs_high, T_powers)

    # Compute a7 to ensure S/R continuity with original species
    a7_low = s_R_original - s_R_low
    a7_high = s_R_original - s_R_high

    low_coeffs = [
        float(a1_low),
        float(a2_low),
        float(a3_low),
        float(a4_low),
        float(a5_low),
        float(a6_low),
        float(a7_low),
    ]
    high_coeffs = [
        float(a1_high),
        float(a2_high),
        float(a3_high),
        float(a4_high),
        float(a5_high),
        float(a6_high),
        float(a7_high),
    ]

    return low_coeffs, high_coeffs


def _build_spline_basis_matrix(
    temperatures: Float64[Array, "n"],
    knot_temperature: Float64[Array, ""],
) -> Float64[Array, "n 6"]:
    """
    Build spline basis design matrix for least-squares fitting of NASA coefficients.

    Constructs the matrix :math:`X` for the least-squares problem where:

    .. math::

        \\frac{C_p}{R} = X \\cdot \\mathbf{coeffs}

    This implements the spline basis from OpenSMOKE++ for smooth thermodynamic fits.

    The design matrix has 6 columns for basis functions:

    .. math::

        X[:, 0:5] &= [1, T, T^2, T^3, T^4] \\quad \\text{(standard polynomial basis)} \\\\
        X[:, 5] &= \\begin{cases}
            0 & \\text{if } T \\leq T_{\\text{knot}} \\\\
            (T - T_{\\text{knot}})^4 & \\text{if } T > T_{\\text{knot}}
        \\end{cases} \\quad \\text{(smooth spline correction)}

    The spline basis function :math:`(T-T_{\\text{knot}})^4` ensures:

    - :math:`C_p/R` is continuous at :math:`T_{\\text{knot}}` (function value matches)
    - :math:`\\frac{dC_p}{dT}` is continuous at :math:`T_{\\text{knot}}` (first derivative matches)
    - :math:`\\frac{d^2C_p}{dT^2}` is continuous at :math:`T_{\\text{knot}}` (second derivative matches)
    - :math:`\\frac{d^3C_p}{dT^3}` is continuous at :math:`T_{\\text{knot}}` (third derivative matches)

    This automatically enforces smooth derivatives without explicit constraints.

    Parameters
    ----------
    temperatures : Float64[Array, "n"]
        Temperature points for fitting [K]
    knot_temperature : Float64[Array, ""]
        Intermediate temperature for spline correction [K]

    Returns
    -------
    Float64[Array, "n 6"]
        Design matrix with shape (n_temperatures, 6)

    Notes
    -----
    The spline term :math:`(T-T_{\\text{knot}})^4` is zero for :math:`T \\leq T_{\\text{knot}}`, ensuring:

    - For :math:`T \\leq T_{\\text{knot}}`: :math:`C_p/R = a_0 + a_1 T + a_2 T^2 + a_3 T^3 + a_4 T^4`
    - For :math:`T > T_{\\text{knot}}`: :math:`C_p/R = a_0 + a_1 T + \\cdots + \\alpha(T-T_{\\text{knot}})^4`

    When :math:`(T-T_{\\text{knot}})^4` is expanded to polynomial basis for :math:`T > T_{\\text{knot}}`,
    the high-T coefficients are computed as spline-corrected versions of the low-T coefficients.
    """
    n_points = len(temperatures)
    X = jnp.zeros((n_points, 6))

    T_powers = temperature_powers(temperatures)

    # Spline correction: zero for T < knot_temperature, (T-knot_temperature)^4 for T >= knot_temperature
    spline_correction = jnp.where(temperatures >= knot_temperature, (temperatures - knot_temperature) ** 4, 0.0)

    X = jnp.concatenate([T_powers, spline_correction[:, None]], axis=1)
    return X
