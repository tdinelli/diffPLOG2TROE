"""
Copyright (c) 2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import equinox as eqx
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float64

from KiRATE.kinetics import Arrhenius
from KiRATE.kinetics.utils import validate_chebyshev_parameters


class Chebyshev(eqx.Module):
    """
    Chebyshev polynomial rate constant calculator for pressure-dependent reactions.

    This class implements Chebyshev polynomial representations of pressure-dependent
    rate constants, providing a compact and efficient way to represent k(T,P) over
    a wide range of temperatures and pressures using bivariate polynomial expansions.

    The Chebyshev formulation is given by:

    .. math::
        \\log_{10} k(T, P) = \\sum_{t=0}^{N_T-1} \\sum_{p=0}^{N_P-1} \\alpha_{tp} \\phi_t(\\tilde{T}) \\phi_p(\\tilde{P})

    where:
        - k(T, P) is the pressure-dependent rate constant
        - :math:`\\alpha_{t, p}` are Chebyshev polynomial coefficients (fitted to data)
        - :math:`\\phi_n(x) = cos(n \\cdot arccos(x))` are Chebyshev polynomials of the first kind
        - T̃, P̃ are reduced temperature and pressure mapped to [-1, 1]

    **Reduced Variables:**

    .. math::
        \\tilde{T} &= \\frac{2T^{-1} - T_{min}^{-1} - T_{max}^{-1}}{T_{max}^{-1} - T_{min}^{-1}} \\\\
        \\tilde{P} &= \\frac{2\\log_{10}P - \\log_{10}P_{min} - \\log_{10}P_{max}}{\\log_{10}P_{max} - \\log_{10}P_{min}}

    These map [T_min, T_max] and [P_min, P_max] to the interval [-1, 1] where
    Chebyshev polynomials are defined.

    Parameters
    ----------
    chebyshev_coefficients : Float64[Array, "nt np"]
        2D array of Chebyshev coefficients :math:`\\alpha_{t, p}` with shape
        (N_T, N_P). The shape determines the polynomial orders automatically.
    T_limits : tuple[float, float], optional
        Temperature range (T_min, T_max) in Kelvin, by default (300.0, 2500.0),
        these default values are the same one adopted by CHEMKIN.
    P_limits : tuple[float, float], optional
        Pressure range (P_min, P_max) in atmospheres, by default (0.001, 100.0),
        these default values are the same one adopted by CHEMKIN.
    name : str, optional
        Human-readable reaction name, by default ""

    Attributes
    ----------
    _chebyshev_coefficients : Float64[Array, "nt np"]
        Chebyshev polynomial coefficients as JAX array
    _T_min, _T_max : Float64[Array, ""]
        Temperature limits in Kelvin
    _P_min, _P_max : Float64[Array, ""]
        Pressure limits in atmospheres
    _log10P_min, _log10P_max : Float64[Array, ""]
        Pre-computed log10 of pressure limits
    _name : str
        Reaction name (static field)

    Notes
    -----
    **Advantages of Chebyshev Representation:**

    - **Compact**: Few coefficients represent complex T-P dependence
    - **Efficient**: Fast polynomial evaluation (no transcendental functions)
    - **Accurate**: Optimal approximation properties of Chebyshev polynomials
    - **Flexible**: Can represent any smooth k(T,P) surface

    **Important Limitations:**

    - **Intepretability**: The coefficients are just the result of pure
        mathematical operations and do not have a real physical meaning
    - **Extrapolation**: Chebyshev polynomials are only defined on [-1, 1]
    - **Oscillations**: May oscillate outside [T_min, T_max] × [P_min, P_max]
    - **Non-physical**: Extrapolated values may be negative or unphysical

    References
    ----------
    .. [1] This is what is reported in the CHEMKIN manual
           Jeff Ing, Chad Sheng, and Joseph W. Bozzelli,
           personal communication, 2002.
    """

    _chebyshev_coefficients: Float64[Array, "nt np"]
    _log10P_min: Float64[Array, ""]
    _log10P_max: Float64[Array, ""]
    _T_min: Float64[Array, ""]
    _T_max: Float64[Array, ""]
    _P_min: Float64[Array, ""]
    _P_max: Float64[Array, ""]
    _name: str = eqx.field(static=True, default="")
    _k0: Arrhenius | None = None

    def __init__(
        self,
        chebyshev_coefficients: Float64[Array, "nt np"],
        T_limits: tuple[float, float] = (300.0, 2500.0),
        P_limits: tuple[float, float] = (0.001, 100.0),
        k0_parameters: dict[str, float] | None = None,
        name: str = "",
    ) -> None:
        """
        Initialize the Chebyshev rate constant representation.

        Parameters
        ----------
        chebyshev_coefficients : Float64[Array, "nt np"]
            2D array of Chebyshev coefficients with shape (N_T, N_P).
            The shape automatically determines the polynomial orders.
            These are typically obtained by fitting to master equation or experimental data.
        T_limits : tuple[float, float], optional
            Temperature range (T_min, T_max) in Kelvin, by default (300.0, 2500.0).
            Defines the valid interpolation range. Extrapolation outside this range
            is strongly discouraged.
        P_limits : tuple[float, float], optional
            Pressure range (P_min, P_max) in atmospheres, by default (0.001, 100.0).
            Defines the valid interpolation range.
        k0_parameters : dict[str, float], optional
            Low pressure limit rate constant, by default None
        name : str, optional
            Human-readable reaction name, by default ""

        Raises
        ------
        ValueError
            If limits are invalid or any parameters violate physical constraints
        """
        self._name = name

        # Validate all parameters using utility function
        validate_chebyshev_parameters(chebyshev_coefficients, T_limits, P_limits)

        # Store coefficients as JAX array
        self._chebyshev_coefficients = chebyshev_coefficients

        # Store temperature limits
        self._T_min, self._T_max = jnp.float64(T_limits[0]), jnp.float64(T_limits[1])

        # Store pressure limits
        self._P_min, self._P_max = jnp.float64(P_limits[0]), jnp.float64(P_limits[1])

        # Pre-compute log10 of pressure limits (used in every rate constant evaluation)
        self._log10P_min = jnp.log10(self._P_min)
        self._log10P_max = jnp.log10(self._P_max)

        # Optional low pressure limit rate constant
        if k0_parameters is not None:
            self._k0 = Arrhenius(parameters=k0_parameters, name=f"{name} (k0)")
        else:
            self._k0 = None

    # ==================================================================================
    # CHEMKIN string parser
    @classmethod
    def from_chemkin(cls, input_string: str) -> "Chebyshev":
        """
        Parse a CHEMKIN-format Chebyshev entry (not yet implemented).

        Parameters
        ----------
        input_string : str
            CHEMKIN-formatted Chebyshev reaction string

        Raises
        ------
        NotImplementedError
            This method is a placeholder for future implementation
        """
        raise NotImplementedError("CHEMKIN parsing for Chebyshev s not yet implemented")

    # ==================================================================================
    # Rate constant methods
    @eqx.filter_jit
    def rate_constant(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "nt"],
        P: float | Float64[Array, ""] | Float64[Array, "np"],
        is_violation_allowed: bool = False,
    ) -> Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]:
        """
        Calculate the Chebyshev rate constant at given temperature(s) and pressure(s).

        This method evaluates the bivariate Chebyshev polynomial expansion to compute
        pressure-dependent rate constants over the specified T-P ranges.

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "nt"]
            Temperature(s) in Kelvin. Accepts scalars or 1D arrays.
        P : float | Float64[Array, ""] | Float64[Array, "np"]
            Pressure(s) in atmospheres. Accepts scalars or 1D arrays.
        is_violation_allowed : bool, optional
            If True, clip T and P to valid ranges [T_min, T_max] and [P_min, P_max].
            If False, allow extrapolation (may produce unphysical results).
            By default False.

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]
            Rate constant(s) with shape matching input broadcasting:

            - Scalar T, Scalar P -> Scalar output
            - Vector T, Scalar P -> Vector output (length nt)
            - Scalar T, Vector P -> Vector output (length np)
            - Vector T, Vector P -> Matrix output (shape: np x nt)

        Notes
        -----
        **Chebyshev Evaluation Algorithm:**

        1. **Clipping** (if is_violation_allowed=True):
           Restrict :math:`T \\in [T_{min}, T_{max}]` and :math:`P \\in [P_{min}, P_{max}]`

        2. **Reduced Variable Calculation:**
           Map to [-1, 1] for Chebyshev polynomial evaluation:

           .. math::
               \\tilde{T} &= \\frac{2T^{-1} - T_{min}^{-1} - T_{max}^{-1}}{T_{max}^{-1} - T_{min}^{-1}} \\\\
               \\tilde{P} &= \\frac{2\\log_{10}P - \\log_{10}P_{min} - \\log_{10}P_{max}}{\\log_{10}P_{max} - \\log_{10}P_{min}}

        3. **Polynomial Evaluation:**
           Compute bivariate sum using :math:`\\phi_n(x) = cos(n \\cdot arccos(x))`

        4. **Conversion:**
           :math:`k(T, P) = 10^{(polynomial_value)}`

        **Extrapolation Warning:**

        When is_violation_allowed=False and inputs are outside valid ranges,
        Chebyshev polynomials may oscillate wildly, producing negative or
        unphysical rate constants. **Always prefer clipping for safety**.
        """
        # Convert inputs to JAX arrays
        T = jnp.asarray(T, dtype=jnp.float64)
        P = jnp.asarray(P, dtype=jnp.float64)

        # Step 1: Optional clipping to valid ranges (differentiable with jnp.where)
        Tc = jnp.where(is_violation_allowed, jnp.clip(T, self._T_min, self._T_max), T)
        Pc = jnp.where(is_violation_allowed, jnp.clip(P, self._P_min, self._P_max), P)

        # Step 2: Compute reduced temperature
        # Maps [T_min, T_max] to [-1, 1] using inverse temperature transformation
        T_tilde = (2.0 / Tc - 1.0 / self._T_min - 1.0 / self._T_max) / (1.0 / self._T_max - 1.0 / self._T_min)

        # Step 3: Compute reduced pressure
        # Maps [log10(P_min), log10(P_max)] to [-1, 1]
        P_tilde = (2.0 * jnp.log10(Pc) - self._log10P_min - self._log10P_max) / (self._log10P_max - self._log10P_min)

        # Step 4: Evaluate Chebyshev polynomial
        if jnp.isscalar(P) or P.ndim == 0:
            # Scalar pressure - evaluate directly
            return self._single_P_rate_constant(T_tilde, P_tilde)
        else:
            # Vector pressure - vectorize over pressure dimension
            vec_func = vmap(lambda p: self._single_P_rate_constant(T_tilde, p))
            return vec_func(P_tilde)

    @eqx.filter_jit
    def _single_P_rate_constant(
        self,
        T_tilde: float | Float64[Array, ""] | Float64[Array, "nt"],
        P_tilde: Float64[Array, ""],
    ) -> Float64[Array, ""] | Float64[Array, "nt"]:
        """
        Evaluate Chebyshev rate constant for a single pressure value.

        This internal method computes the bivariate Chebyshev polynomial expansion
        for a fixed pressure. It is called by `rate_constant()` and vectorized over
        pressure when needed.

        Parameters
        ----------
        T_tilde : float | Float64[Array, ""] | Float64[Array, "nt"]
            Reduced temperature(s) bounded to [-1, 1] (scalar or 1D array)
        P_tilde : Float64[Array, ""]
            Reduced pressure bounded to [-1, 1] (scalar only)

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "nt"]
            Rate constant(s) k(T,P), shape matches T_tilde input

        Notes
        -----
        **Algorithm Steps:**

        1. **Generate polynomial indices:**
           Create index arrays n = [0, 1, ..., N_T-1] and m = [0, 1, ..., N_P-1]

        2. **Evaluate Chebyshev polynomials:**
           Compute :math:`\\phi_n(\\tilde{T})` and :math:`\\phi_m(\\tilde{P})`

        3. **Compute bivariate sum:**
           Calculate
           :math:`\\sum_n \\sum_m \\alpha_{nm} \\cdot \\phi_n(\\tilde{T}) \\cdot \\phi_m(\\tilde{P})`

        4. **Convert from log space:**
           Return k = 10^(sum)

        **Broadcasting Behavior:**

        - If T_tilde is scalar -> output is scalar
        - If T_tilde is (nt,) ->  output is (nt,)

        This method is always called with scalar P_tilde. For vectorized pressure
        evaluation, rate_constant() uses vmap to apply this method across pressures.
        """
        # Extract polynomial orders from coefficient array shape
        N, M = self._chebyshev_coefficients.shape

        # Step 1: Generate polynomial indices
        # Create integer arrays for polynomial degrees: [0, 1, 2, ..., N-1] and [0, 1, 2, ..., M-1]
        n_indices = jnp.arange(N)  # Temperature polynomial degrees: [0, 1, 2, ..., N-1]
        m_indices = jnp.arange(M)  # Pressure polynomial degrees: [0, 1, 2, ..., M-1]

        # Step 2: Evaluate Chebyshev polynomials of the first kind
        # Shape manipulation: [:, None] broadcasts for vectorized evaluation
        phi_n = self.chebyshev_polynomial(n_indices[:, None], T_tilde)  # Shape: (N, *T.shape)
        phi_m = self.chebyshev_polynomial(m_indices[:, None], P_tilde)  # Shape: (M,) since P_tilde is scalar

        # Step 3: Compute weighted bivariate sum
        # Using Einstein summation notation for efficient tensor contraction:
        # sum_nm_n_m -> ...: contracts over n (temperature) and m (pressure) dimensions
        # Result shape matches broadcasted temperature dimensions
        sum_result = jnp.einsum("nm,n...,m...->...", self._chebyshev_coefficients, phi_n, phi_m)

        # Step 4: Convert from log10 space to linear space
        return jnp.power(10.0, sum_result)

    @staticmethod
    @eqx.filter_jit
    def chebyshev_polynomial(n: int | Float64[Array, "..."], x: float | Float64[Array, "..."]) -> Float64[Array, "..."]:
        """
        Evaluate Chebyshev polynomials of the first kind at given points.

        This static method computes T_n(x) = cos(n·arccos(x)) for Chebyshev polynomials
        of the first kind. These polynomials are orthogonal on [-1, 1] and provide
        optimal approximation properties for interpolation.

        Parameters
        ----------
        n : int | Float64[Array, "..."]
            Polynomial degree(s). Can be scalar or array of degrees.
        x : float | Float64[Array, "..."]
            Evaluation point(s) in [-1, 1]. Can be scalar or array.

        Returns
        -------
        Float64[Array, "..."]
            Chebyshev polynomial value(s) T_n(x)
            Shape determined by broadcasting n and x

        Notes
        -----
        **Mathematical Definition:**

        Chebyshev polynomials of the first kind are defined by:

        .. math::
            T_n(x) = \\cos(n \\cdot \\arccos(x)) \\quad \\text{for } x \\in [-1, 1]

        **Recurrence Relation** (alternative formulation, not used here):

        .. math::
            T_0(x) &= 1 \\\\
            T_1(x) &= x \\\\
            T_{n+1}(x) &= 2x \\cdot T_n(x) - T_{n-1}(x)

        **Numerical Stability:**

        Input clipping to [-1, 1] prevents numerical issues with arccos:
        - arccos(x) is undefined for |x| > 1 (NaN result)
        - Small floating-point errors can push x slightly outside [-1, 1]
        - Clipping ensures robust evaluation without changing valid inputs

        .. math::
            \\frac{dT_n(x)}{dx} = \\frac{n \\sin(n \\arccos(x))}{\\sqrt{1-x^2}}

        References
        ----------
        .. [1] Mason, J. C. and Handscomb, D. C. "Chebyshev Polynomials."
               Chapman and Hall/CRC, 2002.
        .. [2] Trefethen, L. N. "Approximation Theory and Approximation Practice."
               SIAM, 2013.
        """
        # Clip x to valid domain [-1, 1] to prevent arccos(x) from returning NaN
        # This handles floating-point errors that might push x slightly outside [-1, 1]
        # For valid inputs in [-1, 1], clipping has no effect
        x_clipped = jnp.clip(x, -1.0, 1.0)

        # Evaluate Chebyshev polynomial using trigonometric definition
        # T_n(x) = cos(n * arccos(x))
        return jnp.cos(n * jnp.arccos(x_clipped))

    # ==================================================================================
    # String Representations and Debugging
    def __str__(self) -> str:
        """
        Generate CHEMKIN-format string representation of Chebyshev reaction.

        Returns a multi-line string in CHEMKIN-II format for writing to mechanism files.
        The format follows the CHEMKIN convention for Chebyshev reactions.

        Returns
        -------
        str
            CHEMKIN-format reaction string with temperature/pressure ranges and coefficients

        Notes
        -----
        **CHEMKIN Chebyshev Format:**

        ::

            REACTION_NAME          0.0 0.0 0.0
             TCHEB / T_min T_max                                   /
             PCHEB / P_min P_max                                   /
             CHEB  / N M  coeff_1 coeff_2 coeff_3 coeff_4 coeff_5  /
             CHEB  /      coeff_6 coeff_7 coeff_8 coeff_9 coeff_10 /
             ...

        Where:
        - N = polynomial order in temperature
        - M = polynomial order in pressure
        - Coefficients are listed row-major (flattened), 5 per line
        """
        lines = []

        # Line 1: Reaction name and placeholder Arrhenius parameters
        lines.append(f"{self._name}\t\t0.0 0.0 0.0")

        # Line 2: Temperature range specification
        # Format: TCHEB / T_min T_max /
        lines.append(f" TCHEB / {self._T_min:.2f} {self._T_max:.2f} /")

        # Line 3: Pressure range specification
        # Format: PCHEB / P_min P_max /
        lines.append(f" PCHEB / {self._P_min:.2f} {self._P_max:.2f} /")

        # Lines 4+: Chebyshev coefficients (5 per line, flattened row-major)
        # First line includes dimensions "CHEB / N M", subsequent lines are "CHEB /"
        N, M = self._chebyshev_coefficients.shape
        flattened = self._chebyshev_coefficients.flatten()  # Row-major flattening

        # Iterate over coefficients in chunks of 5
        for i, chunk_start in enumerate(range(0, len(flattened), 5)):
            chunk = flattened[chunk_start : chunk_start + 5]

            # First line includes polynomial dimensions (N, M)
            # Subsequent lines omit dimensions
            prefix = f" CHEB / {N} {M}" if i == 0 else " CHEB /"

            # Format coefficients in scientific notation with 5 decimal places
            coeffs_str = "".join(f" {coeff:.5e}" for coeff in chunk)
            lines.append(f"{prefix}{coeffs_str} /")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """
        Generate detailed string representation for debugging and inspection.

        Returns
        -------
        str
            String representation showing all key parameters
        """
        return (
            f"Chebyshev\n"
            f" name={self._name},\n"
            f" order_T={self.order_T},\n"
            f" order_P={self.order_P},\n"
            f" T_limits={self.T_limits},\n"
            f" P_limits={self.P_limits}\n)"
        )

    # ==================================================================================
    # Properties for parameters access
    @property
    def coefficients(self) -> Float64[Array, "nt np"]:
        """
        Chebyshev polynomial coefficients.

        Returns
        -------
        Float64[Array, "nt np"]
            2D array of coefficients with shape (order_T, order_P)
        """
        return self._chebyshev_coefficients

    @property
    def T_limits(self) -> tuple[Float64[Array, ""], Float64[Array, ""]]:
        """
        Temperature range (T_min, T_max) in Kelvin.

        Returns
        -------
        tuple[Float64[Array, ""], Float64[Array, ""]]
            (T_min, T_max) defining valid interpolation range
        """
        return (self._T_min, self._T_max)

    @property
    def P_limits(self) -> tuple[Float64[Array, ""], Float64[Array, ""]]:
        """
        Pressure range (P_min, P_max) in atmospheres.

        Returns
        -------
        tuple[Float64[Array, ""], Float64[Array, ""]]
            (P_min, P_max) defining valid interpolation range
        """
        return (self._P_min, self._P_max)

    @property
    def order_T(self) -> int:
        """
        Polynomial order in temperature dimension.

        Returns
        -------
        int
            Number of temperature polynomial terms (N_T)
        """
        return self._chebyshev_coefficients.shape[0]

    @property
    def order_P(self) -> int:
        """
        Polynomial order in pressure dimension.

        Returns
        -------
        int
            Number of pressure polynomial terms (N_P)
        """
        return self._chebyshev_coefficients.shape[1]

    @property
    def name(self) -> str:
        """
        Reaction name.

        Returns
        -------
        str
            Human-readable reaction name
        """
        return self._name

    @property
    def k0(self) -> Arrhenius | None:
        """
        Low pressure limit Arrhenius rate constant.

        Returns
        -------
        Arrhenius | None
            Arrhenius object, or None if not provided.

        Notes
        -----
        This is used for the Mixture Rule treatment and is not a standard
        in CHEMKIN format. It is not directly used in PLOG rate constant
        calculations but may be useful for compatibility with certain kinetics
        frameworks.
        """
        return self._k0
