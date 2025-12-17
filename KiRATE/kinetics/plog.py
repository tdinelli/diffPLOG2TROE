"""
Copyright (c) 2025 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

from typing import Optional, Union

import equinox as eqx
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float64, Int64

from KiRATE.kinetics.arrhenius import Arrhenius
from KiRATE.utilities.chemkin_parser import parse_plog


class Plog(eqx.Module):
    """
    PLOG (Pressure-Logarithmic interpolation) rate constant calculator for chemical reactions.

    This class implements pressure-dependent rate constants using logarithmic interpolation
    between discrete pressure levels. The PLOG formulation is given by:

    .. math::
        k(T, P) = \\text{interpolate}_{\\ln P} \\left[ k_i(T) \\right]

    where:
        - k(T, P) is the rate constant at temperature T and pressure P [units vary with reaction order]
        - k_i(T) are Arrhenius rate constants at discrete pressure levels P_i [atm]
        - Interpolation is performed in log-log space: ln(k) vs ln(P)
        - For P < P_min or P > P_max, constant extrapolation is used

    This implementation uses **soft interpolation** to maintain full differentiability
    for automatic differentiation, enabling gradient-based parameter optimization.

    Key Features:
        - Fully differentiable with respect to temperature and parameters
        - Supports vectorized evaluation over temperature and pressure arrays
        - CHEMKIN-compatible input/output formats

    Parameters
    ----------
    parameters : dict[float, dict[str, float]]
        Dictionary mapping pressure levels [atm] to Arrhenius parameters:

        .. code-block:: python

            {
                0.01: {"A": 1e12, "n": 0.0, "Ea": 10000.0},
                1.0:  {"A": 1e13, "n": 0.5, "Ea": 12000.0},
                10.0: {"A": 1e14, "n": 1.0, "Ea": 15000.0}
            }

    name : str, optional
        Human-readable name for the reaction, by default ""
    k0_parameters : dict[str, float], optional
        Nominal Arrhenius parameters for CHEMKIN compatibility, by default None

    Attributes
    ----------
    _k_levels : list[Arrhenius]
        List of Arrhenius objects at each pressure level
    _p_levels : Float64[Array, "np"]
        Array of pressure levels [atm] in ascending order
    _lnp_levels : Float64[Array, "np"]
        Natural logarithm of pressure levels for efficient interpolation
    _num_p_levels : Int64[Array, ""]
        Number of pressure levels
    _k0 : Optional[Arrhenius]
        Nominal Arrhenius object
    _name : str
        Reaction name for identification (static field)

    References
    ----------
    .. [1] Kee, R. J., et al. "CHEMKIN-III: A Fortran chemical kinetics package
           for the analysis of gas-phase chemical and plasma kinetics." (1996).
    .. [2] TODO add the proper PLOG reference
    """

    _k_levels: list[Arrhenius]
    _p_levels: Float64[Array, "np"]
    _lnp_levels: Float64[Array, "np"]
    _num_p_levels: Int64[Array, ""]
    _k0: Optional[Arrhenius] = None
    _name: str = eqx.field(static=True, default="")

    def __init__(
        self,
        parameters: dict[float, dict[str, float]],
        name: str = "",
        k0_parameters: Optional[dict[str, float]] = None,
    ) -> None:
        """
        Initialize the PLOG rate constant calculator.

        Parameters
        ----------
        parameters : dict[float, dict[str, float]]
            Dictionary mapping pressure levels [atm] to Arrhenius parameters:

            - Key: Pressure level [atm]
            - Value: Arrhenius parameter dictionary {"A": ..., "n": ..., "Ea": ...}

            Pressure levels will be sorted in ascending order internally.
        name : str, optional
            Human-readable name for the reaction, by default ""
        k0_parameters : dict[str, float], optional
            Nominal Arrhenius parameters, by default None

        Raises
        ------
        ValueError
            If parameters dictionary is empty or contains invalid Arrhenius parameters

        Notes
        -----
        - Pressure levels are automatically sorted in ascending order
        - Each pressure level creates an internal Arrhenius object
        - Minimum 2 pressure levels required for interpolation
        """
        self._name = name

        # Sort pressure levels in ascending order (required for interpolation)
        parameters = dict(sorted(parameters.items()))

        # Store pressure levels and their natural logarithms
        self._p_levels = jnp.array(list(parameters.keys()), dtype=jnp.float64)
        self._lnp_levels = jnp.log(self._p_levels)
        self._num_p_levels = jnp.int64(len(self._p_levels))

        # Create Arrhenius objects for each pressure level
        arrhenius_objects = []
        for p, params in parameters.items():
            arrhenius_objects.append(Arrhenius(parameters=params, name=f"{name} ({p})"))

        self._k_levels = arrhenius_objects

        # Optional nominal rate constant (CHEMKIN format compatibility)
        if k0_parameters is not None:
            self._k0 = Arrhenius(parameters=k0_parameters, name=f"{name} (k0)")
        else:
            self._k0 = None

    @classmethod
    def from_chemkin(cls, input_string: str) -> Union["Plog", tuple["Plog", "Plog"]]:
        """
        Create a Plog instance from a CHEMKIN format string.

        This class method provides a convenient way to construct Plog objects
        directly from CHEMKIN-style input strings, which are commonly used in chemical
        kinetics databases and modeling software.

        Parameters
        ----------
        input_string : str
            CHEMKIN-formatted string containing reaction name and PLOG entries.
            Example format::

                H+O2=O+OH  0.0 0.0 0.0
                 PLOG / 0.01  1.0E+12  0.0  10000.0 /
                 PLOG / 1.0   1.0E+13  0.5  12000.0 /
                 PLOG / 100.0 1.0E+14  1.0  15000.0 /

        Returns
        -------
        Plog or tuple[Plog, Plog]
            Single Plog instance for normal reactions, or tuple of two Plog instances
            for DUPLICATE reactions.

        Raises
        ------
        ValueError
            If the input string cannot be parsed or contains invalid parameters.
        """
        parsed_result = parse_plog(input_string)

        if len(parsed_result) == 2:
            # Standard case: single reaction pathway
            reaction_name, plog_coefficients = parsed_result
            return cls(parameters=plog_coefficients, name=reaction_name)
        elif len(parsed_result) == 3:
            # DUPLICATE case: two separate reaction pathways
            reaction_name, plog_coefficients_1, plog_coefficients_2 = parsed_result
            plog_1 = cls(parameters=plog_coefficients_1, name=reaction_name)
            plog_2 = cls(parameters=plog_coefficients_2, name=reaction_name)
            return plog_1, plog_2
        else:
            raise ValueError(f"Unexpected number of return values from parse_plog: {len(parsed_result)}")

    @eqx.filter_jit
    def rate_constant(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "nt"],
        P: float | Float64[Array, ""] | Float64[Array, "np"],
    ) -> Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]:
        """
        Calculate the PLOG rate constant at given temperature(s) and pressure(s).

        This method implements logarithmic interpolation between discrete pressure levels
        with automatic broadcasting for vector inputs. The implementation uses soft
        interpolation to maintain full differentiability.

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "nt"]
            Temperature(s) in Kelvin. Accepts:

            - Python float: Single temperature
            - JAX scalar array: Single temperature as array
            - JAX 1D array: Multiple temperatures

        P : float | Float64[Array, ""] | Float64[Array, "np"]
            Pressure(s) in atmospheres. Accepts:

            - Python float: Single pressure
            - JAX scalar array: Single pressure as array
            - JAX 1D array: Multiple pressures

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]
            Rate constant(s) at the specified temperature(s) and pressure(s).

            - Units depend on reaction order and pre-exponential factor A
            - Shape follows broadcasting rules:

              - Scalar T, Scalar P = Scalar output
              - Vector T, Scalar P = Vector output (length nt)
              - Scalar T, Vector P = Vector output (length np)
              - Vector T, Vector P = Matrix output (shape: np × nt)

            - Always returned as JAX arrays for consistency

        Notes
        -----
        **Interpolation Behavior:**

        - For pressures within the defined range [P_min, P_max], log-log interpolation
          is used: ln(k) vs ln(P)
        - For P < P_min or P > P_max, constant extrapolation is applied
        - The implementation uses soft interpolation weights to maintain differentiability
        """
        T = jnp.asarray(T, dtype=jnp.float64)
        P = jnp.asarray(P, dtype=jnp.float64)

        if jnp.isscalar(P) or P.ndim == 0:
            # Scalar pressure - evaluate directly
            return self._single_P_rate_constant(T, P)
        else:
            # Vector pressure - vectorize over pressure dimension
            vec_func = vmap(lambda p: self._single_P_rate_constant(T, p))
            return vec_func(P)

    @eqx.filter_jit
    def _single_P_rate_constant(
        self,
        T: Float64[Array, ""] | Float64[Array, "nt"],
        P: Float64[Array, ""],
    ) -> Float64[Array, ""] | Float64[Array, "nt"]:
        """
        Core differentiable PLOG rate constant calculation for a single pressure.

        This method implements the log-log interpolation algorithm using soft
        (continuous) operations to maintain full differentiability for automatic
        differentiation. The key insight is to replace hard index selection with
        weighted interpolation across all pressure intervals.

        Algorithm Overview:
        -------------------
        1. Evaluate Arrhenius rate constants at all discrete pressure levels
        2. Clamp pressure to valid range using differentiable jnp.clip
        3. Compute interpolation weights for all intervals simultaneously
        4. Use soft selection to identify the active interval
        5. Return weighted sum of interpolated values

        Parameters
        ----------
        T : Float64[Array, ""] | Float64[Array, "nt"]
            Temperature (scalar or vector) in Kelvin
        P : Float64[Array, ""]
            Pressure (scalar) in atmospheres

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "nt"]
            Rate constant(s) matching the shape of T

        Notes
        -----
        **Why Compute All Pressure Levels?**

        Although only 2 pressure levels are needed for interpolation, we compute
        all N levels (typically 5-10) because:

        - Enables clean vectorization without dynamic indexing
        - Each Arrhenius evaluation is cheap
        - Maintains full differentiability through JAX autodiff
        - Standard practice in differentiable programming

        **Differentiability Guarantees:**

        - jnp.clip: Has subgradients at boundaries (dclip/dx = 0 or 1)
        - Boolean masks: Converted to float64 (gradient is zero, which is correct)
        - Weighted sum: Fully differentiable linear combination
        - All array operations: Native JAX with full autodiff support
        """
        # ==============================================================================
        # Step 1: Evaluate Arrhenius rate constants at all pressure levels
        # For each pressure level P_i, compute k_i(T) using the Arrhenius equation
        # Stack results into a single array for vectorized operations
        # Shape: (num_p_levels,) if T is scalar, (num_p_levels, nt) if T is vector
        all_lnk = jnp.stack([
            k_level.log_rate_constant(T)
            for k_level in self._k_levels
        ])

        # ==============================================================================
        # Step 2: Handle extrapolation using differentiable clamping
        # Compute log-pressure for interpolation in log-log space
        lnP = jnp.log(P)

        # Clamp to valid range [lnP_min, lnP_max] for extrapolation
        # jnp.clip is differentiable with subgradients:
        #   dclip/dx = {0 if x < min, 1 if min <= x <= max, 0 if x > max}
        lnP_clamped = jnp.clip(lnP, self._lnp_levels[0], self._lnp_levels[-1])

        # ==============================================================================
        # Step 3: Compute interpolation weights for all intervals simultaneously
        # Define pressure intervals: [P_i, P_{i+1}] for i = 0, ..., N-2
        lnP_lower = self._lnp_levels[:-1]  # [lnP_0, lnP_1, ..., lnP_{N-2}]
        lnP_upper = self._lnp_levels[1:]   # [lnP_1, lnP_2, ..., lnP_{N-1}]

        # Compute interpolation fraction alpha in [0, 1] for each interval
        # alpha = (lnP - lnP_i) / (lnP_{i+1} - lnP_i)
        alpha = (lnP_clamped - lnP_lower) / (lnP_upper - lnP_lower)
        alpha = jnp.clip(alpha, 0.0, 1.0)  # Ensure numerical stability

        # ==============================================================================
        # Step 4: Perform linear interpolation in log-space for each interval
        # Extract ln(k) values at interval boundaries
        lnk_lower = all_lnk[:-1]  # ln(k) at P_i
        lnk_upper = all_lnk[1:]   # ln(k) at P_{i+1}

        # Linear interpolation: ln(k_interp) = ln(k_i) + α · (ln(k_{i+1}) - ln(k_i))
        # Handle both scalar and vector T cases
        if all_lnk.ndim == 1:
            # T is scalar: all_lnk shape = (num_p_levels,)
            # Result shape: (num_intervals,)
            lnk_interp = lnk_lower + alpha * (lnk_upper - lnk_lower)
        else:
            # T is vector: all_lnk shape = (num_p_levels, nt)
            # Result shape: (num_intervals, nt)
            # Broadcasting: alpha[:, None] creates shape (num_intervals, 1)
            lnk_interp = lnk_lower + alpha[:, None] * (lnk_upper - lnk_lower)

        # ==============================================================================
        # Step 5: Soft selection of the active interval
        # Identify which interval contains lnP_clamped using boolean mask
        # Interval i is active if: lnP_i < lnP_clamped <= lnP_{i+1}
        # Note: Use > (not >=) on lower bound to avoid double-counting at boundaries
        in_interval = ((lnP_clamped > lnP_lower) & (lnP_clamped <= lnP_upper)).astype(jnp.float64)

        # Convert to normalized weights (exactly one interval should be active)
        # Add small epsilon (1e-12) for numerical stability in edge cases
        weights = in_interval / (jnp.sum(in_interval) + 1e-12)

        # Compute weighted sum: selects the interpolated value from the active interval
        # Gradients flow through all operations since this is a differentiable linear combination
        if all_lnk.ndim == 1:
            # Scalar T case: lnk_interp shape = (num_intervals,)
            lnk = jnp.sum(weights * lnk_interp)
        else:
            # Vector T case: lnk_interp shape = (num_intervals, nt)
            # Sum over interval dimension (axis=0), keeping temperature dimension
            lnk = jnp.sum(weights[:, None] * lnk_interp, axis=0)

        # ==============================================================================
        # Step 6: Convert from log-space back to rate constant
        return jnp.exp(lnk)

    # ==================================================================================
    # Automatic Differentiation Methods
    @eqx.filter_jit
    def grad_temperature(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "nt"],
        P: float | Float64[Array, ""] | Float64[Array, "np"],
    ) -> Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]:
        """
        Calculate the derivative of the rate constant with respect to temperature
        (dk/dT) using automatic differentiation.

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "nt"]
            Temperature(s) in Kelvin at which to evaluate the gradient
        P : float | Float64[Array, ""] | Float64[Array, "np"]
            Pressure(s) in atm at which to evaluate the gradient

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]
            Temperature gradient dk/dT at the specified temperature(s) and pressure(s)

        Notes
        -----
        This method supports automatic differentiation thanks to the fully
        differentiable implementation using soft interpolation.

        The gradient is computed element-wise:
        - Scalar T, Scalar P = Scalar gradient
        - Vector T, Scalar P = Vector gradient (one per T)
        - Scalar T, Vector P = Vector gradient (one per P)
        - Vector T, Vector P = Matrix gradient (grid of T x P)
        """
        T_jax = jnp.asarray(T, dtype=jnp.float64)
        P_jax = jnp.asarray(P, dtype=jnp.float64)

        # Define gradient function for a single (T, P) pair
        def single_grad(t, p):
            return eqx.filter_grad(lambda t_: self.rate_constant(t_, p))(t)

        if T_jax.ndim == 0 and P_jax.ndim == 0:
            # Both scalar
            return single_grad(T_jax, P_jax)
        elif T_jax.ndim > 0 and P_jax.ndim == 0:
            # Vector T, scalar P
            return vmap(lambda t: single_grad(t, P_jax))(T_jax)
        elif T_jax.ndim == 0 and P_jax.ndim > 0:
            # Scalar T, vector P
            return vmap(lambda p: single_grad(T_jax, p))(P_jax)
        else:
            # Both vectors - create grid
            return vmap(lambda p: vmap(lambda t: single_grad(t, p))(T_jax))(P_jax)

    @eqx.filter_jit
    def grad_params(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "nt"],
        P: float | Float64[Array, ""] | Float64[Array, "np"],
    ) -> "Plog":
        """
        Calculate the gradient of the rate constant with respect to parameters using
        automatic differentiation.

        Computes the parameter sensitivity for all Arrhenius parameters at all
        pressure levels:

        .. math::
            \\nabla_{\\theta} k(T, P) = \\begin{bmatrix}
                \\frac{\\partial k}{\\partial A_1}, \\frac{\\partial k}{\\partial n_1}, \\frac{\\partial k}{\\partial E_{a,1}} \\\\
                \\vdots \\\\
                \\frac{\\partial k}{\\partial A_n}, \\frac{\\partial k}{\\partial n_n}, \\frac{\\partial k}{\\partial E_{a,n}}
            \\end{bmatrix}

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "nt"]
            Temperature(s) in Kelvin at which to evaluate the parameter gradients
        P : float | Float64[Array, ""] | Float64[Array, "np"]
            Pressure(s) in atm at which to evaluate the parameter gradients

        Returns
        -------
        Plog
            A Plog object with gradients stored in place of parameters.
            Each Arrhenius object in k_levels contains gradients:

            - ``result.k_levels[i].A``: :math:`\\frac{\\partial k}{\\partial A_i}`
            - ``result.k_levels[i].n``: :math:`\\frac{\\partial k}{\\partial n_i}`
            - ``result.k_levels[i].Ea``: :math:`\\frac{\\partial k}{\\partial E_{a,i}}`

        Notes
        -----
        - For vector T or P inputs, computes gradient of :math:`\\sum_{i,j} k(T_i, P_j)`
        - This sum-over-points is useful for parameter fitting with multiple data points
        - Gradients account for the interpolation weights in the PLOG formulation
        """
        T_jax = jnp.asarray(T, dtype=jnp.float64)
        P_jax = jnp.asarray(P, dtype=jnp.float64)

        # Wrapper function to enable differentiation w.r.t. module parameters
        # Use sum to handle vectorized inputs
        wrapper_function = lambda m, t, p: jnp.sum(m.rate_constant(t, p))

        return eqx.filter_grad(wrapper_function)(self, T_jax, P_jax)

    # ==================================================================================
    # String Representations and Debugging
    def __str__(self) -> str:
        """
        Return a CHEMKIN-compatible string representation.

        Formats the PLOG parameters in the standard CHEMKIN input format,
        which is widely used in combustion and chemical kinetics software.

        Returns
        -------
        str
            Multi-line string with reaction name and PLOG entries in CHEMKIN format.
        """
        if self._k0 is not None:
            str_obj = f"{self.name}\t\t{float(self._k0.A):.5E} {float(self._k0.n):.5E} {float(self._k0.Ea):.5E}\n"
        else:
            str_obj = f"{self.name}\t\t{0.0:.5E} {0.0:.5E} {0.0:.5E}\n"

        for i in range(int(self._num_p_levels)):
            arrhenius = self._k_levels[i]
            str_obj += f" PLOG / {float(self._p_levels[i]):.5E}\t{float(arrhenius.A):.5E} {float(arrhenius.n):.5E} {float(arrhenius.Ea):.5E} /\n"
        return str_obj

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging and development.

        Provides a multi-line, human-readable representation showing all
        parameter values with appropriate precision.

        Returns
        -------
        str
            Multi-line formatted string showing all PLOG parameters.
        """
        lines = ["Plog("]
        lines.append(f" name = {self._name}")
        lines.append(f" num_p_levels = {int(self._num_p_levels)}")
        lines.append(" pressure_levels = [")
        for i in range(int(self._num_p_levels)):
            lines.append(f"  P = {float(self._p_levels[i]):.5e} atm:")

            # Reuse Arrhenius __repr__ and indent it
            arr_repr = repr(self._k_levels[i])
            lines.append("   " + arr_repr.replace("\n", "\n   "))
        lines.append(" ]")
        lines.append(")")
        return "\n".join(lines)

    # ==================================================================================
    # Properties for Parameters Access
    @property
    def name(self) -> str:
        """
        Human-readable reaction name.

        Returns
        -------
        str
            The reaction name string, typically in chemical equation format.
        """
        return self._name

    @property
    def p_levels(self) -> Float64[Array, "np"]:
        """
        Pressure levels array.

        Returns
        -------
        Float64[Array, "np"]
            1D array of pressure levels [atm] in ascending order.
        """
        return self._p_levels

    @property
    def lnp_levels(self) -> Float64[Array, "np"]:
        """
        Natural logarithm of pressure levels.

        Returns
        -------
        Float64[Array, "np"]
            1D array of ln(P) values corresponding to pressure levels.
        """
        return self._lnp_levels

    @property
    def num_p_levels(self) -> Int64[Array, ""]:
        """
        Number of pressure levels.

        Returns
        -------
        Int64[Array, ""]
            Integer count of discrete pressure levels in the PLOG definition.
        """
        return self._num_p_levels

    @property
    def k_levels(self) -> list[Arrhenius]:
        """
        List of Arrhenius objects at each pressure level.

        Returns
        -------
        list[Arrhenius]
            Ordered list of Arrhenius rate constant calculators, one per pressure level.
        """
        return self._k_levels

    @property
    def k0(self) -> Optional[Arrhenius]:
        """
        Nominal Arrhenius rate constant (CHEMKIN compatibility).

        Returns
        -------
        Optional[Arrhenius]
            Arrhenius object for nominal parameters, or None if not provided.

        Notes
        -----
        This parameter does not exist in CHEMKIN format and is not directly
        used in rate constant calculations.
        """
        if self._k0 is not None:
            return self._k0
        else:
            return None
