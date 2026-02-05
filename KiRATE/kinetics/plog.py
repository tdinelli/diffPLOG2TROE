"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

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
    parameters : dict[float, dict[str, float] | list[dict[str, float]]]
        Dictionary mapping pressure levels [atm] to Arrhenius parameters:

        .. code-block:: python

            {
                0.01: {"A": 1e12, "n": 0.0, "Ea": 10000.0},
                1.0:  {"A": 1e13, "n": 0.5, "Ea": 12000.0},
                10.0: {"A": 1e14, "n": 1.0, "Ea": 15000.0}
            }

        For non-Arrhenius behavior, multiple Arrhenius expressions can be summed:

        .. code-block:: python

            {
                0.01: [
                    {"A": 1.44e14, "n": -0.93, "Ea": 1700.0},
                    {"A": 4.07e15, "n": -6.73, "Ea": -14031.0}
                ],
                1.0: [
                    {"A": 1.38e17, "n": -1.64, "Ea": 4750.0},
                    {"A": 6.20e13, "n": -0.78, "Ea": 3522.0}
                ]
            }

    name : str, optional
        Human-readable name for the reaction, by default ""
    k0_parameters : dict[str, float], optional
        Low pressure limit rate constant this is needed only for the Mixture
        Rule treatment and is not a standard in CHEMKIN or elsewhere.

    Attributes
    ----------
    _arrhenius_levels : list[list[Arrhenius]]
        List of Arrhenius object lists at each pressure level. Each pressure level
        can have multiple Arrhenius terms that are summed together.
    _p_levels : Float64[Array, "np"]
        Array of pressure levels [atm] in ascending order
    _lnp_levels : Float64[Array, "np"]
        Natural logarithm of pressure levels for efficient interpolation
    _num_p_levels : Int64[Array, ""]
        Number of pressure levels
    _k0 : Arrhenius | None
        Low pressure limit rate constant
    _name : str
        Reaction name for identification (static field)

    References
    ----------
    .. [1] X. Gou, J. A. Miller, W. Sun, and Y. Ju. Implementation of PLOG
           function in Chemkin II and III.
           https://engine.princeton.edu/model-reduction/, 2011.
    """

    _arrhenius_levels: list[list[Arrhenius]]
    _p_levels: Float64[Array, "np"]
    _lnp_levels: Float64[Array, "np"]
    _num_p_levels: Int64[Array, ""]
    _k0: Arrhenius | None = None
    _name: str = eqx.field(static=True, default="")

    def __init__(
        self,
        parameters: dict[float, dict[str, float] | list[dict[str, float]]],
        name: str = "",
        k0_parameters: dict[str, float] | None = None,
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
            Low pressure limit rate constant, by default None

        Raises
        ------
        ValueError
            If parameters dictionary is empty or contains invalid Arrhenius parameters

        Notes
        -----
        - Pressure levels are automatically sorted in ascending order
        - Each pressure level can have one or more Arrhenius terms
        - Multiple terms at a pressure are summed: :math:`k(T,P) = \\sum_i A_i T^{n_i} exp(-Ea_i / RT)`
        - Minimum 2 pressure levels required for interpolation
        """
        self._name = name

        # Sort pressure levels in ascending order (required for interpolation)
        parameters = dict(sorted(parameters.items()))

        # Store pressure levels and their natural logarithms
        # .keys() inherently remove the duplicate because in a dictionary you
        # cant define multiple elements with the same key this why we switched
        # to lists
        self._p_levels = jnp.array(list(parameters.keys()), dtype=jnp.float64)
        self._lnp_levels = jnp.log(self._p_levels)
        self._num_p_levels = jnp.int64(len(self._p_levels))

        # Create Arrhenius objects for each pressure level
        # Each pressure level can have multiple Arrhenius terms (stored as list)
        arrhenius_objects = []
        for p, params in parameters.items():
            # Handle both single dict and list of dicts for backward compatibility
            if isinstance(params, dict):
                # Single Arrhenius term - wrap in list for consistency
                arrhenius_list = [Arrhenius(parameters=params, name=f"{name} ({p})")]
            else:
                # Multiple Arrhenius terms to be summed
                arrhenius_list = [
                    Arrhenius(parameters=param_dict, name=f"{name} ({p}, term {i + 1})")
                    for i, param_dict in enumerate(params)
                ]
            arrhenius_objects.append(arrhenius_list)

        self._arrhenius_levels = arrhenius_objects

        # Optional low pressure limit rate constant
        if k0_parameters is not None:
            self._k0 = Arrhenius(parameters=k0_parameters, name=f"{name} (k0)")
        else:
            self._k0 = None

    @classmethod
    def from_chemkin(cls, input_string: str) -> "Plog":
        """
        Create Plog instance(s) from a CHEMKIN format string.

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

            For non-Arrhenius behavior (multiple PLOG entries at same pressure)::

                O+C10H7CH3=CH3C10H6OH  1.0e+17  -1.64  4750.0
                 PLOG / 0.01  1.44e+14  -0.93   1700.0 /
                 PLOG / 0.01  4.07e+15  -6.73  -14031.0 /
                 PLOG / 1.0   1.38e+17  -1.64   4750.0 /
                 PLOG / 1.0   6.20e+13  -0.78   3522.0 /

        Returns
        -------
        Plog
            Single Plog instance with summed Arrhenius terms at each pressure level

        Raises
        ------
        ValueError
            If the input string cannot be parsed or contains invalid parameters.
        """
        reaction_name, plog_parameters = parse_plog(input_string)

        # Parser returns dict[float, list[dict[str, float]]]
        # Pass directly to __init__ which handles both single dict and list of dicts
        return cls(parameters=plog_parameters, name=reaction_name)

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
            Temperature(s) in Kelvin.
        P : float | Float64[Array, ""] | Float64[Array, "np"]
            Pressure(s) in atmospheres.

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "nt"] | Float64[Array, "np"] | Float64[Array, "nt np"]
            Rate constant(s) at the specified temperature(s) and pressure(s).

            - Units depend on reaction order and pre-exponential factor A
            - Shape follows broadcasting rules:

              - Scalar T, Scalar P -> Scalar output
              - Vector T, Scalar P -> Vector output (length nt)
              - Scalar T, Vector P -> Vector output (length np)
              - Vector T, Vector P -> Matrix output (shape: np × nt)

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

        if jnp.isscalar(P) or P.ndim == 0:  # Scalar pressure - evaluate directly
            return self._single_P_rate_constant(T, P)
        else:  # Vector pressure - vectorize over pressure dimension
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
        # For each pressure level P_i, compute k_i(T) by summing all Arrhenius terms
        # Then take log for interpolation in log-log space
        # Shape: (num_p_levels,) if T is scalar, (num_p_levels, nt) if T is vector
        all_k = jnp.array(
            [
                jnp.sum(jnp.array([arr.rate_constant(T) for arr in arrhenius_list]), axis=0)
                for arrhenius_list in self._arrhenius_levels
            ]
        )
        all_lnk = jnp.log(all_k)

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
        lnP_upper = self._lnp_levels[1:]  # [lnP_1, lnP_2, ..., lnP_{N-1}]

        # Compute interpolation fraction alpha in [0, 1] for each interval
        # alpha = (lnP - lnP_i) / (lnP_{i+1} - lnP_i)
        alpha = (lnP_clamped - lnP_lower) / (lnP_upper - lnP_lower)
        alpha = jnp.clip(alpha, 0.0, 1.0)  # Ensure numerical stability

        # ==============================================================================
        # Step 4: Perform linear interpolation in log-space for each interval
        # Extract ln(k) values at interval boundaries
        lnk_lower = all_lnk[:-1]  # ln(k) at P_i
        lnk_upper = all_lnk[1:]  # ln(k) at P_{i+1}

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
        # Interval i is active if: lnP_i <= lnP_clamped <= lnP_{i+1}
        # Use >= on lower bound to handle exact pressure level matches
        in_interval = ((lnP_clamped >= lnP_lower) & (lnP_clamped <= lnP_upper)).astype(jnp.float64)

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
            arrhenius_list = self._arrhenius_levels[i]
            for arrhenius in arrhenius_list:
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

            # Show all Arrhenius terms at this pressure level
            arrhenius_list = self._arrhenius_levels[i]
            for j, arrhenius in enumerate(arrhenius_list):
                if len(arrhenius_list) > 1:
                    lines.append(f"   Term {j + 1}:")
                # Reuse Arrhenius __repr__ and indent it
                arr_repr = repr(arrhenius)
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
    def arrhenius_levels(self) -> list[list[Arrhenius]]:
        """
        List of Arrhenius object lists at each pressure level.

        Returns
        -------
        list[list[Arrhenius]]
            Ordered list of Arrhenius rate constant calculator lists. Each pressure
            level can have multiple Arrhenius terms that are summed together.
        """
        return self._arrhenius_levels

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
