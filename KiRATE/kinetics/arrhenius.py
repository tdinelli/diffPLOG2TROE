"""
Copyright (c) 2025 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import re
from typing import Dict, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float64

from KiRATE.kinetics.utils import validate_arrhenius_parameters
from KiRATE.utilities.physical_constants import constants


class Arrhenius(eqx.Module):
    """
    Arrhenius rate constant calculator for chemical reactions.

    This class implements the Arrhenius equation to calculate temperature-dependent
    rate constants for chemical reactions. The Arrhenius equation is given by:

    .. math::
        k(T) = A \\cdot T^n \\cdot \\exp\\left(\\frac{-Ea}{R \\cdot T}\\right)

    where:
        - k(T) is the rate constant at temperature T [units vary with reaction order]
        - A is the pre-exponential factor (frequency factor) [units vary with reaction order]
        - T is the absolute temperature [K]
        - n is the temperature exponent [dimensionless]
        - Ea is the activation energy [cal/mol]
        - R is the universal gas constant [cal/mol/K]

    This implementation supports both standard Arrhenius (n=0) and modified Arrhenius
    (n!=0) forms, with automatic differentiation for sensitivity analysis.

    Parameters
    ----------
    parameters : Dict[str, float]
        Dictionary containing Arrhenius parameters with keys:

        - "A": Pre-exponential factor (units depend on reaction order)
        - "n": Temperature exponent (dimensionless)
        - "Ea": Activation energy (cal/mol)
    name : str, optional
        Human-readable name for the reaction, by default ""

    Attributes
    ----------
    _A : Float64[Array, ""]
        Pre-exponential factor stored as JAX scalar
    _n : Float64[Array, ""]
        Temperature exponent stored as JAX scalar
    _Ea : Float64[Array, ""]
        Activation energy stored as JAX scalar
    _name : str
        Reaction name for identification (static field)
    """

    _A: Float64[Array, ""]
    _n: Float64[Array, ""]
    _Ea: Float64[Array, ""]
    _name: str = eqx.field(static=True, default="")

    def __init__(self, parameters: Dict[str, float], name: str = "") -> None:
        """
        Initialize the Arrhenius rate constant calculator.

        Parameters
        ----------
        parameters : Dict[str, float]
            Dictionary containing Arrhenius parameters:
            - "A": Pre-exponential factor (units depend on reaction order)
            - "n": Temperature exponent (dimensionless)
            - "Ea": Activation energy (cal/mol)
        name : str, optional
            Human-readable name for the reaction, by default ""

        Raises
        ------
        ValueError
            If required parameters are missing or contain invalid values
        UserWarning
            If parameter values are outside typical ranges (validation warnings)

        Examples
        --------
        >>> # Standard Arrhenius (n=0)
        >>> params = {"A": 1e13, "n": 0.0, "Ea": 15000}
        >>> standard_arrhenius = Arrhenius(params, name="Standard Arrhenius")

        >>> # Modified Arrhenius (n!=0)
        >>> params = {"A": 1e12, "n": 0.5, "Ea": 10000}
        >>> mod_arrhenius = Arrhenius(params, name="Modified Arrhenius")
        """
        # ==============================================================================
        # Validate input parameters
        eqx.filter_pure_callback(
            validate_arrhenius_parameters,
            parameters,
            result_shape_dtypes=None,
        )

        # ==============================================================================
        # Store reaction name
        self._name = name

        # ==============================================================================
        # Pre-exponential factor
        self._A = jnp.float64(parameters["A"])

        # ==============================================================================
        # Temperature exponent
        self._n = jnp.float64(parameters["n"])

        # ==============================================================================
        # Activation energy
        self._Ea = jnp.float64(parameters["Ea"])

    @classmethod
    def from_chemkin(cls, input_string: str) -> "Arrhenius":
        """
        Create an Arrhenius instance from a CHEMKIN format string.

        This class method provides a convenient way to construct Arrhenius objects
        directly from CHEMKIN-style input strings, which are commonly used in
        chemical kinetics databases and modeling software.

        Parameters
        ----------
        input_string : str
            CHEMKIN-formatted string containing reaction name and parameters.

        Returns
        -------
        Arrhenius
            New Arrhenius instance with parsed parameters and reaction name.

        Raises
        ------
        ValueError
            If the input string cannot be parsed or contains invalid parameters.

        Examples
        --------
        Basic usage:

        >>> chemkin_str = "H+O2=O+OH    1.14e+14    0.000    15286.00"
        >>> reaction = Arrhenius.from_chemkin(chemkin_str)
        >>> print(reaction.name)  # "H+O2=O+OH"
        >>> print(reaction.A)     # 1.14e+14

        See Also
        --------
        parse_chemkin_entry : Static method used internally for parsing
        """
        name, params = cls.parse_chemkin_entry(input_string)
        return cls(name=name, parameters=params)

    @eqx.filter_jit
    def rate_constant(
        self,
        T: Union[float, Float64[Array, ""], Float64[Array, "n"]],
    ) -> Union[Float64[Array, ""], Float64[Array, "n"]]:
        """
        Calculate the Arrhenius rate constant at given temperature(s).

        This method implements the core Arrhenius equation with automatic
        broadcasting for vector inputs.

        Parameters
        ----------
        T : Union[float, Float64[Array, ""], Float64[Array, "n"]]
            Temperature(s) in Kelvin. Accepts:

            - Python float: Single temperature
            - JAX scalar array: Single temperature as array
            - JAX 1D array: Multiple temperatures

            Note: Python floats are automatically promoted by JAX, but explicit
            array conversion may be needed for gradient operations.

        Returns
        -------
        Union[Float64[Array, ""], Float64[Array, "n"]]
            Rate constant(s) at the specified temperature(s).

            - Units depend on reaction order and pre-exponential factor A
            - Shape matches input temperature array
            - Always returned as JAX arrays for consistency

        Notes
        -----
        - This method is JIT-compiled for optimal performance
        - Supports automatic differentiation w.r.t. both T and parameters
        - Broadcasting rules follow standard JAX conventions
        - For very high/low temperatures, numerical overflow/underflow may occur

        Examples
        --------
        Single temperature:

        >>> import jax.numpy as jnp
        >>> params = {"A": 1.0e13, "n": 0.0, "Ea": 15000.0}
        >>> reaction = Arrhenius(params, name="H + O2 -> OH + O")
        >>> T_single = 1000.0
        >>> k = reaction.rate_constant(T_single)
        >>> print(f"k = {k:.3e}")

        Multiple temperatures:

        >>> T_array = jnp.array([800.0, 1000.0, 1200.0])
        >>> k_array = reaction.rate_constant(T_array)
        >>> print(f"k = {k_array}")
        """
        T = jnp.asarray(T, dtype=jnp.float64)
        return self._A * jnp.power(T, self._n) * jnp.exp(-self._Ea / constants.R_cal_mol / T)

    def convert_to_standard_arrhenius(self) -> Dict:
        """
        Convert a three-parameter Arrhenius model to standard two-parameter form.

        This method fits the current three-parameter model (A, n, Ea) to a
        two-parameter model (A', 0, Ea') using least squares regression over
        a temperature range. Useful for compatibility with systems that only
        support standard Arrhenius form.

        Returns
        -------
        Dict
            Comprehensive dictionary containing:

            - "refitted_arrhenius": New Arrhenius instance with n=0
            - "fit_statistics": Dictionary with fitting diagnostics:
                - "residuals": Sum of squared residuals from least squares fit
                - "rank": Rank of the design matrix
                - "singular_values": Singular values from SVD decomposition
                - "condition_number": Matrix condition number (numerical stability)
                - "r_squared": Coefficient of determination (goodness of fit)

        Raises
        ------
        ValueError
            If the current model already has n≈0 (within numerical tolerance)

        Notes
        -----
        - Fitting is performed over temperature range [298, 3000] K with 300 points
        - Uses linear regression in ln(k) vs 1/T space (Arrhenius linearization)
        - R^2 values > 0.99 typically indicate good fit quality
        - Large condition numbers (>1e12) may indicate numerical issues
        - This method is still under active development for robustness improvements

        Examples
        --------
        Convert modified Arrhenius to standard form:

        >>> # Original 3-parameter model
        >>> params_3p = {"A": 1e13, "n": 0.5, "Ea": 15000}
        >>> arrh_3p = Arrhenius(params_3p, name="Original")

        >>> # Convert to 2-parameter model
        >>> result = arrh_3p.convert_to_standard_arrhenius()
        >>> arrh_2p = result["refitted_arrhenius"]
        >>> stats = result["fit_statistics"]

        >>> print(f"Original: A={arrh_3p.A:.2e}, n={arrh_3p.n:.3f}, Ea={arrh_3p.Ea:.0f}")
        >>> print(f"Refitted: A={arrh_2p.A:.2e}, n={arrh_2p.n:.3f}, Ea={arrh_2p.Ea:.0f}")
        >>> print(f"Fit quality R² = {stats['r_squared']:.6f}")
        """
        # ==============================================================================
        # Check if conversion is needed
        if jnp.isclose(self._n, 0.0):
            raise ValueError(f"Model already in standard Arrhenius form (n = {self._n:.5f}). Conversion is not needed!")

        # ==============================================================================
        # Generate temperature points for refitting
        T = jnp.linspace(298.0, 3000.0, 300)

        # ==============================================================================
        # Calculate rate constants using current model
        k_original = self.rate_constant(T)
        log_k = jnp.log(k_original)
        inv_T = 1.0 / T

        # ==============================================================================
        # Set up design matrix for 2-parameter fit (n=0)
        X = jnp.vstack([jnp.ones_like(inv_T), -inv_T]).T

        # ==============================================================================
        # Perform least squares regression
        beta, residuals, rank, singular_values = jnp.linalg.lstsq(X, log_k, rcond=None)

        # ==============================================================================
        # Extract fitted parameters: ln(A) and Ea/R
        refitted_A = jnp.exp(beta[0])
        refitted_Ea = beta[1] * constants.R_cal_mol

        # ==============================================================================
        # Create new Arrhenius instance
        refitted_arrhenius = Arrhenius(
            parameters={"A": float(refitted_A), "n": 0.0, "Ea": float(refitted_Ea)},
            name=f"{self._name} (refitted)",
        )

        # ==============================================================================
        # Calculate R2 for fit quality assessment this is done in the log space
        # otherwise this does not make any sense
        k_refitted = refitted_arrhenius.rate_constant(T)
        log_k_refitted = jnp.log(k_refitted)
        ss_res_log = jnp.sum((log_k - log_k_refitted) ** 2)
        ss_tot_log = jnp.sum((log_k - jnp.mean(log_k)) ** 2)
        r_squared = 1 - (ss_res_log / ss_tot_log)

        # ==============================================================================
        # Comprehensive result dictionary
        result = {
            "refitted_arrhenius": refitted_arrhenius,
            "fit_statistics": {
                "residuals": float(residuals[0]) if len(residuals) > 0 else None,
                "rank": int(rank),
                "singular_values": [float(sv) for sv in singular_values],
                "condition_number": float(singular_values[0] / singular_values[-1]),
                "r_squared": float(r_squared),
            },
        }

        return result

    # ==================================================================================
    # Automatic Differentiation Methods
    @eqx.filter_jit
    def grad_temperature(
        self,
        T: Union[float, Float64[Array, ""], Float64[Array, "n"]],
    ) -> Union[Float64[Array, ""], Float64[Array, "n"]]:
        """
        Calculate the temperature gradient dk/dT using automatic differentiation.

        Computes the derivative of the rate constant with respect to temperature,
        which is useful for sensitivity analysis, temperature optimization, and
        understanding reaction temperature dependence.

        Parameters
        ----------
        T : Union[float, Float64[Array, ""], Float64[Array, "n"]]
            Temperature(s) in Kelvin at which to evaluate the gradient.

        Returns
        -------
        Union[Float64[Array, ""], Float64[Array, "n"]]
            Temperature gradient dk/dT at the specified temperature(s).

            - Units: [rate_constant_units]/K
            - Shape matches input temperature array
            - Positive values indicate rate increases with temperature
            - Negative values indicate rate decreases with temperature (rare)

        Notes
        -----
        - For analytical formula, use: dk/dT = k(T) * (n/T - Ea/(RT²))

        Examples
        --------
        Temperature sensitivity at a single point:

        >>> import jax.numpy as jnp
        >>> params = {"A": 1.0e13, "n": 0.0, "Ea": 15000.0}
        >>> reaction = Arrhenius(params, name="H + O2 -> OH + O")
        >>> T = 1000.0
        >>> dk_dT = reaction.grad_temperature(T)
        >>> k = reaction.rate_constant(T)
        >>> relative_sensitivity = dk_dT * T / k
        >>> print(f"Rate changes by {relative_sensitivity:.2f} per 1% temperature increase")

        Temperature sensitivity over a range:

        >>> T_range = jnp.linspace(500, 2000, 50)
        >>> sensitivities = reaction.grad_temperature(T_range)
        >>> max_sensitivity_idx = jnp.argmax(jnp.abs(sensitivities))
        >>> print(f"Maximum sensitivity at T = {T_range[max_sensitivity_idx]:.0f} K")
        """
        T_jax = jnp.asarray(T, dtype=jnp.float64)  # Ensure differentiability with eqx.filter_grad

        if T_jax.ndim == 0:
            return eqx.filter_grad(self.rate_constant)(T_jax)
        else:
            return vmap(lambda t: eqx.filter_grad(self.rate_constant)(t))(T_jax)

    @eqx.filter_jit
    def grad_params(
        self,
        T: Union[float, Float64[Array, ""], Float64[Array, "n"]],
    ) -> "Arrhenius":
        """
        Calculate parameter gradients dk/dθ using automatic differentiation.

        Computes the derivatives of the rate constant with respect to all
        Arrhenius parameters (A, n, Ea). Essential for parameter estimation,
        uncertainty quantification, and sensitivity analysis.

        Parameters
        ----------
        T : Union[float, Float64[Array, ""], Float64[Array, "n"]]
            Temperature(s) in Kelvin at which to evaluate the parameter gradients.

        Returns
        -------
        Arrhenius
            Arrhenius object containing gradients in place of parameters:

            - result.A: dk/dA (sensitivity to pre-exponential factor)
            - result.n: dk/dn (sensitivity to temperature exponent)
            - result.Ea: dk/dEa (sensitivity to activation energy)

            Access via properties: result.A, result.n, result.Ea

        Examples
        --------
        Basic parameter sensitivity:

        >>> T = 1000.0
        >>> param_grads = reaction.grad_params(T)
        >>> print(f"dk/dA = {param_grads.A:.2e}")
        >>> print(f"dk/dn = {param_grads.n:.2e}")
        >>> print(f"dk/dEa = {param_grads.Ea:.2e}")

        Relative parameter sensitivities:

        >>> k = reaction.rate_constant(T)
        >>> rel_sens_A = param_grads.A * arrh.A / k
        >>> rel_sens_n = param_grads.n * arrh.n / k
        >>> rel_sens_Ea = param_grads.Ea * arrh.Ea / k
        >>> print(f"Most sensitive parameter: A" if abs(rel_sens_A) > max(abs(rel_sens_n), abs(rel_sens_Ea)) else "n" if abs(rel_sens_n) > abs(rel_sens_Ea) else "Ea")

        Parameter uncertainty propagation:

        >>> # Assuming parameter uncertainties: σ_A, σ_n, σ_Ea
        >>> sigma_A, sigma_n, sigma_Ea = 1e11, 0.1, 500  # example uncertainties
        >>> sigma_k_squared = (param_grads.A * sigma_A)**2 + (param_grads.n * sigma_n)**2 + (param_grads.Ea * sigma_Ea)**2
        >>> sigma_k = jnp.sqrt(sigma_k_squared)
        >>> print(f"Rate constant uncertainty: ±{sigma_k:.2e}")
        """
        T_jax = jnp.asarray(T, dtype=jnp.float64)
        wrapper_function = lambda m, t: m.rate_constant(t)
        if T_jax.ndim == 0:
            return eqx.filter_grad(wrapper_function)(self, T_jax)
        else:
            return vmap(lambda t: eqx.filter_grad(wrapper_function)(self, t))(T_jax)

    # ==================================================================================
    # CHEMKIN string parser
    @staticmethod
    def parse_chemkin_entry(input_string: str) -> Tuple[str, Dict[str, float]]:
        """
        Parse a CHEMKIN-formatted string into reaction name and parameters.

        This static method extracts reaction names and Arrhenius parameters from
        CHEMKIN-style input strings. It handles various formatting conventions
        and whitespace variations commonly found in kinetics databases.

        Parameters
        ----------
        input_string : str
            CHEMKIN-formatted string containing reaction and parameters.
            Expected format: "reaction_name A n Ea"

            The reaction name can contain various operators and spacing:
            - Equality operators: =, =>, <=>
            - Species separation: +

        Returns
        -------
        Tuple[str, Dict[str, float]]
            A tuple containing:
            - reaction_name (str): Normalized reaction name
            - parameters (Dict[str, float]): Dictionary with keys "A", "n", "Ea"

        Raises
        ------
        ValueError
            If the input string is empty, malformed, or contains insufficient data.
            Specifically raised when:
            - Empty or whitespace-only input
            - Fewer than 4 parts (name + 3 parameters) found
            - Numerical parameters cannot be parsed as floats

        See Also
        --------
        from_chemkin : Class method that uses this parser to create Arrhenius instances
        """
        # Strip whitespace and split into lines
        lines = [line.strip() for line in input_string.strip().split("\n") if line.strip()]

        if not lines:
            raise ValueError("Empty chemkin representation")

        # Take the first non-empty line
        line = lines[0]

        # Split by whitespace to get all parts
        parts = line.split()

        if len(parts) < 4:
            raise ValueError("Not enough parts found. Expected reaction name and 3 parameters.")

        # The last 3 parts should be the numerical parameters
        try:
            A = float(parts[-3])
            n = float(parts[-2])
            Ea = float(parts[-1])
        except ValueError:
            raise ValueError("Could not parse numerical parameters")

        # Everything except the last 3 parts is the reaction name
        reaction_parts = parts[:-3]
        reaction_name = " ".join(reaction_parts)

        # Clean up the reaction name by normalizing whitespace around operators
        # Handle =, =>, <=> operators - preserve them as-is
        reaction_name = re.sub(r"\s*(<=>|=>|=)\s*", r"\1", reaction_name)

        # Normalize spaces around + signs
        reaction_name = re.sub(r"\s*\+\s*", "+", reaction_name)

        return (reaction_name, {"A": A, "n": n, "Ea": Ea})

    # ==================================================================================
    # String Representations and Debugging
    def __str__(self) -> str:
        """
        Return a CHEMKIN-compatible string representation.

        Formats the Arrhenius parameters in the standard CHEMKIN input format,
        which is widely used in combustion and chemical kinetics software.

        Returns
        -------
        str
            Tab-separated string with reaction name and parameters:
            Format: "{name}\\t\\t{A:.5e} {n:.5f} {Ea:.5e}"

        Examples
        --------
        >>> params = {"A": 1e13, "n": 0.5, "Ea": 15000}
        >>> reaction = Arrhenius(params, name="H + O2 => OH + O")
        >>> print(str(reaction))
        H + O2 => OH + O        1.00000e+13 0.50000 1.50000e+04
        """
        return f"{self._name}\t\t{float(self._A):.5E} {float(self._n):.5E} {float(self._Ea):.5E}"

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging and development.

        Provides a multi-line, human-readable representation showing all
        parameter values with appropriate precision.

        Returns
        -------
        str
            Multi-line formatted string showing all Arrhenius parameters.

        Examples
        --------
        >>> params = {"A": 1e13, "n": 0.5, "Ea": 15000}
        >>> reaction = Arrhenius(params, name="H + O2 => OH + O")
        >>> print(repr(reaction))
        Arrhenius(
         name = H + O2 -> OH + O
         A    = 1.000e+13
         n    = 0.500
         Ea   = 1.500e+04
        )

        >>> # Useful in Jupyter notebooks and debugging sessions
        >>> reaction # This calls __repr__ automatically
        """
        return f"Arrhenius(\n name = {self._name}\n A    = {float(self._A):.5e}\n n    = {float(self._n):.5e}\n Ea   = {float(self._Ea):.5e}\n)"

    # ==================================================================================
    # Properties for parameters access
    @property
    def A(self) -> Float64[Array, ""]:
        """
        Pre-exponential factor (frequency factor).

        Returns
        -------
        Float64[Array, ""]
            Pre-exponential factor A from the Arrhenius equation. Units depend
            on the reaction order.

        Examples
        --------
        >>> reaction.A
        Array(1.e+13, dtype=float64)

        >>> # Convert to float for display
        >>> float(reaction.A)
        1e13
        """
        return self._A

    @property
    def n(self) -> Float64[Array, ""]:
        """
        Temperature exponent (modified Arrhenius parameter).

        Returns
        -------
        Float64[Array, ""]
            Temperature exponent n from the Arrhenius equation (dimensionless).

        Examples
        --------
        >>> reaction.n
        Array(0.5, dtype=float64)
        """
        return self._n

    @property
    def Ea(self) -> Float64[Array, ""]:
        """
        Activation energy.

        Returns
        -------
        Float64[Array, ""]
            Activation energy Ea in cal/mol.

        Examples
        --------
        >>> reaction.Ea
        Array(15000., dtype=float64)
        """
        return self._Ea

    @property
    def name(self) -> str:
        """
        Human-readable reaction name.

        Returns
        -------
        str
            The reaction name string, typically in chemical equation format.

        Notes
        -----
        - Static field that doesn't participate in JAX transformations
        - Used for identification and output formatting
        - Can contain operators: =, =>, <=> for different reaction types
        - Empty string by default if not specified during initialization

        Examples
        --------
        >>> reaction.name
        'H+O2=O+OH'

        >>> # Can be modified after creation
        >>> reaction_copy = eqx.tree_at(lambda x: x._name, reaction, "New Name")
        >>> reaction_copy.name
        'New Name'
        """
        return self._name

    @property
    def is_three_params(self) -> bool:
        """
        Check if this is a three-parameter (modified) Arrhenius model.

        Returns
        -------
        bool
            True if n != 0 (three-parameter/modified Arrhenius form).
            False if n == 0 (two-parameter/standard Arrhenius form).

        Examples
        --------
        Standard Arrhenius (n=0):

        >>> params = {"A": 1e13, "n": 0.0, "Ea": 15000}
        >>> reaction = Arrhenius(params)
        >>> reaction.is_three_params
        False

        Modified Arrhenius (n!=0):

        >>> params = {"A": 1e13, "n": 0.5, "Ea": 15000}
        >>> reaction = Arrhenius(params)
        >>> reaction.is_three_params
        True
        """
        return not jnp.isclose(self._n, 0.0, atol=1e-13)
