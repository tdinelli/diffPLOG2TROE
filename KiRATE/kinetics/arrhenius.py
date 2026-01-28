"""
Copyright (c) 2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import equinox as eqx
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float64

from KiRATE.kinetics.utils import validate_arrhenius_parameters
from KiRATE.utilities import constants, parse_reaction_line


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
    (n!=0) forms, with automatic differentiation capabilities.

    Parameters
    ----------
    parameters : dict[str, float]
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
    _lnA : Float64[Array, ""]
        Natural logarithm of the pre-exponential factor stored as JAX scalar
    _EaR : Float64[Array, ""]
        Activation energy divided by the ideal gas constant stored as JAX scalar
    _name : str
        Reaction name for identification (static field)
    """

    _A: Float64[Array, ""]
    _n: Float64[Array, ""]
    _Ea: Float64[Array, ""]

    _lnA: Float64[Array, ""]
    _EaR: Float64[Array, ""]

    _name: str = eqx.field(static=True, default="")

    def __init__(self, parameters: dict[str, float], name: str = "") -> None:
        """
        Initialize the Arrhenius rate constant calculator.

        Parameters
        ----------
        parameters : dict[str, float]
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
        """
        self._name = name

        # Parameters validation
        validate_arrhenius_parameters(parameters)

        # Parameters storage
        self._A = jnp.float64(parameters["A"])
        self._n = jnp.float64(parameters["n"])
        self._Ea = jnp.float64(parameters["Ea"])

        # Additional parameters (cached transformed values)
        self._lnA = jnp.log(self._A)
        self._EaR = self._Ea / constants.R_cal_mol

    @classmethod
    def from_chemkin(cls, input_string: str) -> "Arrhenius":
        """
        Create an Arrhenius instance from a CHEMKIN format string.

        This class method provides a convenient way to construct Arrhenius objects
        directly from CHEMKIN-style input strings, which are commonly used in chemical
        kinetics databases and modeling software.

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
        """
        name, params = parse_reaction_line(input_string)

        return cls(name=name, parameters=params)

    # ==================================================================================
    # Rate constant methods
    @eqx.filter_jit
    def rate_constant(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Calculate the Arrhenius rate constant at given temperature(s).

        This method implements the core Arrhenius equation with automatic broadcasting
        for vector inputs.

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) in Kelvin. Accepts:

            - Python float: Single temperature
            - JAX scalar array: Single temperature as array
            - JAX 1D array: Multiple temperatures

            Note: Python floats are automatically promoted by JAX, but explicit
            array conversion may be needed for gradient operations.

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Rate constant(s) at the specified temperature(s).

            - Units depend on reaction order and pre-exponential factor A
            - Shape matches input temperature array
            - Always returned as JAX arrays for consistency
        """
        T = jnp.asarray(T, dtype=jnp.float64)

        return self._A * jnp.power(T, self._n) * jnp.exp(-self._Ea / constants.R_cal_mol / T)

    @eqx.filter_jit
    def log_rate_constant(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Calculate the logarithm of the Arrhenius rate constant at given temperature(s).

        This method implements the core Arrhenius equation with automatic broadcasting
        for vector inputs.

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) in Kelvin. Accepts:

            - Python float: Single temperature
            - JAX scalar array: Single temperature as array
            - JAX 1D array: Multiple temperatures

            Note: Python floats are automatically promoted by JAX, but explicit
            array conversion may be needed for gradient operations.

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Natural logarithm of the rate constant(s) at the specified temperature(s).

            - Units depend on reaction order and pre-exponential factor A
            - Shape matches input temperature array
            - Always returned as JAX arrays for consistency
        """
        T = jnp.asarray(T, dtype=jnp.float64)

        return self._lnA + self._n * jnp.log(T) - self._EaR / T

    # ==================================================================================
    # Automatic Differentiation Methods
    @eqx.filter_jit
    def grad_temperature(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Calculate the derivative of the rate constant with respect to temperature
        (dk/dT) using automatic differentiation.

        For the modified arrhenius expression it is possible to derive an analytical
        formula:

        .. math::
            \\frac{dk(T)}{dT} = k(T) \\cdot \\left(\\frac{n}{T} - \\frac{Ea}{R \\cdot T^2}\\right)

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) in Kelvin at which to evaluate the gradient.

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Temperature gradient dk/dT at the specified temperature(s).
        """
        T_jax = jnp.asarray(T, dtype=jnp.float64)  # Ensure differentiability with eqx.filter_grad

        if T_jax.ndim == 0:
            return eqx.filter_grad(self.rate_constant)(T_jax)
        else:
            return vmap(lambda t: eqx.filter_grad(self.rate_constant)(t))(T_jax)

    @eqx.filter_jit
    def grad_params(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> "Arrhenius":
        """
        Calculate the gradient of the rate constant with respect to parameters using
        automatic differentiation.

        Computes the parameter sensitivity vector:

        .. math::
            \\nabla_{\\theta} k(T) = \\begin{bmatrix}
                \\frac{\\partial k}{\\partial A} \\\\
                \\frac{\\partial k}{\\partial n} \\\\
                \\frac{\\partial k}{\\partial E_a}
            \\end{bmatrix}

        where :math:`\\theta = (A, n, E_a)` are the Arrhenius parameters.

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) in Kelvin at which to evaluate the parameter gradients.

            - **Scalar**: Returns gradients at a single temperature
            - **Vector**: Returns gradients of the sum :math:`\\sum_i k(T_i)`, useful for
                          fitting to multiple experimental points simultaneously

        Returns
        -------
        Arrhenius
            An Arrhenius object with gradients stored in place of parameters:

            - ``result.A``: :math:`\\frac{\\partial k}{\\partial A}` - Sensitivity to pre-exponential factor [dimensionless if A has same units as k]
            - ``result.n``: :math:`\\frac{\\partial k}{\\partial n}` - Sensitivity to temperature exponent [same units as k]
            - ``result.Ea``: :math:`\\frac{\\partial k}{\\partial E_a}` - Sensitivity to activation energy [k·mol/cal]

        Notes
        -----
        For the Arrhenius equation :math:`k(T) = A T^n \\exp(-E_a / RT)`, the analytical gradients are:

        .. math::
            \\frac{\\partial k}{\\partial A} &= \\frac{k}{A} \\\\
            \\frac{\\partial k}{\\partial n} &= k \\ln(T) \\\\
            \\frac{\\partial k}{\\partial E_a} &= -\\frac{k}{RT}

        When T is a vector, this method computes :math:`\\nabla_{\\theta} \\sum_i k(T_i)`, which is
        equivalent to summing individual gradients: :math:`\\sum_i \\nabla_{\\theta} k(T_i)`.
        """
        T_jax = jnp.asarray(T, dtype=jnp.float64)

        # This is needed to perform the differentiation wrt to the params
        # since they are stored as fields in this class
        wrapper_function = lambda m, t: m.rate_constant(t)

        if T_jax.ndim == 0:
            return eqx.filter_grad(wrapper_function)(self, T_jax)
        else:
            return vmap(lambda t: eqx.filter_grad(wrapper_function)(self, t))(T_jax)

    @eqx.filter_jit
    def grad_ln_k_transformed_params(
        self,
        T: float | Float64[Array, ""] | Float64[Array, "n"],
    ) -> Float64[Array, "3"] | Float64[Array, "n 3"]:
        """
        Compute gradient of ln(k) with respect to transformed parameters
        (i.e., [ln(A), n, Ea/R]).

        .. math::
            \\nabla_{\\theta'} \\ln k(T) = \\begin{bmatrix}
                \\frac{\\partial \\ln k}{\\partial \\ln A} \\\\
                \\frac{\\partial \\ln k}{\\partial n} \\\\
                \\frac{\\partial \\ln k}{\\partial (E_a/R)}
            \\end{bmatrix}

        where :math:`\\theta' = (\\ln A, n, E_a/R)` are the transformed parameters.

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) in Kelvin at which to evaluate the gradients.

        Returns
        -------
        Float64[Array, "3"] | Float64[Array, "n 3"]
            Gradient vector(s) in transformed parameter space:

            - Shape (3,) for scalar T: [∂ln(k)/∂ln(A), ∂ln(k)/∂n, ∂ln(k)/∂(Ea/R)]
            - Shape (n, 3) for vector T: gradient at each temperature

        Notes
        -----
        Eventough we are working in jax and automatic differntiation is available I
        think that for this specific use case its simpler to use the analytical version.

        The analytical result for the gradient is:

        .. math::
            \\mathbf{g}(T) = \\begin{bmatrix} 1 \\\\ \\ln(T) \\\\ -1/T \\end{bmatrix}

        This gradient vector is used in the delta method for uncertainty propagation:

        .. math::
            \\sigma^2[\\ln k(T)] = \\mathbf{g}(T)^T \\boldsymbol{\\Sigma}' \\mathbf{g}(T)
        """
        T_array = jnp.asarray(T, dtype=jnp.float64)

        # Analytical gradient formula: [1, ln(T), -1/T]
        g = jnp.stack([jnp.ones_like(T_array), jnp.log(T_array), -jnp.reciprocal(T_array)], axis=-1)

        # For scalar input, return shape (3,); for vector input, return shape (n, 3)
        return jnp.squeeze(g) if T_array.ndim == 0 else g

    # ==================================================================================
    # Uncertainty quantification/propagation methods
    def rate_constant_uncertainty(
        self,
        temperature: float | Float64[Array, ""] | Float64[Array, "n"],
        cov_matrix: Float64[Array, "n_params n_params"],
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Compute uncertainty in ln(k) using covariance matrix (delta method).

        This method propagates parameter uncertainties from a covariance matrix
        to compute the standard deviation of ln(k) at given temperature(s).

        Parameters
        ----------
        temperature : float | Float64[Array, "n"]
            Temperature(s) in Kelvin at which to evaluate uncertainty
        cov_matrix : Float64[Array, "n_params n_params"]
            Covariance matrix in transformed parameter space [ln(A), n, Ea/R]

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Standard deviation σ(ln k) at each temperature

        Notes
        -----
        Uses the delta method to propagate parameter uncertainties:

        .. math::
            \\sigma^2[\\ln k(T)] = \\mathbf{g}(T)^T \\boldsymbol{\\Sigma} \\mathbf{g}(T)

        where :math:`\\mathbf{g}(T) = [1, \\ln(T), -1/T]` is the gradient vector
        from ``grad_ln_k_transformed_params()`` and :math:`\\boldsymbol{\\Sigma}` is
        the covariance matrix.

        When performing Monte Carlo sampling from the covariance matrix,
        Nagy and Turányi (2011) recommend truncating the distribution at
        :math:`\\pm 2 \\sigma` or :math:`\\pm 3 \\sigma` to avoid physically
        unrealistic parameter combinations (e.g., negative activation
        energies, extremely large pre-exponential factors).

        References
        ----------
        .. [1] T. Nagy and T. Turányi, "Uncertainty of Arrhenius parameters",
               Int. J. Chem. Kinet., 43, 359-378 (2011). DOI: 10.1002/kin.20551
        """
        T = jnp.asarray(temperature, dtype=jnp.float64)

        # Delta method: sigma^2(ln k) = g^T Σ g
        g = self.grad_ln_k_transformed_params(T)
        var_ln_k = jnp.sum(g @ cov_matrix * g, axis=1)
        sigma_ln_k = jnp.sqrt(var_ln_k)

        return sigma_ln_k

    def rate_constant_bounds(
        self,
        temperature: float | Float64[Array, ""] | Float64[Array, "n"],
        uncertainty_factor: float,
    ) -> tuple[Float64[Array, ""] | Float64[Array, "n"], Float64[Array, ""] | Float64[Array, "n"]]:
        """
        Compute direct bounds on k(T) from uncertainty factor.

        Applies the uncertainty factor directly to the rate constant:

        .. math::
            k_{lower}(T) = k_{nominal}(T) / 10^f

            k_{upper}(T) = k_{nominal}(T) \\times 10^f

        Parameters
        ----------
        temperature : float | Float64[Array, "n"]
            Temperature(s) in Kelvin
        uncertainty_factor : float
            Multiplicative uncertainty factor (base 10). For example, f=2 means
            k can vary by a factor of 100 (10^2).

        Returns
        -------
        k_lower, k_upper : tuple of Float64 arrays
            Lower and upper bounds on k(T) at each temperature

        Notes
        -----
        This is the simplest approach for uncertainty - it assumes the uncertainty
        factor applies uniformly to k(T) at all temperatures. Commonly used when
        parameter uncertainties are unknown but an overall uncertainty factor is specified.
        """
        k_nominal = self.rate_constant(temperature)
        factor = 10.0**uncertainty_factor
        return k_nominal / factor, k_nominal * factor

    def parameter_bounds(
        self,
        uncertainty_factor: float,
        method: str = "correlated",
        T_low: float = 300,
        T_high: float = 2500.0,
    ) -> dict[str, tuple[Float64[Array, ""], Float64[Array, ""]]]:
        """
        Compute parameter bounds for optimization/sampling from uncertainty factor.

        These bounds constrain parameter space during optimization or provide
        sampling ranges for Monte Carlo uncertainty propagation.

        Parameters
        ----------
        uncertainty_factor : float
            Multiplicative uncertainty factor (base 10)
        method : str, default="correlated"
            Method for computing bounds:

            - "correlated": Correlated bounds accounting for compensation effects
            - "independent": Independent parameter perturbations

        T_low : float, default=300.0
            Lower reference temperature (K)
        T_high : float, default=2500.0
            Upper reference temperature (K)

        Returns
        -------
        dict[str, tuple[Float64[Array, ""], Float64[Array, ""]]]
            Parameter bounds with keys:

            - "lnA": (min, max) bounds on ln(A)
            - "n": (min, max) bounds on n
            - "EaR": (min, max) bounds on Ea/R

        Notes
        -----
        **Correlated method**: Back-calculates parameter bounds ensuring k(T) stays
        within the uncertainty factor at T_low and T_high, accounting for
        parameter compensation. Gives wider bounds.

        **Independent method**: Independent perturbations to each parameter:

        .. math::
            \\ln(A) &\\in [\\ln(A_0) \\pm f\\ln(10)] \\\\
            n &\\in [n_0 \\pm f\\ln(10)/\\ln(T_{high})] \\\\
            E_a/R &\\in [(E_a/R)_0 \\pm f\\ln(10) \\cdot T_{low}]
        """
        if method == "correlated":
            return self._parameter_bounds_correlated(uncertainty_factor, T_low, T_high)
        elif method == "independent":
            return self._parameter_bounds_independent(uncertainty_factor, T_low, T_high)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from: 'correlated', 'independent'")

    def _parameter_bounds_correlated(
        self,
        f: float,
        T_low: float,
        T_high: float,
    ) -> dict[str, tuple[Float64[Array, ""], Float64[Array, ""]]]:
        """Correlated method: parameter bounds accounting for compensation effects."""

        T_range = jnp.array([T_low, T_high], dtype=jnp.float64)
        log_T = jnp.log(T_range)
        inv_T = 1.0 / T_range

        # Precompute common terms
        delta_ln_A = f * jnp.log(10.0)
        log_T_diff = log_T[0] - log_T[1]
        inv_T_diff = inv_T[0] - inv_T[1]
        T_ratio = T_range[0] / T_range[1]

        # Step 1: Bounds on ln(A)
        lnA_min = self._lnA - delta_ln_A
        lnA_max = self._lnA + delta_ln_A

        # Step 2: Limiting rate constants at both temperatures
        # Using lnA_min gives lower bounds at T_low and T_high
        lnk_with_lnA_min = lnA_min + self._n * log_T - self._EaR * inv_T
        lnk_lb_Tlow, lnk_ub_Tlow = lnk_with_lnA_min[0], lnk_with_lnA_min[1]

        # Using lnA_max gives upper bounds at T_low and T_high
        lnk_with_lnA_max = lnA_max + self._n * log_T - self._EaR * inv_T
        lnk_lb_Thigh, lnk_ub_Thigh = lnk_with_lnA_max[0], lnk_with_lnA_max[1]

        # Step 3: Back-calculate n bounds
        EaR_term = self._EaR * inv_T_diff
        n_1 = (lnk_ub_Tlow - lnk_lb_Thigh - EaR_term) / log_T_diff
        n_2 = (lnk_lb_Tlow - lnk_ub_Thigh - EaR_term) / log_T_diff
        n_min = jnp.minimum(n_1, n_2)
        n_max = jnp.maximum(n_1, n_2)

        # Step 4: Back-calculate Ea/R bounds
        common_n_term = self._n * (log_T[1] - T_ratio * log_T[0])
        inv_T_factor = 1.0 / (1.0 - T_ratio)

        lnA_1 = (lnk_lb_Thigh - T_ratio * lnk_ub_Tlow - common_n_term) * inv_T_factor
        lnA_2 = (lnk_ub_Thigh - T_ratio * lnk_lb_Tlow - common_n_term) * inv_T_factor

        EaR_1 = T_range[0] * (lnA_1 + self._n * log_T[0] - lnk_ub_Tlow)
        EaR_2 = T_range[0] * (lnA_2 + self._n * log_T[0] - lnk_lb_Tlow)

        EaR_min = jnp.minimum(EaR_1, EaR_2)
        EaR_max = jnp.maximum(EaR_1, EaR_2)

        return {
            "lnA": (lnA_min, lnA_max),
            "n": (n_min, n_max),
            "EaR": (EaR_min, EaR_max),
        }

    def _parameter_bounds_independent(
        self,
        f: float,
        T_low: float,
        T_high: float,
    ) -> dict[str, tuple[Float64[Array, ""], Float64[Array, ""]]]:
        """Independent method: independent parameter perturbations."""
        delta_lnA = f * jnp.log(10.0)
        delta_n = delta_lnA / jnp.log(T_high)
        delta_EaR = delta_lnA * T_low

        return {
            "lnA": (self._lnA - delta_lnA, self._lnA + delta_lnA),
            "n": (self._n - delta_n, self._n + delta_n),
            "EaR": (self._EaR - delta_EaR, self._EaR + delta_EaR),
        }

    # ==================================================================================
    # Utilities methods
    def convert_to_standard_arrhenius(self) -> "Arrhenius":
        """
        Convert a three-parameter Arrhenius model to standard two-parameter form.

        This method fits the current three-parameter model (A, n, Ea) to a
        two-parameter model (A', 0, Ea') using least squares regression over
        a temperature range. Useful for compatibility with systems that only
        support standard Arrhenius form.

        Returns
        -------
        TODO

        Raises
        ------
        ValueError
            If the current model already has n≈0 (within numerical tolerance)
        """
        # Check if conversion is needed
        if jnp.isclose(self._n, 0.0):
            raise ValueError(f"Model already in standard Arrhenius form (n = {self._n:.5f}). Conversion is not needed!")

        # Generate temperature points for refitting
        T = jnp.linspace(300.0, 3000.0, 300)

        # Calculate rate constants using current model
        k_original = self.rate_constant(T)
        log_k = jnp.log(k_original)
        inv_T = 1.0 / T

        # Set up design matrix for 2-parameter fit (n=0)
        X = jnp.vstack([jnp.ones_like(inv_T), -inv_T]).T

        # Perform least squares regression
        beta, *_ = jnp.linalg.lstsq(X, log_k, rcond=None)

        # Extract fitted parameters: ln(A) and Ea/R
        refitted_A = jnp.exp(beta[0])
        refitted_Ea = beta[1] * constants.R_cal_mol

        # Create new Arrhenius instance
        refitted_arrhenius = Arrhenius(
            parameters={"A": float(refitted_A), "n": 0.0, "Ea": float(refitted_Ea)},
            name=self._name,
        )

        return refitted_arrhenius

    # ==================================================================================
    # String representations and debugging
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
        """
        params = [
            ("name", self._name, "s"),
            ("A", float(self._A), ".5e"),
            ("n", float(self._n), ".5e"),
            ("Ea", float(self._Ea), ".5e"),
        ]

        lines = ["Arrhenius("]
        for key, value, fmt in params:
            lines.append(f" {key:4s} = {value:{fmt}}")
        lines.append(")")
        return "\n".join(lines)

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
        """
        return self._A

    @property
    def lnA(self) -> Float64[Array, ""]:
        """
        Natural logarithm of the pre-exponential factor.

        Returns
        -------
        Float64[Array, ""]
            Natural logarithm of the pre-exponential factor A
            from the Arrhenius equation. Units depend on the
            reaction order.
        """
        return self._lnA

    @property
    def n(self) -> Float64[Array, ""]:
        """
        Temperature exponent (modified Arrhenius parameter).

        Returns
        -------
        Float64[Array, ""]
            Temperature exponent n from the Arrhenius equation (dimensionless).
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
        """
        return self._Ea

    @property
    def EaR(self) -> Float64[Array, ""]:
        """
        Activation energy divided by the ideal gas constant.

        Returns
        -------
        Float64[Array, ""]
            Activation energy divide by the ideal gas constant
            Ea/R in K.
        """
        return self._EaR

    @property
    def parameters(self) -> dict[str, float]:
        """
        Dictionary containing the entire set of arrhenius parameters.

        Returns
        -------
        dict[str, float]
            Dictionary of arrhenius parameters: {"A": ..., "n": ..., "Ea": ...}
        """
        return {"A": float(self._A), "n": float(self._n), "Ea": float(self._Ea)}

    @property
    def name(self) -> str:
        """
        Human-readable reaction name.

        Returns
        -------
        str
            The reaction name.
        """
        return self._name
