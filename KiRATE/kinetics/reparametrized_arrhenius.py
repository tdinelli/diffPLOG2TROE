"""
Copyright (c) 2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float64

from KiRATE.utilities.physical_constants import constants


class ReparametrizedArrhenius(eqx.Module):
    """
    Reparametrized (centered) Arrhenius rate constant calculator.

    This class implements a reparametrized form of the modified Arrhenius equation
    that provides better numerical conditioning and reduced parameter correlation
    during fitting. By centering the equation around a reference temperature, it
    addresses statistical and numerical issues common in the standard Arrhenius form.

    The reparametrized rate constant is given by:

    .. math::
        k(T) = k_{ref} \\cdot \\left(\\frac{T}{T_{ref}}\\right)^n \\cdot
        \\exp\\left(-\\frac{E_a}{R}\\left(\\frac{1}{T} - \\frac{1}{T_{ref}}\\right)\\right)

    where:
        - :math:`k_{ref}` is the rate constant at the reference temperature
        - :math:`T_{ref}` is a user-chosen reference temperature (typically near
          the center of the experimental range)
        - :math:`n` is the temperature exponent (same physical meaning as standard form)
        - :math:`E_a` is the activation energy (same physical meaning as standard form)

    This form is mathematically equivalent to the standard Arrhenius equation:

    .. math::
        k(T) = A \\cdot T^n \\cdot \\exp\\left(-\\frac{E_a}{RT}\\right)

    with the conversion:

    .. math::
        A = k_{ref} \\cdot T_{ref}^{-n} \\cdot \\exp\\left(\\frac{E_a}{R \\cdot T_{ref}}\\right)

    Advantages over Standard Form
    ------------------------------
    1. **Reduced Parameter Correlation**: In the standard form, the pre-exponential
       factor A and activation energy Ea are often highly correlated during fitting,
       leading to large uncertainties and numerical instability.

    2. **Improved Numerical Conditioning**: The centered form provides better
       conditioning for least-squares and gradient-based optimization algorithms.

    3. **Better Parameter Uncertainty**: Reduced correlation leads to more reliable
       uncertainty estimates and confidence intervals.

    4. **Centered Parameterization**: Reference point near experimental data improves
       extrapolation behavior and parameter interpretability.

    Attributes
    ----------
    _k_ref : Float64[Array, ""]
        Rate constant at the reference temperature [units depend on reaction order]
    _n : Float64[Array, ""]
        Temperature exponent (dimensionless)
    _Ea : Float64[Array, ""]
        Activation energy [cal/mol]
    _T_ref : Float64[Array, ""]
        Reference temperature [K]
    _name : str
        Human-readable reaction name

    See Also
    --------
    Arrhenius : Standard modified Arrhenius rate constant

    Notes
    -----
    **CHEMKIN Compatibility:**

    The CHEMKIN format does not natively support reparametrized Arrhenius equations,
    so this class does not provide a ``from_chemkin()`` method. To use CHEMKIN data:
        1. Parse with ``Arrhenius.from_chemkin()``
        2. Convert to reparametrized form using ``ReparametrizedArrhenius.from_standard_form()``

    **When to Use:**

    - **Use Reparametrized Form**: When fitting parameters to experimental data,
      especially if the standard form shows high A-Ea correlation
    - **Use Standard Form**: For general rate calculations or when reading from
      databases (most kinetics data is stored in standard form)
    """

    _k_ref: Float64[Array, ""]
    _n: Float64[Array, ""]
    _Ea: Float64[Array, ""]
    _T_ref: Float64[Array, ""]
    _name: str = eqx.field(static=True, default="")

    def __init__(self, parameters: dict[str, float], name: str = "") -> None:
        """
        Initialize the reparametrized Arrhenius rate constant calculator.

        Parameters
        ----------
        parameters : dict[str, float]
            Reparametrized Arrhenius parameters with keys:
                - "k_ref": Rate constant at T_ref [units depend on reaction order]
                - "n": Temperature exponent (dimensionless)
                - "Ea": Activation energy [cal/mol]
                - "T_ref": Reference temperature [K]
        name : str, optional
            Human-readable reaction name, by default ""

        Notes
        -----
        **Choosing T_ref:**

        The reference temperature should be chosen strategically:
            - Center of experimental temperature range (reduces extrapolation error)
            - Temperature where measurements are most accurate
            - Typical choice: mean or median of experimental temperatures

        **Parameter Interpretation:**

        Unlike the standard form where A has no direct physical meaning, k_ref
        represents the actual rate constant at a specific temperature (T_ref),
        making it more intuitive for parameter estimation.
        """
        self._name = name
        self._k_ref = jnp.float64(parameters["k_ref"])
        self._n = jnp.float64(parameters["n"])
        self._Ea = jnp.float64(parameters["Ea"])
        self._T_ref = jnp.float64(parameters["T_ref"])

    @eqx.filter_jit
    def rate_constant(
        self, T: float | Float64[Array, ""] | Float64[Array, "n"]
    ) -> Float64[Array, ""] | Float64[Array, "n"]:
        """
        Calculate the rate constant at given temperature(s).

        Evaluates the reparametrized Arrhenius equation:

        .. math::
            k(T) = k_{ref} \\cdot \\left(\\frac{T}{T_{ref}}\\right)^n \\cdot
            \\exp\\left(-\\frac{E_a}{R}\\left(\\frac{1}{T} - \\frac{1}{T_{ref}}\\right)\\right)

        Parameters
        ----------
        T : float | Float64[Array, ""] | Float64[Array, "n"]
            Temperature(s) in Kelvin. Accepts scalars or 1D arrays.

        Returns
        -------
        Float64[Array, ""] | Float64[Array, "n"]
            Rate constant(s) at the given temperature(s).
            Units depend on the reaction order (same as k_ref).

            - Scalar T → Scalar output
            - Vector T → Vector output (same length as input)
        """
        T = jnp.asarray(T, dtype=jnp.float64)
        return (
            self._k_ref
            * jnp.pow(T / self._T_ref, self._n)
            * jnp.exp(-(self._Ea / constants.R_cal_mol) * (1.0 / T - 1.0 / self._T_ref))
        )

    def to_standard_form(self) -> dict[str, float]:
        """
        Convert reparametrized parameters to standard Arrhenius form.

        Transforms the centered representation to the standard modified Arrhenius
        parameters that can be used with the ``Arrhenius`` class.

        Returns
        -------
        dict[str, float]
            Standard Arrhenius parameters with keys:
                - "A": Pre-exponential factor [units depend on reaction order]
                - "n": Temperature exponent (unchanged, dimensionless)
                - "Ea": Activation energy (unchanged, cal/mol)

        Notes
        -----
        **Conversion Formula:**

        The pre-exponential factor A is computed as:

        .. math::
            A = k_{ref} \\cdot T_{ref}^{-n} \\cdot \\exp\\left(\\frac{E_a}{R \\cdot T_{ref}}\\right)

        The parameters n and Ea remain unchanged between forms.
        """
        A_standard = self._k_ref * (self._T_ref ** (-self._n)) * jnp.exp(self._Ea / constants.R_cal_mol / self._T_ref)
        return {"A": float(A_standard), "n": float(self._n), "Ea": float(self._Ea)}

    @classmethod
    def from_standard_form(
        cls,
        parameters: dict[str, float],
        T_ref: float | Float64[Array, ""],
        name: str = "",
    ) -> "ReparametrizedArrhenius":
        """
        Create ReparametrizedArrhenius from standard Arrhenius parameters.

        This class method converts standard Arrhenius parameters (A, n, Ea) to the
        reparametrized form centered at a specified reference temperature.

        Parameters
        ----------
        parameters : dict[str, float]
            Standard Arrhenius parameters with keys:
                - "A": Pre-exponential factor [units depend on reaction order]
                - "n": Temperature exponent (dimensionless)
                - "Ea": Activation energy [cal/mol]

        T_ref : float | Float64[Array, ""]
            Reference temperature [K] for the reparametrized form.
            Should be chosen near the center of the experimental temperature range.

        name : str, optional
            Human-readable reaction name, by default ""

        Returns
        -------
        ReparametrizedArrhenius
            New ReparametrizedArrhenius instance centered at T_ref
        """
        T_ref = jnp.float64(T_ref)
        k_ref = parameters["A"] * (T_ref ** parameters["n"]) * jnp.exp(-parameters["Ea"] / constants.R_cal_mol / T_ref)
        centered_params = {"k_ref": float(k_ref), "n": parameters["n"], "Ea": parameters["Ea"], "T_ref": T_ref}

        return cls(parameters=centered_params, name=name)

    def __str__(self) -> str:
        """
        Return a tab-separated string representation.

        Returns
        -------
        str
            Tab-separated string with reaction name and reparametrized parameters:
            Format: "{name}\\t\\t{k_ref:.5e} {n:.5e} {Ea:.5e} {T_ref:.1f}"

        Notes
        -----
        This format is useful for logging and simple text output. For structured
        data export, use ``to_standard_form()`` and standard CHEMKIN formatting.
        """
        k_ref_original = self._k_ref
        Ea_original = self._Ea
        return f"{self._name}\t\t{k_ref_original:.5e} {self._n:.5e} {Ea_original:.5e} {self._T_ref:.1f}"

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging.

        Provides a multi-line, human-readable representation showing all
        reparametrized parameters with appropriate precision.

        Returns
        -------
        str
            Multi-line formatted string showing all parameters including T_ref.
        """
        return (
            f"ReparametrizedArrhenius("
            f"\n name = {self._name}"
            f"\n Tref = {self._T_ref:.5f}"
            f"\n kref = {self._k_ref:.5E}"
            f"\n n    = {self._n:.5E}"
            f"\n Ea   = {self._Ea:.5E}"
            "\n)"
        )

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

    @property
    def k_ref(self) -> Float64[Array, ""]:
        """
        Rate constant at the reference temperature.

        Returns
        -------
        Float64[Array, ""]
            k_ref: Rate constant evaluated at T_ref.
            Units depend on reaction order (same as standard A × T_ref^n × exp(-Ea/RT_ref)).
        """
        return self._k_ref

    @property
    def A(self) -> Float64[Array, ""]:
        """
        Equivalent pre-exponential factor in standard Arrhenius form.

        Computes the standard form pre-exponential factor A from the
        reparametrized parameters.

        Returns
        -------
        Float64[Array, ""]
            Pre-exponential factor A. Units depend on reaction order.

        Notes
        -----
        This is computed on-the-fly using:

        .. math::
            A = k_{ref} \\cdot T_{ref}^{-n} \\cdot \\exp\\left(\\frac{E_a}{R \\cdot T_{ref}}\\right)

        This property is useful for compatibility with code expecting standard
        Arrhenius parameters without explicit conversion.
        """
        return self._k_ref * (self._T_ref ** (-self._n)) * jnp.exp(self._Ea / constants.R_cal_mol / self._T_ref)

    @property
    def n(self) -> Float64[Array, ""]:
        """
        Temperature exponent (modified Arrhenius parameter).

        Returns
        -------
        Float64[Array, ""]
            Temperature exponent n (dimensionless).

        Notes
        -----
        This parameter has the same value and meaning in both standard and
        reparametrized forms.
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

        Notes
        -----
        This parameter has the same value and meaning in both standard and
        reparametrized forms.
        """
        return self._Ea

    @property
    def T_ref(self) -> Float64[Array, ""]:
        """
        Reference temperature for the reparametrized form.

        Returns
        -------
        Float64[Array, ""]
            Reference temperature T_ref in Kelvin.

        Notes
        -----
        This temperature defines the centering point of the reparametrized equation.
        At this temperature, the rate constant equals k_ref exactly.
        """
        return self._T_ref
