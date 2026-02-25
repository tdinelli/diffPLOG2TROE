"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details

Pure-species dynamic viscosity from Chapman-Enskog kinetic theory.

Implements the single-species viscosity using Lennard-Jones 12-6 potential
parameters following the CHEMKIN / OpenSMOKE++ formulation.

References
----------
.. [1] Kee, R. J., Rupley, F. M., and Miller, J. A.
       "CHEMKIN-II: A Fortran Chemical Kinetics Package for the Analysis of
       Gas-Phase Chemical Kinetics." Sandia Report SAND89-8009 (1989).

.. [2] Kee, R. J., Dixon-Lewis, G., Warnatz, J., Coltrin, M. E., and Miller, J. A.
       "A Fortran Computer Code Package for the Evaluation of Gas-Phase
       Multicomponent Transport Properties." Sandia Report SAND86-8246 (1986).
"""

import jax.numpy as jnp
from jaxtyping import Array, Float64

from KiRATE.species.transport.collision_integrals import omega22
from KiRATE.utilities import constants

# Prefactor in the viscosity formula
# Derived from Chapman-Enskog: eta = (5/16) * sqrt(pi*m*kB*T) / (pi*sigma^2*Omega22)
# In CHEMKIN/OpenSMOKE++ reduced units: 26.693e-7 [kg/(m·s) / (sqrt(g/mol)·sqrt(K) / Angstrom^2)]
_COEFF_ETA: Float64[Array, ""] = jnp.float64(26.693e-7)


def mu_star(
    mu_debye: Float64[Array, ""],
    epsilon_over_k: Float64[Array, ""],
    sigma: Float64[Array, ""],
) -> Float64[Array, ""]:
    """
    Reduced dipole moment mu*.

    Converts a dipole moment in Debye to the dimensionless reduced form
    used in collision integral calculations (equation 5.5 in CHEMKIN Theory Manual):

    .. math::
        \\mu^* = \\frac{\\mu}{\\sqrt{\\epsilon \\sigma^3}}

    where :math:`\\mu` is converted from Debye to :math:`\\sqrt{\\text{Angstrom}^3 \\cdot \\text{erg}}`
    and :math:`\\epsilon = (\\epsilon/k_B) \\cdot k_B`.

    Parameters
    ----------
    mu_debye : Float64[Array, ""]
        Dipole moment [Debye]
    epsilon_over_k : Float64[Array, ""]
        Lennard-Jones well depth divided by Boltzmann constant [K]
    sigma : Float64[Array, ""]
        Lennard-Jones collision diameter [Angstrom]

    Returns
    -------
    Float64[Array, ""]
        Dimensionless reduced dipole moment [-]

    Notes
    -----
    Values below :math:`10^{-32}` Debye are treated as zero to avoid numerical issues,
    mirroring the MU_MIN check in OpenSMOKE++.
    """
    mu_cgs = mu_debye * 1.0e-6  # [Debye] to [sqrt(A^3·erg)]
    epsilon_cgs = epsilon_over_k * constants.KB_CGS  # [K] to [erg]
    mu_star_val = mu_cgs / jnp.sqrt(epsilon_cgs * sigma**3)

    return jnp.where(mu_debye < 1.0e-32, 0.0, mu_star_val)


def delta_star(mu_star_val: Float64[Array, ""]) -> Float64[Array, ""]:
    """
    Reduced dipole moment delta* used as collision integral table coordinate.

    Computes the dimensionless dipole parameter (equation 5.6 in CHEMKIN Theory Manual):

    .. math::
        \\delta^* = 0.5 (\\mu^*)^2

    Parameters
    ----------
    mu_star_val : Float64[Array, ""]
        Dimensionless reduced dipole moment from :func:`mu_star`

    Returns
    -------
    Float64[Array, ""]
        Reduced dipole moment delta* [-]
    """
    return 0.5 * mu_star_val**2


def species_viscosity(
    T: Float64[Array, ""],
    MW: Float64[Array, ""],
    sigma: Float64[Array, ""],
    epsilon_over_k: Float64[Array, ""],
    mu_debye: Float64[Array, ""],
) -> Float64[Array, ""]:
    """
    Dynamic viscosity of a pure species.

    Computes viscosity using Chapman-Enskog kinetic theory with Lennard-Jones 12-6
    potential following equation 5.1 from CHEMKIN Theory Manual:

    .. math::
        \\eta_k = \\frac{26.693 \\times 10^{-7} \\sqrt{M_k T}}{\\sigma_k^2 \\Omega^{(2,2)}(T^*_k, \\delta^*_k)}

    where the reduced temperature is :math:`T^*_k = T / (\\epsilon_k / k_B)` and
    the reduced dipole moment is :math:`\\delta^*_k = 0.5 (\\mu^*_k)^2`.

    Parameters
    ----------
    T : Float64[Array, ""]
        Temperature [K]
    MW : Float64[Array, ""]
        Molecular weight [g/mol]
    sigma : Float64[Array, ""]
        Lennard-Jones collision diameter [Angstrom]
    epsilon_over_k : Float64[Array, ""]
        Lennard-Jones well depth divided by Boltzmann constant [K]
    mu_debye : Float64[Array, ""]
        Dipole moment [Debye]. Use 0.0 for nonpolar species.

    Returns
    -------
    Float64[Array, ""]
        Dynamic viscosity [kg/(m·s)]

    Notes
    -----
    For polar molecules (:math:`\\mu > 0`), the collision integral :math:`\\Omega^{(2,2)}`
    accounts for dipole-dipole interactions through the :math:`\\delta^*` parameter.

    The prefactor :math:`26.693 \\times 10^{-7}` equals :math:`(5/16)\\sqrt{\\pi}` in
    CHEMKIN reduced units. See OpenSMOKE++ ``InitializeViscosity()`` and
    ``SingleSpeciesViscosity()`` for implementation details.

    References
    ----------
    .. [1] Kee, R. J., Rupley, F. M., and Miller, J. A.
           "CHEMKIN-II: A Fortran Chemical Kinetics Package for the Analysis of
           Gas-Phase Chemical Kinetics." Sandia Report SAND89-8009 (1989).
    """
    mstar = mu_star(mu_debye, epsilon_over_k, sigma)
    dstar = delta_star(mstar)
    t_star = T / epsilon_over_k  # reduced temperature [-]

    coeff = _COEFF_ETA * jnp.sqrt(MW) / sigma**2  # prefactor [kg/m/s / sqrt(K)]
    o22 = omega22(t_star, dstar)  # collision integral  [-]

    return coeff * jnp.sqrt(T) / o22  # [kg/m/s]
