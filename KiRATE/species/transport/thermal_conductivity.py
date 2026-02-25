"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details

Pure-species thermal conductivity from Chapman-Enskog / Mason-Monchick theory.

Implements the single-species thermal conductivity following the CHEMKIN /
OpenSMOKE++ formulation (equations 5.17–5.34). The model decomposes the heat flux
into translational, rotational, and vibrational contributions, each weighted
by a dedicated shape factor.

References
----------
.. [1] Kee, R. J., Dixon-Lewis, G., Warnatz, J., Coltrin, M. E., and Miller, J. A.
       "A Fortran Computer Code Package for the Evaluation of Gas-Phase
       Multicomponent Transport Properties." Sandia Report SAND86-8246 (1986).
"""

import jax.numpy as jnp
from jaxtyping import Array, Float64

from KiRATE.species.transport.collision_integrals import omega11
from KiRATE.species.transport.viscosity import delta_star, mu_star, species_viscosity
from KiRATE.utilities import constants

# Prefactor for self-diffusion D_kk at P=1 bar
# [m^2/s * bar / (sqrt(g/mol) * K^1.5 / A^2)]
_COEFF_DKK: Float64[Array, ""] = jnp.float64(2.6693e-7)

# Parker-Brau-Jonkman constants for rotational relaxation correction
_ZROTA: Float64[Array, ""] = jnp.float64(0.5 * jnp.pi**1.5)  # pi^1.5 / 2
_ZROTB: Float64[Array, ""] = jnp.float64(2.0 + 0.25 * jnp.pi**2)  # 2 + pi^2/4
_ZROTC: Float64[Array, ""] = jnp.float64(jnp.pi**1.5)  # pi^1.5

# Frequently used combinations
_2_OVER_PI: Float64[Array, ""] = jnp.float64(2.0 / jnp.pi)  # 2/pi
_5_OVER_3R: Float64[Array, ""] = jnp.float64(5.0 / (3.0 * constants.R_J_mol_K))  # 5/(3R)


def parker_brau_jonkman(T: Float64[Array, ""], epsilon_over_k: Float64[Array, ""]) -> Float64[Array, ""]:
    """
    Parker-Brau-Jonkman correction factor F(T).

    Computes the temperature-dependent correction factor used to adjust the
    rotational relaxation number from its reference value at 298 K (equation 5.34):

    .. math::
        F(T) = 1 + \\sqrt{\\frac{\\epsilon/k_B}{T}} \\left(\\frac{\\pi^{3/2}}{2}
               + \\sqrt{\\frac{\\epsilon/k_B}{T}}\\left(2 + \\frac{\\pi^2}{4}
               + \\pi^{3/2}\\sqrt{\\frac{\\epsilon/k_B}{T}}\\right)\\right)

    Parameters
    ----------
    T : Float64[Array, ""]
        Temperature [K]
    epsilon_over_k : Float64[Array, ""]
        Lennard-Jones well depth divided by Boltzmann constant [K]

    Returns
    -------
    Float64[Array, ""]
        Dimensionless correction factor [-]
    """
    aux = jnp.sqrt(epsilon_over_k / T)

    return 1.0 + aux * (_ZROTA + aux * (_ZROTB + aux * _ZROTC))


def z_rot(
    T: Float64[Array, ""],
    epsilon_over_k: Float64[Array, ""],
    z_rot_298: Float64[Array, ""],
) -> Float64[Array, ""]:
    """
    Rotational relaxation collision number at temperature T.

    Corrects the tabulated value at 298 K to the actual temperature using
    the Parker-Brau-Jonkman model (equation 5.33):

    .. math::
        Z_{\\text{rot}}(T) = Z_{\\text{rot}}(298\\,\\text{K}) \\frac{F(298\\,\\text{K})}{F(T)}

    Parameters
    ----------
    T : Float64[Array, ""]
        Temperature [K]
    epsilon_over_k : Float64[Array, ""]
        Lennard-Jones well depth divided by Boltzmann constant [K]
    z_rot_298 : Float64[Array, ""]
        Rotational relaxation number at 298 K [-]

    Returns
    -------
    Float64[Array, ""]
        Rotational relaxation collision number at temperature T [-]
    """
    f_298 = parker_brau_jonkman(jnp.float64(298.0), epsilon_over_k)
    f_T = parker_brau_jonkman(T, epsilon_over_k)

    return z_rot_298 * f_298 / f_T


def cv_components(
    geometry: int,
    cv_over_r: Float64[Array, ""],
) -> tuple[Float64[Array, ""], Float64[Array, ""], Float64[Array, ""]]:
    """
    Partition total heat capacity into translational, rotational, and vibrational components.

    Decomposes :math:`C_v` based on molecular geometry following equipartition theorem:

    .. math::
        C_{v,\\text{trans}} &= \\frac{3}{2}R \\quad \\text{(always)} \\\\
        C_{v,\\text{rot}} &= \\begin{cases}
            0 & \\text{monatomic} \\\\
            R & \\text{linear} \\\\
            \\frac{3}{2}R & \\text{nonlinear}
        \\end{cases} \\\\
        C_{v,\\text{vib}} &= C_{v,\\text{total}} - C_{v,\\text{trans}} - C_{v,\\text{rot}}

    Parameters
    ----------
    geometry : int
        Molecular geometry index (0=monatomic, 1=linear, 2=nonlinear)
    cv_over_r : Float64[Array, ""]
        Dimensionless total heat capacity :math:`C_v/R` from NASA polynomials [-]

    Returns
    -------
    tuple[Float64[Array, ""], Float64[Array, ""], Float64[Array, ""]]
        Translational, rotational, and vibrational heat capacities [J/(mol·K)]

    Notes
    -----
    For monatomic species, :math:`C_{v,\\text{vib}} = 0` by construction since there
    are no rotational or vibrational degrees of freedom.
    """
    cv_trans = 1.5 * constants.R_J_mol_K  # (3/2) R — always

    cv_rot = jnp.where(
        geometry == 0,
        0.0,  # monatomic: no rotation
        jnp.where(
            geometry == 1,
            constants.R_J_mol_K,  # linear: R
            1.5 * constants.R_J_mol_K,  # nonlinear: (3/2) R
        ),
    )

    # Vibrational contribution: remainder of total Cv
    cv_total = cv_over_r * constants.R_J_mol_K
    cv_vib = jnp.where(geometry == 0, 0.0, cv_total - cv_trans - cv_rot)

    return cv_trans, cv_rot, cv_vib


def self_diffusion(
    T: Float64[Array, ""],
    MW: Float64[Array, ""],
    sigma: Float64[Array, ""],
    epsilon_over_k: Float64[Array, ""],
    delta_star_val: Float64[Array, ""],
) -> Float64[Array, ""]:
    """
    Self-diffusion coefficient at P = 1 bar.

    Computes the main diagonal element of the binary diffusion matrix (equation 5.31):

    .. math::
        D_{kk} = \\frac{2.6693 \\times 10^{-7}}{\\sqrt{M_k}\\,\\sigma_k^2}
                 \\frac{T^{3/2}}{\\Omega^{(1,1)}(T^*, \\delta^*)}

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
    delta_star_val : Float64[Array, ""]
        Reduced dipole moment :math:`\\delta^* = 0.5 (\\mu^*)^2` [-]

    Returns
    -------
    Float64[Array, ""]
        Self-diffusion coefficient at P = 1 bar [m^2/s]

    Notes
    -----
    This is computed at an arbitrary reference pressure of 1 bar. In the thermal
    conductivity formula, the product :math:`\\rho \\cdot D_{kk}` is pressure-independent
    since :math:`\\rho \\sim P` and :math:`D_{kk} \\sim 1/P`.
    """
    t_star = T / epsilon_over_k
    coeff = _COEFF_DKK / (jnp.sqrt(MW) * sigma**2)
    o11 = omega11(t_star, delta_star_val)

    return coeff * T**1.5 / o11  # [m^2/s] at P=1 bar


def species_thermal_conductivity(
    T: Float64[Array, ""],
    MW: Float64[Array, ""],
    sigma: Float64[Array, ""],
    epsilon_over_k: Float64[Array, ""],
    mu_debye: Float64[Array, ""],
    z_rot_298: Float64[Array, ""],
    geometry: int,
    cv_over_r: Float64[Array, ""],
) -> Float64[Array, ""]:
    """
    Thermal conductivity of a pure species.

    Implements Mason-Monchick / Eucken model with Parker-Brau-Jonkman rotational
    correction (equations 5.17–5.34 of CHEMKIN Theory Manual):

    .. math::
        \\lambda = \\frac{\\eta}{M_k} \\left( f_{\\text{trans}} C_{v,\\text{trans}}
                  + f_{\\text{rot}} C_{v,\\text{rot}}
                  + f_{\\text{vib}} C_{v,\\text{vib}} \\right)

    where the shape factors are:

    .. math::
        f_{\\text{vib}} &= \\frac{\\rho D_{kk}}{\\eta} \\\\
        A &= 2.5 - f_{\\text{vib}} \\\\
        B &= Z_{\\text{rot}} + \\frac{2}{\\pi}\\left(\\frac{5}{3R}C_{v,\\text{rot}} + f_{\\text{vib}}\\right) \\\\
        f_{\\text{trans}} &= 2.5\\left(1 - \\frac{2}{\\pi}\\frac{C_{v,\\text{rot}}}{C_{v,\\text{trans}}}\\frac{A}{B}\\right) \\\\
        f_{\\text{rot}} &= f_{\\text{vib}}\\left(1 + \\frac{2}{\\pi}\\frac{A}{B}\\right)

    For monatomic species (geometry=0), only the translational term contributes (equation 5.30).

    Parameters
    ----------
    T : Float64[Array, ""]
        Temperature [K]
    MW : Float64[Array, ""]
        Molecular weight [kg/mol]
    sigma : Float64[Array, ""]
        Lennard-Jones collision diameter [Angstrom]
    epsilon_over_k : Float64[Array, ""]
        Lennard-Jones well depth divided by Boltzmann constant [K]
    mu_debye : Float64[Array, ""]
        Dipole moment [Debye]. Use 0.0 for nonpolar species.
    z_rot_298 : Float64[Array, ""]
        Rotational relaxation collision number at 298 K [-]
    geometry : int
        Molecular geometry (0=monatomic, 1=linear, 2=nonlinear)
    cv_over_r : Float64[Array, ""]
        Dimensionless heat capacity :math:`C_v/R = C_p/R - 1` from NASA polynomials [-]

    Returns
    -------
    Float64[Array, ""]
        Thermal conductivity [W/(m·K)]

    Notes
    -----
    The final result is in SI units [W/(m·K)] throughout, unlike some formulations
    that require unit conversions at the end.

    References
    ----------
    .. [1] Kee, R. J., Dixon-Lewis, G., Warnatz, J., Coltrin, M. E., and Miller, J. A.
           "A Fortran Computer Code Package for the Evaluation of Gas-Phase
           Multicomponent Transport Properties." Sandia Report SAND86-8246 (1986).
    """
    MW_gmol = MW * 1.0e3  # [kg/mol] → [g/mol]

    mstar = mu_star(mu_debye, epsilon_over_k, sigma)
    dstar = delta_star(mstar)

    # Viscosity eta [kg/m/s]
    eta = species_viscosity(T, MW_gmol, sigma, epsilon_over_k, mu_debye)

    # Rotational relaxation Z_rot(T)
    z_rot_T = z_rot(T, epsilon_over_k, z_rot_298)

    # Cv components [J/mol/K]
    cv_trans, cv_rot, cv_vib = cv_components(geometry, cv_over_r)

    # Self-diffusion D_kk [m^2/s] at P = 1 bar
    d_kk = self_diffusion(T, MW_gmol, sigma, epsilon_over_k, dstar)

    # Species density rho [kg/m^3] at P = 1 bar (eq. 5.23)
    # rho = P * MW / (R * T) with P = 1e5 Pa
    rho = 1.0e5 * MW / (constants.R_J_mol_K * T)

    # Shape factors (eqs. 5.18–5.22)
    f_vib = rho * d_kk / eta  # (5.20)
    A = 2.5 - f_vib  # (5.21)
    B = z_rot_T + _2_OVER_PI * (_5_OVER_3R * cv_rot + f_vib)  # (5.22)
    f_trans = 2.5 * (1.0 - _2_OVER_PI * (cv_rot / cv_trans) * (A / B))  # (5.18)
    f_rot = f_vib * (1.0 + _2_OVER_PI * (A / B))  # (5.19)

    # Thermal conductivity [W/m/K]
    eta_over_mw = eta / (MW_gmol * 1.0e-3)  # [kg/m/s] / [kg/mol]

    # Monatomic: only translational contribution (eq. 5.30)
    lambda_monatomic = eta_over_mw * (f_trans * cv_trans)

    # Polyatomic: all three contributions (eq. 5.17)
    lambda_polyatomic = eta_over_mw * (f_trans * cv_trans + f_rot * cv_rot + f_vib * cv_vib)

    return jnp.where(geometry == 0, lambda_monatomic, lambda_polyatomic)
