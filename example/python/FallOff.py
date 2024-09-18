import matplotlib.pyplot as plt
import jax.numpy as jnp
import sys
sys.path.append("/Users/tdinelli/Documents/GitHub/PLOG_Converter")
from source.FallOff import kinetic_constant_fall_off

"""
CH3 + CH3 (+M) = C2H6 (+M)

Example reported within the CHEMKIN manual
"""

constant_troe = jnp.array([
    [1.135E+36, -5.246, 1704.8],  # LPL
    [6.220E+16, -1.174, 635.80],  # HPL
    [0.405, 1120., 69.6],  # Coefficients "T2" is optional
])

constant_lindemann = jnp.array([
    [1.135E+36, -5.246, 1704.8],  # LPL
    [6.220E+16, -1.174, 635.80],  # HPL
])

kc = []
k0 = []
kInf = []
M = []
kcl = []

T = jnp.linspace(1000., 1000., 300)
P = jnp.logspace(-5, 2, 300)

for i in P:
    _kc, _k0, _kInf, _M = kinetic_constant_fall_off(constant_troe, 1000., float(i))
    kc.append(_kc)
    k0.append(_k0 * _M)
    kInf.append(_kInf)
    M.append(_M)

    _kcl, _, _, _ = kinetic_constant_fall_off(constant_lindemann, 1000., float(i))
    kcl.append(_kcl)

plt.plot(M, kc, 'b-o', label="TROE form")
plt.plot(M, kcl, 'k-o', label="Lindemann form")
plt.plot(M, kInf, "g--", label="HPL")
plt.plot(M, k0, "r--", label="LPL*[M]")
plt.yscale("log")
plt.xscale("log")
plt.ylim([min(kc), kInf[-1]*1.1])
plt.legend()
plt.show()