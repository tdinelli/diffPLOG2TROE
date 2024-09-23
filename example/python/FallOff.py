import matplotlib.pyplot as plt
import jax.numpy as jnp
import sys
sys.path.append("/Users/tdinelli/Documents/GitHub/diffPLOG2TROE")
from diffPLOG2TROE.falloff import kinetic_constant_falloff


"""
CH3 + CH3 (+M) = C2H6 (+M)

Example reported within the CHEMKIN manual
"""

falloff_troe = (
    jnp.array([
        [1.135E+36, -5.246, 1704.8],  # LPL
        [6.220E+16, -1.174, 635.80],  # HPL
        [0.405, 1120., 69.6],  # TROE parameters
    ]),
    1, # Lindemann -> 0, TROE -> 1, SRI -> 2
)

falloff_lindemann = (
    jnp.array([
        [1.135E+36, -5.246, 1704.8],  # LPL
        [6.220E+16, -1.174, 635.80],  # HPL
    ]),
    0, # Lindemann -> 0, TROE -> 1, SRI -> 2
)

kc = []
k0 = []
kInf = []
M = []
kcl = []

T = jnp.linspace(1000., 1000., 300)
P = jnp.logspace(-5, 2, 300)
for i in P:
    _kc, _k0, _kInf, _M = kinetic_constant_falloff(falloff_troe, 1000., float(i))
    kc.append(_kc)
    k0.append(_k0 * _M)
    kInf.append(_kInf)
    M.append(_M)

    _kcl, _, _, _ = kinetic_constant_falloff(falloff_lindemann, 1000., float(i))
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
