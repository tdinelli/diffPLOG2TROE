import matplotlib.pyplot as plt
import jax.numpy as jnp
import sys
sys.path.append("/Users/tdinelli/Documents/GitHub/PLOG_Converter")
from source.Cabr import kinetic_constant_cabr

"""
CH3 + CH3(+M) = C2H5 + H(+M)

Example reported within the CHEMKIN manual
"""

constant_troe = jnp.array([
    [10**12.698, 0.099, 10600, .0], # The fourth term is dummy
    [10**-6.42, 4.838, 7710, .0],   # The fourth term is dummy
    [1.641, 4334, 2725, .0]
], dtype=jnp.float64)

constant_lindemann = jnp.array([
    [10**12.698, 0.099, 10600],
    [10**-6.42, 4.838, 7710],
], dtype=jnp.float64)

kc = []
k0 = []
kInf = []
M = []
kcl = []

T = jnp.linspace(1000., 1000., 300)
P = jnp.logspace(-5, 2, 300)

for i in P:
    _kc, _k0, _kInf, _M = kinetic_constant_cabr(constant_troe, 1000., float(i))
    kc.append(_kc)
    k0.append(_k0)
    kInf.append(_kInf / _M)
    M.append(_M)

    _kcl, _, _, _ = kinetic_constant_cabr(constant_lindemann, 1000., float(i))
    kcl.append(_kcl)


plt.plot(M, kc, 'b-o', label="TROE form")
plt.plot(M, kcl, 'k-o', label="Lindemann form")
plt.plot(M, kInf, "g--", label="HPL / [M]")
plt.plot(M, k0, "r--", label="LPL")
plt.yscale("log")
plt.xscale("log")
plt.ylim([min(kc), k0[-1]*1.1])
plt.legend()
plt.show()
