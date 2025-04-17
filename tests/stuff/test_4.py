import jax.numpy as jnp
import matplotlib.pyplot as plt

from diffPLOG2TROE.boundaries import Boundaries
from diffPLOG2TROE.kinetic_constants import Arrhenius


rate_constant = Arrhenius(name="H2+O=H+OH", params=jnp.array([5.080e04, 2.670e00, 6.292e03]))
T_range = jnp.linspace(300, 2500, 300)
k = rate_constant.kinetic_constant(T_range)

bound = Boundaries(rate_constant=k, T_range=(300, 2500), uncertainty_factor=0.5, n_T=300)
boundaries = bound.get_boundary_rate_constants(T_range)

print(boundaries["ub_params"])
print(boundaries["lb_params"])

plt.semilogy(T_range, boundaries["nominal"], label="nominal")
plt.semilogy(T_range, boundaries["upper"], label="upper")
plt.semilogy(T_range, boundaries["lower"], label="lower")

plt.legend()
plt.show()
