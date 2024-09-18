from jax import jacfwd, jacrev
import jax.numpy as jnp

import sys
sys.path.append("/Users/tdinelli/Documents/GitHub/PLOG_Converter")
from source.PressureLogarithmic import compute_plog
from source.Refitter import rmse_loss_function

plog = jnp.array([
    [1.00E-01, 7.23E+29, -5.32E+00, 110862.4],
    [1.00E+00, 3.50E+30, -5.22E+00, 111163.3],
    [1.00E+01, 1.98E+31, -5.16E+00, 111887.8],
    [1.00E+02, 2.69E+31, -4.92E+00, 112778.7],
], dtype=jnp.float64)
fg = jnp.array([7.22998730e+29, -6.31999979e+00, 1.10862399e+05, 1.17519213e+31, -6.17806125e+00, 1.11669702e+05,
                0.00000000e+00, 1.51236600e+00, 2.72669994e+10, 5.11885045e+04], dtype=jnp.float64)
T_range = jnp.array([500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500])
P_range = jnp.array([20, 30, 40, 50])
k_plog = compute_plog(plog, T_range, P_range)
data = (T_range, P_range, k_plog)

f = lambda p: rmse_loss_function(p, data)
def hessian(f):
    return jacfwd(jacrev(f))

H = hessian(f)(fg)
print("jacfwd result, with shape", H.shape)
print(H)
