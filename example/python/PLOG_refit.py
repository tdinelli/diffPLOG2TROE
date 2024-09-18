from jax import value_and_grad, jacfwd, jacrev, grad
import jax.numpy as jnp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import sys
sys.path.append("/Users/tdinelli/Documents/GitHub/PLOG_Converter")
from source.PressureLogarithmic import compute_plog
from source.FallOff import compute_fall_off
from source.Refitter import rmse_loss_function, ratio_loss_function, compute_pressure_limits

plog = jnp.array([
    [1.00E-01, 7.23E+29, -5.32E+00, 110862.4],
    [1.00E+00, 3.50E+30, -5.22E+00, 111163.3],
    [1.00E+01, 1.98E+31, -5.16E+00, 111887.8],
    [1.00E+02, 2.69E+31, -4.92E+00, 112778.7],
], dtype=jnp.float64)

# T_range = jnp.array([500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500])
# P_range = jnp.array([0.1, 0.4, 0.7, 1, 2, 5, 10, 20, 30, 40, 50, 70, 100])
# P_range = jnp.array([0.1, 0.4, 0.7, 1, 2, 5])
# P_range = jnp.array([20, 30, 40, 50])
T_range = jnp.linspace(500, 2500, 20)
P_range = jnp.logspace(-1, 2, 10)

# fg = jnp.array([7.22998730e+29, -6.31999979e+00, 1.10862399e+05, 1.17519213e+31, -6.17806125e+00, 1.11669702e+05,
# 0.00000000e+00, 1.51236600e+00, 2.72669994e+10, 5.11885045e+04], dtype=jnp.float64)
fg = jnp.array([1.e+30, .0, 1.e+05, 1.e+30, .0, 1.e+05, 0.00000000e+00, 1.51236600e+00, 2.72669994e+10, 5.11885045e+04], dtype=jnp.float64)
fg = compute_pressure_limits(plog, fg, T_range, P_range)

k_plog = compute_plog(plog, T_range, P_range)
data = (T_range, P_range, k_plog)

# loss_function = rmse_loss_function
loss_function = ratio_loss_function

f = lambda p, data: loss_function(p, data)
def hessian(f):
    return jacfwd(jacrev(f))

jacobian = grad(loss_function)
loss_val, loss_grad = value_and_grad(loss_function)(fg, data)
jacos = jacobian(fg, data)
hesso = hessian(f)(fg, data)

print(f"First guess value of the parameters {fg}")
print(f"Initial value of the Loss Function {loss_val}")

# NELDER-MEAD
res = minimize(
    loss_function,
    fg,
    args=(data,),
    method='nelder-mead',
    # method='powell',
    options={'xatol': 1e-8, 'maxfev': 1000000, 'disp': True}
)

# BFGS (Doesn't work well)
# res = minimize(
#     loss_function,
#     fg,
#     args=(data,),
#     jac=jacobian,
#     method='BFGS',
#     options={'xtol': 1e-8, 'disp': True}
# )

# Newton Conjugate gradient
# res = minimize(
#     loss_function,
#     fg,
#     args=(data,),
#     jac=jacobian,
#     hess=hessian(f),
#     method='Newton-CG',
#     options={'xtol': 1e-20, 'disp': True, 'return_all': True}
# )

# res = minimize(
#     loss_function,
#     fg,
#     args=(data,),
#     jac=jacobian,
#     hess=hessian(f),
#     method='trust-ncg',
#     options={'gtol': 1e-8, 'maxiter': 100000, 'disp': True}
# )

loss_val, loss_grad = value_and_grad(loss_function)(res.x, data)
print(f"Final value of the Loss Function {loss_val}")

print(res.x)

optimized_constant = jnp.array([
    [res.x[0], res.x[1], res.x[2], 0.0],
    [res.x[3], res.x[4], res.x[5], 0.0],
    [res.x[6], res.x[7], res.x[8], res.x[9]]
])
k_troe = compute_fall_off(optimized_constant, T_range, P_range)

ratio = k_troe / k_plog
fig, ax = plt.subplots()

c = ax.imshow(ratio, cmap='hot', interpolation='nearest')
# ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)

plt.show()
