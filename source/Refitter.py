import time
import numpy as np
from jax import jit, grad, debug
import jax.numpy as jnp
import jaxopt
# Internal modules
from .ArrheniusBase import arrhenius_fit
from .PressureLogarithmic import kinetic_constant_plog, compute_plog
from .FallOff import compute_fall_off


def refit_setup(plog: jnp.ndarray, fit_type: str):
    fg = jnp.array([1.380332E+33, -4.32, 1.105952E+05, 1.18231785E+37, 9.63743640E-01, 1.50102675E+05, 5.79496168E-04,
                    1.22837204E+02, 1.0e+30, 1.0e+30])
    # fg = jnp.array([1.0e10, -1, 1.0E+05, 1.0E+10, 1.0, 1.0E+05, 5.0e-4, 1.0e2, 1.0e+30, 1.0e+30])

    # Useless at the moment
    # if fit_type == "FallOff":
    #     isFallOff = True
    # elif fit_type == "CABR":
    #     isCabr = True
    # else:
    #     raise ValueError("Unknown fit type! Available are: FallOff | CABR")

    n_range_points = 50
    T_range = jnp.linspace(500, 2500, n_range_points)
    # -1, 2 as extreme values in logspace base 10, means that the first pressure value is equal to 0.1 atm and the
    # last one to 100 atm before was -> self.P_range = jnp.linspace(0.1, 100, n_range_points)
    P_range = jnp.logspace(-1, 2, n_range_points)

    # Memory allocation
    k_plog = jnp.empty(shape=(n_range_points, n_range_points))

    start_time = time.time()
    k_plog = compute_plog(plog, T_range, P_range)
    end_time = time.time()
    execution_time = end_time - start_time

    print("==============================================")
    print(" PLOG to FallOff/CABR refritter          :)   ")
    print("==============================================")
    print(" Time to compute the PLOG constant: {:.6f} s\n".format(execution_time))

    return T_range, P_range, k_plog, fg

def compute_pressure_limits(plog: jnp.ndarray, T_range: jnp.ndarray, P_range: jnp.ndarray) -> jnp.ndarray:
    """
    This function compute the First guess Arrhenius parameters for the HPL and LPL. The computation is done in such a
    way that given the pressure and temperature range where we want to refit the PLOG expression into a FallOff or a
    CABR the kinetic constant is computed all along the Temperature interval and at the extreme value of the pressure
    interval
    """
    fg = jnp.array([.0, .0, .0, .0, .0, .0, 0.1, 0.1, 0.1, 0.1])
    k0_fg = jnp.array([kinetic_constant_plog(plog, i, P_range[0]) for i in T_range])
    kInf_fg = jnp.array([kinetic_constant_plog(plog, i, P_range[-1]) for i in T_range])

    A0_fg, b0_fg, Ea0_fg, R2adj0 = arrhenius_fit(k0_fg, T_range)
    AInf_fg, bInf_fg, EaInf_fg, R2adjInf = arrhenius_fit(kInf_fg, T_range)

    fg = fg.at[0].set(A0_fg)
    fg = fg.at[1].set(b0_fg - 1)  # A.F. did that
    fg = fg.at[2].set(Ea0_fg)

    fg = fg.at[3].set(AInf_fg)
    fg = fg.at[4].set(bInf_fg - 1)  # A.F. did that
    fg = fg.at[5].set(EaInf_fg)

    print(" Computing first guesses for the LPL and HPL")
    print("  * Adjusted R2 for the LPL: {:.3}".format(R2adj0))
    print("    - A: {:.3e}, b: {:.3}, Ea: {:.3e}".format(A0_fg, b0_fg + 1, Ea0_fg))
    print("  * Adjusted R2 for the HPL: {:.3}".format(R2adjInf))
    print("    - A: {:.3e}, b: {:.3}, Ea: {:.3e}\n".format(AInf_fg, bInf_fg + 1, EaInf_fg))

    return fg


# @jit
# def loss_function(x: jnp.ndarray, T_range: jnp.ndarray, P_range: jnp.ndarray, k_plog: jnp.ndarray) -> jnp.float64:
#     # Unpacking optimization parameters
#     A0, b0, Ea0, AInf, bInf, EaInf, A, T3, T1, T2 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]
#     refitted_constant = jnp.array([
#             [A0, b0, Ea0, 0.0],
#             [AInf, bInf, EaInf, 0.0],
#             [A, T3, T1, T2]
#     ])
#
#     k_troe = compute_fall_off(refitted_constant, T_range, P_range)
#     difference = k_troe - k_plog
#     squared_difference = difference ** 2
#     sum_of_squares = jnp.sum(squared_difference)
#     debug.print("{}", sum_of_squares)
#     return sum_of_squares


# @jit
# def loss_function(x: jnp.ndarray, T_range: jnp.ndarray, P_range: jnp.ndarray, k_plog: jnp.ndarray) -> jnp.float64:
#     # Memory allocation
#     # k_troe = jnp.empty(shape=(k_plog.shape[0], k_plog.shape[1]))
#     # Unpacking optimization parameters
#     A0, b0, Ea0, AInf, bInf, EaInf, A, T3, T1, T2 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]
#     refitted_constant = jnp.array([
#             [A0, b0, Ea0, 0.0],
#             [AInf, bInf, EaInf, 0.0],
#             [A, T3, T1, T2]
#     ])
#
#     k_troe = compute_fall_off(refitted_constant, T_range, P_range)
#
#     # The following divide will set the values where self.k_plog == 0 equal to 0 in order to handle by 0 division
#     # self.ratio = jnp.divide(self.k_troe, self.k_plog, out=jnp.zeros_like(self.k_troe), where=self.k_plog != 0)
#     ratio = k_troe / k_plog
#     squared_errors = (ratio - 1)**2
#     obj = jnp.mean(squared_errors)
#     # debug.print("{}", obj)
#     return obj


def jaxopt_loss_function(x: jnp.ndarray, data: tuple, l2reg: float = 0.01) -> jnp.float64:
    T_range, P_range, k_plog = data
    # Memory allocation
    # k_troe = jnp.empty(shape=(k_plog.shape[0], k_plog.shape[1]))
    # Unpacking optimization parameters
    A0, b0, Ea0, AInf, bInf, EaInf, A, T3, T1, T2 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]
    refitted_constant = jnp.array([
            [A0, b0, Ea0, 0.0],
            [AInf, bInf, EaInf, 0.0],
            [A, T3, T1, T2]
    ])

    k_troe = compute_fall_off(refitted_constant, T_range, P_range)

    # The following divide will set the values where self.k_plog == 0 equal to 0 in order to handle by 0 division
    # self.ratio = jnp.divide(self.k_troe, self.k_plog, out=jnp.zeros_like(self.k_troe), where=self.k_plog != 0)
    ratio = k_troe / k_plog
    squared_errors = (ratio - 1)**2
    mse_loss = jnp.mean(squared_errors)

    # Adding L2 regularization
    l2_penalty = 0.5 * l2reg * jnp.sum(x ** 2)

    # Final loss function
    total_loss = mse_loss + l2_penalty

    return total_loss


# def jaxopt_loss_function(x: jnp.ndarray, data: tuple, l2reg: float = 0.01) -> jnp.float64:
#     T_range, P_range, k_plog = data
#     # Memory allocation
#     # k_troe = jnp.empty(shape=(k_plog.shape[0], k_plog.shape[1]))
#     # Unpacking optimization parameters
#     A0, b0, Ea0, AInf, bInf, EaInf, A, T3, T1, T2 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]
#     refitted_constant = jnp.array([
#             [A0, b0, Ea0, 0.0],
#             [AInf, bInf, EaInf, 0.0],
#             [A, T3, T1, T2]
#     ])
#
#     k_troe = compute_fall_off(refitted_constant, T_range, P_range)
#
#     difference = k_troe - k_plog
#     squared_errors = difference ** 2
#     mse_loss = jnp.mean(squared_errors)
#
#     # Adding L2 regularization
#     l2_penalty = 0.5 * l2reg * jnp.sum(x ** 2)
#
#     # Final loss function
#     total_loss = mse_loss + l2_penalty
#
#     return total_loss


def fit(plog):
    np.set_printoptions(precision=6, suppress=True)
    T_range, P_range, k_plog, fg = refit_setup(plog, "FallOff")
    # fg = compute_pressure_limits(plog, T_range, P_range)

    # ---------------------
    # JAXopt
    # ---------------------
    MAX_ITER = 500
    data = (T_range, P_range, k_plog)
    solver = jaxopt.BFGS(fun=jaxopt_loss_function, maxiter=MAX_ITER)
    # solver = jaxopt.GradientDescent(fun=jaxopt_loss_function, maxiter=MAX_ITER)

    res = solver.run(fg, data)
    print("First guess for the parameters:   {}".format(fg))
    print("Optimized parameters:             {}".format(res[0]))


    # ---------------------
    # SciPY optimize
    # ---------------------
    # print("First guess for the parameters:{}".format(fg))
    # print("Initial value of the LOSS function: ", loss_function(fg, T_range, P_range, k_plog)) # This line act also as a warm start
    #
    # minimizer_options = {"maxfev": 1000, "adaptive": True}
    # minimizer_options = {"maxfev": 1000}
    # minimizer_options = {"disp": True, "eps": 1e-10}
    # # loss_gradient = np.asarray(grad(loss_function, argnums=0)(fg, T_range, P_range, k_plog))
    # result = minimize(loss_function, fg, args=(T_range, P_range, k_plog), method="SLSQP", options=minimizer_options)
    # print(result)

    # ---------------------
    # OPTAX (Experimental)
    # ---------------------
    # LOG_SCALED = False
    # MAX_ITER = 2000
    # params = fg
    # schedule = optax.linear_schedule(init_value=0.001, end_value=0.01, transition_steps=1000)
    #
    # # solver = optax.adabelief(learning_rate=schedule)
    # solver = optax.chain(
    #     # optax.clip_by_global_norm(.1),  # Limit the norm of the gradients
    #     # optax.adabelief(learning_rate=schedule)
    #     optax.sgd(learning_rate=0.001)
    # )
    #
    # if LOG_SCALED is True:
    #     scaled_params = jnp.where(params > 0, jnp.log(params), params)
    #     print("First guess for the parameters:   {}".format(params))
    #     print("   scaled parameters:             {}".format(scaled_params))
    #     print("Initial value of the LOSS function: ", loss_function(scaled_params, T_range, P_range, k_plog))
    #     opt_state = solver.init(scaled_params)
    #
    #     for _ in range(MAX_ITER):
    #         loss_gradient = grad(loss_function, argnums=0)(scaled_params, T_range, P_range, k_plog)
    #         updates, opt_state = solver.update(loss_gradient, opt_state, scaled_params)
    #         scaled_params = optax.apply_updates(scaled_params, updates)
    #         params = jnp.where(scaled_params > 0, jnp.exp(scaled_params), scaled_params)
    #         if _ % 100 == 0:
    #             row = f"{_:<10}"
    #             row += f"{loss_function(params, T_range, P_range, k_plog):<30.6f}"
    #             for i in range(10):
    #                 row += f"{loss_gradient[i]:<30.6f}"
    #             print(row)
    #
    # else:
    #     print("First guess for the parameters:{}".format(params))
    #     print("Initial value of the LOSS function: ", loss_function(params, T_range, P_range, k_plog))
    #     opt_state = solver.init(params)
    #
    #     print(f"\n{'#it':<10}{'LOSS':<30}{'GRAD1':<30}{'GRAD2':<30}{'GRAD3':<30}{'GRAD4':<30}{'GRAD5':<30}{'GRAD6':<30}{'GRAD7':<30}{'GRAD8':<30}{'GRAD9':<30}{'GRAD10':<30}")
    #     for _ in range(MAX_ITER):
    #         loss_gradient = grad(loss_function, argnums=0)(params, T_range, P_range, k_plog)
    #         updates, opt_state = solver.update(loss_gradient, opt_state, params)
    #         params = optax.apply_updates(params, updates)
    #         if _ % 100 == 0:
    #             row = f"{_:<10}"
    #             row += f"{loss_function(params, T_range, P_range, k_plog):<30.6f}"
    #             for i in range(10):
    #                 row += f"{loss_gradient[i]:<30.6f}"
    #             print(row)
    #
    # print("Optimized parameters: {}".format(params))
    # print("Final value of the LOSS function: {}".format(loss_function(params, T_range, P_range, k_plog)))
