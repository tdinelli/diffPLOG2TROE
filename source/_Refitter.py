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


# def refit_setup(plog: jnp.ndarray, fit_type: str):
#     fg = jnp.array([1.2295E+33, 6.0615E-01, 2.9645E+5, 1.07077E+37, 5.34428E-01, 3.50434E+5, .0, 1.512366, 2.72669994E+10, 5.11885045E+04], dtype=jnp.float64)
#
#     # Useless at the moment
#     # if fit_type == "FallOff":
#     #     isFallOff = True
#     # elif fit_type == "CABR":
#     #     isCabr = True
#     # else:
#     #     raise ValueError("Unknown fit type! Available are: FallOff | CABR")
#
#     # n_range_points = 50
#     # T_range = jnp.linspace(500, 2500, n_range_points)
#     # # -1, 2 as extreme values in logspace base 10, means that the first pressure value is equal to 0.1 atm and the
#     # # last one to 100 atm before was -> self.P_range = jnp.linspace(0.1, 100, n_range_points)
#     # # P_range = jnp.logspace(-1, 2, n_range_points)
#     # P_range = jnp.logspace(1, 1, n_range_points)
#     # # Memory allocation
#     # k_plog = jnp.empty(shape=(n_range_points, n_range_points))
#
#     # From AF excel just for testing
#     T_range = jnp.array([500, 750,1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500])
#     P_range = jnp.array([0.1, 0.4, 0.7, 1, 2, 5, 10, 20, 30, 40, 50, 70, 100])
#     P_range = jnp.array([0.1, 0.4, 0.7, 1, 2, 5])
#
#     start_time = time.time()
#     k_plog = compute_plog(plog, T_range, P_range)
#     end_time = time.time()
#     execution_time = end_time - start_time
#
#     print("==============================================")
#     print(" PLOG to FallOff/CABR refritter          :)   ")
#     print("==============================================")
#     print(" Time to compute the PLOG constant: {:.6f} s\n".format(execution_time))
#
#     return T_range, P_range, k_plog, fg


# def fit(plog):
#     T_range, P_range, k_plog, fg = refit_setup(plog, "FallOff")
#     fg = compute_pressure_limits(plog, fg, T_range, P_range)
#
#     print(" First guess for the parameters: {}".format(fg))
#     # ---------------------
#     # JAXopt
#     # ---------------------
#     MAX_ITER = 300
#     TOL = 0.1
#     MAX_STEPSIZE = 1.0
#
#     # data = (T_range, P_range, k_plog)
#
#     # ONLY TROE PARAMS
#     data = (T_range, P_range, k_plog, fg) # only troe
#     fg = fg[5:-1]
#
#     solver = jaxopt.BFGS(fun=onlytroe_loss_function, maxiter=MAX_ITER)
#     # solver = jaxopt.GradientDescent(fun=jaxopt_loss_function, maxiter=MAX_ITER, tol=0.00001)
#     # solver = jaxopt.PolyakSGD(fun=jaxopt_loss_function, maxiter=MAX_ITER, tol=0.00001)
#
#     # solver = jaxopt.NonlinearCG(
#     #     fun=onlytroe_loss_function,
#     #     maxiter=MAX_ITER,
#     #     tol=TOL,
#     #     method="polak-ribiere",
#     #     linesearch="zoom",
#     #     linesearch_init="increase",
#     #     condition=None,
#     #     maxls=15,
#     #     decrease_factor=None,
#     #     increase_factor=1.2,
#     #     max_stepsize=MAX_STEPSIZE,
#     #     min_stepsize=1e-06,
#     #     implicit_diff=True,
#     #     implicit_diff_solve=None,
#     #     unroll="auto",
#     #     verbose=1
#     # )
#
#     val, res = solver.run(fg, data)
#
#     print(val)
#     optimized_constant = jnp.array([
#         [val[0], val[1], val[2], 0.0],
#         [val[3], val[4], val[5], 0.0],
#         [val[6], val[7], val[8], val[9]]
#     ])
#     k_troe = compute_fall_off(optimized_constant, T_range, P_range)
#
#     return (k_troe, k_plog)
