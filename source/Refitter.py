import time
from jax import jit, grad, debug
import jax.numpy as jnp
import jaxopt
# Internal modules
from .ArrheniusBase import arrhenius_fit
from .PressureLogarithmic import kinetic_constant_plog, compute_plog
from .FallOff import compute_fall_off


def refit_setup(plog: jnp.ndarray, fit_type: str):
    fg = jnp.array([1.2295E+33, 6.0615E-01, 2.9645E+5, 1.07077E+37, 5.34428E-01, 3.50434E+5, .0, 1.512366, 2.72669994E+10, 5.11885045E+04], dtype=jnp.float64)

    # Useless at the moment
    # if fit_type == "FallOff":
    #     isFallOff = True
    # elif fit_type == "CABR":
    #     isCabr = True
    # else:
    #     raise ValueError("Unknown fit type! Available are: FallOff | CABR")

    # n_range_points = 50
    # T_range = jnp.linspace(500, 2500, n_range_points)
    # # -1, 2 as extreme values in logspace base 10, means that the first pressure value is equal to 0.1 atm and the
    # # last one to 100 atm before was -> self.P_range = jnp.linspace(0.1, 100, n_range_points)
    # # P_range = jnp.logspace(-1, 2, n_range_points)
    # P_range = jnp.logspace(1, 1, n_range_points)
    # # Memory allocation
    # k_plog = jnp.empty(shape=(n_range_points, n_range_points))

    # From AF excel just for testing
    T_range = jnp.array([500, 750,1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500])
    P_range = jnp.array([0.1, 0.4, 0.7, 1, 2, 5, 10, 20, 30, 40, 50, 70, 100])
    P_range = jnp.array([0.1, 0.4, 0.7, 1, 2, 5])

    start_time = time.time()
    k_plog = compute_plog(plog, T_range, P_range)
    end_time = time.time()
    execution_time = end_time - start_time

    print("==============================================")
    print(" PLOG to FallOff/CABR refritter          :)   ")
    print("==============================================")
    print(" Time to compute the PLOG constant: {:.6f} s\n".format(
        execution_time))

    return T_range, P_range, k_plog, fg


def compute_pressure_limits(plog: jnp.ndarray, fg: jnp.ndarray, T_range: jnp.ndarray, P_range: jnp.ndarray) -> jnp.ndarray:
    """
    This function compute the First guess Arrhenius parameters for the HPL and LPL. The computation is done in such a
    way that given the pressure and temperature range where we want to refit the PLOG expression into a FallOff or a
    CABR the kinetic constant is computed all along the Temperature interval and at the extreme value of the pressure
    interval
    """
    k0_fg = jnp.array([kinetic_constant_plog(plog, i, P_range[0])
                      for i in T_range])
    kInf_fg = jnp.array([kinetic_constant_plog(
        plog, i, P_range[-1]) for i in T_range])

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


@jit
def jaxopt_loss_function(x_norm: jnp.ndarray, data: tuple, l2reg: jnp.float64 = 0.) -> jnp.float64:
    # Unpacking data
    T_range, P_range, k_plog = data

    # Denormalize optimization parameters
    # x = denormalize_params(x_norm)
    x = x_norm

    # Unpacking optimization parameters
    A0, b0, Ea0, AInf, bInf, EaInf, A, T3, T1, T2 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]
    refitted_constant = jnp.array([
        [A0, b0, Ea0, 0.0],
        [AInf, bInf, EaInf, 0.0],
        [A, T3, T1, T2]
    ])

    k_troe = compute_fall_off(refitted_constant, T_range, P_range)

    # debug.print("PLOG -> {}", k_plog[:, 0])
    # debug.print("TROE -> {}", k_troe[:, 0])

    ratio = k_troe / k_plog
    squared_errors = (ratio - 1)**2
    mse_loss = jnp.mean(squared_errors)

    # Adding L2 regularization
    l2_penalty = 0.5 * l2reg * jnp.sum(x ** 2)

    # Final loss function
    total_loss = mse_loss + l2_penalty

    # debug.print("Params -> {}", x_norm)
    # debug.print("Params unnorm -> {}", x)
    debug.print("Loss -> {}\n", total_loss)
    return total_loss


def normalize_params(params: jnp.ndarray) -> jnp.ndarray:
    # This is log1p NOT log
    return jnp.sign(params) * jnp.log1p(jnp.abs(params))


def denormalize_params(params_norm: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(params_norm) * (jnp.expm1(jnp.abs(params_norm)))


def fit(plog):
    T_range, P_range, k_plog, fg = refit_setup(plog, "FallOff")
    fg = compute_pressure_limits(plog, fg, T_range, P_range)

    normalized_fg = fg
    print(" First guess for the parameters: {}".format(fg))
    # print("  - Normalized parameters: {}".format(normalized_fg))
    # print("  - De normalized parameters: {}".format(denormalize_params(normalized_fg)))

    # ---------------------
    # JAXopt
    # ---------------------
    MAX_ITER = 300
    data = (T_range, P_range, k_plog)
    solver = jaxopt.BFGS(fun=jaxopt_loss_function, maxiter=MAX_ITER)
    # solver = jaxopt.GradientDescent(fun=jaxopt_loss_function, maxiter=MAX_ITER)
    # solver = jaxopt.PolyakSGD(fun=jaxopt_loss_function, maxiter=MAX_ITER)

    res = solver.run(normalized_fg, data)
    print(" Optimized parameters: {}".format(res[0]))
    print(res)
