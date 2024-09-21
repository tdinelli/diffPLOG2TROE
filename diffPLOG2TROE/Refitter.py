from jax import jit
import jax.numpy as jnp
# Internal modules
from .arrhenius_base import arrhenius_fit
from .pressure_logarithmic import kinetic_constant_plog
from .fall_off import compute_falloff


def compute_pressure_limits(plog: jnp.ndarray, fg: jnp.ndarray, T_range: jnp.ndarray, P_range: jnp.ndarray) -> jnp.ndarray:
    """
    This function compute the First guess Arrhenius parameters for the HPL and LPL. The computation is done in such a
    way that given the pressure and temperature range where we want to refit the PLOG expression into a FallOff or a
    CABR the kinetic constant is computed all along the Temperature interval and at the extreme value of the pressure
    interval
    """
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


@jit
def rmse_loss_function(x: jnp.ndarray, data: tuple) -> jnp.float64:
    # Unpacking data
    T_range, P_range, k_plog = data
    # Unpacking optimization parameters
    A0, b0, Ea0, AInf, bInf, EaInf, A, T3, T1, T2 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]
    refitted_constant = jnp.array([
        [A0, b0, Ea0, 0.0],
        [AInf, bInf, EaInf, 0.0],
        [A, T3, T1, T2]
    ], dtype = jnp.float64)

    k_troe = compute_fall_off(refitted_constant, T_range, P_range)

    squared_errors = (k_troe - k_plog)**2
    mse_loss = jnp.mean(squared_errors)

    return mse_loss


@jit
def ratio_loss_function(x: jnp.ndarray, data: tuple) -> jnp.float64:
    # Unpacking data
    T_range, P_range, k_plog = data
    # Unpacking optimization parameters
    A0, b0, Ea0, AInf, bInf, EaInf, A, T3, T1, T2 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]

    refitted_constant = jnp.array([
        [A0, b0, Ea0, 0.0],
        [AInf, bInf, EaInf, 0.0],
        [A, T3, T1, T2]
    ], dtype=jnp.float64)

    k_troe = compute_fall_off(refitted_constant, T_range, P_range)

    ratio = jnp.divide(k_troe , k_plog)
    squared_errors = (ratio - 1)**2
    mse_loss = jnp.mean(squared_errors)

    return mse_loss
