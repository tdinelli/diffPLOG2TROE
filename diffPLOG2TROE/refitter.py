#################################################### Modules ###########################################################
from functools import partial
from typing import NamedTuple
# JAX
from jax import jit, lax, debug
import chex
import jax.numpy as jnp
# OPTAX
import optax
import optax.tree_utils as otu
# Internal modules
from .arrhenius_base import arrhenius_fit
from .pressure_logarithmic import compute_plog, kinetic_constant_plog
from .falloff import compute_falloff
########################################################################################################################


def compute_pressure_limits(plog: jnp.ndarray, fg: jnp.ndarray, T_range: jnp.ndarray, P_range: jnp.ndarray) -> jnp.ndarray:
    """
    This function compute the First guess Arrhenius parameters for the HPL and LPL. The computation is done in such a
    way that given the pressure and temperature range where we want to refit the PLOG expression into a FallOff or a
    CABR the kinetic constant is computed all along the Temperature interval and at the extreme value of the pressure
    interval
    """
    k0_fg = jnp.array([kinetic_constant_plog(plog, i, P_range[0]) for i in T_range])
    kInf_fg = jnp.array([kinetic_constant_plog(plog, i, P_range[-1]) for i in T_range])

    A0_fg, b0_fg, Ea0_fg, R2adj0 = arrhenius_fit(k0_fg, T_range, None)
    AInf_fg, bInf_fg, EaInf_fg, R2adjInf = arrhenius_fit(kInf_fg, T_range, None)

    fg = fg.at[0].set(A0_fg)
    fg = fg.at[1].set(b0_fg)
    fg = fg.at[2].set(Ea0_fg)

    fg = fg.at[3].set(AInf_fg)
    fg = fg.at[4].set(bInf_fg)
    fg = fg.at[5].set(EaInf_fg)

    print(" Computing first guesses for the LPL and HPL")
    print("  * Adjusted R2 for the LPL: {:.7}".format(R2adj0))
    print("    - A: {:.7e}, b: {:.7}, Ea: {:.7e}".format(A0_fg, b0_fg, Ea0_fg))
    print("  * Adjusted R2 for the HPL: {:.7}".format(R2adjInf))
    print("    - A: {:.7e}, b: {:.7}, Ea: {:.7e}\n".format(AInf_fg, bInf_fg, EaInf_fg))

    return fg


@jit
def rmse_loss_function(x: jnp.ndarray, data: tuple) -> jnp.float64:
    def full_troe() -> tuple:
        A0, b0, Ea0, AInf, bInf, EaInf, A, T3, T1, T2 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]
        refitted_constant = (
            jnp.array([
                [AInf, bInf, EaInf, 0.0],
                [A0, b0, Ea0, 0.0],
                [A, T3, T1, T2]], dtype=jnp.float64
            ),
            1
        )

        return refitted_constant

    def troe_params_only() -> tuple:
        A0 = additional[0]
        b0 = additional[1]
        Ea0 = additional[2]
        AInf = additional[3]
        bInf = additional[4]
        EaInf = additional[5]
        A, T3, T1, T2 = x[0], x[1], x[2], x[3]

        refitted_constant = (
            jnp.array([
                [AInf, bInf, EaInf, 0.0],
                [A0, b0, Ea0, 0.0],
                [A, T3, T1, T2]], dtype=jnp.float64
            ),
            1
        )

        return refitted_constant

    def kinetic_params_only() -> tuple:
        A0, b0, Ea0, AInf, bInf, EaInf = x[0], x[1], x[2], x[3], x[4], x[5]
        A, T3, T1, T2 = additional[0], additional[1], additional[2], additional[3]

        refitted_constant = (
            jnp.array([
                [AInf, bInf, EaInf, 0.0],
                [A0, b0, Ea0, 0.0],
                [A, T3, T1, T2]], dtype=jnp.float64
            ),
            1
        )

        return refitted_constant

    # Normalize the data to always have 4 elements
    T_range, P_range, k_plog, additional = data if len(data) == 4 else (*data, [1]*6)

    refitted_constant = lax.cond(
        len(data) == 3,
        lambda: full_troe(),
        lambda: lax.cond(
            len(additional) == 4,
            lambda: kinetic_params_only(),
            lambda: troe_params_only(),
        )
    )

    k_troe = compute_falloff(refitted_constant, T_range, P_range)

    lnk_plog = jnp.log(k_plog)
    lnk_troe = jnp.log(k_troe)

    squared_errors = (lnk_troe - lnk_plog) ** 2
    mse_loss = jnp.mean(squared_errors)

    return mse_loss


def refit_plog(plog: jnp.ndarray, P: jnp.float64):
    def find_closest_index(array, value):
        differences = jnp.abs(array - value)
        return jnp.argmin(differences)

    T_range = jnp.linspace(500, 2500, 100)
    P_range = jnp.array([P])
    k_plog = compute_plog(plog, T_range, P_range)[0]
    _pressure_levels = plog[:, 0]

    idx_fg = find_closest_index(_pressure_levels, P)
    if P in _pressure_levels:
        A, b, Ea, R2adj = plog[idx_fg][1], plog[idx_fg][2], plog[idx_fg][3], 1.0
        first_guess = jnp.array([plog[idx_fg][1], plog[idx_fg][2], plog[idx_fg][3], 1.0])
    else:
        first_guess = jnp.array([jnp.log(plog[idx_fg][1]), plog[idx_fg][2], plog[idx_fg][3]])
        A, b, Ea, R2adj = arrhenius_fit(k_plog, T_range, first_guess)
        first_guess = jnp.array([plog[idx_fg][1], plog[idx_fg][2], plog[idx_fg][3]])

    return A, b, Ea, R2adj, first_guess


# Definition of the L-BFGS solver adapted from: https://optax.readthedocs.io/en/latest/_collections/examples/lbfgs.html
def run_lbfgs(init_params, fun, opt, max_iter, tol, *args):
    value_and_grad_fun = optax.value_and_grad_from_state(partial(fun, *args))
    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=partial(fun, *args)
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params, final_state


class InfoState(NamedTuple):
    iter_num: chex.Numeric


def print_info(frequency=1):
    def init_fn(params):
        del params
        return InfoState(iter_num=0)

    def update_fn(updates, state, params, *, value, grad, **extra_args):
        del params, extra_args

        def print_fn(_):
            debug.print(
                'Iteration: {i}, Value: {v}, Gradient norm: {e}',
                i=state.iter_num,
                v=value,
                e=otu.tree_l2_norm(grad)
            )
            return updates, InfoState(iter_num=state.iter_num + 1)

        def no_print_fn(_):
            return updates, InfoState(iter_num=state.iter_num + 1)

        should_print = (state.iter_num % frequency) == 0
        return lax.cond(should_print, print_fn, no_print_fn, operand=None)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
