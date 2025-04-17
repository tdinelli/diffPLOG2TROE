import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from optimization_utils import plot_optimization_results, print_optimization_table

from diffPLOG2TROE.kinetic_constants import FallOff, Plog


# ========================================================================
# Initial input variables
T_range = jnp.linspace(500, 2500, 100)
P_range = jnp.logspace(jnp.log10(0.1), jnp.log10(100), 100)
plog_dictionary = {
    "name": "NH2+H=NH3",
    "type": "plog",
    "rate-constant": {
        "coefficients": [
            [1.00000e-01, 1.21300e27, -4.95900e00, 2.80709e03],
            [1.00000e00, 5.86700e27, -4.86700e00, 3.10796e03],
            [1.00000e01, 3.31400e28, -4.79800e00, 3.83238e03],
            [1.00000e02, 4.51200e28, -4.56300e00, 4.72319e03],
        ]
    },
}
starting_plog = Plog(plog_dictionary)

# ========================================================================
# Generate the training data
k_plog = starting_plog.kinetic_constant(T_range, P_range)


def ciao(params):
    falloff = FallOff(
        name="",
        hpl_params=jnp.array([1.500e14, 0.167, 0.000e00]),
        lpl_params=jnp.array([jnp.exp(params[0]), params[1], params[2] * 1.987]),
        falloff_params=jnp.array([params[3], params[4], params[5], params[6]]),
        falloff_type="troe",
    )
    predictions = falloff.kinetic_constant(T_range, P_range)[0]
    return jnp.mean((jnp.log(predictions) - jnp.log(k_plog)) ** 2)


def project_single_param_box(params, param_idx, lower, upper):
    """
    Project a single parameter to a box constraint while leaving others unchanged.

    Args:
        params: Parameter array
        param_idx: Index of the parameter to constrain
        lower: Lower bound for the constrained parameter
        upper: Upper bound for the constrained parameter

    Returns:
        Updated parameter array with the constraint applied
    """
    # Get the parameter to constrain
    param_value = params[param_idx]

    # Apply box constraint
    constrained_value = jnp.clip(param_value, lower, upper)

    # Create updated parameters
    updated_params = params.at[param_idx].set(constrained_value)

    return updated_params


def optimize(omega, num_iterations=1000, learning_rate=0.01):
    def loss_fn(params):
        return ciao(params)

    objective = loss_fn
    params = omega

    lower_bound = 0.0  # Lower bound for params[3]
    upper_bound = 1.0  # Upper bound for params[3]
    project = lambda p: project_single_param_box(p, param_idx=3, lower=lower_bound, upper=upper_bound)

    optimizer = optax.lbfgs(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def update_step(params, opt_state):
        # Compute gradients
        loss, grads = jax.value_and_grad(objective)(params)

        # Apply optimizer update
        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            params=params,
            value=loss,
            grad=grads,
            value_fn=loss_fn,
        )
        params = optax.apply_updates(params, updates)

        # Apply projection to enforce constraints
        params = project(params)  # Now using our projection function

        return params, opt_state, loss, grads

    losses = []
    param_history = [params.copy()]
    gradient_history = []

    for i in range(num_iterations):
        params, opt_state, loss, grads = update_step(params, opt_state)
        losses.append(loss)
        param_history.append(params.copy())
        gradient_history.append(grads)

        if i % 200 == 0:
            print(f"Iteration {i}, Loss: {loss:.6f}, params[3]: {params[3]:.6f}")
            # Print gradient norm
            grad_norm = jnp.linalg.norm(grads)
            print(f"Gradient norm: {grad_norm:.6e}")

    return params, losses, jnp.array(param_history), jnp.array(gradient_history)


def main():
    # ========================================================================
    # Initial guess for the optimization
    initial_guess = jnp.array(
        [jnp.log(9.954e29), -3.959, 2.807e03 / 1.987, 0.500, 9.800e02, 2.800e02, 2.100e03],
        # [jnp.log(9.954e29), -3.959, 2.807e03 / 1.987, 0.500, 9.800e02, 1e30, 1e30],
        dtype=jnp.float64,
    )

    constrained_params, constrained_losses, constrained_history, gradient_history = optimize(
        initial_guess, num_iterations=2000, learning_rate=0.01
    )

    print("\n=== Final Results ===")
    print(f"Constrained final params[3]: {constrained_params[3]:.6f}")
    print(f"Constrained final loss: {ciao(constrained_params):.6f}")

    # Parameter names for reference
    param_names = ["ln(A) [cm³/mol·s]", "n", "E/R [K]", "α", "T*** [K]", "T* [K]", "T** [K]"]

    # Print optimization table
    print_optimization_table(
        constrained_losses, constrained_history, gradient_history, param_names, iterations_to_print=20
    )

    # Plot all optimization data
    figures = plot_optimization_results(constrained_losses, constrained_history, gradient_history, param_names)

    # Create the heatmap plot
    fig = plt.figure(figsize=(12, 8))
    ax3 = fig.add_subplot(111)

    # Create and plot the TROE falloff model
    falloff = FallOff(
        name="NH2+H=NH3",
        hpl_params=jnp.array([1.500e14, 0.167, 0.000e00]),
        lpl_params=jnp.array([jnp.exp(constrained_params[0]), constrained_params[1], constrained_params[2] * 1.987]),
        falloff_params=jnp.array(
            [constrained_params[3], constrained_params[4], constrained_params[5], constrained_params[6]]
        ),
        falloff_type="troe",
    )
    k_troe = falloff.kinetic_constant(T_range, P_range)[0]
    ratio = k_troe.T / k_plog.T

    # Plot the heatmap
    im = ax3.imshow(
        ratio,
        origin="lower",
        aspect="auto",
        extent=[P_range[0], P_range[-1], T_range[0], T_range[-1]],
        # cmap="Pastel2",
        cmap="jet",
        vmin=0.5,
        vmax=1.5,
    )
    ax3.set_xscale("log")
    ax3.set_xlabel("Pressure [bar]", fontweight="bold")
    ax3.set_ylabel("Temperature [K]", fontweight="bold")
    ax3.set_title("Ratio of TROE/PLOG Rate Constants", fontweight="bold")
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label("k_troe/k_plog", fontweight="bold")

    fig.tight_layout()
    plt.savefig("troe_plog_ratio.png", dpi=300)
    plt.show()

    print("\n=== Constrained Falloff Model ===")
    falloff_constrained = FallOff(
        name="NH2+H=NH3",
        hpl_params=jnp.array([1.500e14, 0.167, 0.000e00]),
        lpl_params=jnp.array([jnp.exp(constrained_params[0]), constrained_params[1], constrained_params[2] * 1.987]),
        falloff_params=jnp.array(
            [constrained_params[3], constrained_params[4], constrained_params[5], constrained_params[6]]
        ),
        falloff_type="troe",
    )
    print(f"{falloff_constrained}")


if __name__ == "__main__":
    main()
