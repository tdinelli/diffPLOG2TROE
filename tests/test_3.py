import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

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


def optimize(omega, constraint_type="single_param_box", num_iterations=1000, learning_rate=0.01):
    def loss_fn(params):
        return ciao(params)

    objective = loss_fn
    params = omega

    # Set up projection based on constraint type
    if constraint_type == "single_param_box":
        # Box constraint only for params[3]
        # Adjust these bounds based on your needs
        lower_bound = 0.0  # Lower bound for params[3]
        upper_bound = 1.0  # Upper bound for params[3]
        project = lambda p: project_single_param_box(p, param_idx=3, lower=lower_bound, upper=upper_bound)
    else:
        # No projection (unconstrained)
        project = lambda p: p

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
        return params, opt_state, loss

    losses = []
    param_history = [params.copy()]

    for i in range(num_iterations):
        params, opt_state, loss = update_step(params, opt_state)
        losses.append(loss)
        param_history.append(params.copy())

        if i % 200 == 0:
            print(f"Iteration {i}, Loss: {loss:.6f}, params[3]: {params[3]:.6f}")

    return params, losses, jnp.array(param_history)


def main():
    # ========================================================================
    # Initial guess for the optimization
    initial_guess = jnp.array(
        [jnp.log(9.954e29), -3.959, 2.807e03 / 1.987, 0.500, 9.800e02, 2.800e02, 2.100e03],
        # [jnp.log(9.954e29), -3.959, 2.807e03 / 1.987, 0.500, 9.800e02, 1e30, 1e30],
        dtype=jnp.float64,
    )

    # First run unconstrained optimization
    print("\n=== Running unconstrained optimization ===")
    unconstrained_params, unconstrained_losses, unconstrained_history = optimize(
        initial_guess, constraint_type="none", num_iterations=2000, learning_rate=0.01
    )

    # Then run constrained optimization
    print("\n=== Running constrained optimization (box constraint on params[3]) ===")
    constrained_params, constrained_losses, constrained_history = optimize(
        initial_guess, constraint_type="single_param_box", num_iterations=2000, learning_rate=0.01
    )

    # Print final results
    print("\n=== Final Results ===")
    print(f"Unconstrained final params[3]: {unconstrained_params[3]:.6f}")
    print(f"Constrained final params[3]: {constrained_params[3]:.6f}")
    print(f"Unconstrained final loss: {ciao(unconstrained_params):.6f}")
    print(f"Constrained final loss: {ciao(constrained_params):.6f}")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot losses
    axes[0, 0].plot(unconstrained_losses, label="Unconstrained")
    axes[0, 0].plot(constrained_losses, label="Constrained")
    axes[0, 0].set_title("Loss vs. Iterations")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot param[3] trajectory
    axes[0, 1].plot([h[3] for h in unconstrained_history], label="Unconstrained")
    axes[0, 1].plot([h[3] for h in constrained_history], label="Constrained")
    axes[0, 1].axhline(y=0.0, color="r", linestyle="--", alpha=0.7)  # Lower bound
    axes[0, 1].axhline(y=1.0, color="r", linestyle="--", alpha=0.7)  # Upper bound
    axes[0, 1].set_title("params[3] vs. Iterations")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("params[3] value")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot ratio maps for both solutions
    for idx, (params, title) in enumerate(
        [
            (unconstrained_params, "Unconstrained Solution Ratio (k_troe/k_plog)"),
            (constrained_params, "Constrained Solution Ratio (k_troe/k_plog)"),
        ]
    ):
        falloff = FallOff(
            name="NH2+H=NH3",
            hpl_params=jnp.array([1.500e14, 0.167, 0.000e00]),
            lpl_params=jnp.array([jnp.exp(params[0]), params[1], params[2] * 1.987]),
            falloff_params=jnp.array([params[3], params[4], params[5], params[6]]),
            falloff_type="troe",
        )
        k_troe = falloff.kinetic_constant(T_range, P_range)[0]
        ratio = k_troe.T / k_plog.T

        im = axes[1, idx].imshow(
            ratio,
            origin="lower",
            aspect="auto",
            extent=[P_range[0], P_range[-1], T_range[0], T_range[-1]],
            cmap="Pastel2",
            vmin=0.5,
            vmax=1.5,
        )
        axes[1, idx].set_title(title)
        axes[1, idx].set_xscale("log")
        axes[1, idx].set_xlabel("Pressure [bar]", fontweight="bold")
        axes[1, idx].set_ylabel("Temperature [K]", fontweight="bold")
        plt.colorbar(im, ax=axes[1, idx])

    fig.tight_layout()
    plt.show()

    # Print the final falloff models
    print("\n=== Unconstrained Falloff Model ===")
    falloff_unconstrained = FallOff(
        name="NH2+H=NH3",
        hpl_params=jnp.array([1.500e14, 0.167, 0.000e00]),
        lpl_params=jnp.array(
            [jnp.exp(unconstrained_params[0]), unconstrained_params[1], unconstrained_params[2] * 1.987]
        ),
        falloff_params=jnp.array(
            [unconstrained_params[3], unconstrained_params[4], unconstrained_params[5], unconstrained_params[6]]
        ),
        falloff_type="troe",
    )
    print(f"{falloff_unconstrained}")

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
