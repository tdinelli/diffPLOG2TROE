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
        hpl_params=jnp.array([params[0], params[1], params[2]]),
        lpl_params=jnp.array([params[3], params[4], params[5]]),
        falloff_params=jnp.array([params[6], params[7], params[8], params[9]]),
        falloff_type="troe",
    )
    predictions = falloff.kinetic_constant(T_range, P_range)[0]
    return jnp.mean((jnp.log(predictions) - jnp.log(k_plog)) ** 2)


def optimize(omega, constraint_type="", num_iterations=1000, learning_rate=0.01):
    def loss_fn(params):
        return ciao(params)

    objective = loss_fn
    params = omega

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def update_step(params, opt_state):
        # Compute gradients
        loss, grads = jax.value_and_grad(objective)(params)
        # Apply optimizer update
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        # Apply projection to enforce constraints
        # params = project(params)
        return params, opt_state, loss

    losses = []
    for i in range(num_iterations):
        params, opt_state, loss = update_step(params, opt_state)
        losses.append(loss)
        if i % 200 == 0:
            print(f"Iteration {i}, Loss: {loss:.6f}")

    return params, losses


def main():
    # ========================================================================
    # Initial guess for the optimization
    initial_guess = jnp.array(
        [1.500e14, 0.167, 0.000e00, 9.954e29, -3.959, 2.807e03, 0.500, 9.800e02, 2.800e02, 2.100e03], dtype=jnp.float64
    )

    results = {}
    params, losses = optimize(initial_guess, num_iterations=100000, learning_rate=0.01)
    print(f"Final parameters: {params}")
    results["None"] = (params, losses)

    # Check loss and constraint satisfaction
    final_loss = ciao(params)
    print(f"Final loss: {final_loss:.6f}")

    fig, ax = plt.subplots(1, 2, figsize=(9, 8))
    ax = ax.flatten()

    # Fig 1
    ax[0].plot(losses)
    ax[0].set_yscale("log")

    # Fig 2
    k_plog = starting_plog.kinetic_constant(T_range, P_range)
    falloff = FallOff(
        name="",
        hpl_params=jnp.array([params[0], params[1], params[2]]),
        lpl_params=jnp.array([params[3], params[4], params[5]]),
        falloff_params=jnp.array([params[6], params[7], params[8], params[9]]),
        falloff_type="troe",
    )
    k_troe = falloff.kinetic_constant(T_range, P_range)[0]

    ratio = k_troe.T / k_plog.T

    im = ax[1].imshow(
        ratio, origin="lower", aspect="auto", extent=[P_range[0], P_range[-1], T_range[0], T_range[-1]], cmap="Pastel2"
    )

    ax[1].set_xscale("log")
    ax[1].set_xlabel("Pressure [bar]", fontweight="bold")
    ax[1].set_ylabel("Temperature [K]", fontweight="bold")
    cbar = plt.colorbar(im, orientation="horizontal", pad=0.2)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
