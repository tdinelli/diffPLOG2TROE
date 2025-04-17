import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy.optimize import minimize


# Set random seed for reproducibility
jax.config.update("jax_enable_x64", True)  # Use double precision for better numerical stability


# Define the Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
def rosenbrock(params):
    """
    Rosenbrock function for arbitrary dimensions.
    For 2D: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    """
    x_vals = params[:-1]
    y_vals = params[1:]
    return jnp.sum((1 - x_vals) ** 2 + 100 * (y_vals - x_vals**2) ** 2)


# Define constraint types and their projection functions
def get_projection(constraint_type, **kwargs):
    """Return the appropriate projection function for the given constraint type."""
    if constraint_type == "box":
        lower = kwargs.get("lower", -1.5)
        upper = kwargs.get("upper", 1.5)
        return lambda p: optax.projections.projection_box(p, lower, upper)

    elif constraint_type == "l2_ball":
        radius = kwargs.get("radius", 1.5)
        return lambda p: optax.projections.projection_l2_ball(p, scale=radius)

    elif constraint_type == "l1_ball":
        radius = kwargs.get("radius", 1.5)
        return lambda p: optax.projections.projection_l1_ball(p, scale=radius)

    elif constraint_type == "non_negative":
        return optax.projections.projection_non_negative

    elif constraint_type == "simplex":
        scale = kwargs.get("scale", 1.5)
        return lambda p: optax.projections.projection_simplex(p, scale=scale)

    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")


# Optimize using projected gradient descent with Optax
def optimize_with_optax(
    objective_fn, initial_params, constraint_type, num_iterations=1000, learning_rate=0.01, **constraint_kwargs
):
    """
    Optimize using projected gradient descent with Optax.

    Args:
        objective_fn: The objective function to minimize
        initial_params: Initial parameters
        constraint_type: Type of constraint to apply
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate for the optimizer
        **constraint_kwargs: Additional arguments for the constraint

    Returns:
        Tuple of (optimized_params, loss_history, param_history)
    """
    # Get projection function
    project = get_projection(constraint_type, **constraint_kwargs)

    # Apply initial projection to ensure we start in the feasible region
    params = project(initial_params)

    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Storage for tracking
    loss_history = []
    param_history = [params.copy()]

    # Define update step
    @jax.jit
    def update_step(params, opt_state):
        loss, grads = jax.value_and_grad(objective_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        # Apply projection
        params = project(params)
        return params, opt_state, loss

    # Run optimization
    for i in range(num_iterations):
        params, opt_state, loss = update_step(params, opt_state)
        loss_history.append(loss)
        param_history.append(params.copy())

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss:.6f}")

    return params, jnp.array(loss_history), jnp.array(param_history)


# Optimize using BFGS with projections
def optimize_with_bfgs(objective_fn, initial_params, constraint_type, **constraint_kwargs):
    """
    Optimize using BFGS with projections.

    Args:
        objective_fn: The objective function to minimize
        initial_params: Initial parameters
        constraint_type: Type of constraint to apply
        **constraint_kwargs: Additional arguments for the constraint

    Returns:
        Tuple of (optimized_params, param_history)
    """
    # Get projection function
    project = get_projection(constraint_type, **constraint_kwargs)

    # Apply initial projection to ensure we start in the feasible region
    x0 = project(initial_params)

    # Storage for tracking
    param_history = [x0.copy()]

    # Create callback function to track iterations and apply projections
    def callback(x):
        # Apply projection to maintain feasibility
        x_proj = project(x)
        param_history.append(x_proj.copy())
        return False  # Continue optimization

    # Run BFGS optimization
    result = minimize(objective_fn, x0, method="BFGS", callback=callback, options={"maxiter": 100})

    # Apply final projection to ensure feasibility
    final_params = project(result.x)

    return final_params, jnp.array(param_history)


# Function to create contour plot of Rosenbrock function with optimization trajectories
def plot_rosenbrock_contour(param_histories, constraint_type, constraint_kwargs, method_names):
    """
    Plot the Rosenbrock function contour and optimization trajectories.

    Args:
        param_histories: List of parameter histories from different optimization methods
        constraint_type: Type of constraint applied
        constraint_kwargs: Constraint parameters
        method_names: Names of optimization methods for the legend
    """
    plt.figure(figsize=(12, 10))

    # Create grid for contour plot
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Compute Rosenbrock function values
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = rosenbrock(jnp.array([X[j, i], Y[j, i]]))

    # Plot contours of Rosenbrock function (log scale for better visualization)
    contour = plt.contourf(X, Y, np.log10(Z + 1), 50, cmap="viridis", alpha=0.7)
    plt.colorbar(contour, label="log10(f(x,y) + 1)")

    # Plot constraint boundary if applicable
    if constraint_type == "box":
        lower = constraint_kwargs.get("lower", -2.0)
        upper = constraint_kwargs.get("upper", 2.0)
        plt.axhline(y=lower, color="r", linestyle="--", alpha=0.7)
        plt.axhline(y=upper, color="r", linestyle="--", alpha=0.7)
        plt.axvline(x=lower, color="r", linestyle="--", alpha=0.7)
        plt.axvline(x=upper, color="r", linestyle="--", alpha=0.7)

    elif constraint_type == "l2_ball":
        radius = constraint_kwargs.get("radius", 2.0)
        circle = plt.Circle((0, 0), radius, color="r", fill=False, linestyle="--", alpha=0.7)
        plt.gca().add_patch(circle)

    elif constraint_type == "l1_ball":
        radius = constraint_kwargs.get("radius", 3.0)
        # Plot diamond shape for L1 ball
        l1_x = np.array([radius, 0, -radius, 0, radius])
        l1_y = np.array([0, radius, 0, -radius, 0])
        plt.plot(l1_x, l1_y, "r--", alpha=0.7)

    # Plot optimization trajectories
    colors = ["blue", "green", "purple", "orange"]
    for i, (param_history, method_name) in enumerate(zip(param_histories, method_names)):
        if param_history.ndim == 3:  # If we have multiple runs
            for j in range(param_history.shape[0]):
                path = param_history[j]
                plt.plot(path[:, 0], path[:, 1], "-", color=colors[i], alpha=0.3)
            # Plot the mean trajectory
            mean_path = param_history.mean(axis=0)
            plt.plot(mean_path[:, 0], mean_path[:, 1], "-", color=colors[i], linewidth=2, label=f"{method_name} (mean)")
        else:  # Single run
            plt.plot(param_history[:, 0], param_history[:, 1], "-o", color=colors[i], markersize=4, label=method_name)

    # Plot global minimum
    plt.plot(1, 1, "r*", markersize=10, label="Global Minimum")

    # Add labels and legend
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Rosenbrock Function Optimization with {constraint_type} Constraint")
    plt.legend()
    plt.grid(True)

    return plt


# Run experiments with different starting points
def run_experiments(constraint_type="box", num_runs=5, dimension=2):
    """
    Run optimization experiments with different random starting points.

    Args:
        constraint_type: Type of constraint to apply
        num_runs: Number of random initializations
        dimension: Dimension of the problem (default is 2D Rosenbrock)

    Returns:
        Tuple of results and constraint parameters
    """
    # Initialize the random key here (inside the function)
    key = jax.random.PRNGKey(42)

    # Set constraint parameters based on constraint type
    constraint_kwargs = {}
    if constraint_type == "box":
        constraint_kwargs = {"lower": -1.5, "upper": 1.5}
    elif constraint_type == "l2_ball":
        constraint_kwargs = {"radius": 1.5}
    elif constraint_type == "l1_ball":
        constraint_kwargs = {"radius": 2.0}
    elif constraint_type == "simplex":
        constraint_kwargs = {"scale": 1.0}

    # Storage for results
    optax_results = []
    bfgs_results = []

    # Run optimizations with different random starting points
    for run in range(num_runs):
        print(f"\n=== Run {run + 1}/{num_runs} ===")

        # Generate random starting point
        key, subkey = jax.random.split(key)
        initial_params = jax.random.uniform(subkey, (dimension,), minval=-1.0, maxval=1.5)
        initial_params = jnp.array([-1, 3], dtype=jnp.float64)

        print(f"Initial parameters: {initial_params}")

        # Run Optax optimization
        print("\n--- Optax Projected Gradient ---")
        optax_final, optax_losses, optax_path = optimize_with_optax(
            rosenbrock, initial_params, constraint_type, num_iterations=500, learning_rate=0.01, **constraint_kwargs
        )

        # Run BFGS optimization
        print("\n--- BFGS with Projection ---")
        bfgs_final, bfgs_path = optimize_with_bfgs(rosenbrock, initial_params, constraint_type, **constraint_kwargs)

        # Print final results
        print(f"\nOptax final parameters: {optax_final}, loss: {rosenbrock(optax_final):.6f}")
        print(f"BFGS final parameters: {bfgs_final}, loss: {rosenbrock(bfgs_final):.6f}")

        # Store results
        optax_results.append((optax_final, optax_losses, optax_path))
        bfgs_results.append((bfgs_final, bfgs_path))

    return optax_results, bfgs_results, constraint_kwargs


# Main function to run the example
def main():
    # Constraint types to test
    constraint_types = ["box", "l2_ball", "l1_ball", "non_negative", "simplex"]

    for constraint_type in constraint_types:
        print(f"\n=== Testing {constraint_type} constraint ===")

        # Run experiments
        optax_results, bfgs_results, constraint_kwargs = run_experiments(constraint_type, num_runs=1)

        # For 2D problem, plot optimization trajectories
        if constraint_type in ["box", "l2_ball", "l1_ball"]:
            # Extract parameter histories
            optax_paths = [result[2] for result in optax_results]
            bfgs_paths = [result[1] for result in bfgs_results]

            # Plot trajectories
            plt = plot_rosenbrock_contour(
                [optax_paths[0], bfgs_paths[0]],  # Just use first run for clarity
                constraint_type,
                constraint_kwargs,
                ["Optax Projected Gradient", "BFGS with Projection"],
            )

            plt.tight_layout()
            plt.show()

        # Compare final losses
        optax_final_losses = [rosenbrock(result[0]) for result in optax_results]
        bfgs_final_losses = [rosenbrock(result[0]) for result in bfgs_results]

        print("\n=== Final Results ===")
        print(f"Optax mean final loss: {np.mean(optax_final_losses):.6f}")
        print(f"BFGS mean final loss: {np.mean(bfgs_final_losses):.6f}")

        # Plot convergence for Optax (BFGS doesn't provide loss history)
        if len(optax_results) > 0:
            plt.figure(figsize=(10, 6))
            for i, result in enumerate(optax_results):
                _, losses, _ = result
                plt.semilogy(losses, alpha=0.7, label=f"Run {i + 1}")

            plt.title(f"Convergence with {constraint_type} Constraint (Optax)")
            plt.xlabel("Iteration")
            plt.ylabel("Loss (log scale)")
            plt.grid(True)
            if len(optax_results) > 1:
                plt.legend()
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
