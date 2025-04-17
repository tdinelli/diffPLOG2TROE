import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt


# Generate synthetic least squares problem
def generate_problem(m=100, n=10, seed=42):
    """Generate a synthetic least squares problem Ax = b + noise."""
    key = jax.random.PRNGKey(seed)
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

    # Create matrix A and true solution x_true
    A = jax.random.normal(subkey1, (m, n))
    x_true = jax.random.normal(subkey2, (n,))

    # Generate noisy observations
    noise = 0.1 * jax.random.normal(subkey3, (m,))
    b = jnp.dot(A, x_true) + noise

    return A, b, x_true


# Define the least squares objective
def least_squares_loss(params, A, b):
    """Compute the mean squared error between Ax and b."""
    predictions = jnp.dot(A, params)
    return jnp.mean((predictions - b) ** 2)


# Solve constrained least squares with Optax
def solve_constrained_least_squares(A, b, constraint_type="non_negative", num_iterations=1000, learning_rate=0.01):
    """
    Solve constrained least squares problems using Optax with built-in projections.

    Parameters:
    - A, b: Least squares problem data
    - constraint_type: Type of constraint ("non_negative", "simplex", "box", "l1_ball", "l2_ball")
    - num_iterations: Number of optimization iterations
    - learning_rate: Learning rate for the optimizer

    Returns:
    - params: Optimized parameters
    - losses: Loss history
    """
    m, n = A.shape

    # Create the loss function
    def loss_fn(params):
        return least_squares_loss(params, A, b)

    # Initialize parameters based on constraint type
    if constraint_type == "simplex": # Start with uniform distribution for simplex constraint
        params = jnp.ones(n) / n
    else:
        params = jnp.zeros(n)

    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Set up projection function based on constraint type
    if constraint_type == "non_negative":
        project = optax.projections.projection_non_negative
    elif constraint_type == "simplex":
        project = lambda p: optax.projections.projection_simplex(p, scale=1.0)
    elif constraint_type == "box":
        lower_bound = -0.5
        upper_bound = 0.5
        project = lambda p: optax.projections.projection_box(p, lower_bound, upper_bound)
    elif constraint_type == "l1_ball":
        l1_radius = 2.0
        project = lambda p: optax.projections.projection_l1_ball(p, scale=l1_radius)
    elif constraint_type == "l2_ball":
        l2_radius = 1.0
        project = lambda p: optax.projections.projection_l2_ball(p, scale=l2_radius)
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")

    # Function for L1 regularization (for sparsity)
    if constraint_type == "l1_ball":
        # Additional L1 regularization for sparsity in the l1_ball case
        l1_weight = 0.1

        def regularized_loss(p):
            return loss_fn(p) + l1_weight * jnp.sum(jnp.abs(p))

        objective = regularized_loss
    else:
        objective = loss_fn

    # JIT-compiled update step with projection
    @jax.jit
    def update_step(params, opt_state):
        # Compute gradients
        loss, grads = jax.value_and_grad(objective)(params)
        # Apply optimizer update
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        # Apply projection to enforce constraints
        params = project(params)
        return params, opt_state, loss

    # Run optimization
    losses = []
    for i in range(num_iterations):
        params, opt_state, loss = update_step(params, opt_state)
        losses.append(loss)
        if i % 200 == 0:
            print(f"Iteration {i}, Loss: {loss:.6f}")

    return params, losses


# Example usage
def main():
    # Generate problem
    A, b, x_true = generate_problem(m=100, n=10)

    # Solve with different constraints
    constraints = ["non_negative", "simplex", "box", "l1_ball", "l2_ball"]
    results = {}

    for constraint in constraints:
        print(f"\n=== Solving with {constraint} constraints ===")
        params, losses = solve_constrained_least_squares(A, b, constraint_type=constraint)
        results[constraint] = (params, losses)

        # Check loss and constraint satisfaction
        loss = least_squares_loss(params, A, b)
        print(f"Final loss: {loss:.6f}")

        if constraint == "non_negative":
            print(f"All non-negative: {jnp.all(params >= 0)}")
        elif constraint == "simplex":
            print(f"Sum of parameters: {jnp.sum(params):.6f}")
            print(f"All non-negative: {jnp.all(params >= 0)}")
        elif constraint == "box":
            in_box = jnp.all(params >= -0.5) and jnp.all(params <= 0.5)
            print(f"All parameters in box constraints: {in_box}")
        elif constraint == "l1_ball":
            l1_norm = jnp.sum(jnp.abs(params))
            print(f"L1 norm: {l1_norm:.6f} (should be ≤ 2.0)")
        elif constraint == "l2_ball":
            l2_norm = jnp.sqrt(jnp.sum(params**2))
            print(f"L2 norm: {l2_norm:.6f} (should be ≤ 1.0)")

    # Compute unconstrained solution for comparison
    unconstrained_sol = jnp.linalg.lstsq(A, b, rcond=None)[0]
    unconstrained_loss = least_squares_loss(unconstrained_sol, A, b)
    print(f"\nUnconstrained solution loss: {unconstrained_loss:.6f}")

    # Plot results
    plt.figure(figsize=(15, 10))

    # Plot losses
    plt.subplot(2, 2, 1)
    for constraint, (_, losses) in results.items():
        plt.plot(losses, label=constraint)
    plt.title("Loss vs. Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot parameter values
    plt.subplot(2, 2, 2)
    x_indices = range(len(x_true))
    plt.plot(x_indices, x_true, "ko-", label="True", linewidth=2)
    plt.plot(x_indices, unconstrained_sol, "k--", label="Unconstrained")

    for constraint, (params, _) in results.items():
        plt.plot(x_indices, params, "o-", label=constraint)

    plt.title("Parameter Values")
    plt.xlabel("Parameter Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # Plot L1 norms
    plt.subplot(2, 2, 3)
    labels = []
    l1_norms = []

    # Add true and unconstrained solutions
    labels.extend(["True", "Unconstrained"])
    l1_norms.extend([jnp.sum(jnp.abs(x_true)), jnp.sum(jnp.abs(unconstrained_sol))])

    # Add constrained solutions
    for constraint, (params, _) in results.items():
        labels.append(constraint)
        l1_norms.append(jnp.sum(jnp.abs(params)))

    plt.bar(labels, l1_norms)
    plt.title("L1 Norm Comparison")
    plt.ylabel("L1 Norm")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")

    # Plot L2 norms
    plt.subplot(2, 2, 4)
    labels = []
    l2_norms = []

    # Add true and unconstrained solutions
    labels.extend(["True", "Unconstrained"])
    l2_norms.extend([jnp.sqrt(jnp.sum(x_true**2)), jnp.sqrt(jnp.sum(unconstrained_sol**2))])

    # Add constrained solutions
    for constraint, (params, _) in results.items():
        labels.append(constraint)
        l2_norms.append(jnp.sqrt(jnp.sum(params**2)))

    plt.bar(labels, l2_norms)
    plt.title("L2 Norm Comparison")
    plt.ylabel("L2 Norm")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
