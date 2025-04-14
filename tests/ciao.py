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
    Solve constrained least squares problems using Optax.

    Parameters:
    - A, b: Least squares problem data
    - constraint_type: Type of constraint ("non_negative", "sum_one", "box", "sparse")
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
    if constraint_type == "sum_one":
        # Start with feasible parameters for sum constraint
        params = jnp.ones(n) / n
    else:
        params = jnp.zeros(n)

    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Configure constraint handling based on constraint type
    if constraint_type == "non_negative":
        # Projected gradient method for non-negativity
        def project_params(p):
            return jnp.maximum(0, p)

        @jax.jit
        def update_step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            # Apply projection to enforce non-negativity
            params = project_params(params)
            return params, opt_state, loss

    elif constraint_type == "sum_one":
        # Augmented Lagrangian for sum(x) = 1
        lagrange_mult = 0.0
        penalty = 10.0

        def augmented_lagrangian(p, lm, pen):
            obj = loss_fn(p)
            constraint = jnp.sum(p) - 1.0
            return obj + lm * constraint + 0.5 * pen * constraint**2

        @jax.jit
        def update_inner(params, opt_state, lm, pen):
            def obj_fn(p):
                return augmented_lagrangian(p, lm, pen)

            loss, grads = jax.value_and_grad(obj_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        # Run optimization with augmented Lagrangian
        losses = []
        for outer_iter in range(10):  # Outer iterations to update Lagrangian
            # Inner optimization loop
            for inner_iter in range(num_iterations // 10):
                params, opt_state, loss = update_inner(params, opt_state, lagrange_mult, penalty)
                losses.append(loss)

            # Update Lagrangian multiplier
            constraint_violation = jnp.sum(params) - 1.0
            lagrange_mult = lagrange_mult + penalty * constraint_violation

            # Optionally increase penalty
            penalty = min(2 * penalty, 1e6)

            print(f"Outer iter {outer_iter}, Loss: {loss:.6f}, Sum(x): {jnp.sum(params):.6f}")

        return params, losses

    elif constraint_type == "box":
        # Box constraints: -0.5 ≤ x ≤ 0.5
        lower_bounds = -0.5 * jnp.ones(n)
        upper_bounds = 0.5 * jnp.ones(n)

        def project_box(p):
            return jnp.clip(p, lower_bounds, upper_bounds)

        @jax.jit
        def update_step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            # Apply projection to enforce box constraints
            params = project_box(params)
            return params, opt_state, loss

    elif constraint_type == "sparse":
        # L1 regularization for sparsity
        l1_weight = 0.1

        def regularized_loss(p):
            return loss_fn(p) + l1_weight * jnp.sum(jnp.abs(p))

        @jax.jit
        def update_step(params, opt_state):
            loss, grads = jax.value_and_grad(regularized_loss)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")

    # Run optimization for non-Lagrangian methods
    if constraint_type != "sum_one":
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
    constraints = ["non_negative", "sum_one", "box", "sparse"]
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
        elif constraint == "sum_one":
            print(f"Sum of parameters: {jnp.sum(params):.6f}")
        elif constraint == "box":
            in_box = jnp.all(params >= -0.5) and jnp.all(params <= 0.5)
            print(f"All parameters in box constraints: {in_box}")
        elif constraint == "sparse":
            num_zeros = jnp.sum(jnp.abs(params) < 1e-4)
            print(f"Number of near-zero elements: {num_zeros} out of {len(params)}")

    # Compute unconstrained solution for comparison
    unconstrained_sol = jnp.linalg.lstsq(A, b, rcond=None)[0]
    unconstrained_loss = least_squares_loss(unconstrained_sol, A, b)
    print(f"\nUnconstrained solution loss: {unconstrained_loss:.6f}")

    # Plot results
    plt.figure(figsize=(12, 10))

    # Plot losses
    plt.subplot(2, 2, 1)
    for constraint, (_, losses) in results.items():
        plt.plot(losses, label=constraint)
    plt.title("Loss vs. Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()

    # Plot parameter values
    plt.subplot(2, 2, 2)
    x_indices = range(len(x_true))
    plt.plot(x_indices, x_true, "k-", label="True")
    plt.plot(x_indices, unconstrained_sol, "k--", label="Unconstrained")

    for constraint, (params, _) in results.items():
        plt.plot(x_indices, params, "o-", label=constraint)

    plt.title("Parameter Values")
    plt.xlabel("Parameter Index")
    plt.ylabel("Value")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
