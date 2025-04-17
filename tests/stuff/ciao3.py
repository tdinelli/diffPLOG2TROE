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


# Constraint functions for Augmented Lagrangian
def equality_constraint(params, target_sum=1.0):
    """Equality constraint: sum(x) = target_sum"""
    return jnp.sum(params) - target_sum


def inequality_constraint(params, lower_bound=0.0):
    """Inequality constraint: x >= lower_bound"""
    # Returns array of constraint violations (positive values indicate violations)
    return lower_bound - params


def box_constraint_lower(params, lower_bound=-0.5):
    """Lower bound constraint: x >= lower_bound"""
    return lower_bound - params


def box_constraint_upper(params, upper_bound=0.5):
    """Upper bound constraint: x <= upper_bound"""
    return params - upper_bound


def l1_constraint(params, radius=2.0):
    """L1 norm constraint: ||x||_1 <= radius"""
    return jnp.sum(jnp.abs(params)) - radius


def l2_constraint(params, radius=1.0):
    """L2 norm constraint: ||x||_2 <= radius"""
    return jnp.sqrt(jnp.sum(params**2)) - radius


# Augmented Lagrangian solver
def solve_augmented_lagrangian(
    A,
    b,
    constraint_type="sum_to_one",
    outer_iterations=20,
    inner_iterations=100,
    learning_rate=0.01,
    initial_penalty=10.0,
    penalty_increase_factor=2.0,
):
    """
    Solve constrained least squares using the Augmented Lagrangian method.

    Parameters:
    - A, b: Least squares problem data
    - constraint_type: Type of constraint ("sum_to_one", "non_negative", "box", "l1_norm", "l2_norm")
    - outer_iterations: Number of multiplier update iterations
    - inner_iterations: Number of inner optimization iterations per outer iteration
    - learning_rate: Learning rate for the inner optimizer
    - initial_penalty: Initial penalty parameter (rho)
    - penalty_increase_factor: Factor to increase penalty by after each outer iteration

    Returns:
    - params: Optimized parameters
    - losses: Loss history
    - constraint_violations: History of constraint violations
    """
    m, n = A.shape

    # Initialize parameters
    params = jnp.zeros(n)

    # Set up constraint function based on constraint_type
    if constraint_type == "sum_to_one":
        equality_fn = equality_constraint
        inequality_fn = None
    elif constraint_type == "non_negative":
        equality_fn = None
        inequality_fn = inequality_constraint
    elif constraint_type == "box":
        equality_fn = None
        inequality_fn = lambda p: jnp.concatenate([box_constraint_lower(p), box_constraint_upper(p)])
    elif constraint_type == "l1_norm":
        equality_fn = None
        inequality_fn = l1_constraint
    elif constraint_type == "l2_norm":
        equality_fn = None
        inequality_fn = l2_constraint
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")

    # Initialize Lagrange multipliers and penalty parameter
    eq_multiplier = 0.0 if equality_fn is not None else None

    if inequality_fn is not None:
        # Test to see shape of inequality constraints
        test_ineq = inequality_fn(params)
        if isinstance(test_ineq, jnp.ndarray):
            ineq_multiplier = jnp.zeros_like(test_ineq)
        else:
            ineq_multiplier = 0.0
    else:
        ineq_multiplier = None

    penalty = initial_penalty

    # Create optimizer for inner minimization
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Define the augmented Lagrangian function
    def augmented_lagrangian(p, eq_mult, ineq_mult, pen):
        # Original objective
        obj_value = least_squares_loss(p, A, b)

        # Equality constraint terms
        eq_term = 0.0
        if equality_fn is not None:
            eq_constr = equality_fn(p)
            eq_term = eq_mult * eq_constr + 0.5 * pen * eq_constr**2

        # Inequality constraint terms
        ineq_term = 0.0
        if inequality_fn is not None:
            ineq_constr = inequality_fn(p)

            # Handle scalar or vector inequality constraints
            if isinstance(ineq_constr, jnp.ndarray):
                # Element-wise ReLU for vector constraints
                ineq_term_parts = ineq_mult * ineq_constr + 0.5 * pen * jnp.maximum(0, ineq_constr) ** 2
                ineq_term = jnp.sum(ineq_term_parts)
            else:
                # Scalar constraint
                ineq_term = ineq_mult * ineq_constr + 0.5 * pen * jnp.maximum(0, ineq_constr) ** 2

        return obj_value + eq_term + ineq_term

    # JIT-compiled inner optimization step
    @jax.jit
    def inner_step(p, opt_st, eq_m, ineq_m, pen):
        def loss_fn(params):
            return augmented_lagrangian(params, eq_m, ineq_m, pen)

        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, new_opt_st = optimizer.update(grads, opt_st)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_opt_st, loss

    # Storage for tracking progress
    all_losses = []
    constraint_violations = []

    # Outer iterations - update multipliers and penalty
    for outer_iter in range(outer_iterations):
        # Inner optimization - minimize augmented Lagrangian
        for inner_iter in range(inner_iterations):
            params, opt_state, loss = inner_step(params, opt_state, eq_multiplier, ineq_multiplier, penalty)
            all_losses.append(loss)

        # Compute constraint violations for reporting
        eq_violation = 0.0 if equality_fn is None else jnp.abs(equality_fn(params))

        if inequality_fn is not None:
            ineq_values = inequality_fn(params)
            if isinstance(ineq_values, jnp.ndarray):
                ineq_violation = jnp.sum(jnp.maximum(0, ineq_values))
            else:
                ineq_violation = jnp.maximum(0, ineq_values)
        else:
            ineq_violation = 0.0

        total_violation = eq_violation + ineq_violation
        constraint_violations.append(total_violation)

        # Update Lagrange multipliers
        if equality_fn is not None:
            eq_constr_value = equality_fn(params)
            eq_multiplier = eq_multiplier + penalty * eq_constr_value

        if inequality_fn is not None:
            ineq_constr_value = inequality_fn(params)

            if isinstance(ineq_constr_value, jnp.ndarray):
                ineq_multiplier = jnp.maximum(0, ineq_multiplier + penalty * ineq_constr_value)
            else:
                ineq_multiplier = jnp.maximum(0, ineq_multiplier + penalty * ineq_constr_value)

        # Increase penalty parameter
        penalty = penalty * penalty_increase_factor

        # Print progress
        objective = least_squares_loss(params, A, b)
        print(f"Outer iter {outer_iter}, Loss: {objective:.6f}, Constraint violation: {total_violation:.6f}")

    return params, all_losses, constraint_violations


# Example usage
def main():
    # Generate problem
    A, b, x_true = generate_problem(m=100, n=10)

    # Define constraint types to test
    constraint_types = ["sum_to_one", "non_negative", "box", "l1_norm", "l2_norm"]
    results = {}

    for constraint_type in constraint_types:
        print(f"\n=== Solving with {constraint_type} constraints ===")
        params, losses, violations = solve_augmented_lagrangian(
            A, b, constraint_type=constraint_type, outer_iterations=10, inner_iterations=200
        )
        results[constraint_type] = (params, losses, violations)

        # Check constraint satisfaction
        if constraint_type == "sum_to_one":
            print(f"Sum of parameters: {jnp.sum(params):.6f} (should be 1.0)")
        elif constraint_type == "non_negative":
            print(f"Min parameter value: {jnp.min(params):.6f} (should be ≥ 0)")
        elif constraint_type == "box":
            print(f"Parameter range: [{jnp.min(params):.6f}, {jnp.max(params):.6f}] (should be in [-0.5, 0.5])")
        elif constraint_type == "l1_norm":
            print(f"L1 norm: {jnp.sum(jnp.abs(params)):.6f} (should be ≤ 2.0)")
        elif constraint_type == "l2_norm":
            print(f"L2 norm: {jnp.sqrt(jnp.sum(params**2)):.6f} (should be ≤ 1.0)")

        # Check loss
        loss = least_squares_loss(params, A, b)
        print(f"Final objective value: {loss:.6f}")

    # Compute unconstrained solution for comparison
    unconstrained_sol = jnp.linalg.lstsq(A, b, rcond=None)[0]
    unconstrained_loss = least_squares_loss(unconstrained_sol, A, b)
    print(f"\nUnconstrained solution loss: {unconstrained_loss:.6f}")

    # Plot results
    plt.figure(figsize=(15, 12))

    # Plot objective values
    plt.subplot(2, 2, 1)
    for constraint_type, (_, losses, _) in results.items():
        plt.plot(losses[:500], label=constraint_type)  # Plot first 500 iterations for clarity
    plt.title("Augmented Lagrangian Value vs. Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Augmented Lagrangian Value")
    plt.legend()
    plt.grid(True)

    # Plot constraint violations
    plt.subplot(2, 2, 2)
    for constraint_type, (_, _, violations) in results.items():
        plt.semilogy(violations, label=constraint_type)
    plt.title("Constraint Violation vs. Outer Iterations")
    plt.xlabel("Outer Iteration")
    plt.ylabel("Constraint Violation (log scale)")
    plt.legend()
    plt.grid(True)

    # Plot parameter values
    plt.subplot(2, 2, 3)
    x_indices = range(len(x_true))
    plt.plot(x_indices, x_true, "ko-", label="True", linewidth=2)
    plt.plot(x_indices, unconstrained_sol, "k--", label="Unconstrained")

    for constraint_type, (params, _, _) in results.items():
        plt.plot(x_indices, params, "o-", label=constraint_type)

    plt.title("Parameter Values")
    plt.xlabel("Parameter Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # Plot final objective values
    plt.subplot(2, 2, 4)
    constraint_types_with_unconstrained = constraint_types + ["unconstrained"]
    objective_values = [least_squares_loss(results[ct][0], A, b) for ct in constraint_types]
    objective_values.append(unconstrained_loss)

    plt.bar(constraint_types_with_unconstrained, objective_values)
    plt.title("Final Objective Values")
    plt.xlabel("Constraint Type")
    plt.ylabel("Objective Value")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
