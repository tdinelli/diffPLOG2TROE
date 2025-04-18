import jax
import jax.numpy as jnp
from jax import grad, hessian, jit
import matplotlib.pyplot as plt
import optax
import numpy as np

# Set random seed for reproducibility
key = jax.random.PRNGKey(42)


# Generate synthetic data: exponential decay with noise
def true_function(x, a, b, c):
    return a * jnp.exp(-b * x) + c


# True parameters
true_params = jnp.array([3.0, 0.5, 1.0])  # [a, b, c]

# Generate x data points
n_points = 50
x_data = jnp.linspace(0, 10, n_points)

# Generate noisy observations with same seed as before
noise_level = 0.3
key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape=(n_points,)) * noise_level
y_data = true_function(x_data, *true_params) + noise


# Helper function to compute covariance matrix and related statistics
def compute_covariance_stats(params, loss_fn, x_data, y_data):
    H = hessian(loss_fn)(params, x_data, y_data)
    residual_variance = loss_fn(params, x_data, y_data)
    cov_matrix = jnp.linalg.inv(H) * residual_variance
    std_errors = jnp.sqrt(jnp.diag(cov_matrix))

    # Correlation matrix
    diag_sqrt = jnp.sqrt(jnp.diag(cov_matrix))
    corr_matrix = cov_matrix / jnp.outer(diag_sqrt, diag_sqrt)

    # Eigenvalues
    eigenvalues, _ = jnp.linalg.eigh(cov_matrix)
    condition_number = jnp.max(eigenvalues) / jnp.min(eigenvalues)

    # Relative errors
    relative_errors = std_errors / jnp.abs(params)

    return {
        "cov_matrix": cov_matrix,
        "std_errors": std_errors,
        "corr_matrix": corr_matrix,
        "eigenvalues": eigenvalues,
        "condition_number": condition_number,
        "relative_errors": relative_errors,
    }


# Function to run optimization and return results
def run_optimization(
    model_fn,
    initial_params,
    x_data,
    y_data,
    num_iterations=1000,
    print_interval=100,
    learning_rate=0.1,
    param_names=None,
):
    # Define loss function
    def loss_fn(params, x, y):
        predictions = model_fn(params, x)
        return jnp.mean((predictions - y) ** 2)

    # Optimization setup
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(initial_params)

    # Gradient function
    grad_loss = jit(grad(loss_fn))

    # Update function
    @jit
    def update(params, opt_state):
        g = grad_loss(params, x_data, y_data)
        updates, opt_state = optimizer.update(g, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # Run optimization
    params = initial_params
    losses = []

    for i in range(num_iterations):
        params, opt_state = update(params, opt_state)
        if i % print_interval == 0:
            current_loss = loss_fn(params, x_data, y_data)
            losses.append(current_loss)
            print(f"Iteration {i}, Loss: {current_loss:.6f}, Params: {params}")

    final_loss = loss_fn(params, x_data, y_data)
    print(f"Final Loss: {final_loss:.6f}, Final Params: {params}")

    # Compute covariance and stats
    stats = compute_covariance_stats(params, loss_fn, x_data, y_data)

    # Print summary
    print("\nParameter estimates:")
    for i, name in enumerate(param_names if param_names else [f"p{i}" for i in range(len(params))]):
        print(
            f"{name}: {params[i]:.4f} ± {stats['std_errors'][i]:.4f} (Relative Error: {stats['relative_errors'][i]:.2%})"
        )

    print("\nCorrelation Matrix:")
    print(stats["corr_matrix"])

    print(f"\nCondition Number: {stats['condition_number']:.2f}")

    return params, stats, losses, loss_fn


# ================== Original Parameterization ==================
print("\n====== ORIGINAL PARAMETERIZATION ======")
print("Model: a * exp(-b * x) + c")


def original_model(params, x):
    a, b, c = params
    return a * jnp.exp(-b * x) + c


original_initial = jnp.array([2.0, 0.3, 0.5])
original_params, original_stats, original_losses, original_loss_fn = run_optimization(
    original_model, original_initial, x_data, y_data, param_names=["a", "b", "c"]
)

# ================== Half-Life Parameterization ==================
print("\n====== HALF-LIFE PARAMETERIZATION ======")
print("Model: a * exp(-ln(2) * x / t_half) + c")


def halflife_model(params, x):
    a, t_half, c = params
    return a * jnp.exp(-jnp.log(2) * x / t_half) + c


# Convert original parameters to half-life parameterization
# t_half = ln(2)/b
true_halflife_params = jnp.array(
    [
        true_params[0],  # a remains the same
        jnp.log(2) / true_params[1],  # t_half = ln(2)/b
        true_params[2],  # c remains the same
    ]
)

halflife_initial = jnp.array([2.0, jnp.log(2) / 0.3, 0.5])
halflife_params, halflife_stats, halflife_losses, halflife_loss_fn = run_optimization(
    halflife_model, halflife_initial, x_data, y_data, param_names=["a", "t_half", "c"]
)

# ================== Initial/Final Value Parameterization ==================
print("\n====== INITIAL/FINAL VALUE PARAMETERIZATION ======")
print("Model: (y0 - y_inf) * exp(-b * x) + y_inf")


def initial_final_model(params, x):
    y0, y_inf, b = params
    return (y0 - y_inf) * jnp.exp(-b * x) + y_inf


# Convert original parameters to initial/final parameterization
# y0 = a + c (initial value at x=0)
# y_inf = c (asymptotic value as x→∞)
# a = y0 - y_inf
true_if_params = jnp.array(
    [true_params[0] + true_params[2], true_params[2], true_params[1]]  # y0 = a + c  # y_inf = c  # b remains the same
)

if_initial = jnp.array([2.0 + 0.5, 0.5, 0.3])  # y0 = a+c, y_inf = c, b
if_params, if_stats, if_losses, if_loss_fn = run_optimization(
    initial_final_model, if_initial, x_data, y_data, param_names=["y0", "y_inf", "b"]
)

# ================== Log Decay Rate Parameterization ==================
print("\n====== LOG DECAY RATE PARAMETERIZATION ======")
print("Model: a * exp(-exp(log_b) * x) + c")


def log_decay_model(params, x):
    a, log_b, c = params
    return a * jnp.exp(-jnp.exp(log_b) * x) + c


# Convert original parameters to log decay rate
# log_b = ln(b)
true_log_params = jnp.array(
    [
        true_params[0],  # a remains the same
        jnp.log(true_params[1]),  # log_b = ln(b)
        true_params[2],  # c remains the same
    ]
)

log_initial = jnp.array([2.0, jnp.log(0.3), 0.5])  # a, log(b), c
log_params, log_stats, log_losses, log_loss_fn = run_optimization(
    log_decay_model, log_initial, x_data, y_data, param_names=["a", "log_b", "c"]
)

# ================== Visualize Comparison ==================
# Plot data and fitted models
plt.figure(figsize=(10, 6))
x_fine = jnp.linspace(0, 10, 200)

# Data points
plt.scatter(x_data, y_data, alpha=0.5, label="Data", color="black")

# True function
y_true = true_function(x_fine, *true_params)
plt.plot(x_fine, y_true, "k--", label="True function", linewidth=2)

# Original model
y_original = original_model(original_params, x_fine)
plt.plot(x_fine, y_original, label="Original: a*exp(-b*x)+c")

# Half-life model
y_halflife = halflife_model(halflife_params, x_fine)
plt.plot(x_fine, y_halflife, label="Half-life: a*exp(-ln(2)*x/t_half)+c")

# Initial/Final model
y_if = initial_final_model(if_params, x_fine)
plt.plot(x_fine, y_if, label="Initial/Final: (y0-y_inf)*exp(-b*x)+y_inf")

# Log decay model
y_log = log_decay_model(log_params, x_fine)
plt.plot(x_fine, y_log, label="Log decay: a*exp(-exp(log_b)*x)+c")

plt.title("Comparison of Different Parameterizations")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("parameterization_comparison.png")

# ================== Compare Parameter Uncertainties ==================
plt.figure(figsize=(10, 6))

# Extract relative errors for each parameterization
rel_errors = {
    "Original": original_stats["relative_errors"],
    "Half-Life": halflife_stats["relative_errors"],
    "Initial/Final": if_stats["relative_errors"],
    "Log Decay": log_stats["relative_errors"],
}

# Set up the bar chart
param_indices = np.arange(3)
bar_width = 0.2
opacity = 0.8

# Plot bars for each parameterization
plt.bar(param_indices - bar_width * 1.5, rel_errors["Original"], bar_width, alpha=opacity, label="Original")
plt.bar(param_indices - bar_width / 2, rel_errors["Half-Life"], bar_width, alpha=opacity, label="Half-Life")
plt.bar(param_indices + bar_width / 2, rel_errors["Initial/Final"], bar_width, alpha=opacity, label="Initial/Final")
plt.bar(param_indices + bar_width * 1.5, rel_errors["Log Decay"], bar_width, alpha=opacity, label="Log Decay")

# Add labels and formatting
plt.xlabel("Parameter Index")
plt.ylabel("Relative Error")
plt.title("Comparison of Parameter Relative Errors")
plt.xticks(param_indices, ["Parameter 1", "Parameter 2", "Parameter 3"])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("parameter_uncertainty_comparison.png")

# ================== Compare Condition Numbers ==================
plt.figure(figsize=(8, 5))
condition_numbers = [
    original_stats["condition_number"],
    halflife_stats["condition_number"],
    if_stats["condition_number"],
    log_stats["condition_number"],
]

parameterizations = ["Original", "Half-Life", "Initial/Final", "Log Decay"]

plt.bar(parameterizations, condition_numbers, color="skyblue")
plt.ylabel("Condition Number (lower is better)")
plt.title("Condition Number Comparison")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("condition_number_comparison.png")

# ================== Summary Table ==================
print("\n====== PARAMETERIZATION COMPARISON SUMMARY ======")
print(f"{'Parameterization':<20} {'Cond. Number':<15} {'Max Rel. Error':<15} {'Avg Corr.':<15}")
print("-" * 65)


def avg_abs_correlation(corr_matrix):
    # Remove diagonal elements
    n = corr_matrix.shape[0]
    sum_corr = 0
    count = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                sum_corr += abs(corr_matrix[i, j])
                count += 1
    return sum_corr / count if count > 0 else 0


print(
    f"{'Original':<20} {original_stats['condition_number']:<15.2f} {jnp.max(original_stats['relative_errors']):<15.2%} {avg_abs_correlation(original_stats['corr_matrix']):<15.2f}"
)
print(
    f"{'Half-Life':<20} {halflife_stats['condition_number']:<15.2f} {jnp.max(halflife_stats['relative_errors']):<15.2%} {avg_abs_correlation(halflife_stats['corr_matrix']):<15.2f}"
)
print(
    f"{'Initial/Final':<20} {if_stats['condition_number']:<15.2f} {jnp.max(if_stats['relative_errors']):<15.2%} {avg_abs_correlation(if_stats['corr_matrix']):<15.2f}"
)
print(
    f"{'Log Decay':<20} {log_stats['condition_number']:<15.2f} {jnp.max(log_stats['relative_errors']):<15.2%} {avg_abs_correlation(log_stats['corr_matrix']):<15.2f}"
)


# Convert best parameterization back to original parameters for comparison
def convert_to_original_params(params, param_type):
    if param_type == "original":
        return params
    elif param_type == "halflife":
        a, t_half, c = params
        b = jnp.log(2) / t_half
        return jnp.array([a, b, c])
    elif param_type == "initial_final":
        y0, y_inf, b = params
        a = y0 - y_inf
        c = y_inf
        return jnp.array([a, b, c])
    elif param_type == "log":
        a, log_b, c = params
        b = jnp.exp(log_b)
        return jnp.array([a, b, c])


# Find best parameterization based on condition number and max relative error
parameterizations = [
    ("Original", original_params, "original"),
    ("Half-Life", halflife_params, "halflife"),
    ("Initial/Final", if_params, "initial_final"),
    ("Log Decay", log_params, "log"),
]

stats_list = [original_stats, halflife_stats, if_stats, log_stats]

# Calculate a composite score (lower is better)
scores = []
for i, (name, params, param_type) in enumerate(parameterizations):
    # Normalize condition number and max relative error to 0-1 scale
    max_cond = max(s["condition_number"] for s in stats_list)
    max_rel_err = max(jnp.max(s["relative_errors"]) for s in stats_list)

    norm_cond = stats_list[i]["condition_number"] / max_cond
    norm_rel_err = jnp.max(stats_list[i]["relative_errors"]) / max_rel_err

    # Composite score: weighted sum (lower is better)
    score = 0.4 * norm_cond + 0.6 * norm_rel_err
    scores.append((name, score, params, param_type))

# Sort by score (lowest first)
scores.sort(key=lambda x: x[1])
best_name, best_score, best_params, best_param_type = scores[0]

print("\n====== BEST PARAMETERIZATION ======")
print(f"Best parameterization: {best_name}")
print(f"Composite score (lower is better): {best_score:.4f}")

# Convert best params to original space for comparison
best_original_params = convert_to_original_params(best_params, best_param_type)
print("\nBest params converted to original parameter space:")
print(f"a: {best_original_params[0]:.4f}")
print(f"b: {best_original_params[1]:.4f}")
print(f"c: {best_original_params[2]:.4f}")
print(f"True: [3.0, 0.5, 1.0]")

# ================== Recommendations ==================
print("\n====== RECOMMENDATIONS ======")
print(f"Based on the analysis, the {best_name} parameterization provides the best parameter estimation.")
print("This is due to:")
print(
    f"1. Lower condition number: {stats_list[scores.index((best_name, best_score, best_params, best_param_type))]['condition_number']:.2f}"
)
print(
    f"2. Lower maximum relative error: {jnp.max(stats_list[scores.index((best_name, best_score, best_params, best_param_type))]['relative_errors']):.2%}"
)

print("\nThe key advantages of this parameterization are:")
if best_name == "Half-Life":
    print("- More intuitive physical interpretation (time to decay by half)")
    print("- Better numerical stability when decay rate is small")
elif best_name == "Initial/Final":
    print("- Direct estimation of observable quantities (initial and final values)")
    print("- Reduced correlation between amplitude and offset")
elif best_name == "Log Decay":
    print("- More uniform uncertainty across parameter space")
    print("- Better handling of decay rates spanning multiple orders of magnitude")

print("\nFor similar exponential decay problems, consider this parameterization first.")
