import jax
import jax.numpy as jnp
from jax import grad, hessian, jit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats

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

# Generate noisy observations
noise_level = 0.3
key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape=(n_points,)) * noise_level
y_data = true_function(x_data, *true_params) + noise


# Define our model function
def model(params, x):
    a, b, c = params
    return a * jnp.exp(-b * x) + c


# Define the loss function (mean squared error)
def loss_fn(params, x, y):
    predictions = model(params, x)
    return jnp.mean((predictions - y) ** 2)


# Compute the gradient and Hessian of the loss function
grad_loss = jit(grad(loss_fn))
hessian_loss = jit(hessian(loss_fn))

# Initialize parameters
initial_params = jnp.array([2.0, 0.3, 0.5])  # Initial guess


# Optimization using JAX's built-in optimizer
@jit
def update(params, opt_state):
    g = grad_loss(params, x_data, y_data)
    updates, opt_state = optimizer.update(g, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state


# Use optax for optimization
import optax

learning_rate = 0.1
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(initial_params)

# Run optimization
params = initial_params
losses = []
num_iterations = 1000

for i in range(num_iterations):
    params, opt_state = update(params, opt_state)
    if i % 100 == 0:
        current_loss = loss_fn(params, x_data, y_data)
        losses.append(current_loss)
        print(f"Iteration {i}, Loss: {current_loss:.6f}, Params: {params}")

print(f"Final optimized parameters: {params}")
print(f"True parameters: {true_params}")

# Compute the Hessian at the optimized parameters
H = hessian_loss(params, x_data, y_data)

# The covariance matrix is proportional to the inverse of the Hessian
# Scale factor is the residual variance (estimated from MSE)
residual_variance = loss_fn(params, x_data, y_data)
cov_matrix = jnp.linalg.inv(H) * residual_variance

# Extract standard errors (square root of diagonal elements)
std_errors = jnp.sqrt(jnp.diag(cov_matrix))

print("\nCovariance Matrix:")
print(cov_matrix)

print("\nStandard Errors:")
print(std_errors)

# Calculate correlation matrix
diag_sqrt = jnp.sqrt(jnp.diag(cov_matrix))
corr_matrix = cov_matrix / jnp.outer(diag_sqrt, diag_sqrt)

print("\nCorrelation Matrix:")
print(corr_matrix)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = jnp.linalg.eigh(cov_matrix)

print("\nEigenvalues:")
print(eigenvalues)

# Condition number (ratio of largest to smallest eigenvalue)
condition_number = jnp.max(eigenvalues) / jnp.min(eigenvalues)
print(f"\nCondition Number: {condition_number:.2f}")

# Compute relative standard errors
relative_errors = std_errors / jnp.abs(params)
print("\nRelative Standard Errors:")
print(relative_errors)

# Plot the data and fitted curve
plt.figure(figsize=(12, 6))

# Plot 1: Data and fitted curve
plt.subplot(1, 2, 1)
x_fine = jnp.linspace(0, 10, 200)
y_fit = model(params, x_fine)
y_true = true_function(x_fine, *true_params)

plt.scatter(x_data, y_data, alpha=0.7, label="Data")
plt.plot(x_fine, y_fit, "r-", label="Fitted curve")
plt.plot(x_fine, y_true, "g--", label="True curve")
plt.legend()
plt.title("Data and Model Fit")
plt.xlabel("x")
plt.ylabel("y")


# Plot 2: Parameter correlation visualization
def plot_cov_ellipse(cov, pos, ax=None, nstd=2, **kwargs):
    if ax is None:
        ax = plt.gca()

    eigenvals, eigenvecs = jnp.linalg.eigh(cov)
    # Sort eigenvalues and corresponding eigenvectors
    idx = jnp.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    angle = jnp.degrees(jnp.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width, height = 2 * nstd * jnp.sqrt(eigenvals)

    ellip = Ellipse(xy=pos, width=width, height=height, angle=angle, **kwargs)
    ax.add_artist(ellip)
    return ellip


# Plot parameter correlations for a and b
plt.subplot(1, 2, 2)
# Extract the 2x2 covariance matrix for parameters a and b
cov_ab = cov_matrix[:2, :2]
pos = (params[0], params[1])  # a and b parameter values

# Plot the ellipse
plot_cov_ellipse(cov_ab, pos, ax=plt.gca(), nstd=1, alpha=0.3, color="blue", label="1σ confidence")
plot_cov_ellipse(cov_ab, pos, ax=plt.gca(), nstd=2, alpha=0.2, color="blue", label="2σ confidence")

plt.scatter(params[0], params[1], c="red", s=50, marker="x", label="Optimized parameters")
plt.scatter(true_params[0], true_params[1], c="green", s=50, marker="o", label="True parameters")

plt.xlabel("Parameter a")
plt.ylabel("Parameter b")
plt.title("Parameter Confidence Ellipses")
plt.legend()
plt.tight_layout()

plt.savefig("parameter_optimization.png")
plt.show()

# Create a bar plot for parameter correlations
plt.figure(figsize=(10, 6))
param_names = ["a", "b", "c"]
n_params = len(param_names)

# Create correlation heatmap
plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")

# Add correlation values to the plot
for i in range(n_params):
    for j in range(n_params):
        plt.text(
            j,
            i,
            f"{corr_matrix[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
        )

plt.xticks(range(n_params), param_names)
plt.yticks(range(n_params), param_names)
plt.title("Parameter Correlation Matrix")
plt.tight_layout()

plt.savefig("parameter_correlations.png")
plt.show()

# Monte Carlo simulation to validate the covariance matrix
print("\nRunning Monte Carlo simulation to verify covariance matrix...")

# Generate many parameter samples from the estimated covariance
num_samples = 1000
key, subkey = jax.random.split(key)

# Using multivariate normal with the estimated covariance
param_samples = jax.random.multivariate_normal(subkey, mean=params, cov=cov_matrix, shape=(num_samples,))

# Plot the parameter distributions
plt.figure(figsize=(15, 5))
for i, name in enumerate(param_names):
    plt.subplot(1, 3, i + 1)

    # Histogram of samples
    plt.hist(param_samples[:, i], bins=30, alpha=0.5, density=True)

    # Theoretical normal distribution
    x = np.linspace(params[i] - 3 * std_errors[i], params[i] + 3 * std_errors[i], 100)
    plt.plot(x, stats.norm.pdf(x, params[i], std_errors[i]), "r-")

    # Add lines for true and estimated parameter values
    plt.axvline(params[i], color="blue", linestyle="-", label="Estimated")
    plt.axvline(true_params[i], color="green", linestyle="--", label="True")

    plt.title(f"Parameter {name} Distribution")
    plt.xlabel(f"{name} value")
    plt.ylabel("Density")
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.savefig("parameter_distributions.png")
plt.show()

# Summary of results
print("\n========== OPTIMIZATION RESULTS ==========")
print(f"Loss value: {loss_fn(params, x_data, y_data):.6f}")
print("\nParameter estimates:")
for i, name in enumerate(param_names):
    print(f"{name}: {params[i]:.4f} ± {std_errors[i]:.4f} (true: {true_params[i]:.4f})")

print("\nParameter correlations:")
for i in range(n_params):
    for j in range(i + 1, n_params):
        print(f"Corr({param_names[i]}, {param_names[j]}): {corr_matrix[i, j]:.4f}")

print(f"\nCondition number: {condition_number:.2f}")

# Evaluate optimization quality
print("\n========== OPTIMIZATION QUALITY ==========")
good_optimization = True

# Check 1: Parameters close to true values
param_error = jnp.abs((params - true_params) / true_params)
if jnp.any(param_error > 0.2):  # More than 20% error
    print("⚠️ Some parameters significantly differ from true values")
    good_optimization = False
else:
    print("✓ Parameters close to true values")

# Check 2: Standard errors reasonable
if jnp.any(relative_errors > 0.3):  # More than 30% relative error
    print("⚠️ Some parameters have high relative standard errors")
    good_optimization = False
else:
    print("✓ Standard errors reasonable")

# Check 3: Condition number
if condition_number > 1000:
    print("⚠️ High condition number indicates potential ill-conditioning")
    good_optimization = False
else:
    print("✓ Condition number reasonable")

# Check 4: High correlations
if jnp.any(jnp.abs(corr_matrix - jnp.eye(n_params)) > 0.9):
    print("⚠️ High parameter correlations detected")
    good_optimization = False
else:
    print("✓ Parameter correlations acceptable")

if good_optimization:
    print("\n✓ OPTIMIZATION APPEARS SUCCESSFUL")
else:
    print("\n⚠️ OPTIMIZATION MAY NEED IMPROVEMENT")
