import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def calculate_hessian_jax(func, params):
    # Define gradient function
    grad_func = jax.grad(func)

    # Define the Hessian as the Jacobian of the gradient
    hessian_func = jax.jacfwd(grad_func)

    # Compute the Hessian
    return hessian_func(params)


def estimate_covariance(hessian, regularization=1e-8):
    try:
        # Ensure symmetry
        hessian_np = (hessian + hessian.T) / 2

        # Add small regularization to diagonal for numerical stability
        n = hessian_np.shape[0]
        hessian_reg = hessian_np + jnp.eye(n) * regularization

        # Invert the Hessian to get the covariance matrix
        return jnp.linalg.inv(hessian_reg)
    except:
        # If still singular after regularization, use pseudoinverse
        print("Warning: Hessian is singular even with regularization, using pseudoinverse")
        hessian_np = (hessian + hessian.T) / 2
        return jnp.linalg.pinv(hessian_np)


def analyze_parameter_covariance(func, optimal_params, param_names, regularization=1e-6):
    # Calculate Hessian at the optimum using JAX
    hessian = calculate_hessian_jax(func, optimal_params)

    # Print the condition number of the Hessian
    try:
        cond_num = jnp.linalg.cond(hessian)
        print(f"Hessian condition number: {cond_num:.2e}")
        if cond_num > 1e10:
            print("Warning: Hessian is poorly conditioned, results may be unstable")
    except:
        print("Could not compute condition number of Hessian")

    # Estimate covariance matrix with regularization
    cov_matrix = estimate_covariance(hessian, regularization)

    # Calculate the correlation matrix
    std_devs = jnp.sqrt(jnp.abs(jnp.diag(cov_matrix)))  # Use abs to avoid negative values from numerical issues
    corr_matrix = jnp.zeros_like(cov_matrix)

    for i in range(len(optimal_params)):
        for j in range(len(optimal_params)):
            # Handle cases where standard deviation is zero or NaN
            if jnp.isnan(std_devs[i]) or jnp.isnan(std_devs[j]) or std_devs[i] == 0 or std_devs[j] == 0:
                corr_matrix = corr_matrix.at[i, j].set(jnp.nan)
            else:
                corr_val = cov_matrix[i, j] / (std_devs[i] * std_devs[j])
                # Cap correlation at [-1, 1] to handle numerical issues
                corr_val = jnp.clip(corr_val, -1.0, 1.0)
                corr_matrix = corr_matrix.at[i, j].set(corr_val)

    # Calculate confidence intervals (95%)
    confidence_intervals = []
    for i, param in enumerate(optimal_params):
        if jnp.isnan(std_devs[i]):
            ci = (jnp.nan, jnp.nan)
        else:
            ci = (param - 1.96 * std_devs[i], param + 1.96 * std_devs[i])
        confidence_intervals.append(ci)

    # Return all information
    info = {
        "hessian": hessian,
        "covariance": cov_matrix,
        "correlation": corr_matrix,
        "std_devs": std_devs,
        "confidence_intervals": confidence_intervals,
    }

    # Identify parameters with valid and invalid uncertainties
    valid_indices = []
    invalid_indices = []
    for i, std in enumerate(std_devs):
        if jnp.isnan(std) or std == 0:
            invalid_indices.append(i)
        else:
            valid_indices.append(i)

    print(f"\nParameters with valid uncertainties: {[param_names[i] for i in valid_indices]}")
    print(f"Parameters with invalid uncertainties: {[param_names[i] for i in invalid_indices]}")

    # Plot the correlation matrix (masking NaN values)
    plt.figure(figsize=(10, 8))
    mask = jnp.isnan(corr_matrix)

    # Convert to numpy for seaborn compatibility
    corr_matrix_np = np.array(corr_matrix)
    mask_np = np.array(mask)

    sns.heatmap(
        corr_matrix_np,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        xticklabels=param_names,
        yticklabels=param_names,
        mask=mask_np,
    )
    plt.title("Parameter Correlation Matrix")
    plt.tight_layout()
    plt.savefig("parameter_correlation.png", dpi=300)

    # Plot correlation as a network diagram only for valid correlations
    plot_correlation_network(corr_matrix, param_names)

    # Print parameter uncertainties
    print("\n=== Parameter Uncertainties (Standard Errors) ===")
    for i, name in enumerate(param_names):
        uncertainty = "NaN" if jnp.isnan(std_devs[i]) else f"{std_devs[i]:.6f}"
        print(f"{name}: {optimal_params[i]:.6f} ± {uncertainty}")

    # Print parameter confidence intervals
    print("\n=== Parameter 95% Confidence Intervals ===")
    for i, name in enumerate(param_names):
        ci = confidence_intervals[i]
        if jnp.isnan(ci[0]) or jnp.isnan(ci[1]):
            print(f"{name}: (NaN, NaN)")
        else:
            print(f"{name}: ({ci[0]:.6f}, {ci[1]:.6f})")

    return info


def plot_correlation_network(corr_matrix, param_names, threshold=0.2):
    try:
        import networkx as nx

        # Create graph
        G = nx.Graph()

        # Add nodes only for parameters with valid correlations
        valid_params = []
        for i, name in enumerate(param_names):
            if not jnp.all(jnp.isnan(corr_matrix[i, :])):
                G.add_node(name)
                valid_params.append(i)

        # Add edges for correlations above threshold, skipping NaN values
        for i in valid_params:
            for j in valid_params:
                if i < j:  # Avoid duplicate edges
                    corr = corr_matrix[i, j]
                    if not jnp.isnan(corr) and abs(corr) > threshold:
                        G.add_edge(param_names[i], param_names[j], weight=float(corr))

        if len(G.edges()) == 0:
            print("No significant parameter correlations found, skipping network plot")
            return

        # Set up plot
        plt.figure(figsize=(10, 8))

        # Create position layout
        pos = nx.spring_layout(G, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.8)

        # Draw edges with colors based on correlation sign and width based on strength
        edge_colors = []
        edge_widths = []

        for u, v, data in G.edges(data=True):
            corr = data["weight"]
            edge_colors.append("blue" if corr > 0 else "red")
            edge_widths.append(abs(corr) * 5)

        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12)

        # Add edge labels (correlation values)
        edge_labels = {(u, v): f"{data['weight']:.2f}" for u, v, data in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

        plt.title(f"Parameter Correlation Network (|corr| > {threshold})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("correlation_network.png", dpi=300)
    except ImportError:
        print("networkx package not available, skipping network plot")


def plot_parameter_pairs(optimal_params, cov_matrix, param_names):
    """
    Plot pairwise distributions assuming a multivariate normal, handling NaN values.

    Args:
        optimal_params: Optimal parameters found by optimization
        cov_matrix: Covariance matrix
        param_names: List of parameter names
    """
    # Identify parameters with valid uncertainties
    valid_indices = []
    for i in range(len(optimal_params)):
        if not jnp.isnan(jnp.diag(cov_matrix)[i]) and jnp.diag(cov_matrix)[i] > 0:
            valid_indices.append(i)

    # Skip if no valid parameters
    if len(valid_indices) == 0:
        print("No parameters have valid uncertainties, skipping pair plots")
        return

    # Filter to include only valid parameters
    filtered_params = jnp.array([optimal_params[i] for i in valid_indices])
    filtered_names = [param_names[i] for i in valid_indices]

    # Extract valid submatrix of covariance
    n_valid = len(valid_indices)
    filtered_cov = jnp.zeros((n_valid, n_valid))
    for i in range(n_valid):
        for j in range(n_valid):
            filtered_cov = filtered_cov.at[i, j].set(cov_matrix[valid_indices[i], valid_indices[j]])

    # Create plots only for valid parameters
    fig, axes = plt.subplots(n_valid, n_valid, figsize=(15, 15))

    # If only one parameter is valid, reshape axes for indexing
    if n_valid == 1:
        axes = jnp.array([[axes]])

    # Generate multivariate normal samples around the optimum
    try:
        # Try to generate samples - might fail if covariance is still ill-conditioned
        key = jax.random.PRNGKey(0)
        samples = jax.random.multivariate_normal(key, mean=filtered_params, cov=filtered_cov, shape=(1000,))

        # Plot each parameter pair
        for i in range(n_valid):
            for j in range(n_valid):
                ax = axes[i, j]

                if i == j:
                    # Diagonal: plot histogram
                    ax.hist(samples[:, i], bins=20, alpha=0.7)
                    ax.axvline(filtered_params[i], color="r", linestyle="--")

                    # Add parameter name
                    if i == 0:
                        ax.set_title(filtered_names[i])
                else:
                    # Off-diagonal: plot scatter
                    ax.scatter(samples[:, j], samples[:, i], alpha=0.1, s=1)
                    ax.scatter([filtered_params[j]], [filtered_params[i]], color="r", s=20)

                    # Add parameter names
                    if i == n_valid - 1:
                        ax.set_xlabel(filtered_names[j])
                    if j == 0:
                        ax.set_ylabel(filtered_names[i])

        plt.tight_layout()
        plt.savefig("parameter_pairs.png", dpi=300)

    except Exception as e:
        print(f"Error generating multivariate samples: {e}")
        print("Skipping pair plots due to ill-conditioned covariance matrix")


def find_identifiability_issues(hessian, param_names, threshold=1e-6):
    # Compute eigendecomposition of the Hessian
    try:
        eigvals, eigvecs = jnp.linalg.eigh(hessian)

        # Sort eigenvalues and eigenvectors
        idx = jnp.argsort(jnp.abs(eigvals))
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Identify small eigenvalues (indicating poor identifiability)
        small_indices = jnp.where(jnp.abs(eigvals) < threshold)[0]

        if len(small_indices) > 0:
            print("\n=== Parameter Identifiability Analysis ===")
            print(f"Found {len(small_indices)} small eigenvalues (< {threshold})")

            # For each small eigenvalue, show the parameters contributing the most
            for i, idx in enumerate(small_indices):
                eigen_mode = eigvecs[:, idx]

                # Get absolute contributions
                contributions = jnp.abs(eigen_mode)

                # Sort parameters by contribution
                sorted_idx = jnp.argsort(-contributions)

                print(f"\nEigenvalue {i + 1}: {eigvals[idx]:.2e}")
                print("Major contributing parameters:")

                # Show top contributors
                for j in range(min(3, len(param_names))):
                    param_idx = sorted_idx[j]
                    print(f"  {param_names[param_idx]}: {contributions[param_idx]:.4f}")

            # Suggestions for improving identifiability
            print("\nSuggestions for improving parameter identifiability:")
            print("1. Fix some parameters to literature values")
            print("2. Reparameterize the model to reduce correlations")
            print("3. Collect more data or different experimental conditions")
            print("4. Use Bayesian methods with informative priors")

        return {"eigenvalues": eigvals, "eigenvectors": eigvecs, "small_eigenvalue_indices": small_indices}
    except:
        print("Could not perform eigendecomposition of the Hessian")
        return None


def main(loss_func, optimal_params, param_names, regularization=1e-6):
    # Analyze parameter covariance
    print("\n=== Analyzing Parameter Covariance ===")
    info = analyze_parameter_covariance(loss_func, optimal_params, param_names, regularization)

    # Analyze parameter identifiability
    find_identifiability_issues(info["hessian"], param_names)

    # Plot parameter pairs for valid parameters
    print("\n=== Plotting Parameter Pairs ===")
    plot_parameter_pairs(optimal_params, info["covariance"], param_names)

    print("\nCovariance analysis complete!")

    return info
