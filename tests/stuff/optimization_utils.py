"""
Utility functions for optimization data visualization and analysis.
"""

import jax.numpy as jnp


def print_optimization_table(losses, param_history, grad_history, param_names, iterations_to_print=10):
    """
    Print a tabular summary of the optimization process.

    Args:
        losses: Array of loss values for each iteration
        param_history: Array of parameter values for each iteration
        grad_history: Array of gradient values for each iteration
        param_names: List of parameter names
        iterations_to_print: Number of evenly spaced iterations to print
    """
    num_iterations = len(losses)

    # Select evenly spaced iterations to print
    if num_iterations <= iterations_to_print:
        indices = jnp.arange(num_iterations)
    else:
        indices = jnp.linspace(0, num_iterations - 1, iterations_to_print, dtype=int)

    # Create header
    header = ["Iter", "Loss", "||Grad||"]
    for i, name in enumerate(param_names):
        header.extend([f"Param[{i}]", f"∇Param[{i}]"])

    # Print header
    header_format = " | ".join(["{:^10}" for _ in header])
    separator = "-" * (12 * len(header))

    print("\n=== Optimization Trajectory Summary ===")
    print(separator)
    print(header_format.format(*header))
    print(separator)

    # Print rows
    for idx in indices:
        row = [idx, losses[idx], jnp.linalg.norm(grad_history[idx])]
        for i in range(len(param_names)):
            row.extend([param_history[idx + 1][i], grad_history[idx][i]])

        # Format rows with appropriate precision
        formatted_row = []
        for i, val in enumerate(row):
            if i == 0:  # Iteration index
                formatted_row.append(f"{val:10d}")
            elif i == 1:  # Loss value
                formatted_row.append(f"{val:10.4e}")
            elif i == 2:  # Gradient norm
                formatted_row.append(f"{val:10.4e}")
            elif i % 2 == 1:  # Parameter gradients
                formatted_row.append(f"{val:10.4e}")
            else:  # Parameter values
                formatted_row.append(f"{val:10.4f}")

        row_format = " | ".join(["{}" for _ in formatted_row])
        print(row_format.format(*formatted_row))

    # Also print the last iteration if not already included
    if num_iterations - 1 not in indices:
        idx = num_iterations - 1
        row = [idx, losses[idx], jnp.linalg.norm(grad_history[idx])]
        for i in range(len(param_names)):
            row.extend([param_history[idx + 1][i], grad_history[idx][i]])

        formatted_row = []
        for i, val in enumerate(row):
            if i == 0:
                formatted_row.append(f"{val:10d}")
            elif i == 1:
                formatted_row.append(f"{val:10.4e}")
            elif i == 2:
                formatted_row.append(f"{val:10.4e}")
            elif i % 2 == 1:
                formatted_row.append(f"{val:10.4e}")
            else:
                formatted_row.append(f"{val:10.4f}")

        row_format = " | ".join(["{}" for _ in formatted_row])
        print(separator)
        print(row_format.format(*formatted_row))

    print(separator)

    # Save table to CSV for further analysis
    try:
        import pandas as pd

        # Create a DataFrame with all iterations
        data = {
            "Iteration": range(num_iterations),
            "Loss": losses,
            "Gradient_Norm": [jnp.linalg.norm(g) for g in grad_history],
        }

        # Add parameters and gradients
        for i in range(len(param_names)):
            data[f"Param_{i}"] = [p[i] for p in param_history[1:]]  # Skip initial guess
            data[f"Grad_{i}"] = [g[i] for g in grad_history]

        # Create DataFrame and save
        df = pd.DataFrame(data)
        csv_filename = "optimization_trajectory.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Complete optimization trajectory saved to {csv_filename}")
    except ImportError:
        print("pandas not available, CSV export skipped")
    except Exception as e:
        print(f"Error exporting to CSV: {e}")


def plot_optimization_results(losses, param_history, grad_history, param_names, save_figures=True):
    """
    Generate and optionally save comprehensive plots of optimization results.

    Args:
        losses: Array of loss values for each iteration
        param_history: Array of parameter values for each iteration
        grad_history: Array of gradient values for each iteration
        param_names: List of parameter names
        save_figures: Whether to save figures to disk

    Returns:
        Dictionary containing references to generated figures
    """
    import matplotlib.pyplot as plt

    figures = {}

    # Create figure for parameter plots
    fig_params = plt.figure(figsize=(15, 12))
    figures["parameters"] = fig_params

    # Create a grid layout for all parameters plus the loss plot
    num_params = len(param_names)

    # Plot loss function
    ax_loss = fig_params.add_subplot(4, 2, 1)
    ax_loss.plot(losses, "b-")
    ax_loss.set_title("Loss vs. Iterations")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale("log")
    ax_loss.grid(True)

    # Plot each parameter trajectory
    for i in range(num_params):
        ax = fig_params.add_subplot(4, 2, i + 2)
        param_values = [h[i] for h in param_history[1:]]  # Skip initial guess
        ax.plot(param_values, "g-")
        ax.set_title(f"Parameter {i}: {param_names[i]}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.grid(True)

        # Add special constraints for alpha parameter if applicable
        if i == 3 and "α" in param_names[i]:
            ax.axhline(y=0.0, color="r", linestyle="--", alpha=0.7)
            ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.7)

    fig_params.tight_layout()
    if save_figures:
        plt.savefig("parameter_history.png", dpi=300)

    # Create a figure for gradient plots
    fig_grads = plt.figure(figsize=(15, 12))
    figures["gradients"] = fig_grads

    # Plot each parameter's gradient
    for i in range(num_params):
        ax = fig_grads.add_subplot(4, 2, i + 1)
        grad_values = [g[i] for g in grad_history]
        ax.plot(grad_values, "r-")
        ax.set_title(f"Gradient {i}: ∇{param_names[i]}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Gradient")
        ax.grid(True)

        # Add a horizontal line at y=0 for reference
        ax.axhline(y=0.0, color="k", linestyle="--", alpha=0.5)

        # Use symmetric log scale for better visualization of gradients
        ax.set_yscale("symlog", linthresh=1e-6)

    # Add a plot of gradient magnitudes
    ax = fig_grads.add_subplot(4, 2, 8)
    grad_magnitudes = [jnp.linalg.norm(g) for g in grad_history]
    ax.plot(grad_magnitudes, "b-")
    ax.set_title("Gradient Magnitude ||∇L||")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Magnitude")
    ax.set_yscale("log")
    ax.grid(True)

    fig_grads.tight_layout()
    if save_figures:
        plt.savefig("gradient_history.png", dpi=300)

    return figures
