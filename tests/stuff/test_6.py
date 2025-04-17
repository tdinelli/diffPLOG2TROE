import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

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


# Define priors
def model(T_range, P_range, k_plog_obs):
    # Prior distributions for each parameter
    log_A = numpyro.sample("log_A", dist.Normal(jnp.log(1e30), 5.0))
    n = numpyro.sample("n", dist.Normal(-4.0, 1.0))
    E_over_R = numpyro.sample("E_over_R", dist.Normal(1400.0, 500.0))

    # Alpha has a constraint between 0 and 1
    alpha = numpyro.sample("alpha", dist.Beta(2.0, 2.0))

    # Other Troe parameters
    T3 = numpyro.sample("T3", dist.LogNormal(jnp.log(1000.0), 1.0))
    T1 = numpyro.sample("T1", dist.LogNormal(jnp.log(300.0), 1.0))
    T2 = numpyro.sample("T2", dist.LogNormal(jnp.log(2000.0), 1.0))

    # Combine parameters
    params = jnp.array([log_A, n, E_over_R, alpha, T3, T1, T2])

    # Observation model - calculate predicted rates
    def calculate_rates(params):
        falloff = FallOff(
            name="",
            hpl_params=jnp.array([1.500e14, 0.167, 0.000e00]),
            lpl_params=jnp.array([jnp.exp(params[0]), params[1], params[2] * 1.987]),
            falloff_params=jnp.array([params[3], params[4], params[5], params[6]]),
            falloff_type="troe",
        )
        predictions = falloff.kinetic_constant(T_range, P_range)[0]
        return predictions

    k_troe_pred = calculate_rates(params)

    # Likelihood - use log-normal distribution to model multiplicative error
    sigma = numpyro.sample("sigma", dist.LogNormal(-2.0, 1.0))  # Error scale parameter

    # Flatten arrays for observation model
    log_k_troe_pred = jnp.log(k_troe_pred).flatten()
    log_k_plog_obs = jnp.log(k_plog_obs).flatten()

    # Log-normal likelihood
    numpyro.sample("obs", dist.Normal(log_k_troe_pred, sigma), obs=log_k_plog_obs)


def run_mcmc(T_range, P_range, k_plog, num_warmup=1000, num_samples=2000, num_chains=4):
    """
    Run MCMC to sample from the posterior distribution of parameters.

    Args:
        T_range: Temperature range array
        P_range: Pressure range array
        k_plog: Observed rate constants from PLOG
        num_warmup: Number of warmup steps
        num_samples: Number of posterior samples
        num_chains: Number of MCMC chains

    Returns:
        MCMC object with samples
    """
    # Set up the sampler with NUTS
    nuts_kernel = NUTS(model)

    # Run MCMC
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )

    # Execute the run
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key, T_range, P_range, k_plog)

    return mcmc


def analyze_posterior(mcmc):
    """
    Analyze and visualize the posterior distributions.

    Args:
        mcmc: MCMC object with samples
    """
    # Print summary statistics
    mcmc.print_summary()

    # Convert to arviz object for better visualization
    data = az.from_numpyro(mcmc)

    # Plot posterior distributions
    az.plot_posterior(data)
    plt.tight_layout()
    plt.savefig("posterior_distributions.png", dpi=300)

    # Plot trace plots to check convergence
    az.plot_trace(data)
    plt.tight_layout()
    plt.savefig("trace_plots.png", dpi=300)

    # Plot pair plots to see correlations
    az.plot_pair(data, var_names=["log_A", "n", "E_over_R", "alpha", "T3", "T1", "T2"], kind="kde")
    plt.tight_layout()
    plt.savefig("parameter_correlations.png", dpi=300)

    # Get samples as a dictionary
    samples = mcmc.get_samples()

    return samples


def evaluate_model_with_samples(samples, T_range, P_range, k_plog, num_draws=100):
    """
    Evaluate the model using parameter samples from the posterior.

    Args:
        samples: Dictionary of parameter samples
        T_range: Temperature range array
        P_range: Pressure range array
        k_plog: Observed rate constants
        num_draws: Number of posterior draws to plot
    """
    # Get a subset of samples to visualize
    keys = ["log_A", "n", "E_over_R", "alpha", "T3", "T1", "T2"]
    num_samples = len(samples[keys[0]])
    indices = jnp.linspace(0, num_samples - 1, num_draws, dtype=int)

    # Create parameter arrays for each sample
    all_params = []
    for i in indices:
        params = jnp.array(
            [
                samples[keys[0]][i],
                samples[keys[1]][i],
                samples[keys[2]][i],
                samples[keys[3]][i],
                samples[keys[4]][i],
                samples[keys[5]][i],
                samples[keys[6]][i],
            ]
        )
        all_params.append(params)

    # Choose a few T,P pairs to visualize
    t_indices = jnp.linspace(0, len(T_range) - 1, 5, dtype=int)
    p_indices = jnp.linspace(0, len(P_range) - 1, 5, dtype=int)

    # Create figure for predictions
    fig, axes = plt.subplots(len(t_indices), len(p_indices), figsize=(15, 12))

    # Add a figure title
    fig.suptitle("Rate Constant Posterior Predictions at Different T,P Points", fontsize=16)

    # Plot histograms of predicted rates for each T,P pair
    for t_idx, t_i in enumerate(t_indices):
        for p_idx, p_i in enumerate(p_indices):
            # Calculate rate constants for all parameter samples
            pred_rates = []
            for params in all_params:
                falloff = FallOff(
                    name="",
                    hpl_params=jnp.array([1.500e14, 0.167, 0.000e00]),
                    lpl_params=jnp.array([jnp.exp(params[0]), params[1], params[2] * 1.987]),
                    falloff_params=jnp.array([params[3], params[4], params[5], params[6]]),
                    falloff_type="troe",
                )
                k_troe = falloff.kinetic_constant(T_range[t_i : t_i + 1], P_range[p_i : p_i + 1])[0]
                pred_rates.append(k_troe[0][0])

            # Plot histogram of predicted rates
            ax = axes[t_idx, p_idx]
            ax.hist(pred_rates, bins=20, alpha=0.6)
            ax.set_title(f"T={T_range[t_i]:.0f}K, P={P_range[p_i]:.2f}bar")

            # Add reference line for PLOG value
            ax.axvline(k_plog[t_i, p_i], color="red", linestyle="--")

            # Use log scale since rates can span many orders of magnitude
            ax.set_xscale("log")

            # Only show y-axis for the leftmost plots
            if p_idx > 0:
                ax.set_yticklabels([])

            # Only show x-axis label for the bottom plots
            if t_idx == len(t_indices) - 1:
                ax.set_xlabel("Rate constant")

            # Only show y-axis label for the leftmost plots
            if p_idx == 0:
                ax.set_ylabel("Frequency")

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the title
    plt.savefig("posterior_predictions.png", dpi=300)

    # Create a heatmap of the ratio uncertainty
    plot_uncertainty_map(samples, T_range, P_range, k_plog)

    return all_params


def plot_uncertainty_map(samples, T_range, P_range, k_plog):
    """
    Create a heatmap showing the uncertainty in the ratio of predicted to observed rates.

    Args:
        samples: Dictionary of parameter samples
        T_range: Temperature range array
        P_range: Pressure range array
        k_plog: Observed rate constants
    """
    # Get a subset of samples to use (e.g., 100 samples)
    keys = ["log_A", "n", "E_over_R", "alpha", "T3", "T1", "T2"]
    num_samples = len(samples[keys[0]])
    subset_size = min(100, num_samples)  # Use at most 100 samples for speed
    indices = jnp.linspace(0, num_samples - 1, subset_size, dtype=int)

    # Storage for ratios from all samples
    all_ratios = []

    # Calculate ratios for each sample
    for i in indices:
        params = jnp.array(
            [
                samples[keys[0]][i],
                samples[keys[1]][i],
                samples[keys[2]][i],
                samples[keys[3]][i],
                samples[keys[4]][i],
                samples[keys[5]][i],
                samples[keys[6]][i],
            ]
        )

        # Calculate rate constants
        falloff = FallOff(
            name="",
            hpl_params=jnp.array([1.500e14, 0.167, 0.000e00]),
            lpl_params=jnp.array([jnp.exp(params[0]), params[1], params[2] * 1.987]),
            falloff_params=jnp.array([params[3], params[4], params[5], params[6]]),
            falloff_type="troe",
        )
        k_troe = falloff.kinetic_constant(T_range, P_range)[0]
        ratio = k_troe.T / k_plog.T
        all_ratios.append(ratio)

    # Convert to array for easier manipulation
    all_ratios = jnp.array(all_ratios)

    # Calculate statistics across samples
    mean_ratio = jnp.mean(all_ratios, axis=0)
    std_ratio = jnp.std(all_ratios, axis=0)
    cv_ratio = std_ratio / mean_ratio  # Coefficient of variation = std / mean

    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot mean ratio
    im1 = axes[0].imshow(
        mean_ratio,
        origin="lower",
        aspect="auto",
        extent=[P_range[0], P_range[-1], T_range[0], T_range[-1]],
        cmap="Pastel2",
        vmin=0.5,
        vmax=1.5,
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Pressure [bar]", fontweight="bold")
    axes[0].set_ylabel("Temperature [K]", fontweight="bold")
    axes[0].set_title("Mean Ratio (TROE/PLOG)", fontweight="bold")
    plt.colorbar(im1, ax=axes[0])

    # Plot standard deviation of the ratio
    im2 = axes[1].imshow(
        std_ratio,
        origin="lower",
        aspect="auto",
        extent=[P_range[0], P_range[-1], T_range[0], T_range[-1]],
        cmap="viridis",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Pressure [bar]", fontweight="bold")
    axes[1].set_ylabel("Temperature [K]", fontweight="bold")
    axes[1].set_title("Standard Deviation of Ratio", fontweight="bold")
    plt.colorbar(im2, ax=axes[1])

    # Plot coefficient of variation (relative uncertainty)
    im3 = axes[2].imshow(
        cv_ratio,
        origin="lower",
        aspect="auto",
        extent=[P_range[0], P_range[-1], T_range[0], T_range[-1]],
        cmap="plasma",
    )
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Pressure [bar]", fontweight="bold")
    axes[2].set_ylabel("Temperature [K]", fontweight="bold")
    axes[2].set_title("Coefficient of Variation (std/mean)", fontweight="bold")
    plt.colorbar(im3, ax=axes[2])

    fig.tight_layout()
    plt.savefig("ratio_uncertainty.png", dpi=300)


def main():
    # Set up JAX to use double precision
    jax.config.update("jax_enable_x64", True)

    # Run MCMC
    print("Running MCMC sampling...")
    mcmc = run_mcmc(T_range, P_range, k_plog, num_warmup=1000, num_samples=2000, num_chains=4)

    # Analyze posterior
    print("\nAnalyzing posterior distribution...")
    samples = analyze_posterior(mcmc)

    # Evaluate model with posterior samples
    print("\nEvaluating model with posterior samples...")
    evaluate_model_with_samples(samples, T_range, P_range, k_plog)

    print("\nPosterior analysis complete!")


if __name__ == "__main__":
    main()
