"""
Copyright (c) 2025 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optimistix as optx
from jaxtyping import Array
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from ..kinetics.reparametrized_arrhenius import ReparametrizedArrhenius
from ..utilities.physical_constants import constants


class ArrheniusFittingResults:
    """
    Container for Arrhenius fitting results and statistics.

    Attributes
    ----------
    arrhenius : ReparametrizedArrhenius
        Fitted Arrhenius object
    fitted_values : Array
        Predicted rate constants at input temperatures
    residuals : Array
        Log-space residuals (log(k_pred) - log(k_data))
    rms_residual : float
        Root mean square of residuals
    std_errors : Optional[Array]
        Standard errors of fitted parameters
    cov_matrix : Optional[Array]
        Covariance matrix of fitted parameters
    corr_matrix : Optional[Array]
        Correlation matrix of fitted parameters
    max_correlation : Optional[float]
        Maximum absolute correlation coefficient
    condition_number : Optional[float]
        Condition number of covariance matrix
    success : bool
        Whether the fitting was successful
    T_ref : Optional[float]
        Reference temperature (for centered parametrization only)
    """

    def __init__(
        self,
        arrhenius: ReparametrizedArrhenius,
        fitted_values: Array,
        residuals: Array,
        rms_residual: float,
        T_ref: Optional[float] = None,
        std_errors: Optional[Array] = None,
        cov_matrix: Optional[Array] = None,
        corr_matrix: Optional[Array] = None,
        success: bool = True,
    ):
        self.arrhenius = arrhenius
        self.fitted_values = fitted_values
        self.residuals = residuals
        self.rms_residual = rms_residual
        self.T_ref = T_ref
        self.std_errors = std_errors
        self.cov_matrix = cov_matrix
        self.corr_matrix = corr_matrix
        self.success = success

        # ==============================================================================
        # Compute additional statistics
        if corr_matrix is not None:
            self.max_correlation = jnp.max(jnp.abs(corr_matrix - jnp.eye(len(corr_matrix))))
        else:
            self.max_correlation = None

        if cov_matrix is not None:
            eigenvals = jnp.linalg.eigvals(cov_matrix)
            self.condition_number = jnp.max(eigenvals) / jnp.min(eigenvals)
        else:
            self.condition_number = None

    def print_summary(self, verbose: bool = True) -> None:
        """
        Print a summary of the fitting results.

        Parameters
        ----------
        verbose : bool
            If True, print detailed statistics including correlations
        """
        print(f"Arrhenius Fitting Results: {self.arrhenius.name}")
        print("=" * 50)

        # Print parameters based on type
        if isinstance(self.arrhenius, ReparametrizedArrhenius):
            k_ref = jnp.exp(self.arrhenius.lnk_ref)
            Ea = self.arrhenius.EaR * constants.R_cal_mol
            print(f"Parametrization: Centered (T_ref = {self.arrhenius.T_ref:.1f} K)")
            print(f"k_ref = {k_ref:.3e}")
            print(f"n     = {self.arrhenius.n:.3f}")
            print(f"Ea    = {Ea:.1f} cal/mol")
        else:
            A = jnp.exp(self.arrhenius.lnA)
            Ea = self.arrhenius.EaR * constants.R_cal_mol
            print(f"Parametrization: Standard")
            print(f"A  = {A:.3e}")
            print(f"n  = {self.arrhenius.n:.3f}")
            print(f"Ea = {Ea:.1f} cal/mol")

        print(f"\nFit Quality:")
        print(f"RMS residual = {self.rms_residual:.4f}")
        print(f"Success = {self.success}")

        if verbose and self.std_errors is not None:
            print(f"\nParameter Uncertainties:")
            if isinstance(self.arrhenius, ReparametrizedArrhenius):
                k_ref_std = k_ref * self.std_errors[0]  # Convert from log-space
                print(f"k_ref: ±{k_ref_std:.3e} ({k_ref_std / k_ref * 100:.1f}%)")
                print(f"n:     ±{self.std_errors[1]:.3f} ({abs(self.std_errors[1] / self.arrhenius.n) * 100:.1f}%)")
                print(
                    f"Ea:    ±{self.std_errors[2] * constants.R_cal_mol:.1f} cal/mol ({abs(self.std_errors[2] * constants.R_cal_mol / Ea) * 100:.1f}%)"
                )
            else:
                A_std = A * self.std_errors[0]  # Convert from log-space
                print(f"A:  ±{A_std:.3e} ({A_std / A * 100:.1f}%)")
                print(f"n:  ±{self.std_errors[1]:.3f} ({abs(self.std_errors[1] / self.arrhenius.n) * 100:.1f}%)")
                print(
                    f"Ea: ±{self.std_errors[2] * constants.R_cal_mol:.1f} cal/mol ({abs(self.std_errors[2] * constants.R_cal_mol / Ea) * 100:.1f}%)"
                )

        if verbose and self.corr_matrix is not None:
            print(f"\nCorrelation Analysis:")
            print(f"Max |correlation| = {self.max_correlation:.3f}")
            if self.condition_number is not None:
                print(f"Condition number = {self.condition_number:.2e}")
                if self.condition_number > 1e12:
                    print("  Very high condition number - near singular covariance")
                elif self.condition_number > 1e6:
                    print("  High condition number - potential numerical issues")
                else:
                    print("  Good numerical conditioning")


def fit_centered_arrhenius(
    T: Array,
    k_data: Array,
    T_ref: Optional[float] = None,
    initial_guess: Optional[Dict[str, float]] = None,
    name: str = "",
    rtol: float = 1e-10,
    atol: float = 1e-10,
    max_steps: int = 1000,
) -> ArrheniusFittingResults:
    """
    Fit centered Arrhenius parameters to experimental data for improved numerical stability.

    Parameters
    ----------
    T : Array
        Temperature data in Kelvin
    k_data : Array
        Rate constant data
    T_ref : Optional[float]
        Reference temperature. If None, uses mean temperature
    initial_guess : Optional[Dict[str, float]]
        Initial parameter guesses {"k_ref": float, "n": float, "Ea": float}
        If None, uses heuristic estimates
    name : str
        Name for the resulting ReparametrizedArrhenius object
    rtol, atol : float
        Relative and absolute tolerances for optimization

    Returns
    -------
    ArrheniusFittingResults
        Fitting results container
    """
    # ==================================================================================
    # Validate input data
    _validate_input_data(T, k_data)

    # ==================================================================================
    # Set reference temperature
    if T_ref is None:
        T_ref = float(jnp.mean(T))
    elif T_ref <= 0:
        raise ValueError("Reference temperature must be positive")

    # ==================================================================================
    # Generate initial guess if not provided
    if initial_guess is None:
        initial_guess = _generate_initial_guess(T, k_data, T_ref)
        print(
            f"Estimated initial guess:\n k_ref: {initial_guess['k_ref']}, n: {initial_guess['n']}, Ea: {initial_guess['Ea']}"
        )

    # ==================================================================================
    # Convert to optimization parameters
    initial_params = jnp.array(
        [
            jnp.log(initial_guess["k_ref"]),
            initial_guess["n"],
            initial_guess["Ea"] / constants.R_cal_mol,
        ]
    )

    # ==================================================================================
    # Set up solver
    solver = optx.LevenbergMarquardt(rtol=rtol, atol=atol, verbose=frozenset({"loss", "step_size"}))

    # ==================================================================================
    # Solve
    solution = optx.least_squares(
        fn=_residual_function,
        solver=solver,
        y0=initial_params,
        args=(T, k_data, T_ref),
        max_steps=max_steps,
    )

    if solution.result != optx.RESULTS.successful:
        print(f"Warning: Optimization failed with status: {solution.result}")
        success = False
    else:
        success = True

    # ==================================================================================
    # Extract fitted parameters
    lnk_ref_fit, n_fit, EaR_fit = solution.value
    k_ref_fit = jnp.exp(lnk_ref_fit)
    Ea_fit = EaR_fit * constants.R_cal_mol

    # ==================================================================================
    # Create ReparametrizedArrhenius object
    parameters = {"k_ref": float(k_ref_fit), "n": float(n_fit), "Ea": float(Ea_fit), "T_ref": T_ref}
    arrhenius = ReparametrizedArrhenius(parameters, name)

    # ==================================================================================
    # Compute fitted values and residuals
    fitted_values = arrhenius.rate_constant(T)
    residuals = jnp.log(fitted_values) - jnp.log(k_data)
    rms_residual = float(jnp.sqrt(jnp.mean(residuals**2)))

    # ==================================================================================
    # Compute uncertainties
    std_errors, cov_matrix, corr_matrix, _ = _compute_parameter_uncertainty(solution)

    return ArrheniusFittingResults(
        arrhenius=arrhenius,
        fitted_values=fitted_values,
        residuals=residuals,
        rms_residual=rms_residual,
        T_ref=T_ref,
        std_errors=std_errors,
        cov_matrix=cov_matrix,
        corr_matrix=corr_matrix,
        success=success,
    )


def _validate_input_data(T: Array, k_data: Array) -> None:
    """
    Validate input temperature and rate constant data.

    Parameters
    ----------
    T : Array
        Temperature data
    k_data : Array
        Rate constant data

    Raises
    ------
    ValueError
        If data is invalid
    """
    if len(T) != len(k_data):
        raise ValueError("Temperature and rate constant arrays must have the same length")

    if len(T) < 3:
        raise ValueError("Need at least 3 data points for parameter fitting")

    if jnp.any(T <= 0):
        raise ValueError("All temperatures must be positive")

    if jnp.any(k_data <= 0):
        raise ValueError("All rate constants must be positive")

    if jnp.any(~jnp.isfinite(T)) or jnp.any(~jnp.isfinite(k_data)):
        raise ValueError("All data must be finite")


def _compute_parameter_uncertainty(
    solution: optx.Solution,
) -> Tuple[Optional[Array], Optional[Array], Optional[Array], float]:
    """
    Compute parameter uncertainties and correlation matrix from optimization solution.

    Parameters
    ----------
    solution : optx.Solution
        Solution object from optimistix least squares solver

    Returns
    -------
    Tuple[Optional[Array], Optional[Array], Optional[Array], float]
        Standard errors, covariance matrix, correlation matrix, and residual variance
    """
    try:
        # ==============================================================================
        # Get the final Jacobian from the solution
        final_jacobian = solution.state.f_info.jac

        # ==============================================================================
        # Convert to dense matrix
        if hasattr(final_jacobian, "as_dense"):
            J = final_jacobian.as_dense()
        else:
            # ==========================================================================
            # For FunctionLinearOperator, evaluate it properly
            n_params = len(solution.value)

            # ==========================================================================
            # Create unit vectors and apply the Jacobian
            J_cols = []
            for i in range(n_params):
                unit_vec = jnp.zeros(n_params)
                unit_vec = unit_vec.at[i].set(1.0)
                col = final_jacobian.mv(unit_vec)
                J_cols.append(col)

            J = jnp.column_stack(J_cols)

        residuals = solution.state.f_info.residual

        # ==============================================================================
        # Estimate residual variance
        dof = len(residuals) - len(solution.value)
        if dof > 0:
            residual_variance = jnp.sum(residuals**2) / dof
        else:
            residual_variance = jnp.sum(residuals**2)

        # ==============================================================================
        # Covariance matrix
        JtJ = J.T @ J
        cov_matrix = residual_variance * jnp.linalg.inv(JtJ)
        std_errors = jnp.sqrt(jnp.diag(cov_matrix))

        # ==============================================================================
        # Correlation matrix
        std_matrix = jnp.outer(std_errors, std_errors)
        corr_matrix = cov_matrix / std_matrix

        return std_errors, cov_matrix, corr_matrix, residual_variance

    except (jnp.linalg.LinAlgError, AttributeError) as e:
        print("Warning: Covariance matrix is singular - parameters may be non-identifiable")
        print(f"Warning: Could not compute parameter uncertainties: {e}")
        return None, None, None, 0.0


def _residual_function(params: Array, args: Tuple) -> Array:
    """
    Residual function for centered Arrhenius parametrization.

    Parameters
    ----------
    params : Array
        Parameters [log(k_ref), n, Ea/R]
    args : Tuple
        (T_data, k_data, T_ref)

    Returns
    -------
    Array
        Log-space residuals
    """
    T_data, k_data, T_ref = args
    lnk_ref, n, EaR = params

    # ==================================================================================
    # Compute predicted rate constants
    k_pred = jnp.exp(lnk_ref + n * jnp.log(T_data / T_ref) - EaR * (1.0 / T_data - 1.0 / T_ref))

    return jnp.log(k_pred) - jnp.log(k_data)


def _generate_initial_guess(
    T: Array,
    k_data: Array,
    T_ref: Optional[float] = None,
) -> Dict[str, float]:
    """
    Generate initial parameter guesses using linear regression heuristics.

    Parameters
    ----------
    T : Array
        Temperature data
    k_data : Array
        Rate constant data
    T_ref : Optional[float]
        Reference temperature for centered parametrization

    Returns
    -------
    Dict[str, float]
        Initial parameter guesses
    """
    # ==============================================================================
    # Linear regression on log(k) vs 1/T for Ea estimate
    inv_T = 1.0 / T
    log_k = jnp.log(k_data)
    X = jnp.column_stack([jnp.ones_like(inv_T), inv_T])
    coeffs = jnp.linalg.lstsq(X, log_k, rcond=None)[0]

    Ea_guess = float(-coeffs[1] * constants.R_cal_mol)

    if T_ref is None:
        T_ref = float(jnp.mean(T))
    # Estimate k_ref at T_ref from data
    idx_ref = jnp.argmin(jnp.abs(T - T_ref))
    k_ref_guess = float(k_data[idx_ref])
    return {"k_ref": k_ref_guess, "n": 0.0, "Ea": Ea_guess}


def plot_correlation_matrix(
    results: ArrheniusFittingResults,
    figsize: Tuple[float, float] = (6, 5),
    cmap: str = "RdBu_r",
    title: Optional[str] = None,
    show_values: bool = True,
    value_format: str = ".3f",
) -> None:
    """
    Plot the correlation matrix for fitted Arrhenius parameters.

    Parameters
    ----------
    results : ArrheniusFittingResults
        Fitting results containing correlation matrix
    figsize : Tuple[float, float]
        Figure size (width, height)
    cmap : str
        Colormap for the correlation matrix
    title : Optional[str]
        Custom title for the plot
    show_values : bool
        Whether to show correlation values on the matrix
    value_format : str
        Format string for correlation values
    """
    if results.corr_matrix is None:
        print("Warning: No correlation matrix available in results")
        return

    if isinstance(results.arrhenius, ReparametrizedArrhenius):
        param_labels = ["log(k_ref)", "n", "Ea/R"]
        default_title = f"Correlation Matrix - Centered Parametrization\n(T_ref = {results.T_ref:.0f} K)"
    else:
        param_labels = ["log(A)", "n", "Ea/R"]
        default_title = "Correlation Matrix - Standard Parametrization"

    if title is None:
        title = default_title

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot correlation matrix
    im = ax.imshow(results.corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect="equal")

    # Add correlation values as text
    if show_values:
        for i in range(len(param_labels)):
            for j in range(len(param_labels)):
                text = ax.text(
                    j,
                    i,
                    f"{results.corr_matrix[i, j]:{value_format}}",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

    # Set labels and ticks
    ax.set_xticks(range(len(param_labels)))
    ax.set_yticks(range(len(param_labels)))
    ax.set_xticklabels(param_labels)
    ax.set_yticklabels(param_labels)
    ax.set_title(title)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Correlation coefficient")

    # Add statistics text
    # if results.max_correlation is not None:
    #     stats_text = f"Max |correlation| = {results.max_correlation:.3f}"
    #     if results.condition_number is not None:
    #         stats_text += f"\nCondition number = {results.condition_number:.2e}"
    #
    #     ax.text(
    #         0.02,
    #         0.98,
    #         stats_text,
    #         transform=ax.transAxes,
    #         verticalalignment="top",
    #         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    #     )

    plt.tight_layout()
    plt.show()


def plot_confidence_ellipses(
    results: ArrheniusFittingResults,
    T: Optional[Array] = None,
    true_params: Optional[List[float]] = None,
    confidence_levels: List[float] = [0.68, 0.95, 0.99],
    colors: List[str] = ["red", "orange", "yellow"],
    alphas: List[float] = [0.3, 0.2, 0.1],
    figsize: Tuple[float, float] = (15, 5),
) -> None:
    """
    Plot confidence ellipses for pairwise parameter correlations.

    Parameters
    ----------
    results : ArrheniusFittingResults
        Fitting results containing correlation matrix and parameter values
    T : Optional[jnp.Array]
        Temperature data (used for computing true parameter values)
    true_params : Optional[List[float]]
        True parameter values [A, n, Ea] for comparison (if known)
    confidence_levels : List[float]
        Confidence levels for ellipses
    colors : List[str]
        Colors for each confidence level
    alphas : List[float]
        Alpha values for each confidence level
    figsize : Tuple[float, float]
        Figure size
    """
    if results.corr_matrix is None or results.std_errors is None:
        print("Warning: Correlation matrix or standard errors not available")
        return

    param_labels = ["log(k_ref)", "n", "Ea/R"]
    fitted_params = np.array([results.arrhenius.lnk_ref, results.arrhenius.n, results.arrhenius.EaR])
    title_prefix = "Centered Parametrization"

    # Compute true values if provided
    true_values = None
    if true_params is not None and T is not None:
        true_A, true_n, true_Ea = true_params
        true_EaR = true_Ea / constants.R_cal_mol
        true_k_ref = true_A * jnp.power(results.T_ref, true_n) * jnp.exp(-true_EaR / results.T_ref)
        true_values = np.array([jnp.log(true_k_ref), true_n, true_EaR])

    # Convert correlation to covariance matrix
    std_errors_np = np.array(results.std_errors)
    cov_np = results.corr_matrix * np.outer(std_errors_np, std_errors_np)

    # Parameter pairs for plotting
    param_pairs = [(0, 1), (0, 2), (1, 2)]

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for idx, (i, j) in enumerate(param_pairs):
        ax = axes[idx]

        # Extract 2x2 covariance submatrix
        cov_2d = cov_np[np.ix_([i, j], [i, j])]

        # Eigenvalues and eigenvectors for ellipse orientation
        eigenvals, eigenvecs = np.linalg.eigh(cov_2d)

        # Sort by eigenvalue
        order = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]

        # Angle of first eigenvector
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

        # Plot confidence ellipses
        for conf_level, color, alpha in zip(confidence_levels, colors, alphas):
            chi2_val = chi2.ppf(conf_level, df=2)

            width = 2 * np.sqrt(chi2_val * eigenvals[0])
            height = 2 * np.sqrt(chi2_val * eigenvals[1])

            ellipse = Ellipse(
                xy=(fitted_params[i], fitted_params[j]),
                width=width,
                height=height,
                angle=angle,
                facecolor=color,
                alpha=alpha,
                edgecolor=color,
                linewidth=2,
                label=f"{conf_level * 100:.0f}% confidence",
            )
            ax.add_patch(ellipse)

        # Plot fitted point
        ax.plot(fitted_params[i], fitted_params[j], "ko", markersize=8, label="Fitted value", zorder=10)

        # Plot true point if available
        if true_values is not None:
            ax.plot(true_values[i], true_values[j], "r*", markersize=12, label="True value", zorder=10)

        # Labels and formatting
        ax.set_xlabel(param_labels[i])
        ax.set_ylabel(param_labels[j])
        ax.set_title(f"{param_labels[i]} vs {param_labels[j]}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set axis limits based on confidence ellipses
        center_x, center_y = fitted_params[i], fitted_params[j]
        max_std_x = 3 * np.sqrt(cov_2d[0, 0])
        max_std_y = 3 * np.sqrt(cov_2d[1, 1])

        ax.set_xlim(center_x - max_std_x, center_x + max_std_x)
        ax.set_ylim(center_y - max_std_y, center_y + max_std_y)

    plt.suptitle(f"Confidence Ellipses - {title_prefix}", fontsize=16)
    plt.tight_layout()
    plt.show()
