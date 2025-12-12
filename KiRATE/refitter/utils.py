from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Float64

from KiRATE.kinetics import Arrhenius
from KiRATE.utilities import constants


@dataclass
class RefittingResult:
    """
    Results from Arrhenius parameter refitting.

    Attributes
    ----------
    arrhenius : Arrhenius
        Fitted Arrhenius object with optimized parameters
    R2 : float
        Coefficient of determination (R2)
    SSE : float
        Sum of squared errors
    RMSE : float
        Root mean squared error
    MAE : float
        Mean absolute error
    optimality : float
        First-order optimality measure (gradient norm)
    converged : bool
        Whether optimization converged successfully
    n_steps : int
        Number of optimization steps performed
    fun : float
        Final objective function value
    std_errors : Array, optional
        Standard errors of parameters [A, n, Ea]
    cov_matrix : Array, optional
        Covariance matrix of parameters
    corr_matrix : Array, optional
        Correlation matrix of parameters
    """

    arrhenius: Arrhenius
    R2: float
    SSE: float
    RMSE: float
    MAE: float
    optimality: float
    converged: bool
    n_steps: int
    fun: float
    std_errors: Optional[Float64[Array, "n"]] = None
    cov_matrix: Optional[Float64[Array, "n n"]] = None
    corr_matrix: Optional[Float64[Array, "n n"]] = None

    @property
    def A(self) -> float:
        """Pre-exponential factor."""
        return float(self.arrhenius.A)

    @property
    def n(self) -> float:
        """Temperature exponent."""
        return float(self.arrhenius.n)

    @property
    def Ea(self) -> float:
        """Activation energy (cal/mol)."""
        return float(self.arrhenius.Ea)

    @property
    def refitted_rate(self) -> "Arrhenius":
        return self.arrhenius

    @property
    def max_correlation(self) -> Optional[float]:
        """Maximum absolute off-diagonal correlation coefficient."""
        if self.corr_matrix is not None:
            return float(jnp.max(jnp.abs(self.corr_matrix - jnp.eye(len(self.corr_matrix)))))

        return None

    @property
    def condition_number(self) -> Optional[float]:
        """Condition number of covariance matrix."""
        if self.cov_matrix is not None:
            eigenvals = jnp.linalg.eigvals(self.cov_matrix)
            return float(jnp.max(jnp.abs(eigenvals)) / jnp.min(jnp.abs(eigenvals)))

        return None

    def __repr__(self) -> str:
        return (
            f"RefittingResult(A={self.A:.3e}, n={self.n:.3f}, Ea={self.Ea:.1f}, "
            f"R2={self.R2:.5f}, converged={self.converged})"
        )

    def print_summary(self, verbose: bool = True) -> None:
        """
        Print a summary of the fitting results.

        Parameters
        ----------
        verbose : bool, optional
            If True, print detailed statistics including uncertainties and correlations.
            Default is True.
        """
        print("Arrhenius Fitting Results")
        print("=" * 70)

        # Parameters
        print(" - Fitted Parameters:")
        print(f"    A             = {self.A:.6e}")
        print(f"    n             = {self.n:.6f}")
        print(f"    Ea            = {self.Ea:.2f}")

        # Fit quality
        print(" - Fit Quality:")
        print(f"    R2            = {self.R2:.6f}")
        print(f"    RMSE          = {self.RMSE:.6f}")
        print(f"    MAE           = {self.MAE:.6f}")
        print(f"    SSE           = {self.SSE:.6f}")

        # Convergence info
        print(" - Optimization:")
        print(f"    Converged     = {self.converged}")
        print(f"    Steps         = {self.n_steps}")
        print(f"    Optimality    = {self.optimality:.3e}")
        print(f"    Final obj     = {self.fun:.6f}")

        # Detailed statistics
        if verbose and self.std_errors is not None:
            print(" * Parameter Uncertainties:")
            A_std = self.A * self.std_errors[0]  # Convert from log-space
            n_std = self.std_errors[1]
            Ea_std = self.std_errors[2] * constants.R_cal_mol

            print(" * Parameter Uncertainties:")
            print(f"    A:  ±{A_std:.3e} ({A_std / self.A * 100:.1f}%)")
            print(f"    n:  ±{n_std:.6f} ({abs(n_std / self.n) * 100:.1f}%)")
            print(f"    Ea: ±{Ea_std:.2f}({abs(Ea_std / self.Ea) * 100:.1f}%)")

        if verbose and self.corr_matrix is not None:
            print(" * Correlation Analysis:")
            print(f"    Max |correlation| = {self.max_correlation:.4f}")

            if self.condition_number is not None:
                print(f"  Condition number  = {self.condition_number:.3e}")
                if self.condition_number > 1e12:
                    print("    Very high - near singular covariance matrix")
                elif self.condition_number > 1e6:
                    print("    High - potential numerical issues")
                else:
                    print("    Good numerical conditioning")

        print("=" * 70)


def compute_statistics(
    predictions: Float64[Array, "n"],
    observations: Float64[Array, "n"],
) -> dict[str, float]:
    """
    Compute fitting statistics in log space.

    Parameters
    ----------
    predictions : Float64[Array, "n"]
        Predicted values from the model
    observations : Float64[Array, "n"]
        Observed values from experimental data

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - 'R2': Coefficient of determination (R²)
        - 'SSE': Sum of squared errors
        - 'RMSE': Root mean squared error
        - 'MAE': Mean absolute error

    Notes
    -----
    Statistics are computed in log space, which gives equal weight to
    relative errors across different orders of magnitude. This is
    appropriate for rate constants that can span many orders of magnitude.
    """
    residuals = observations - predictions
    SSE = jnp.sum(residuals**2)
    SS_tot = jnp.sum((observations - jnp.mean(observations)) ** 2)
    R2 = float(1.0 - (SSE / SS_tot)) if SS_tot != 0 else 0.0
    RMSE = float(jnp.sqrt(SSE / len(residuals)))
    MAE = float(jnp.mean(jnp.abs(residuals)))

    return {"R2": R2, "SSE": float(SSE), "RMSE": RMSE, "MAE": MAE}
