from dataclasses import dataclass
from typing import Optional, Union

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
    def lnA(self) -> float:
        """Natural logarithm of pre-exponential factor (transformed parameter)."""
        return float(self.arrhenius.lnA)

    @property
    def EaR(self) -> float:
        """Activation energy divided by R in temperature units (K)."""
        return float(self.arrhenius.EaR)

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

    @property
    def transformed_cov_matrix(self) -> Optional[Float64[Array, "3 3"]]:
        """
        Covariance matrix in transformed parameter space [ln(A), Ea/R, n].

        Returns
        -------
        Float64[Array, "3 3"] or None
            Covariance matrix :math:`\\Sigma '` in transformed space, or None if not computed.
            Element (i,j) is Cov(:math:`\\Theta '_i`, :math:`\\Theta '_j`)
            where :math:`\\Theta '` = [ln(A), Ea/R, n].

        Notes
        -----
        The covariance matrix stored internally is already in transformed space:
        - First parameter: ln(A), not A
        - Second parameter: Ea/(R_cal), not Ea
        - Third parameter: n (unchanged)

        This follows the Turányi-Nagy approach where transformed parameters
        approximately follow a multivariate normal distribution.
        """
        return self.cov_matrix

    @property
    def transformed_std_errors(self) -> Optional[Float64[Array, "3"]]:
        """
        Standard errors in transformed parameter space [ln(A), Ea/R, n].

        Returns
        -------
        Float64[Array, "3"] or None
            Standard errors [sigma(ln A), sigma(Ea/R), sigma(n)],
            or None if not computed.

        Notes
        -----
        These are the square roots of the diagonal elements of the transformed
        covariance matrix. They represent 1-sigma uncertainties in the
        transformed parameters.
        """
        return self.std_errors

    def rate_constant_uncertainty(
        self,
        temperature: Union[float, Float64[Array, "n"]],
    ) -> Union[float, Float64[Array, "n"]]:
        """
        Compute uncertainty in rate constant k(T) using Turányi-Nagy formula.

        This method computes the standard deviation of ln(k) at given temperature(s)
        by propagating the covariance matrix through the Arrhenius expression using
        the analytical formula from Turányi and Nagy (2011).

        Parameters
        ----------
        temperature : float or Float64[Array, "n"]
            Temperature(s) in Kelvin at which to compute uncertainty

        Returns
        -------
        float or Float64[Array, "n"]
            Standard deviation :math:`\\sigma`(ln k) in log-space (dimensionless).
            To get relative uncertainty: :math:`\\sigma (k)/k \\approx \\sigma(ln(k))`
            for small uncertainties.

        Raises
        ------
        ValueError
            If covariance matrix was not computed during fitting

        Notes
        -----
        **Mathematical Formulation**

        For the Arrhenius expression in log-space:

        .. math::
            \\ln k(T) = \\ln(A) + n \\ln(T) - \\frac{E_a}{RT}

        The variance of ln(k) is computed via the delta method:

        .. math::
            \\sigma^2[\\ln k(T)] = \\mathbf{g}(T)^T \\Sigma' \\mathbf{g}(T)

        where the gradient vector is:

        .. math::
            \\mathbf{g}(T) = \\begin{bmatrix}
                \\frac{\\partial \\ln k}{\\partial \\ln A} \\\\
                \\frac{\\partial \\ln k}{\\partial n} \\\\
                \\frac{\\partial \\ln k}{\\partial (E_a/R)}
            \\end{bmatrix} = \\begin{bmatrix}
                1 \\\\
                \\ln(T) \\\\
                -1/T
            \\end{bmatrix}

        and :math:`\\Sigma'` is the covariance matrix in transformed parameter
        space [ln(A), n, Ea/R].

        **Handling Fixed Parameters**

        When some parameters are fixed (e.g., n=0), the covariance matrix is reduced
        to only include the free parameters. The gradient is adjusted accordingly.

        **Interpretation**

        - Returns :math:`\\sigma (ln(k))`, the standard deviation of the natural log of k

        **Key Advantages**

        1. **Accounts for parameter correlations**: Uses full covariance matrix
        2. **Analytical formula**: No Monte Carlo sampling needed
        3. **Temperature-dependent uncertainty**: Shows where fit is most/least certain
        4. **Rigorous error propagation**: Based on first-order Taylor expansion

        References
        ----------
        .. [1] T. Nagy and T. Turányi, "Uncertainty of Arrhenius parameters",
               Int. J. Chem. Kinet., 43, 359-378 (2011).
               DOI: 10.1002/kin.20551
        """
        if self.cov_matrix is None:
            raise ValueError(
                "Covariance matrix not available. "
                "Ensure uncertainties were provided during fitting."
            )

        # Delegate to Arrhenius class method for full 3-parameter case
        n_params = self.cov_matrix.shape[0]

        if n_params == 3:
            # All three parameters - use Arrhenius method directly
            return self.arrhenius.rate_constant_uncertainty(
                temperature=temperature, cov_matrix=self.cov_matrix
            )

        # For masked parameters (1 or 2 free params), compute manually
        T = jnp.asarray(temperature, dtype=jnp.float64)
        scalar_input = T.ndim == 0
        T_flat = jnp.atleast_1d(T)

        # Get full gradient and extract relevant columns
        g_full = self.arrhenius.grad_ln_k_transformed_params(T_flat)

        if n_params == 2:
            # Two parameters (e.g., n fixed) [ln(A), Ea/R]
            # Extract columns 0 and 2 (ln(A) and Ea/R)
            g = g_full[:, [0, 2]]
        elif n_params == 1:
            # One parameter (e.g., A and n fixed) [Ea/R]
            # Extract column 2 (Ea/R)
            g = g_full[:, [2]]
        else:
            raise ValueError(f"Unexpected covariance matrix size: {n_params}x{n_params}")

        # Variance: σ²(ln k) = g^T Σ' g
        var_ln_k = jnp.sum(g @ self.cov_matrix * g, axis=1)
        sigma_ln_k = jnp.sqrt(var_ln_k)

        # Return scalar if input was scalar
        if scalar_input:
            return float(sigma_ln_k[0])
        else:
            return sigma_ln_k

    def __repr__(self) -> str:
        return (
            f"RefittingResult(A={self.A:.3e}, n={self.n:.3f}, Ea={self.Ea:.1f}, "
            f"R2={self.R2:.5f}, converged={self.converged})"
        )

    def print_summary(self, verbose: bool = True, show_transformed: bool = False) -> None:
        """
        Print a summary of the fitting results.

        Parameters
        ----------
        verbose : bool, optional
            If True, print detailed statistics including uncertainties and correlations.
            Default is True.
        show_transformed : bool, optional
            If True, also print transformed parameters [ln(A), Ea/R, n] following
            Turányi-Nagy (2011) convention. Default is False.
        """
        print("Arrhenius Fitting Results")
        print("=" * 70)

        # Parameters
        print(" - Fitted Parameters:")
        print(f"    A             = {self.A:.6e}")
        print(f"    n             = {self.n:.6f}")
        print(f"    Ea            = {self.Ea:.2f} cal/mol")

        # Transformed parameters (Turányi-Nagy)
        if show_transformed:
            print("\n - Transformed Parameters [ln(A), Ea/R, n]:")
            print(f"    ln(A)         = {self.lnA:.6f}")
            print(f"    Ea/R          = {self.EaR:.2f} K")
            print(f"    n             = {self.n:.6f}")

        # Fit quality
        print("\n - Fit Quality:")
        print(f"    R2            = {self.R2:.6f}")
        print(f"    RMSE          = {self.RMSE:.6f}")
        print(f"    MAE           = {self.MAE:.6f}")
        print(f"    SSE           = {self.SSE:.6f}")

        # Convergence info
        print("\n - Optimization:")
        print(f"    Converged     = {self.converged}")
        print(f"    Steps         = {self.n_steps}")
        print(f"    Optimality    = {self.optimality:.3e}")
        print(f"    Final obj     = {self.fun:.6f}")

        # Detailed statistics
        if verbose and self.std_errors is not None:
            print("\n * Parameter Uncertainties (Original Space):")
            A_std = self.A * self.std_errors[0]  # Convert from log-space
            n_std = self.std_errors[1]
            Ea_std = self.std_errors[2] * constants.R_cal_mol

            print(f"    A:  ±{A_std:.3e} ({A_std / self.A * 100:.1f}%)")
            if self.n != 0:
                print(f"    n:  ±{n_std:.6f} ({abs(n_std / self.n) * 100:.1f}%)")
            else:
                print(f"    n:  ±{n_std:.6f} (n=0, fixed or negligible)")
            print(f"    Ea: ±{Ea_std:.2f} cal/mol ({abs(Ea_std / self.Ea) * 100:.1f}%)")

            if show_transformed:
                print("\n * Parameter Uncertainties (Transformed Space):")
                print(f"    sigma(ln A)     = ±{self.std_errors[0]:.6f}")
                print(f"    sigma(Ea/R)     = ±{self.std_errors[1]:.2f} K")
                print(f"    sigma(n)        = ±{self.std_errors[2]:.6f}")

        if verbose and self.corr_matrix is not None:
            print("\n * Correlation Analysis:")
            print(f"    Max |corr|      = {self.max_correlation:.4f}")

            if self.condition_number is not None:
                print(f"    Cond. number    = {self.condition_number:.3e}")
                if self.condition_number > 1e12:
                    print("      Very high - near singular covariance matrix")
                    print("      -> Parameters are poorly identifiable")
                    print("      -> Consider: fixing n, reparametrization, or wider T range")
                elif self.condition_number > 1e6:
                    print("      High - potential numerical issues")
                    print("      -> Some parameter correlations may be strong")
                else:
                    print("      Good numerical conditioning")

            # Show correlation matrix if requested
            if show_transformed and self.corr_matrix is not None:
                print("\n * Correlation Matrix:")
                labels = ["ln(A)", "Ea/R ", "n    "]
                print("         ", "  ".join(labels))
                for i, label in enumerate(labels):
                    row_str = f"    {label}"
                    for j in range(3):
                        row_str += f"  {self.corr_matrix[i, j]:6.3f}"
                    print(row_str)

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
        - 'R2': Coefficient of determination (R2)
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
