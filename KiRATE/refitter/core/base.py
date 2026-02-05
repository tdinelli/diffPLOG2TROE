"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details

Base classes and interfaces for rate constant fitting.
"""

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import jax.numpy as jnp
from jaxtyping import Array, Float64

RateConstant = TypeVar("RateConstant")


@dataclass
class FittingResult(Generic[RateConstant]):
    """
    Base class for fitting results.

    This generic base class contains common metrics for all fitting procedures,
    regardless of the rate law being fitted (Arrhenius, PLOG, Fall-off, etc.).

    Attributes
    ----------
    rate_constant : RateConstant
        Fitted rate constant object (Arrhenius, Plog, FallOff, etc.)
    R2 : float
        Coefficient of determination (R²)
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
        Standard errors of parameters
    cov_matrix : Array, optional
        Covariance matrix of parameters
    corr_matrix : Array, optional
        Correlation matrix of parameters
    """

    rate_constant: RateConstant
    R2: float
    SSE: float
    RMSE: float
    MAE: float
    optimality: float
    converged: bool
    n_steps: int
    fun: float
    std_errors: Float64[Array, "n"] | None = None
    cov_matrix: Float64[Array, "n n"] | None = None
    corr_matrix: Float64[Array, "n n"] | None = None

    @property
    def max_correlation(self) -> float | None:
        """Maximum absolute off-diagonal correlation coefficient."""
        if self.corr_matrix is not None:
            return float(jnp.max(jnp.abs(self.corr_matrix - jnp.eye(len(self.corr_matrix)))))
        return None

    @property
    def condition_number(self) -> float | None:
        """Condition number of covariance matrix."""
        if self.cov_matrix is not None:
            eigenvals = jnp.linalg.eigvals(self.cov_matrix)
            return float(jnp.max(jnp.abs(eigenvals)) / jnp.min(jnp.abs(eigenvals)))
        return None


class Fitter(Protocol[RateConstant]):
    """
    Protocol defining the interface for rate constant fitting.

    All specific fitters (ArrheniusFitter, PlogFitter, etc.) should implement
    this protocol. Using Protocol instead of ABC allows compatibility with
    Equinox modules and duck typing.

    Type Parameters
    ---------------
    RateConstant
        The type of rate constant object this fitter produces
        (e.g., Arrhenius, Plog, FallOff)
    """

    def fit(
        self,
        temperature: Float64[Array, "n"],
        rate_constant: Float64[Array, "n"],
        uncertainties: Float64[Array, "n"] | None = None,
        **kwargs,
    ) -> FittingResult[RateConstant]:
        """
        Fit rate constant parameters to data.

        Parameters
        ----------
        temperature : Float64[Array, "n"]
            Temperature values [K]
        rate_constant : Float64[Array, "n"]
            Rate constant values
        uncertainties : Float64[Array, "n"], optional
            Uncertainties in rate constant values
        **kwargs
            Additional fitter-specific parameters

        Returns
        -------
        FittingResult[RateConstant]
            Fitting results with optimized parameters and statistics
        """
        ...

    def initial_guess(
        self,
        temperature: Float64[Array, "n"],
        rate_constant: Float64[Array, "n"],
        uncertainties: Float64[Array, "n"] | None = None,
        **kwargs,
    ) -> Float64[Array, "n_params"]:
        """
        Compute initial parameter guess.

        Parameters
        ----------
        temperature : Float64[Array, "n"]
            Temperature values [K]
        rate_constant : Float64[Array, "n"]
            Rate constant values
        uncertainties : Float64[Array, "n"], optional
            Uncertainties in rate constant values
        **kwargs
            Additional fitter-specific parameters

        Returns
        -------
        Float64[Array, "n_params"]
            Initial parameter guess
        """
        ...
