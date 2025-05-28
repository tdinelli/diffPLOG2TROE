"""
Not tested yet
"""
from typing import Optional, Tuple

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Float64

from ..utilities.custom_types import Matrix64f, ScalarOrVector


class Chebyshev(eqx.Module):
    chebyshev_coefficients: Matrix64f
    T_min: Float64 = 300
    T_max: Float64 = 2500
    P_min: Float64 = 0.001
    P_max: Float64 = 100
    log10P_min = jnp.log10(P_min)
    log10P_max = jnp.log10(P_max)

    def __init__(
            self,
        order_T: int,
        order_P: int,
        chebyshev_coefficients: Matrix64f,
        T_limits: Optional[Tuple[Float64, Float64]] = None,
        P_limits: Optional[Tuple[Float64, Float64]] = None,
    ) -> None:
        if T_limits is not None:
            self._validate_limits(T_limits)
            self.T_min, self.T_max = T_limits
        if P_limits is not None:
            self._validate_limits(P_limits)
            self.P_min, self.P_max = P_limits
            self.log10P_min = jnp.log10(self.P_min)
            self.log10P_max = jnp.log10(self.P_max)

        if chebyshev_coefficients.shape[0] != order_T or chebyshev_coefficients.shape[1] != order_P:
            raise ValueError(
                f"The number of chebyshev coefficients must be equal to {order_T * order_P} given {chebyshev_coefficients.shape[0] * chebyshev_coefficients.shape[1]}"
            )
        self.chebyshev_coefficients = chebyshev_coefficients

    @eqx.filter_jit
    def kinetic_constant(
        self,
        T: ScalarOrVector,
        P: ScalarOrVector,
        is_violation_allowed: bool,
    ) -> ScalarOrVector:
        Tc, Pc = lax.cond(
            is_violation_allowed,
            lambda operands: (
                jnp.clip(operands[0], self.T_min, self.T_max),
                jnp.clip(operands[1], self.P_min, self.P_max),
            ),
            lambda operands: (operands[0], operands[1]),
            (T, P),
        )

        Ttilde = (2.0 / Tc - 1.0 / self.T_min - 1.0 / self.T_max) / (1.0 / self.T_max - 1.0 / self.T_min)
        Ptilde = (2.0 * jnp.log10(Pc) - self.log10P_min - self.log10P_max) / (self.log10P_max - self.log10P_min)

        N, M = self.chebyshev_coefficients.shape

        # ====================================================================
        # Create polynomial order arrays
        n_orders = jnp.arange(1, N + 1)  # [1, 2, ..., N]
        m_orders = jnp.arange(1, M + 1)  # [1, 2, ..., M]

        # ====================================================================
        # Compute Chebyshev polynomials
        # phi_n will have shape (N, *T.shape)
        # phi_m will have shape (M, *P.shape)
        phi_n = self.chebyshev_poly(n_orders[:, None], Ttilde)
        phi_m = self.chebyshev_poly(m_orders[:, None], Ptilde)

        # ====================================================================
        # Compute the weighted double sum using einsum
        # This handles all broadcasting cases automatically:
        # - T scalar, P scalar: result is scalar
        # - T vector, P scalar: result has shape (*T.shape,)
        # - T scalar, P vector: result has shape (*P.shape,)
        # - T vector, P vector: result has shape (*T.shape, *P.shape)
        sum_result = jnp.einsum("nm,n...,m...->...", self.chebyshev_coefficients, phi_n, phi_m)

        return jnp.power(10.0, sum_result)

    @staticmethod
    @eqx.filter_jit
    def chebyshev_poly(n, x):
        """
        Compute Chebyshev polynomial of the first kind T_n(x).

        Args:
            n: Polynomial order (can be array)
            x: Input value (can be array)

        Returns:
            T_n(x) with appropriate broadcasting
        """
        # Use the recurrence relation: T_0(x) = 1, T_1(x) = x
        # T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)

        # Handle the case where n or x might be arrays
        n = jnp.asarray(n)
        x = jnp.asarray(x)

        # For vectorized computation, we'll use the explicit formula
        # T_n(x) = cos(n * arccos(x)) for |x| <= 1
        # For |x| > 1, we use the hyperbolic form
        # Clip x to handle numerical issues near boundaries
        x_clipped = jnp.clip(x, -1.0, 1.0)

        return jnp.cos(n * jnp.arccos(x_clipped))

    @staticmethod
    def _validate_limits(limits: Tuple[float, float]) -> None:
        """Validate that lower_limit ≤ upper_limit."""
        lower_limit, upper_limit = limits
        if lower_limit > upper_limit:
            raise ValueError(
                f"Invalid interval: [{lower_limit}, {upper_limit}] (lower > upper by {lower_limit - upper_limit:.6g})"
            )


    def __str__(self) -> str:
        return ""
