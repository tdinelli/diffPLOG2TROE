from typing import Optional, Tuple

import equinox as eqx
import jax.numpy as jnp
from jax import lax, vmap
from jaxtyping import Float64

from ..utilities.custom_types import Matrix64f, ScalarOrVector


class Chebyshev(eqx.Module):
    chebyshev_coefficients: Matrix64f
    name: str
    log10P_min: Float64
    log10P_max: Float64
    T_min: Float64 = 300
    T_max: Float64 = 2500
    P_min: Float64 = 0.001
    P_max: Float64 = 100

    def __init__(
        self,
        order_T: int,
        order_P: int,
        chebyshev_coefficients: Matrix64f,
        T_limits: Optional[Tuple[Float64, Float64]] = None,
        P_limits: Optional[Tuple[Float64, Float64]] = None,
        name: str = "",
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
        self.name = name

    @eqx.filter_jit
    def kinetic_constant(
        self,
        T: ScalarOrVector,
        P: ScalarOrVector,
        is_violation_allowed: bool = False,
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

        T_tilde = (2.0 / Tc - 1.0 / self.T_min - 1.0 / self.T_max) / (1.0 / self.T_max - 1.0 / self.T_min)
        P_tilde = (2.0 * jnp.log10(Pc) - self.log10P_min - self.log10P_max) / (self.log10P_max - self.log10P_min)

        if jnp.isscalar(P) or P.ndim == 0:  # P is scalar
            return self._single_P_kinetic_constant(T_tilde, P_tilde)
        else:  # P is array
            vec_func = vmap(lambda p: self._single_P_kinetic_constant(T_tilde, p))
            return vec_func(P_tilde)


    @eqx.filter_jit
    def _single_P_kinetic_constant(self, T_tilde: ScalarOrVector, P_tilde: Float64) -> ScalarOrVector:
        N, M = self.chebyshev_coefficients.shape

        # ====================================================================
        # Calculate Chebyshev polynomials
        # phi_n[0] = cos(0 * arccos(x)) = 1, phi_n[1] = cos(1 * arccos(x)) = x, etc.
        n_indices = jnp.arange(N)  # [0, 1, 2, ..., N-1]
        m_indices = jnp.arange(M)  # [0, 1, 2, ..., M-1]

        # ====================================================================
        # Compute Chebyshev polynomials: T_n(x) = cos(n * arccos(x))
        phi_n = self.chebyshev_polynomial(n_indices[:, None], T_tilde)  # Shape: (N, *T.shape)
        phi_m = self.chebyshev_polynomial(m_indices[:, None], P_tilde)  # Shape: (M, *P.shape)

        # ====================================================================
        # Compute the weighted double sum: sum_{n,m} a_{n,m} * phi_n * phi_m
        # Using einsum for efficient broadcasting across all input shapes
        sum_result = jnp.einsum("nm,n...,m...->...", self.chebyshev_coefficients, phi_n, phi_m)

        # ====================================================================
        # Apply conversion and return: 10^(sum)
        return jnp.power(10.0, sum_result)

    @staticmethod
    @eqx.filter_jit
    def chebyshev_polynomial(n, x):
        """
        Compute Chebyshev polynomial of the first kind T_n(x).

        Args:
            n: Polynomial order (can be array)
            x: Input value (can be array)

        Returns:
            T_n(x) with appropriate broadcasting
        """
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
        """Return string representation in CHEMKIN format."""
        lines = []
        # Header line
        lines.append(f"{self.name}\t\t0.0 0.0 0.0")
        # Temperature and pressure ranges
        lines.append(f" TCHEB / {self.T_min:.2f} {self.T_max:.2f} /")
        lines.append(f" PCHEB / {self.P_min:.2f} {self.P_max:.2f} /")

        # Chebyshev coefficients
        N, M = self.chebyshev_coefficients.shape
        flattened = self.chebyshev_coefficients.flatten()

        for i, chunk_start in enumerate(range(0, len(flattened), 5)):
            chunk = flattened[chunk_start:chunk_start + 5]

            # First line includes dimensions, subsequent lines don't
            prefix = f" CHEB / {N} {M}" if i == 0 else " CHEB /"

            # Format coefficients with consistent spacing
            coeffs_str = "".join(f" {coeff:.5e}" for coeff in chunk)
            lines.append(f"{prefix}{coeffs_str} /")

        return "\n".join(lines)
