import equinox as eqx
import jax.numpy as jnp
from jax import lax, vmap

from ..types.common import Integer, Matrix, RateType, Scalar, ScalarOrVector, TupleOfFloat


class Chebyshev(eqx.Module):
    _chebyshev_coefficients: Matrix
    _log10P_min: Scalar
    _log10P_max: Scalar
    _T_min: Scalar
    _T_max: Scalar
    _P_min: Scalar
    _P_max: Scalar
    _name: str = eqx.field(static=True, default="")

    def __init__(
        self,
        order_T: Integer,
        order_P: Integer,
        chebyshev_coefficients: Matrix,
        T_limits: TupleOfFloat = (300.0, 2500.0),
        P_limits: TupleOfFloat = (0.001, 100.0),
        name: str = "",
    ) -> None:
        self._validate_limits(T_limits)
        self._T_min, self._T_max = T_limits

        self._validate_limits(P_limits)
        self._P_min, self._P_max = P_limits
        self._log10P_min = jnp.log10(self._P_min)
        self._log10P_max = jnp.log10(self._P_max)

        if chebyshev_coefficients.shape[0] != order_T or chebyshev_coefficients.shape[1] != order_P:
            raise ValueError(
                f"The number of chebyshev coefficients must be equal to {order_T * order_P} given {chebyshev_coefficients.shape[0] * chebyshev_coefficients.shape[1]}"
            )
        self._chebyshev_coefficients = chebyshev_coefficients
        self._name = name

    @eqx.filter_jit
    def rate_constant(self, T: ScalarOrVector, P: ScalarOrVector, is_violation_allowed: bool = False) -> RateType:
        Tc, Pc = lax.cond(
            is_violation_allowed,
            lambda operands: (
                jnp.clip(operands[0], self._T_min, self._T_max),
                jnp.clip(operands[1], self._P_min, self._P_max),
            ),
            lambda operands: (operands[0], operands[1]),
            (T, P),
        )

        T_tilde = (2.0 / Tc - 1.0 / self._T_min - 1.0 / self._T_max) / (1.0 / self._T_max - 1.0 / self._T_min)
        P_tilde = (2.0 * jnp.log10(Pc) - self._log10P_min - self._log10P_max) / (self._log10P_max - self._log10P_min)

        if jnp.isscalar(P):  # P is scalar
            return self._single_P_rate_constant(T_tilde, P_tilde)
        else:  # P is array
            vec_func = vmap(lambda p: self._single_P_rate_constant(T_tilde, p))
            return vec_func(P_tilde)

    @eqx.filter_jit
    def _single_P_rate_constant(self, T_tilde: ScalarOrVector, P_tilde: Scalar) -> ScalarOrVector:
        N, M = self._chebyshev_coefficients.shape

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
        sum_result = jnp.einsum("nm,n...,m...->...", self._chebyshev_coefficients, phi_n, phi_m)

        # ====================================================================
        # Apply conversion and return: 10^(sum)
        return jnp.power(10.0, sum_result)

    @staticmethod
    @eqx.filter_jit
    def chebyshev_polynomial(n, x):
        # Clip x to handle numerical issues near boundaries
        x_clipped = jnp.clip(x, -1.0, 1.0)
        return jnp.cos(n * jnp.arccos(x_clipped))

    @staticmethod
    def _validate_limits(limits: TupleOfFloat) -> None:
        lower_limit, upper_limit = limits
        if lower_limit > upper_limit:
            raise ValueError(
                f"Invalid interval: [{lower_limit}, {upper_limit}] (lower > upper by {lower_limit - upper_limit:.6g})"
            )

    def __str__(self) -> str:
        lines = []

        # Header line
        lines.append(f"{self._name}\t\t0.0 0.0 0.0")

        # Temperature and pressure ranges
        lines.append(f" TCHEB / {self._T_min:.2f} {self._T_max:.2f} /")
        lines.append(f" PCHEB / {self._P_min:.2f} {self._P_max:.2f} /")

        # Chebyshev coefficients
        N, M = self._chebyshev_coefficients.shape
        flattened = self._chebyshev_coefficients.flatten()

        for i, chunk_start in enumerate(range(0, len(flattened), 5)):
            chunk = flattened[chunk_start : chunk_start + 5]

            # First line includes dimensions, subsequent lines don't
            prefix = f" CHEB / {N} {M}" if i == 0 else " CHEB /"

            # Format coefficients with consistent spacing
            coeffs_str = "".join(f" {coeff:.5e}" for coeff in chunk)
            lines.append(f"{prefix}{coeffs_str} /")

        return "\n".join(lines)
