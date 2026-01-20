"""
Test utilities for unified comparison and error reporting.

Copyright (c) 2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import jax.numpy as jnp


def assert_rate_constants_close(
    test_case,
    calculated,
    expected,
    T_range=None,
    P_range=None,
    atol=1e-10,
    rtol=1e-8,
    test_name="Rate constant comparison",
):
    """
    Unified assertion for rate constant comparisons with detailed error reporting.

    This helper function standardizes the comparison of calculated vs expected
    rate constants across all test files, providing consistent error messages
    with detailed information about where mismatches occur.

    Parameters
    ----------
    test_case : unittest.TestCase
        The test case instance (self) to call assertions on
    calculated : jax.Array
        Calculated rate constants from KiRATE
    expected : jax.Array
        Expected rate constants from reference (e.g., Cantera)
    T_range : jax.Array, optional
        Temperature array for error reporting. If None, indices only are reported.
    P_range : jax.Array, optional
        Pressure array for error reporting. If None, indices only are reported.
    atol : float, optional
        Absolute tolerance for jnp.allclose (default: 1e-10)
    rtol : float, optional
        Relative tolerance for jnp.allclose (default: 1e-8)
    test_name : str, optional
        Descriptive name for the test (default: "Rate constant comparison")

    Raises
    ------
    AssertionError
        If calculated and expected values don't match within tolerances
    """
    # Check shape match first
    test_case.assertEqual(
        calculated.shape,
        expected.shape,
        f"{test_name}: Shape mismatch - calculated {calculated.shape} vs expected {expected.shape}",
    )

    # Calculate relative errors
    rel_errors = jnp.abs(calculated - expected) / jnp.abs(expected)
    max_rel_error = jnp.max(rel_errors)
    max_error_idx = jnp.unravel_index(jnp.argmax(rel_errors), rel_errors.shape)

    # Check if all values are close
    is_close = jnp.allclose(calculated, expected, atol=atol, rtol=rtol)

    # If not close, build detailed error message
    if not is_close:
        # Handle different dimensionalities
        if calculated.ndim == 1:
            # 1D case (temperature only)
            idx = max_error_idx[0] if isinstance(max_error_idx, tuple) else max_error_idx
            error_parts = [f"Max relative error: {max_rel_error:.6e}"]

            if T_range is not None:
                error_parts.append(f"at T_idx={idx} (T={T_range[idx].item():.2f}K)")
            else:
                error_parts.append(f"at index={idx}")

            error_parts.append(f"calculated={calculated[idx].item():.6e}")
            error_parts.append(f"expected={expected[idx].item():.6e}")

        elif calculated.ndim == 2:
            # 2D case (pressure and temperature)
            p_idx, t_idx = max_error_idx
            error_parts = [f"Max relative error: {max_rel_error:.6e}"]

            location_parts = []
            if P_range is not None:
                location_parts.append(f"P_idx={p_idx} (P={P_range[p_idx].item():.6f}bar)")
            else:
                location_parts.append(f"P_idx={p_idx}")

            if T_range is not None:
                location_parts.append(f"T_idx={t_idx} (T={T_range[t_idx].item():.2f}K)")
            else:
                location_parts.append(f"T_idx={t_idx}")

            error_parts.append(f"at {', '.join(location_parts)}")
            error_parts.append(f"calculated={calculated[p_idx, t_idx].item():.6e}")
            error_parts.append(f"expected={expected[p_idx, t_idx].item():.6e}")

        else:
            # Generic case for higher dimensions
            error_parts = [
                f"Max relative error: {max_rel_error:.6e}",
                f"at index {max_error_idx}",
                f"calculated={calculated[max_error_idx].item():.6e}",
                f"expected={expected[max_error_idx].item():.6e}",
            ]

        error_msg = f"{test_name}: " + " | ".join(error_parts)
        test_case.fail(error_msg)

    # Final assertion
    test_case.assertTrue(is_close, f"{test_name}: Values not within tolerance")
