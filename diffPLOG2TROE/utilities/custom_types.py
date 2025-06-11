"""
Type aliases for chemical kinetics modeling.

This module defines commonly used type aliases for arrays and reaction rate
parametrizations used throughout the chemical kinetics library.

Array Types:
    - Float64Vector: 1D array of Float64 values
    - Float64Matrix: 2D array of Float64 values
    - Float64Tensor3D: 3D array of Float64 values
    - ScalarOrVector: Either a scalar Float64 or 1D Float64 array

Rate Parametrization Types:
    - AnyRate: Union of all supported rate parametrization classes
"""

from typing import TypeAlias, Union, Dict

from jaxtyping import Array, Float64


# ============================================================================
# Array Type Aliases
# ============================================================================
#: 1D array of Float64 values (e.g., temperature or pressure arrays)
Array64f: TypeAlias = Float64[Array, "dim"]
Array64f_3: TypeAlias = Float64[Array, "3"]
Array64f_5: TypeAlias = Float64[Array, "5"]

# ============================================================================
#: 2D array of Float64 values (e.g., rate coefficient matrices)
Matrix64f: TypeAlias = Float64[Array, "rows cols"]

# ============================================================================
#: 3D array of Float64 values (e.g., batched matrices for multiple conditions)
Tensor64f: TypeAlias = Float64[Array, "batch height width"]

# ============================================================================
#: Either a scalar Float64 or 1D Float64 array - useful for flexible function inputs
ScalarOrVector: TypeAlias = Union[Float64, Float64[Array, "dim"]]
