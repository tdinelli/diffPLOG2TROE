"""
Copyright (c) 2025 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

from typing import Dict, Tuple, TypeAlias, Union

from jaxtyping import Array, Float64, Int32


# ============================================================================
# Basic number types
# Real: TypeAlias = Union[float, Float64[Array, ""]] Maybe this is more correct dunno yet
Real: TypeAlias = Float64[Array, ""]
Integer: TypeAlias = Int32[Array, ""]

# ============================================================================
# Basic array types
Vector: TypeAlias = Float64[Array, "..."]
Matrix: TypeAlias = Float64[Array, "rows cols"]

# ============================================================================
# Custom common types
Either: TypeAlias = Union[Real, Vector]
ArrayLike: TypeAlias = Union[Real, Vector, Matrix]

# ============================================================================
# Custom parameters types
ParamsDict: TypeAlias = Dict[str, float]
PlogParamsDict: TypeAlias = Dict[float, ParamsDict]
TupleOfFloat: TypeAlias = Tuple[Real, Real]
