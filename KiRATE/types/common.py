"""
Copyright (c) 2025 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

from typing import Dict, Tuple, TypeAlias, Union

import numpy as np
from jaxtyping import Array, Float64, Int64


Number = Union[
    np.number,  # NumPy scalar types
    float,  # Python scalar types
]

ArrayLike = Union[
    Array,  # JAX array type
    np.ndarray,  # NumPy array type
    Number,  # valid scalars
]


# ============================================================================
# Basic number types. We are strictly enforcing double precision here maybe in
# the future this will change
Scalar: TypeAlias = Float64[ArrayLike, ""]
Integer: TypeAlias = Int64[ArrayLike, ""]

# ============================================================================
# Basic array types
Vector: TypeAlias = Float64[ArrayLike, "..."]
Matrix: TypeAlias = Float64[ArrayLike, "rows cols"]

# ============================================================================
# Custom common types
ScalarOrVector: TypeAlias = Union[Scalar, Vector]
RateType: TypeAlias = Union[Scalar, Vector, Matrix]
ParamsDict: TypeAlias = Dict[str, Scalar]

# ============================================================================
# Custom parameters types
PlogParamsDict: TypeAlias = Dict[Scalar, ParamsDict]
TupleOfFloat: TypeAlias = Tuple[Scalar, Scalar]
