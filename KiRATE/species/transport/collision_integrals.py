"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details

Collision integral evaluation for gas-phase transport properties.

Implements Omega^(1,1) and Omega^(2,2) collision integrals using 2D quadratic
(Newton forward) interpolation on tabulated data from:

    TODO: Add references

The tables are the same as those used in CHEMKIN and OpenSMOKE++.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float64

# Reduced dipole moment grid — 8 points
_DELTA_STAR: Float64[Array, "8"] = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5], dtype=jnp.float64)

# Reduced temperature grid — 37 points (non-uniform spacing)
_T_STAR: Float64[Array, "37"] = jnp.array(
    [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.2,
        1.4,
        1.6,
        1.8,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        12.0,
        14.0,
        16.0,
        18.0,
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
        50.0,
        75.0,
        100.0,
    ],
    dtype=jnp.float64,
)

# Omega^(1,1) table — shape (37, 8)
# Rows: T* grid points, Columns: delta* grid points
# Used for: mass diffusivity Djk
_O37_8: Float64[Array, "37 8"] = jnp.array(
    [
        #  d*=0.00   d*=0.25   d*=0.50   d*=0.75   d*=1.00   d*=1.50   d*=2.00   d*=2.50
        [4.008, 4.002, 4.655, 5.520, 6.454, 8.214, 9.824, 11.310],  # T*=0.1
        [3.130, 3.164, 3.355, 3.721, 4.198, 5.230, 6.225, 7.160],  # T*=0.2
        [2.649, 2.657, 2.770, 3.002, 3.319, 4.054, 4.785, 5.483],  # T*=0.3
        [2.314, 2.320, 2.402, 2.572, 2.812, 3.386, 3.972, 4.539],  # T*=0.4
        [2.066, 2.073, 2.140, 2.278, 2.472, 2.946, 3.437, 3.918],  # T*=0.5
        [1.877, 1.885, 1.944, 2.060, 2.225, 2.628, 3.054, 3.747],  # T*=0.6
        [1.729, 1.738, 1.790, 1.893, 2.036, 2.388, 2.763, 3.137],  # T*=0.7
        [1.6122, 1.622, 1.670, 1.760, 1.886, 2.198, 2.535, 2.872],  # T*=0.8
        [1.517, 1.527, 1.572, 1.653, 1.765, 2.044, 2.350, 2.657],  # T*=0.9
        [1.440, 1.450, 1.490, 1.564, 1.665, 1.917, 2.196, 2.4780],  # T*=1.0
        [1.3204, 1.330, 1.364, 1.425, 1.510, 1.720, 1.956, 2.199],  # T*=1.2
        [1.234, 1.240, 1.272, 1.324, 1.394, 1.573, 1.777, 1.990],  # T*=1.4
        [1.168, 1.176, 1.202, 1.246, 1.306, 1.460, 1.640, 1.827],  # T*=1.6
        [1.1166, 1.124, 1.146, 1.185, 1.237, 1.372, 1.530, 1.700],  # T*=1.8
        [1.075, 1.082, 1.102, 1.135, 1.181, 1.300, 1.441, 1.592],  # T*=2.0
        [1.0006, 1.005, 1.020, 1.046, 1.080, 1.170, 1.278, 1.397],  # T*=2.5
        [0.9500, 0.9538, 0.9656, 0.9852, 1.012, 1.082, 1.168, 1.265],  # T*=3.0
        [0.9131, 0.9162, 0.9256, 0.9413, 0.9626, 1.019, 1.090, 1.170],  # T*=3.5
        [0.8845, 0.8871, 0.8948, 0.9076, 0.9252, 0.9720, 1.030, 1.098],  # T*=4.0
        [0.8428, 0.8446, 0.8500, 0.8590, 0.8716, 0.9053, 0.9483, 0.9984],  # T*=5.0
        [0.8130, 0.8142, 0.8183, 0.8250, 0.8344, 0.8598, 0.8927, 0.9316],  # T*=6.0
        [0.7898, 0.7910, 0.7940, 0.7993, 0.8066, 0.8265, 0.8526, 0.8836],  # T*=7.0
        [0.7711, 0.7720, 0.7745, 0.7788, 0.7846, 0.8007, 0.8220, 0.8474],  # T*=8.0
        [0.7555, 0.7562, 0.7584, 0.7619, 0.7667, 0.7800, 0.7976, 0.8189],  # T*=9.0
        [0.7422, 0.7430, 0.7446, 0.7475, 0.7515, 0.7627, 0.7776, 0.7960],  # T*=10.0
        [0.72022, 0.7206, 0.7220, 0.7241, 0.7271, 0.7354, 0.7464, 0.7600],  # T*=12.0
        [0.7025, 0.7030, 0.7040, 0.7055, 0.7078, 0.7142, 0.7228, 0.7334],  # T*=14.0
        [0.68776, 0.6880, 0.6888, 0.6901, 0.6919, 0.6970, 0.7040, 0.7125],  # T*=16.0
        [0.6751, 0.6753, 0.6760, 0.6770, 0.6785, 0.6827, 0.6884, 0.6955],  # T*=18.0
        [0.6640, 0.6642, 0.6648, 0.6657, 0.6669, 0.6704, 0.6752, 0.6810],  # T*=20.0
        [0.6414, 0.6415, 0.6418, 0.6425, 0.6433, 0.6457, 0.6490, 0.6530],  # T*=25.0
        [0.6235, 0.6236, 0.6239, 0.6243, 0.6249, 0.6267, 0.6290, 0.6320],  # T*=30.0
        [0.60882, 0.6089, 0.6091, 0.6094, 0.6100, 0.6112, 0.6130, 0.6154],  # T*=35.0
        [0.5964, 0.5964, 0.5966, 0.5970, 0.5972, 0.5983, 0.6000, 0.6017],  # T*=40.0
        [0.5763, 0.5763, 0.5764, 0.5766, 0.5768, 0.5775, 0.5785, 0.5800],  # T*=50.0
        [0.5415, 0.5415, 0.5416, 0.5416, 0.5418, 0.5420, 0.5424, 0.5430],  # T*=75.0
        [0.5180, 0.5180, 0.5182, 0.5184, 0.5184, 0.5185, 0.5186, 0.5187],  # T*=100.0
    ],
    dtype=jnp.float64,
)

# Omega^(2,2) table — shape (37, 8)
# Rows: T* grid points, Columns: delta* grid points
# Used for: viscosity eta, thermal conductivity lambda
_P37_8: Float64[Array, "37 8"] = jnp.array(
    [
        #  d*=0.00   d*=0.25   d*=0.50   d*=0.75   d*=1.00   d*=1.50   d*=2.00   d*=2.50
        [4.100, 4.266, 4.833, 5.742, 6.729, 8.624, 10.340, 11.890],  # T*=0.1
        [3.263, 3.305, 3.516, 3.914, 4.433, 5.570, 6.637, 7.618],  # T*=0.2
        [2.840, 2.836, 2.936, 3.168, 3.511, 4.329, 5.126, 5.874],  # T*=0.3
        [2.531, 2.522, 2.586, 2.749, 3.004, 3.640, 4.282, 4.895],  # T*=0.4
        [2.284, 2.277, 2.329, 2.460, 2.665, 3.187, 3.727, 4.249],  # T*=0.5
        [2.084, 2.081, 2.130, 2.243, 2.417, 2.862, 3.329, 3.786],  # T*=0.6
        [1.922, 1.924, 1.970, 2.072, 2.225, 2.641, 3.028, 3.435],  # T*=0.7
        [1.7902, 1.795, 1.840, 1.934, 2.070, 2.417, 2.788, 3.156],  # T*=0.8
        [1.682, 1.689, 1.733, 1.820, 1.944, 2.258, 2.596, 2.933],  # T*=0.9
        [1.593, 1.600, 1.644, 1.725, 1.840, 2.124, 2.435, 2.746],  # T*=1.0
        [1.455, 1.465, 1.504, 1.574, 1.670, 1.913, 2.181, 2.450],  # T*=1.2
        [1.355, 1.365, 1.400, 1.461, 1.544, 1.754, 1.989, 2.228],  # T*=1.4
        [1.280, 1.289, 1.321, 1.374, 1.447, 1.630, 1.838, 2.053],  # T*=1.6
        [1.222, 1.231, 1.260, 1.306, 1.370, 1.532, 1.718, 1.912],  # T*=1.8
        [1.176, 1.184, 1.209, 1.250, 1.307, 1.450, 1.618, 1.795],  # T*=2.0
        [1.0933, 1.100, 1.119, 1.150, 1.193, 1.304, 1.435, 1.578],  # T*=2.5
        [1.039, 1.044, 1.060, 1.083, 1.117, 1.204, 1.310, 1.428],  # T*=3.0
        [0.9996, 1.004, 1.016, 1.035, 1.062, 1.133, 1.220, 1.320],  # T*=3.5
        [0.9699, 0.9732, 0.9830, 0.9991, 1.021, 1.080, 1.153, 1.236],  # T*=4.0
        [0.9268, 0.9291, 0.9360, 0.9473, 0.9628, 1.005, 1.058, 1.120],  # T*=5.0
        [0.8962, 0.8979, 0.9030, 0.9114, 0.9230, 0.9545, 0.9955, 1.044],  # T*=6.0
        [0.8727, 0.8741, 0.8780, 0.8845, 0.8935, 0.9180, 0.9505, 0.9893],  # T*=7.0
        [0.8538, 0.8549, 0.8580, 0.8632, 0.8703, 0.8900, 0.9164, 0.9482],  # T*=8.0
        [0.8379, 0.8388, 0.8414, 0.8456, 0.8515, 0.8680, 0.8895, 0.9160],  # T*=9.0
        [0.8243, 0.8251, 0.8273, 0.8308, 0.8356, 0.8493, 0.8676, 0.8900],  # T*=10.0
        [0.8018, 0.8024, 0.8039, 0.8065, 0.8100, 0.8200, 0.8337, 0.8504],  # T*=12.0
        [0.7836, 0.7840, 0.7852, 0.7872, 0.7899, 0.7976, 0.8080, 0.8212],  # T*=14.0
        [0.7683, 0.7687, 0.7696, 0.7710, 0.7733, 0.7794, 0.7880, 0.7983],  # T*=16.0
        [0.7552, 0.7554, 0.7562, 0.7575, 0.7592, 0.7640, 0.7710, 0.7797],  # T*=18.0
        [0.7436, 0.7438, 0.7445, 0.7455, 0.7470, 0.7512, 0.7570, 0.7642],  # T*=20.0
        [0.71982, 0.7200, 0.7204, 0.7211, 0.7221, 0.7250, 0.7289, 0.7339],  # T*=25.0
        [0.7010, 0.7011, 0.7014, 0.7020, 0.7026, 0.7047, 0.7076, 0.7112],  # T*=30.0
        [0.68545, 0.6855, 0.6860, 0.6860, 0.6867, 0.6883, 0.6905, 0.6930],  # T*=35.0
        [0.6723, 0.6724, 0.6726, 0.6730, 0.6733, 0.6745, 0.6760, 0.6784],  # T*=40.0
        [0.6510, 0.6510, 0.6512, 0.6513, 0.6516, 0.6524, 0.6534, 0.6546],  # T*=50.0
        [0.6140, 0.6140, 0.6143, 0.6145, 0.6147, 0.6148, 0.6148, 0.6147],  # T*=75.0
        [0.5887, 0.5889, 0.5894, 0.5900, 0.5903, 0.5901, 0.5895, 0.5885],  # T*=100.0
    ],
    dtype=jnp.float64,
)


def _quadratic_interp(
    x: Float64[Array, ""],
    x1: Float64[Array, ""],
    x2: Float64[Array, ""],
    x3: Float64[Array, ""],
    y1: Float64[Array, ""],
    y2: Float64[Array, ""],
    y3: Float64[Array, ""],
) -> Float64[Array, ""]:
    """
    3-point Newton forward quadratic interpolation.

    Given three nodes (x1,y1), (x2,y2), (x3,y3) and a query point x,
    returns the value of the interpolating polynomial at x.

    This is the same formula used in OpenSMOKE++ CollisionIntegral11/22.

    Parameters
    ----------
    x           : query point
    x1, x2, x3 : node abscissae (must satisfy x1 < x2 < x3)
    y1, y2, y3 : node ordinates

    Returns
    -------
    Float64[Array, ""]
        Interpolated value at x
    """
    a2 = (y2 - y1) * (1 / (x2 - x1))
    a3 = (y3 - y1 - a2 * (x3 - x1)) * (1.0 / ((x3 - x1) * (x3 - x2)))

    return y1 + (x - x1) * (a2 + a3 * (x - x2))


def _locate(sorted_vector: Float64[Array, "n"], value: Float64[Array, ""]) -> Float64[Array, ""]:
    """
    Returns index i such that sorted_vec[i] <= val < sorted_vec[i+1].

    Equivalent to OpenSMOKE's LocateInSortedVector.
    Result is clamped to [0, len-3] so that i, i+1, i+2 are always valid.

    Parameters
    ----------
    sorted_vec : 1-D sorted array
    val        : query value

    Returns
    -------
    int
        Index into sorted_vec
    """
    idx = jnp.searchsorted(sorted_vector, value, side="right") - 1

    return jnp.clip(idx, 0, sorted_vector.shape[0] - 3)


def _collision_integral(
    t_star: Float64[Array, ""],
    d_star: Float64[Array, ""],
    table: Float64[Array, "37 8"],
    fallback_c0: Float64[Array, ""],
    fallback_c1: Float64[Array, ""],
    fallback_c2: Float64[Array, ""],
    fallback_c3: Float64[Array, ""],
) -> Float64[Array, ""]:
    """
    2D quadratic interpolation on a 37×8 collision integral table.

    Three cases (mirroring the OpenSMOKE logic) are all evaluated and selected
    via jnp.where — required for JAX JIT compatibility.

    Case 1: T* > _T_STAR[-1]
        Polynomial fallback: c0 + c1*T* + c2*T*^2 + c3*T*^3

    Case 2: |δ*| <= 1e-5  (nonpolar species)
        1-D quadratic interpolation on the first table column only.

    Case 3: General (polar or intermediate)
        Interpolate in T* for three consecutive δ* columns,
        then interpolate in δ* across those three values.

    Parameters
    ----------
    t_star                    : reduced temperature T* = T / (ε/k_B)
    d_star                    : reduced dipole moment δ*
    table                     : 37×8 collision integral table
    fallback_c0..fallback_c3  : polynomial coefficients for T* > _T_STAR[-1]

    Returns
    -------
    float
        Collision integral value Ω
    """
    # Clamp T* to minimum (mirrors the OpenSMOKEpp warning + reassignment at tjk=0.09)
    t_star = jnp.clip(t_star, 0.09, None)

    # Locate T* and δ* in their respective grids
    _it = _locate(_T_STAR, t_star)  # index into _T_STAR; it, it+1, it+2 valid
    _id = _locate(_DELTA_STAR, d_star)  # index into _DELTA_STAR; id_, id_+1, id_+2 valid

    # For δ* clamp to [0, 5] (mirrors: if djk < delta[2] → idStar=1; if > delta[7] → idStar=6)
    _id = jnp.clip(_id, 0, 5)

    # Node abscissae
    ts1, ts2, ts3 = _T_STAR[_it], _T_STAR[_it + 1], _T_STAR[_it + 2]
    ds1, ds2, ds3 = _DELTA_STAR[_id], _DELTA_STAR[_id + 1], _DELTA_STAR[_id + 2]

    # Case 1 (polynomial): fallback for high T*
    result_fallback = fallback_c0 + t_star * (fallback_c1 + t_star * (fallback_c2 + t_star * fallback_c3))

    # Case 2 (nonpolar): interpolate first column only
    result_1d = _quadratic_interp(t_star, ts1, ts2, ts3, table[_it, 0], table[_it + 1, 0], table[_it + 2, 0])

    # Case 3 (general): interpolate T* for 3 δ* columns, then in δ*
    y1 = _quadratic_interp(t_star, ts1, ts2, ts3, table[_it, _id], table[_it + 1, _id], table[_it + 2, _id])
    y2 = _quadratic_interp(t_star, ts1, ts2, ts3, table[_it, _id + 1], table[_it + 1, _id + 1], table[_it + 2, _id + 1])
    y3 = _quadratic_interp(t_star, ts1, ts2, ts3, table[_it, _id + 2], table[_it + 1, _id + 2], table[_it + 2, _id + 2])
    result_2d = _quadratic_interp(d_star, ds1, ds2, ds3, y1, y2, y3)

    # Select result (all branches evaluated, no Python-level branching)
    result = jnp.where(jnp.abs(d_star) <= 1.0e-5, result_1d, result_2d)
    result = jnp.where(t_star > _T_STAR[-1], result_fallback, result)

    return result


def omega11(t_star: Float64[Array, ""], d_star: Float64[Array, ""]) -> Float64[Array, ""]:
    """
    Collision integral Omega^(1,1)(T*, delta*).

    Used in the computation of binary mass diffusivities Djk.

    Parameters
    ----------
    t_star : Float64[Array, ""]
        Reduced temperature T* = T / (epsilon_jk / k_B)  [-]
    d_star : Float64[Array, ""]
        Reduced dipole moment delta* = 0.5 * mu*^2        [-]

    Returns
    -------
    Float64[Array, ""]
        Omega^(1,1) collision integral  [-]

    Notes
    -----
    For T* > 100 (last tabulated point), falls back to:
        Omega^(1,1) = 0.623 - 0.00136*T* + 3.46e-6*T*^2 - 3.43e-9*T*^3
    """
    return _collision_integral(
        t_star,
        d_star,
        _O37_8,
        fallback_c0=jnp.float64(0.623),
        fallback_c1=jnp.float64(-0.136e-2),
        fallback_c2=jnp.float64(0.346e-5),
        fallback_c3=jnp.float64(-0.343e-8),
    )


def omega22(t_star: Float64[Array, ""], d_star: Float64[Array, ""]) -> Float64[Array, ""]:
    """
    Collision integral Omega^(2,2)(T*, delta*).

    Used in the computation of viscosity eta and thermal conductivity lambda.

    Parameters
    ----------
    t_star : float
        Reduced temperature T* = T / (epsilon / k_B)   [-]
    d_star : float
        Reduced dipole moment delta* = 0.5 * mu*^2     [-]

    Returns
    -------
    float
        Omega^(2,2) collision integral  [-]

    Notes
    -----
    For T* > 100 (last tabulated point), falls back to:
        Omega^(2,2) = 0.703 - 0.00146*T* + 3.57e-6*T*^2 - 3.43e-9*T*^3
    """
    return _collision_integral(
        t_star,
        d_star,
        _P37_8,
        fallback_c0=jnp.float64(0.7030),
        fallback_c1=jnp.float64(-0.146e-2),
        fallback_c2=jnp.float64(0.357e-5),
        fallback_c3=jnp.float64(-0.343e-8),
    )
