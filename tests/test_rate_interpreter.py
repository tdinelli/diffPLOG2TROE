import sys


sys.path.append("/Users/tdinelli/Documents/GitHub/diffPLOG2TROE")
import jax.numpy as jnp

from diffPLOG2TROE.rate_interpreter import parse_rate_constant


def test_parse_plog():
    jax_plog_constant = jnp.array(
        [
            [1.000000e-03, 4.758000e18, -3.817000e00, 1.767600e04],
            [3.000000e-03, 2.225000e20, -4.149000e00, 1.903700e04],
            [9.900000e-03, 7.564000e21, -4.434000e00, 2.032500e04],
            [2.960000e-02, 9.107000e24, -5.189000e00, 2.241900e04],
            [9.870000e-02, 3.144000e29, -6.376000e00, 2.523300e04],
            [2.961000e-01, 1.150000e32, -7.037000e00, 2.666200e04],
            [9.869000e-01, 1.069000e36, -8.107000e00, 2.906400e04],
            [2.960700e00, 2.438000e36, -8.153000e00, 2.933600e04],
            [9.869000e00, 6.663000e35, -7.919000e00, 2.921700e04],
            [2.960700e01, 1.723000e38, -8.506000e00, 3.127300e04],
            [9.869000e01, 3.007000e41, -9.290000e00, 3.396600e04],
            [2.960700e02, 6.767000e36, -7.832000e00, 3.161300e04],
            [9.869000e02, 1.897000e38, -8.047000e00, 3.424000e04],
        ],
        dtype=jnp.float64,
    )

    rate_constant = {
        "name": "HOCO=H+CO2",
        "type": "PLOG",
        "parameters": [
            [1.000000e-03, 4.758000e18, -3.817000e00, 1.767600e04],
            [3.000000e-03, 2.225000e20, -4.149000e00, 1.903700e04],
            [9.900000e-03, 7.564000e21, -4.434000e00, 2.032500e04],
            [2.960000e-02, 9.107000e24, -5.189000e00, 2.241900e04],
            [9.870000e-02, 3.144000e29, -6.376000e00, 2.523300e04],
            [2.961000e-01, 1.150000e32, -7.037000e00, 2.666200e04],
            [9.869000e-01, 1.069000e36, -8.107000e00, 2.906400e04],
            [2.960700e00, 2.438000e36, -8.153000e00, 2.933600e04],
            [9.869000e00, 6.663000e35, -7.919000e00, 2.921700e04],
            [2.960700e01, 1.723000e38, -8.506000e00, 3.127300e04],
            [9.869000e01, 3.007000e41, -9.290000e00, 3.396600e04],
            [2.960700e02, 6.767000e36, -7.832000e00, 3.161300e04],
            [9.869000e02, 1.897000e38, -8.047000e00, 3.424000e04],
        ],
    }

    assert jnp.array_equal(jax_plog_constant, parse_rate_constant(rate_constant))


def test_parse_falloff():
    jax_falloff_troe = (
        jnp.array(
            [
                [6.220e16, -1.174, 635.80, 0.000],  # HPL
                [1.135e36, -5.246, 1704.8, 0.000],  # LPL
                [0.405, 1120, 69.6, 0.0],
            ],
            dtype=jnp.float64,
        ),
        1,  # Lindemann -> 0, TROE -> 1, SRI -> 2
    )

    rate_constant = {
        "name": "CH3+CH3(+M)=C2H6(+M)",
        "type": "FallOff",
        "is_explicitly_enhanced": False,
        "parameters": {
            "k0": [1.135e36, -5.246, 1704.8],
            "kInf": [6.220e16, -1.174, 635.80],
        },
        "fitting_type": "TROE",
        "fitting_parameters": [0.405, 1120, 69.6, 0.0],
    }

    assert parse_rate_constant(rate_constant)[1] == jax_falloff_troe[1]
    assert jnp.array_equal(parse_rate_constant(rate_constant)[0], jax_falloff_troe[0])


def test_parse_cabr():
    jax_cabr_sri = (
        jnp.array(
            [
                [4.9890e12, 9.9000e-02, 1.060e04, 0.0, 0.0],
                [3.8000e-07, 4.8380e00, 7.710e03, 0.0, 0.0],
                [1.641, 4334, 2725, 1.0, 0.0],
            ],
            dtype=jnp.float64,
        ),
        2,  # Lindemann -> 0, TROE -> 1, SRI -> 2
    )

    rate_constant = {
        "name": "CH3+CH3(+M)=C2H5+H(+M)",
        "type": "CABR",
        "is_explicitly_enhanced": False,
        "parameters": {
            "k0": [4.9890e12, 9.9000e-02, 1.060e04],
            "kInf": [3.8000e-07, 4.8380e00, 7.710e03],
        },
        "fitting_type": "SRI",
        "fitting_parameters": [1.641, 4334, 2725, 1.0, 0.0],
    }

    assert parse_rate_constant(rate_constant)[1] == jax_cabr_sri[1]
    assert jnp.array_equal(parse_rate_constant(rate_constant)[0], jax_cabr_sri[0])
