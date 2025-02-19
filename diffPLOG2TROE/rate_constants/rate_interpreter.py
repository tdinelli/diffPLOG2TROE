from enum import IntEnum
from typing import Dict, List, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array, Float64


class FittingType(IntEnum):
    lindemann = 0
    troe = 1
    sri = 2


def parse_rate_constant(rate_constant: Dict):
    reaction_type = rate_constant["type"]

    parsers = {
        "arrhenius": lambda x: parse_arrhenius_parameters(x["rate-constant"]["coefficients"]),
        "plog": lambda x: parse_plog(x["rate-constant"]["coefficients"]),
        "falloff": parse_falloff,
    }

    unsupported_types = {
        "cabr": "CABR",
        "3body": "3BODY",
        "RPBR": "Reduced Pressure Based",
        "Extended-PLOG": "Extended PLOG",
        "Extended-FallOff": "Extended FallOff",
        "Chebyshev": "Chebyshev",
    }

    if reaction_type in unsupported_types:
        raise ValueError(f"{unsupported_types[reaction_type]} reaction type not supported yet!")

    if reaction_type in parsers:
        return parsers[reaction_type](rate_constant)

    raise ValueError(f"Unsupported reaction type: {reaction_type}")


def parse_arrhenius_parameters(parameters: List[Float64]) -> Array:
    if len(parameters) != 3:
        raise ValueError(f"Arrhenius reactions must have three parameters [A, n, E], got {len(parameters)} parameters")
    return jnp.array(parameters, dtype=jnp.float64)


def parse_plog(parameters: List[List[Float64]]) -> Tuple[Array, Array]:
    pressure_levels = []
    rate_constants = []
    for p_level in parameters:
        if len(p_level) != 4:
            raise ValueError(
                "Plog definition require four parameters [P, A, n, E], got {len(parameters)} parameters at level {i+1}"
            )
        pressure_levels.append(p_level[0])
        rate_constants.append(p_level[1:])
    return (jnp.array(pressure_levels, dtype=jnp.float64), jnp.array(rate_constants, dtype=jnp.float64))


def parse_falloff(rate_constant: Dict) -> Union[Tuple[Array, Array, int], Tuple[Array, Array, Array, int]]:
    k_lpl = parse_arrhenius_parameters(rate_constant["rate-constant"]["lpl-coefficients"])
    k_hpl = parse_arrhenius_parameters(rate_constant["rate-constant"]["hpl-coefficients"])

    if rate_constant["falloff-type"] == "lindemann":
        return (k_hpl, k_lpl, FittingType.lindemann.value)

    params = rate_constant["rate-constant"]["falloff-coefficients"]

    valid_lengths = {"troe": {3, 4}, "sri": {3, 4, 5}}
    if len(params) not in valid_lengths[rate_constant["falloff-type"]]:
        raise ValueError(
            "{} formalism requires {} parameters, got {}".format(
                rate_constant["falloff-type"], valid_lengths[rate_constant["falloff-type"]], len(params)
            )
        )
    params = jnp.pad(jnp.array(params), (0, 5 - len(params)), constant_values=0.0)

    type_value = FittingType[rate_constant["falloff-type"]].value

    return (k_hpl, k_lpl, params, type_value)


# def parse_cabr(rate_constant: Dict) -> Tuple[Array, int]:
#     k_lpl, k_hpl = parse_arrhenius_parameters(rate_constant["parameters"])
#     fitting_type = rate_constant["fitting_type"]
#
#     if fitting_type == "Lindemann":
#         return (
#             create_falloff_array(k_hpl, k_lpl, swap_order=True),
#             FittingType.lindemann.value,
#         )
#
#     params, type_value = parse_fitting_params(rate_constant, fitting_type)
#     return create_falloff_array(k_hpl, k_lpl, params, swap_order=True), type_value
#
#
# def parse_threebody(rate_constant: Dict) -> Array:
#     return jnp.array(rate_constant["parameters"], dtype=jnp.float64)
