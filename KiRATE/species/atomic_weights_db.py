"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import jax.numpy as jnp
from jaxtyping import Array, Float64

# Element symbols in order (index 0 is placeholder 'E')
ELEMENT_SYMBOLS = [
    "E",
    "H",
    "HE",
    "LI",
    "BE",
    "B",
    "C",
    "N",
    "O",
    "F",
    "NE",
    "NA",
    "MG",
    "AL",
    "SI",
    "P",
    "S",
    "CL",
    "AR",
    "K",
    "CA",
    "SC",
    "TI",
    "V",
    "CR",
    "MN",
    "FE",
    "CO",
    "NI",
    "CU",
    "ZN",
    "GA",
    "GE",
    "AS",
    "SE",
    "BR",
    "KR",
    "RB",
    "SR",
    "Y",
    "ZR",
    "NB",
    "MO",
    "TC",
    "RU",
    "RH",
    "PD",
    "AG",
    "CD",
    "IN",
    "SN",
    "SB",
    "TE",
    "I",
    "XE",
    "CS",
    "BA",
    "LA",
    "CE",
    "PR",
    "ND",
    "PM",
    "SM",
    "EU",
    "GD",
    "TB",
    "DY",
    "HO",
    "ER",
    "TM",
    "YB",
    "LU",
    "HF",
    "TA",
    "W",
    "RE",
    "OS",
    "IR",
    "PT",
    "AU",
    "HG",
    "TL",
    "PB",
    "BI",
    "PO",
    "AT",
    "RN",
    "FR",
    "RA",
    "AC",
    "TH",
    "PA",
    "U",
    "NP",
    "PU",
    "AM",
    "CM",
    "BK",
    "CF",
    "ES",
    "FM",
    "MD",
    "NO",
    "LR",
    "D",
    "e",
]

# Atomic weights array (g/mol) - corresponds to ELEMENT_SYMBOLS indices
ATOMIC_WEIGHTS_ARRAY = jnp.array(
    [
        0.0,  # E  (placeholder)
        1.008000016212463,  # H
        4.002999782562256,  # He
        6.93900,  # Li
        9.01220,  # Be
        10.81100,  # B
        12.010999679565430,  # C
        14.0069999694824,  # N
        15.998999595642090,  # O
        18.99840,  # F
        20.18300,  # Ne
        22.98980,  # Na
        24.31200,  # Mg
        26.98150,  # Al
        28.08600,  # Si
        30.97380,  # P
        32.06,  # S
        35.45300,  # Cl
        39.948001861572270,  # Ar
        39.10200,  # K
        40.08000,  # Ca
        44.95600,  # Sc
        47.90000,  # Ti
        50.94200,  # V
        51.99600,  # Cr
        54.93800,  # Mn
        55.84700,  # Fe
        58.93320,  # Co
        58.71000,  # Ni
        63.54000,  # Cu
        65.37000,  # Zn
        69.72000,  # Ga
        72.59000,  # Ge
        74.92160,  # As
        78.96000,  # Se
        79.90090,  # Br
        83.80000,  # Kr
        85.47000,  # Rb
        87.62000,  # Sr
        88.90500,  # Y
        91.22000,  # Zr
        92.90600,  # Nb
        95.94000,  # Mo
        99.00000,  # Tc
        101.07000,  # Ru
        102.90500,  # Rh
        106.40000,  # Pd
        107.87000,  # Ag
        112.40000,  # Cd
        114.82000,  # In
        118.69000,  # Sn
        121.75000,  # Sb
        127.60000,  # Te
        126.90440,  # I
        131.30000,  # Xe
        132.90500,  # Cs
        137.34000,  # Ba
        138.91000,  # La
        140.12000,  # Ce
        140.90700,  # Pr
        144.24000,  # Nd
        145.00000,  # Pm
        150.35000,  # Sm
        151.96000,  # Eu
        157.25000,  # Gd
        158.92400,  # Tb
        162.50000,  # Dy
        164.93000,  # Ho
        167.26000,  # Er
        168.93400,  # Tm
        173.04000,  # Yb
        174.99700,  # Lu
        178.49000,  # Hf
        180.94800,  # Ta
        183.85000,  # W
        186.20000,  # Re
        190.20000,  # Os
        192.20000,  # Ir
        195.09000,  # Pt
        196.96700,  # Au
        200.59000,  # Hg
        204.37000,  # Tl
        207.19000,  # Pb
        208.98000,  # Bi
        210.00000,  # Po
        210.00000,  # At
        222.00000,  # Rn
        223.00000,  # Fr
        226.00000,  # Ra
        227.00000,  # Ac
        232.03800,  # Th
        231.00000,  # Pa
        238.03000,  # U
        237.00000,  # Np
        242.00000,  # Pu
        243.00000,  # Am
        247.00000,  # Cm
        249.00000,  # Bk
        251.00000,  # Cf
        254.00000,  # Es
        253.00000,  # Fm
        0.0,  # Md
        0.0,  # No
        0.0,  # Lr
        2.01410,  # D (Deuterium)
        5.45e-4,  # e (Electron)
    ],
    dtype=jnp.float64,
)

# Dictionary for convenient lookup by element symbol
ATOMIC_WEIGHTS: dict[str, float] = dict(zip(ELEMENT_SYMBOLS, ATOMIC_WEIGHTS_ARRAY))


def get_atomic_weight(element: str) -> Float64[Array, ""]:
    """
    Get atomic weight for a given element symbol.

    Args:
        element: Element symbol (case-insensitive)

    Returns:
        Atomic weight in g/mol

    Raises:
        KeyError: If element is not found in database
    """
    element_upper = element.upper()
    if element_upper not in ATOMIC_WEIGHTS:
        raise KeyError(f"Element '{element}' not found in atomic weights database")
    return jnp.float64(ATOMIC_WEIGHTS[element_upper])


def get_molecular_weight(composition: dict[str, int]) -> Float64[Array, ""]:
    """
    TODO: Update the doc here
    Calculate molecular weight from elemental composition.

    Args:
        composition: Dictionary mapping element symbols to counts

    Returns:
        Molecular weight in g/mol
    """
    mw = jnp.float64(0.0)
    for element, count in composition.items():
        mw += get_atomic_weight(element) * count
    return mw
