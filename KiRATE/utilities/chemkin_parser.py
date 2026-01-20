"""
Copyright (c) 2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import re


def check_reaction_name(reaction_name: str, m_is_allowed: bool = False) -> None:
    """
    This function controls that a given reaction in the CHEMKIN standard has a valid name
    in the sense that if the reaction type is not of a threebody or a falloff or a cabr
    it should not contain any species named M or (+M)
    """
    to_be_controlled = ["M", "+M", "(+M)"]

    if not m_is_allowed:
        for species in to_be_controlled:
            if species in reaction_name:
                raise ValueError(
                    f"Invalid reaction name: '{reaction_name}' contains '{species}' but "
                    "this is only allowed for threebody, falloff, CABR or Mixture Ruled "
                    "like reactions"
                )


def parse_reaction_line(input_string: str, m_is_allowed: bool = False) -> tuple[str, dict[str, float]]:
    """
    Parse a CHEMKIN-formatted string into reaction name and parameters.

    This static method extracts reaction names and Arrhenius parameters from CHEMKIN-style
    input strings. It handles various formatting conventions and whitespace variations
    commonly found in kinetics databases.

    Parameters
    ----------
    input_string : str
        CHEMKIN-formatted string containing reaction and parameters. Expected format:
        "reaction_name A n Ea"

        The reaction name can contain various operators and spacing:
        - Equality operators: =, =>, <=>
        - Species separation: +

    Returns
    -------
    Tuple[str, Dict[str, float]]
        A tuple containing:
        - reaction_name (str): Normalized reaction name
        - parameters (Dict[str, float]): Dictionary with keys "A", "n", "Ea"

    Raises
    ------
    ValueError
        If the input string is empty, malformed, or contains insufficient data.
        Specifically raised when:
        - Empty or whitespace-only input
        - Fewer than 4 parts (name + 3 parameters) found
        - Numerical parameters cannot be parsed as floats

        If the reaction name is not conforming to the CHEMKIN standard
    """
    # Strip whitespace and split into lines
    lines = [line.strip() for line in input_string.strip().split("\n") if line.strip()]

    if not lines:
        raise ValueError("Empty chemkin representation")

    # Take the first non-empty line
    line = lines[0]

    # Remove comments (everything after '!')
    if "!" in line:
        line = line.split("!")[0].strip()

    # Split by whitespace to get all parts
    parts = line.split()

    if len(parts) < 4:
        raise ValueError("Not enough parts found. Expected reaction name and 3 parameters.")

    # The last 3 parts should be the numerical parameters
    try:
        A = float(parts[-3])
        n = float(parts[-2])
        Ea = float(parts[-1])
    except ValueError:
        raise ValueError("Could not parse numerical parameters")

    # Everything except the last 3 parts is the reaction name
    reaction_parts = parts[:-3]
    reaction_name = " ".join(reaction_parts)

    # Clean up the reaction name by normalizing whitespace around operators
    # Handle =, =>, <=> operators - preserve them as-is
    reaction_name = re.sub(r"\s*(<=>|=>|=)\s*", r"\1", reaction_name)

    # Normalize spaces around + signs
    reaction_name = re.sub(r"\s*\+\s*", "+", reaction_name)

    # Validate reaction name
    check_reaction_name(reaction_name, m_is_allowed)

    return (reaction_name, {"A": A, "n": n, "Ea": Ea})


def parse_plog(
    input_string: str,
) -> (
    tuple[str, dict[float, dict[str, float]]] | tuple[str, dict[float, dict[str, float]], dict[float, dict[str, float]]]
):
    """
    Parse a CHEMKIN-format pressure-dependent logarithmic (PLOG) reaction.

    PLOG reactions define Arrhenius parameters that vary with pressure. Each PLOG entry
    specifies an Arrhenius expression valid at a specific pressure.

    Parameters
    ----------
    input_string : str
        CHEMKIN-formatted PLOG reaction string containing:

        - Line 1: Reaction equation with placeholder Arrhenius parameters
        - Lines 2+: PLOG entries in format ``PLOG / P A n Ea /``

        May contain duplicate pressure entries (indicated by DUPLICATE/DUP keyword).

    Returns
    -------
    reaction_name : str
        Chemical equation (e.g., "HOCO=OH+CO")
    plog_coefficients_1 : dict[float, dict[str, float]]
        Primary PLOG coefficients mapping pressure [atm] → {"A": ..., "n": ..., "Ea": ...}
    plog_coefficients_2 : dict[float, dict[str, float]] or None
        Secondary PLOG coefficients for duplicate reactions (only returned if duplicates
        exist). Structure matches ``plog_coefficients_1``.

    Raises
    ------
    ValueError
        If input is empty, first line missing, or PLOG line format is invalid.

    Notes
    -----
    **CHEMKIN PLOG Format:**

    The first line contains the reaction equation and placeholder Arrhenius parameters
    that are **ignored** (actual parameters come from PLOG entries):

    .. code-block:: text

        REACTION_EQUATION    A_placeholder    n_placeholder    Ea_placeholder
        PLOG / P1  A1  n1  Ea1 /
        PLOG / P2  A2  n2  Ea2 /
        ...

    Where:

    - P: Pressure [atm]
    - A: Pre-exponential factor [units vary with reaction order]
    - n: Temperature exponent [dimensionless]
    - Ea: Activation energy [cal/mol]

    **Duplicate Reactions:**
    TODO the explanation here
    """
    # Split input into lines and validate
    lines = input_string.strip().split("\n")
    if not lines:
        raise ValueError("Empty CHEMKIN PLOG representation")

    # Parse reaction name from first line
    # The first line contains the reaction equation and placeholder Arrhenius
    # parameters.
    # NOTE: The Arrhenius parameters on this line are IGNORED - actual
    #       parameters come from the PLOG entries below.
    main_line = lines[0].strip()
    if not main_line:
        raise ValueError("First line must contain reaction equation")

    # Extract reaction name
    reaction_name, _ = parse_reaction_line(main_line)

    # Compile regex for extracting numeric values
    # Matches: integers, floats, scientific notation (e.g., 1.5e-3, -2.4E+10)
    number_pattern = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

    # Parse PLOG entries here I am basically allowing only two duplication in
    # principle there should be more than 2??? Not sure about that :)
    plog_coefficients_1 = {}  # Primary reaction pathway: {pressure: {"A": ..., "n": ..., "Ea": ...}}
    plog_coefficients_2 = {}  # Secondary reaction for duplicates: {pressure: {"A": ..., "n": ..., "Ea": ...}}
    has_duplicates = False

    for line in lines[1:]:
        line = line.strip()

        # Remove comments (everything after '!')
        if "!" in line:
            line = line.split("!")[0].strip()

        # Skip empty lines and DUPLICATE markers these may come from the fact that people
        # actually copy and paste stuff from files
        if not line or "DUP" in line.upper() or "DUPLICATE" in line.upper():
            continue

        # Extract PLOG data from between slashes: PLOG / data /
        tokens = line.split("/")
        if len(tokens) < 2:
            raise ValueError(f"Invalid PLOG line format (expected '/ data /'): {line}")

        # tokens[0] = "PLOG ", tokens[1] = " P A n Ea ", tokens[2] = "" or comment
        plog_data_str = tokens[1].strip()

        # Parse numeric coefficients: [P, A, n, Ea]
        try:
            plog_coefficients = [float(x) for x in number_pattern.findall(plog_data_str)]
        except ValueError as e:
            raise ValueError(f"Error parsing numeric values in PLOG line: {line}") from e

        if len(plog_coefficients) != 4:
            raise ValueError(f"Expected 4 values (P, A, n, Ea) in PLOG line, got {len(plog_coefficients)}: {line}")

        pressure = plog_coefficients[0]  # Pressure [atm]
        arrhenius_params = {"A": plog_coefficients[1], "n": plog_coefficients[2], "Ea": plog_coefficients[3]}

        # Handle duplicate pressure entries
        if pressure in plog_coefficients_1:
            # This pressure already exists, so we have a duplicate reaction
            has_duplicates = True

            if pressure not in plog_coefficients_2:
                # First duplicate: store in secondary dictionary
                plog_coefficients_2[pressure] = arrhenius_params
            else:
                # Multiple duplicates (>2 pathways): not supported
                raise ValueError(f"More than 2 duplicate reactions at pressure {pressure} - not supported")
        else:
            # New pressure: store in primary dictionary
            plog_coefficients_1[pressure] = arrhenius_params

    if has_duplicates:
        return reaction_name, plog_coefficients_1, plog_coefficients_2
    else:
        return reaction_name, plog_coefficients_1


def parse_falloff(
    input_string: str,
) -> tuple[str, dict[str, float], dict[str, float], str, dict[str, float] | None, dict[str, float] | None]:
    """
    Parse a CHEMKIN-format falloff reaction.

    Parameters
    ----------
    input_string : str
        CHEMKIN-formatted falloff reaction string

    Returns
    -------
    tuple
        (reaction_name, hpl_params, lpl_params, falloff_type, falloff_params, efficiencies)
    """
    lines = input_string.strip().split("\n")
    if not lines:
        raise ValueError("Empty CHEMKIN FallOff representation")

    main_line = lines[0].strip()
    if not main_line:
        raise ValueError("First line must contain reaction equation")

    # Extract reaction name and high-pressure limit parameters
    reaction_name, hpl_coefficients = parse_reaction_line(main_line, True)

    number_pattern = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

    # Initialize variables
    lpl_coefficients = None
    falloff_type = "lindemann"  # Default falloff type
    falloff_params = None
    efficiencies = None

    for line in lines[1:]:
        line = line.strip()

        # Remove comments (everything after '!')
        if "!" in line:
            line = line.split("!")[0].strip()

        # Skip empty lines
        if not line:
            continue

        if "LOW" in line:
            # Parse LOW pressure limit coefficients
            tokens = line.split("/")
            if len(tokens) < 2:
                raise ValueError(f"Invalid LOW line format (expected '/ data /'): {line}")

            low_data_str = tokens[1].strip()

            try:
                low_coefficients = [float(x) for x in number_pattern.findall(low_data_str)]
            except ValueError as e:
                raise ValueError(f"Error parsing numeric values in LOW line: {line}") from e

            if len(low_coefficients) != 3:
                raise ValueError(f"Expected 3 values (A, n, Ea) in LOW line, got {len(low_coefficients)}: {line}")

            lpl_coefficients = {"A": low_coefficients[0], "n": low_coefficients[1], "Ea": low_coefficients[2]}

        elif "TROE" in line:
            # Parse TROE parameters
            falloff_type = "troe"
            tokens = line.split("/")
            if len(tokens) < 2:
                raise ValueError(f"Invalid TROE line format (expected '/ data /'): {line}")

            troe_data_str = tokens[1].strip()

            try:
                troe_values = [float(x) for x in number_pattern.findall(troe_data_str)]
            except ValueError as e:
                raise ValueError(f"Error parsing numeric values in TROE line: {line}") from e

            if len(troe_values) not in [3, 4]:
                raise ValueError(f"Expected 3 or 4 values in TROE line, got {len(troe_values)}: {line}")

            # TROE format: A T3 T1 [T2]
            falloff_params = {
                "A": troe_values[0],
                "T3": troe_values[1],
                "T1": troe_values[2],
                "T2": troe_values[3] if len(troe_values) == 4 else 0.0,
            }

        elif "SRI" in line:
            # Parse SRI parameters
            falloff_type = "sri"
            tokens = line.split("/")
            if len(tokens) < 2:
                raise ValueError(f"Invalid SRI line format (expected '/ data /'): {line}")

            sri_data_str = tokens[1].strip()

            try:
                sri_values = [float(x) for x in number_pattern.findall(sri_data_str)]
            except ValueError as e:
                raise ValueError(f"Error parsing numeric values in SRI line: {line}") from e

            if len(sri_values) not in [3, 5]:
                raise ValueError(f"Expected 3 or 5 values in SRI line, got {len(sri_values)}: {line}")

            # SRI format: a b c [d e]
            falloff_params = {
                "a": sri_values[0],
                "b": sri_values[1],
                "c": sri_values[2],
                "d": sri_values[3] if len(sri_values) == 5 else 1.0,
                "e": sri_values[4] if len(sri_values) == 5 else 0.0,
            }

        else:
            # Parse collision efficiencies (e.g., H2O/12.0/ H2/2.0/)
            # Check if line contains efficiency specifications
            efficiency_pattern = re.compile(r"(\w+)\s*/\s*([\d.eE+-]+)\s*/")
            matches = efficiency_pattern.findall(line)
            if matches:
                if efficiencies is None:
                    efficiencies = {}
                for species, efficiency in matches:
                    efficiencies[species] = float(efficiency)

    # Validate that LOW was provided
    if lpl_coefficients is None:
        raise ValueError("FallOff reaction must contain LOW pressure limit parameters")

    return reaction_name, hpl_coefficients, lpl_coefficients, falloff_type, falloff_params, efficiencies


def parse_threebody(input_string: str) -> tuple[str, dict[str, float], dict[str, float] | None]:
    """
    Parse a CHEMKIN-format three-body (termolecular) reaction.

    This function extracts the reaction name, third-order rate constant parameters,
    and optional collision efficiency factors from a CHEMKIN-formatted string.

    Parameters
    ----------
    input_string : str
        CHEMKIN-formatted three-body reaction string with the following format:

        .. code-block:: text

            REACTION_NAME    A   n   Ea
            SPECIES / efficiency / ... / ...      / ! optional

        Example:
            H+OH+M=H2O+M  2.2E+22  -2.0  0.0
            H2O/6.0/ AR/0.38/

        Where:
            - REACTION_NAME contains the species and '+M' indicator
            - A, n, Ea are the modified Arrhenius parameters for k₀(T)
            - efficiency lines specify collision partner efficiencies (optional)
            - Species not listed default to efficiency = 1.0

    Returns
    -------
    tuple[str, dict[str, float], dict[str, float] | None]
        A 3-tuple containing:
            - reaction_name (str): Normalized reaction equation (e.g., "H+OH+M=H2O+M")
            - k0_params (dict[str, float]): Third-order rate constant parameters
              {"A": pre-exponential, "n": temperature exponent, "Ea": activation energy}
            - efficiencies (dict[str, float] | None): Collision efficiency factors
              Maps species names to dimensionless efficiency values, or None if not specified

    Raises
    ------
    ValueError
        If the input string is empty, malformed, or missing required parameters
    """
    lines = input_string.strip().split("\n")
    if not lines:
        raise ValueError("Empty CHEMKIN threebody representation")

    main_line = lines[0].strip()
    if not main_line:
        raise ValueError("First line must contain reaction equation")

    # Extract reaction name and third-order rate constant parameters
    reaction_name, k0_coefficients = parse_reaction_line(main_line, True)

    # Initialize variables
    efficiencies = None
    efficiency_pattern = re.compile(r"(\w+)\s*/\s*([\d.eE+-]+)\s*/")
    for line in lines[1:]:
        line = line.strip()

        # Remove comments (everything after '!')
        if "!" in line:
            line = line.split("!")[0].strip()

        # Skip empty lines
        if not line:
            continue

        # Parse collision efficiencies (e.g., H2O/12.0/ H2/2.0/)
        # Check if line contains efficiency specifications
        matches = efficiency_pattern.findall(line)
        if matches:
            if efficiencies is None:
                efficiencies = {}
            for species, efficiency in matches:
                efficiencies[species] = float(efficiency)

    return reaction_name, k0_coefficients, efficiencies
