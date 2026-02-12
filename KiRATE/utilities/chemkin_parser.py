"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

import re


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
    tuple[str, dict[str, float]]
        A tuple containing:
        - reaction_name (str): Normalized reaction name
        - parameters (dict[str, float]): Dictionary with keys "A", "n", "Ea"

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
) -> tuple[str, dict[float, list[dict[str, float]]]]:
    """
    Parse a CHEMKIN-format pressure-dependent logarithmic (PLOG) reaction.

    PLOG reactions define Arrhenius parameters that vary with pressure. Each PLOG entry
    specifies an Arrhenius expression valid at a specific pressure. Multiple PLOG entries
    at the same pressure are **summed** to capture non-Arrhenius behavior (curved Arrhenius plots).

    Parameters
    ----------
    input_string : str
        CHEMKIN-formatted PLOG reaction string containing:

        - Line 1: Reaction equation with placeholder Arrhenius parameters
        - Lines 2+: PLOG entries in format ``PLOG / P A n Ea /``

        May contain multiple PLOG entries at the same pressure for non-Arrhenius fitting.

    Returns
    -------
    reaction_name : str
        Chemical equation (e.g., "HOCO=OH+CO")
    plog_coefficients : dict[float, list[dict[str, float]]]
        Dictionary mapping pressure [atm] to list of Arrhenius parameter sets.
        Each pressure maps to one or more parameter dictionaries: {"A": ..., "n": ..., "Ea": ...}

        - Single entry per pressure: Standard PLOG interpolation
        - Multiple entries per pressure: Sum of Arrhenius expressions to fit non-Arrhenius behavior

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

    **Multiple Arrhenius Expressions (Non-Arrhenius Behavior):**

    When the same pressure appears multiple times, the rate constant is computed as the
    **sum** of all Arrhenius expressions at that pressure:

    .. math::
        k(T, P) = \\sum_i A_i T^{n_i} \\exp(-E_{a,i} / RT)

    This is used to fit complex, non-Arrhenius temperature dependencies:

    .. code-block:: text

        O+C10H7CH3=CH3C10H6OH    1.0e+17  -1.64  4750.0
        PLOG / 0.01  1.44e+14  -0.93   1700.0 /    ! Term 1
        PLOG / 0.01  4.07e+15  -6.73  -14031.0 /   ! Term 2 (negative Ea)
        PLOG / 0.01  1.07e+35  -6.92   13025.0 /   ! Term 3
        PLOG / 1.0   1.38e+17  -1.64   4750.0 /    ! Term 1
        PLOG / 1.0   6.20e+13  -0.78   3522.0 /    ! Term 2
        ...

    The sum of terms provides flexibility to capture curved Arrhenius plots that arise
    from complex reaction mechanisms or transitions between different rate-limiting steps.
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

    # Parse PLOG entries - group by pathway index
    # Structure: {pressure: [params1, params2, ...]} where each list entry is a pathway
    plog_by_pressure = {}  # {pressure: [{"A": ..., "n": ..., "Ea": ...}, ...]}

    for line in lines[1:]:
        line = line.strip()

        # Remove comments (everything after '!')
        if "!" in line:
            line = line.split("!")[0].strip()

        # Skip empty lines and DUPLICATE markers
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

        # Add to the list of pathways for this pressure
        if pressure not in plog_by_pressure:
            plog_by_pressure[pressure] = []
        plog_by_pressure[pressure].append(arrhenius_params)

    # Return the structure as-is: {pressure: [params1, params2, ...]}
    # Each pressure maps to a list of Arrhenius parameter dictionaries
    # The rate constant at each pressure is the sum of all terms
    return reaction_name, plog_by_pressure


def parse_falloff(
    input_string: str,
) -> tuple[str, dict[str, float], dict[str, float], str, dict[str, float] | None, dict[str, float] | None]:
    """
    Parse a CHEMKIN-format fall-off reaction with pressure-dependent kinetics.

    This function extracts fall-off reaction parameters including high-pressure limit (HPL),
    low-pressure limit (LPL), broadening factor type and parameters, and optional collision
    efficiencies from a CHEMKIN-formatted string.

    Parameters
    ----------
    input_string : str
        CHEMKIN-formatted fall-off reaction string with the following structure:

        .. code-block:: text

            REACTION_NAME    A_hpl   n_hpl   Ea_hpl
            LOW / A_lpl  n_lpl  Ea_lpl /
            TROE / parameters... /           ! or SRI or omit for Lindemann
            SPECIES / efficiency / ...       ! optional

        Example:

        .. code-block:: text

            H+O2(+M)=HO2(+M)  1.48E+12  0.6  0.0
            LOW / 6.37E+20  -1.72  524.8 /
            TROE / 0.8 1E-30 1E+30 /
            H2O/18.0/ AR/0.0/

    Returns
    -------
    reaction_name : str
        Chemical equation with (+M) indicator (e.g., "H+O2(+M)=HO2(+M)")
    hpl_params : dict[str, float]
        High-pressure limit Arrhenius parameters: {"A": ..., "n": ..., "Ea": ...}
    lpl_params : dict[str, float]
        Low-pressure limit Arrhenius parameters: {"A": ..., "n": ..., "Ea": ...}
    falloff_type : str
        Broadening factor type: "lindemann", "troe", "sri", or "tsang"
    falloff_params : dict[str, float] | None
        Broadening factor parameters (type-dependent), or None for Lindemann
    efficiencies : dict[str, float] | None
        Collision efficiency factors mapping species to dimensionless values,
        or None if not specified (default efficiency = 1.0)

    Raises
    ------
    ValueError
        If input is empty, malformed, or missing required LOW parameters
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


def parse_cabr(
    input_string: str,
) -> tuple[str, dict[str, float], dict[str, float], str, dict[str, float] | None, dict[str, float] | None]:
    """
    Parse a CHEMKIN-format Chemically Activated Bimolecular Reaction (CABR).

    This function extracts CABR reaction parameters including low-pressure limit (LPL),
    high-pressure limit (HPL), broadening factor type and parameters, and optional collision
    efficiencies. CABR reactions are the inverse of fall-off reactions, where the main line
    specifies the LPL and the HIGH keyword provides the HPL.

    Parameters
    ----------
    input_string : str
        CHEMKIN-formatted CABR reaction string with the following structure:

        .. code-block:: text

            REACTION_NAME    A_lpl   n_lpl   Ea_lpl
            HIGH / A_hpl  n_hpl  Ea_hpl /
            TROE / parameters... /           ! or SRI or omit for Lindemann
            SPECIES / efficiency / ...       ! optional

        Example:

        .. code-block:: text

            CH3+CH3(+M)=C2H6(+M)  9.2E+16  -1.17  636.0
            HIGH / 1.8E+13  0.0  0.0 /
            TROE / 0.405 1120.0 69.6 /
            H2O/5.0/ CO2/3.0/

    Returns
    -------
    reaction_name : str
        Chemical equation with (+M) indicator (e.g., "CH3+CH3(+M)=C2H6(+M)")
    hpl_params : dict[str, float]
        High-pressure limit Arrhenius parameters: {"A": ..., "n": ..., "Ea": ...}
    lpl_params : dict[str, float]
        Low-pressure limit Arrhenius parameters: {"A": ..., "n": ..., "Ea": ...}
    cabr_type : str
        Broadening factor type: "lindemann", "troe", "sri", or "tsang"
    cabr_params : dict[str, float] | None
        Broadening factor parameters (type-dependent), or None for Lindemann
    efficiencies : dict[str, float] | None
        Collision efficiency factors mapping species to dimensionless values,
        or None if not specified (default efficiency = 1.0)

    Raises
    ------
    ValueError
        If input is empty, malformed, or missing required HIGH parameters
    """
    lines = input_string.strip().split("\n")
    if not lines:
        raise ValueError("Empty CHEMKIN CABR representation")

    main_line = lines[0].strip()
    if not main_line:
        raise ValueError("First line must contain reaction equation")

    # Extract reaction name and high-pressure limit parameters
    reaction_name, lpl_coefficients = parse_reaction_line(main_line, True)

    number_pattern = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

    # Initialize variables
    hpl_coefficients = None
    cabr_type = "lindemann"  # Default cabr type
    cabr_params = None
    efficiencies = None

    for line in lines[1:]:
        line = line.strip()

        # Remove comments (everything after '!')
        if "!" in line:
            line = line.split("!")[0].strip()

        # Skip empty lines
        if not line:
            continue

        if "HIGH" in line:
            # Parse HIGH pressure limit coefficients
            tokens = line.split("/")
            if len(tokens) < 2:
                raise ValueError(f"Invalid HIGH line format (expected '/ data /'): {line}")

            high_data_str = tokens[1].strip()

            try:
                high_coefficients = [float(x) for x in number_pattern.findall(high_data_str)]
            except ValueError as e:
                raise ValueError(f"Error parsing numeric values in HIGH line: {line}") from e

            if len(high_coefficients) != 3:
                raise ValueError(f"Expected 3 values (A, n, Ea) in high line, got {len(high_coefficients)}: {line}")

            hpl_coefficients = {"A": high_coefficients[0], "n": high_coefficients[1], "Ea": high_coefficients[2]}

        elif "TROE" in line:
            # Parse TROE parameters
            cabr_type = "troe"
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
            cabr_params = {
                "A": troe_values[0],
                "T3": troe_values[1],
                "T1": troe_values[2],
                "T2": troe_values[3] if len(troe_values) == 4 else 0.0,
            }

        elif "SRI" in line:
            # Parse SRI parameters
            cabr_type = "sri"
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
            cabr_params = {
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

    # Validate that HIGH was provided
    if hpl_coefficients is None:
        raise ValueError("CABR reaction must contain HIGH pressure limit parameters")

    return reaction_name, hpl_coefficients, lpl_coefficients, cabr_type, cabr_params, efficiencies


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
            - A, n, Ea are the modified Arrhenius parameters for :math:`k_0(T)`
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


def parse_stoichiometry(reaction_name: str) -> dict:
    """
    Parse a CHEMKIN-style reaction string and extract stoichiometric information.

    This function decomposes a reaction string into its constituent species, stoichiometric
    coefficients, and reversibility. It handles various CHEMKIN arrow conventions and
    automatically removes third-body indicators (M, +M, (+M)).

    Parameters
    ----------
    reaction_name : str
        Reaction string in CHEMKIN format. Examples:

        - "2O+M<=>O2+M" (reversible with third-body)
        - "H+O2=>OH+O" (irreversible)
        - "CH2(S)+H2=CH4" (reversible with excited state)
        - "2.5O2+CH4=>2H2O+CO2" (with fractional stoichiometry)
        - "H+O2(+N2)=HO2(+N2)" (explicit collider)

    Returns
    -------
    dict
        Dictionary containing:

        - **species** (list[str]): Unique species names participating in the reaction
        - **reactants** (dict[str, float]): Mapping of reactant species to stoichiometric coefficients
        - **products** (dict[str, float]): Mapping of product species to stoichiometric coefficients
        - **reversible** (bool): True if reversible (= or <=>), False if irreversible (=>)

    Raises
    ------
    ValueError
        If no valid reaction arrow (=, =>, <=>) is found in the reaction string.

    Notes
    -----
    The function handles several edge cases:

    - Generic third-body indicators (M, +M, (+M)) are removed before parsing
    - Explicit colliders (e.g., (+N2), (+AR)) are extracted and added to both sides
    - Duplicate species on the same side are accumulated (e.g., "O+O" becomes {"O": 2.0})
    - Parentheses in species names are preserved (e.g., "CH2(S)" is kept intact)
    - Stoichiometric coefficients can be integers or decimals
    """
    # Normalize and clean the reaction string
    reaction_name = reaction_name.strip()

    # Extract explicit colliders before removing generic third-body indicators
    # Pattern: (+COLLIDER) where COLLIDER is not just M
    # Use set to deduplicate colliders that appear on both sides
    explicit_colliders = set()
    collider_pattern = re.compile(r"\(\+([A-Z][A-Za-z0-9]*)\)")
    for match in collider_pattern.finditer(reaction_name):
        collider = match.group(1)
        if collider != "M":  # Only extract if not generic M
            explicit_colliders.add(collider)

    # Remove explicit collider notation from reaction string
    reaction_name = collider_pattern.sub("", reaction_name)

    # Remove generic third-body indicators
    reaction_name = reaction_name.replace("+M", "")

    # Split reaction into reactants and products based on arrow type
    reactants_str, products_str, reversible = _split_reaction(reaction_name)

    # Parse both sides
    reactants = _parse_species_side(reactants_str)
    products = _parse_species_side(products_str)

    # Add explicit colliders to both reactants and products
    for collider in explicit_colliders:
        reactants[collider] = reactants.get(collider, 0.0) + 1.0
        products[collider] = products.get(collider, 0.0) + 1.0

    # Extract the species names that are part of the reaction
    species = list(set(reactants.keys()) | set(products.keys()))

    return {
        "species": species,
        "reactants": reactants,
        "products": products,
        "reversible": reversible,
    }


def _split_reaction(reaction_name: str) -> tuple[str, str, bool]:
    """
    Split reaction string into reactants, products, and determine reversibility.

    This helper function identifies the reaction arrow type and splits the reaction
    string accordingly. It checks for arrows in order of precedence: <=>, =>, =.

    Parameters
    ----------
    reaction_name : str
        Complete reaction string containing reactants, arrow, and products

    Returns
    -------
    reactants_str : str
        String containing reactant species (left side of arrow)
    products_str : str
        String containing product species (right side of arrow)
    reversible : bool
        True if arrow is = or <=>, False if arrow is =>

    Raises
    ------
    ValueError
        If no valid reaction arrow is found in the input string
    """
    if "<=>" in reaction_name:
        parts = reaction_name.split("<=>", 1)
        return parts[0], parts[1], True
    elif "=>" in reaction_name:
        parts = reaction_name.split("=>", 1)
        return parts[0], parts[1], False
    elif "=" in reaction_name:
        parts = reaction_name.split("=", 1)
        return parts[0], parts[1], True
    else:
        raise ValueError(f"No valid reaction arrow found in: {reaction_name}. Valid are: = | <=> | =>")


def _parse_species_side(side_str: str) -> dict[str, float]:
    """
    Parse one side of the reaction (reactants or products) into stoichiometric coefficients.

    This helper function processes a string containing multiple species (separated by +)
    and extracts each species with its stoichiometric coefficient. If a species appears
    multiple times, coefficients are accumulated.

    Parameters
    ----------
    side_str : str
        String containing species separated by '+' (e.g., "2H+O2" or "CH2(S)+H2")

    Returns
    -------
    dict[str, float]
        Dictionary mapping species names to their total stoichiometric coefficients.
        Species appearing multiple times have their coefficients summed.

    Notes
    -----
    Empty strings or whitespace-only entries are ignored. Parentheses in species
    names are preserved during splitting.
    """
    stoich = {}
    species_list = _split_species(side_str)

    for species in species_list:
        if not species:
            continue

        coeff, species_name = _parse_single_species(species)
        stoich[species_name] = stoich.get(species_name, 0.0) + coeff

    return stoich


def _split_species(side_str: str) -> list[str]:
    """
    Split species string by '+' while respecting parentheses in species names.

    This helper function intelligently splits species on '+' separators while preserving
    parentheses that are part of species names (e.g., excited states, isomers).
    It uses a parenthesis depth counter to avoid splitting inside parenthesized names.

    Parameters
    ----------
    side_str : str
        String containing multiple species separated by '+'. May include species
        with parentheses in their names (e.g., "O+CH2(S)+H2")

    Returns
    -------
    list[str]
        List of individual species strings with leading/trailing whitespace removed.
        Empty strings are excluded.

    Notes
    -----
    The algorithm tracks parenthesis depth to distinguish between:

    - Species separator: "+" at depth 0 (e.g., "H+O2" → ["H", "O2"])
    - Part of species name: "+" at depth > 0 or parentheses (e.g., "CH2(S)" → ["CH2(S)"])
    """
    species_list = []
    current = ""
    paren_depth = 0

    for char in side_str:
        if char == "(":
            paren_depth += 1
            current += char
        elif char == ")":
            paren_depth -= 1
            current += char
        elif char == "+" and paren_depth == 0:
            if current.strip():
                species_list.append(current.strip())
            current = ""
        else:
            current += char

    if current.strip():
        species_list.append(current.strip())

    return species_list


def _parse_single_species(species: str) -> tuple[float, str]:
    """
    Parse a single species string into stoichiometric coefficient and species name.

    This helper function extracts the optional leading stoichiometric coefficient
    from a species string. If no coefficient is present, it defaults to 1.0.

    Parameters
    ----------
    species : str
        Single species string, optionally prefixed with a stoichiometric coefficient.
        Examples: "O", "2O", "1.5CH4", "CH2(S)"

    Returns
    -------
    coefficient : float
        Stoichiometric coefficient (defaults to 1.0 if not specified)
    species_name : str
        Species name with leading/trailing whitespace removed

    Raises
    ------
    ValueError
        If the species string does not match the expected format

    Notes
    -----
    The function uses a regular expression to match the pattern:
    ``^(\\d+\\.?\\d*)?(.+)$``

    This matches an optional numeric coefficient (integer or decimal) followed
    by the species name. The species name must contain at least one character.
    """
    # Match optional coefficient followed by species name
    match = re.match(r"^(\d+\.?\d*)?(.+)$", species)

    if not match:
        raise ValueError(f"Invalid species format: {species}")

    coeff_str, species_name = match.groups()
    coeff = float(coeff_str) if coeff_str else 1.0
    species_name = species_name.strip()

    return coeff, species_name


def parse_species(thermo_string: str) -> tuple[str, dict[str, int], str, float, float, float, list[float], list[float]]:
    """
    Parse CHEMKIN NASA 7-coefficient polynomial thermodynamic data for a species.

    The NASA polynomial format uses a 4-line representation to store thermodynamic
    properties as temperature-dependent polynomials. Two sets of 7 coefficients
    cover different temperature ranges (typically split around 1000K).

    Parameters
    ----------
    thermo_string : str
        4-line CHEMKIN NASA thermo format string with the following structure:

        .. code-block:: text

            Line 1: Species_name    Elements  Phase  Tmin   Tmax   Tmid
            Line 2: a1_high  a2_high  a3_high  a4_high  a5_high
            Line 3: a6_high  a7_high  a1_low   a2_low   a3_low
            Line 4: a4_low   a5_low   a6_low   a7_low

        Where each numeric field is 15 characters wide in Fortran format.

        **Extended Composition Format:**

        For species with many elements, the format supports continuation lines
        using the ``&`` character:

        .. code-block:: text

            Line 1: Species_name  C 0H 0  G  Tmin Tmax Tmid  1&
            Line 2: C  1250 H  812
            Line 3: a1_high  a2_high  a3_high  a4_high  a5_high
            ...

        Continuation lines (after ``&``) contain element-count pairs that are
        merged with the standard composition from Line 1.

    Returns
    -------
    species_name : str
        Chemical species identifier (e.g., "H2O", "CH4")
    elemental_composition : dict[str, int]
        Elemental composition mapping element symbols to atom counts
        (e.g., {"H": 2, "O": 1} for H2O)
    phase : str
        Phase indicator: "G" (gas), "L" (liquid), or "S" (solid)
    Tmin : float
        Minimum valid temperature [K] for polynomial data
    Tmax : float
        Maximum valid temperature [K] for polynomial data
    Tmid : float
        Temperature [K] separating low and high polynomial ranges
    high_coeffs : list[float]
        7 NASA polynomial coefficients valid for T ∈ [Tmid, Tmax]
    low_coeffs : list[float]
        7 NASA polynomial coefficients valid for T ∈ [Tmin, Tmid]

    Raises
    ------
    ValueError
        - If input does not contain exactly 4 lines (after processing continuations)
        - If intermediate temperature (Tmid) is missing or invalid
        - If coefficient parsing fails

    Notes
    -----
    **Missing Intermediate Temperature:**

    If Tmid is not specified in columns 66-75 of line 1, a descriptive error
    is raised. In standard CHEMKIN files, Tmid should come from the THERMO
    section header. The error message includes the temperature range to help
    diagnose the issue.

    **Extended Composition:**

    The ``&`` continuation character allows specification of arbitrarily many
    elements, which is useful for large molecules in combustion chemistry
    (e.g., biodiesel surrogates with C > 20).

    Examples
    --------
    Standard format (4 lines):

    >>> thermo = '''
    ... H2O               H   2O   1     G   200.00   6000.00  1000.00    1
    ... 2.67703787E+00 2.97318329E-03-7.73769690E-07 9.44336689E-11-4.26900959E-15    2
    ... -2.98858938E+04 6.88255571E+00 4.19864056E+00-2.03643410E-03 6.52040211E-06    3
    ... -5.48797062E-09 1.77197817E-12-3.02937267E+04-8.49032208E-01                   4
    ... '''
    >>> name, comp, phase, tmin, tmax, tmid, high, low = parse_species(thermo)
    >>> print(name, comp)
    H2O {'H': 2, 'O': 1}

    Extended composition with continuation:

    >>> thermo_ext = '''
    ... BIGMOL            C   0H   0     G   300.00   4000.00  1000.00    1&
    ... C  100 H  200
    ... ...coefficients...
    ... '''
    """
    # Parse thermodynamic data
    # Split by newline and filter out empty lines
    raw_lines = thermo_string.strip().split("\n")
    lines = [line for line in raw_lines if line.strip()]

    # Handle extended elemental composition with & continuation character
    # If first line ends with &, the following lines contain additional composition data
    extended_composition = {}
    if lines[0].rstrip().endswith("&"):
        # Find all continuation lines
        comp_lines = []
        i = 0
        while i < len(lines) - 1 and lines[i].rstrip().endswith("&"):
            comp_lines.append(lines[i + 1])
            i += 1

        # Parse extended composition from continuation lines
        # Format: element count element count ...
        comp_str = " ".join(comp_lines)
        comp_tokens = comp_str.split()
        for j in range(0, len(comp_tokens), 2):
            if j + 1 < len(comp_tokens):
                element = comp_tokens[j].capitalize()
                try:
                    count = int(comp_tokens[j + 1])
                    if count > 0:
                        extended_composition[element] = count
                except ValueError:
                    pass  # Skip invalid entries

        # Remove continuation lines from the main data
        # lines[0] is the header with &, lines[1:i+1] are composition, lines[i+1:] are coefficients
        lines = [lines[0]] + lines[i + 1 :]

    if len(lines) != 4:
        raise ValueError(f"Expected 4 lines in CHEMKIN thermo format, got {len(lines)}")

    # Remove leading/trailing whitespace but preserve internal structure
    # The CHEMKIN format uses fixed-width fields, so we need to preserve column positions
    # Strip only trailing whitespace, keep leading spaces if present
    lines = [line.rstrip() for line in lines]

    # Parse header line
    header = lines[0]
    species_name = header[0:24].split()[0].strip()

    # Parse elemental composition from header (standard format)
    elemental_composition = parse_composition(header[24:44], 4, 5)

    # Merge with extended composition from continuation lines
    elemental_composition.update(extended_composition)

    # Extract phase
    phase = header[44] if len(header) > 44 else "G"

    # Extract temperature ranges
    Tmin = fort_float(header[45:55])
    Tmax = fort_float(header[55:65])

    # Extract intermediate temperature (Tmid)
    try:
        tmid_str = header[65:75].strip()
        if not tmid_str:
            raise ValueError(
                f"Missing intermediate temperature (Tmid) for species '{species_name}'. "
                f"The CHEMKIN NASA format requires Tmid to be specified in columns 66-75 of line 1. "
                f"This value should typically come from the THERMO section header. "
                f"Temperature range: [{Tmin}, {Tmax}] K"
            )
        Tmid = fort_float(tmid_str)
    except (ValueError, IndexError) as e:
        if "Missing intermediate temperature" in str(e):
            raise
        raise ValueError(
            f"Invalid intermediate temperature (Tmid) for species '{species_name}': '{header[65:75]}'. "
            f"Could not parse Tmid from columns 66-75 of line 1. "
            f"Temperature range: [{Tmin}, {Tmax}] K"
        ) from e

    # Extract NASA polynomial coefficients (high-T first!)
    # CHEMKIN format allows numbers to be written without spaces between them
    # (e.g., "1.23E+02-4.56E-03" is valid). Therefore, we use regex to extract
    # all numbers from each line.

    import re

    # Pattern matches: optional sign, digits with optional decimal, optional exponent
    number_pattern = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")

    # Extract all numbers from lines 2-4
    line2_numbers = []
    for match in number_pattern.findall(lines[1]):
        # Handle Fortran 'D' notation
        num_str = match.replace("D", "E").replace("d", "e")
        line2_numbers.append(float(num_str))

    line3_numbers = []
    for match in number_pattern.findall(lines[2]):
        num_str = match.replace("D", "E").replace("d", "e")
        line3_numbers.append(float(num_str))

    line4_numbers = []
    for match in number_pattern.findall(lines[3]):
        num_str = match.replace("D", "E").replace("d", "e")
        line4_numbers.append(float(num_str))

    # Validate we have the right number of coefficients
    if len(line2_numbers) < 5:
        raise ValueError(f"Line 2 must contain at least 5 coefficients, found {len(line2_numbers)}")
    if len(line3_numbers) < 5:
        raise ValueError(f"Line 3 must contain at least 5 coefficients, found {len(line3_numbers)}")
    if len(line4_numbers) < 4:
        raise ValueError(f"Line 4 must contain at least 4 coefficients, found {len(line4_numbers)}")

    # High-T coefficients: 5 from line 2, 2 from line 3
    high_coeffs = line2_numbers[:5] + line3_numbers[:2]

    # Low-T coefficients: last 3 from line 3, first 4 from line 4
    low_coeffs = line3_numbers[2:5] + line4_numbers[:4]

    return (
        species_name,
        elemental_composition,
        phase,
        Tmin,
        Tmax,
        Tmid,
        high_coeffs,
        low_coeffs,
    )


def parse_composition(elements: str, nElements: int, width: int) -> dict[str, int]:
    """
    Parse elemental composition from fixed-width NASA polynomial entry.

    This helper function extracts elemental symbols and their atom counts from the
    fixed-width format used in CHEMKIN NASA thermodynamic data files.

    Parameters
    ----------
    elements : str
        Fixed-width string containing elemental composition data
        (e.g., "H 2O 1  " with width=5 per element)
    nElements : int
        Number of element entries to parse (typically 4)
    width : int
        Character width per element entry (typically 5)

    Returns
    -------
    dict[str, int]
        Dictionary mapping element symbols (capitalized) to atom counts.
        Only non-zero counts are included.

    Notes
    -----
    Each element entry in the string is formatted as:
    - Positions [0:2]: Element symbol (e.g., "H ", "O ", "C ")
    - Positions [2:width]: Atom count as integer or float

    Empty entries and parsing errors are silently ignored.
    """
    composition = {}
    for i in range(nElements):
        symbol = elements[width * i : width * i + 2].strip()
        count = elements[width * i + 2 : width * i + width].strip()
        if not symbol:
            continue
        try:
            count = int(float(count))
            if count:
                composition[symbol.capitalize()] = count
        except ValueError:
            pass
    return composition


def check_reaction_name(reaction_name: str, m_is_allowed: bool = False) -> None:
    """
    Validate CHEMKIN reaction name for proper use of third-body indicators.

    This function ensures that third-body indicators (M, +M, (+M)) are only used in
    reaction types that support them, preventing invalid reaction specifications. The
    validation uses tokenization to distinguish between standalone "M" as a third-body
    indicator and "M" appearing within species names.

    Parameters
    ----------
    reaction_name : str
        CHEMKIN reaction equation to validate
    m_is_allowed : bool, optional
        If True, allows third-body indicators in the reaction name.
        Set to True for threebody, fall-off, CABR, and mixture-rule reactions.
        By default False (standard elementary reactions)

    Raises
    ------
    ValueError
        If ``m_is_allowed=False`` and the reaction name contains third-body
        indicators (M, +M, or (+M))

    Notes
    -----
    Third-body indicators are only valid for:

    - Three-body reactions: ``H+OH+M=H2O+M``
    - Fall-off reactions: ``H+O2(+M)=HO2(+M)``
    - CABR reactions: ``CH3+CH3(+M)=C2H6(+M)``
    - Mixture-rule reactions with pressure dependence

    Standard elementary reactions should not include M as a third-body indicator.

    **Detection Strategy:**

    The function tokenizes the reaction string by:

    1. Replacing reaction arrows (=, =>, <=>) with ``+``
    2. Splitting by ``+`` to get individual species tokens
    3. Checking if ``"M"`` appears as a complete token

    This approach correctly distinguishes:

    - Valid: ``"H+B2M2=CH3+IC4H8"`` - "M" is part of "B2M2"
    - Invalid: ``"H+M=H2"`` - "M" is a standalone token
    - Valid: ``"CH3M+O2=products"`` - "M" is part of "CH3M"
    - Invalid: ``"H+OH+M=H2O+M"`` - "M" appears as separate tokens
    """
    if not m_is_allowed:
        # Check for (+M) pattern (FallOff/CABR indicator)
        if "(+M)" in reaction_name:
            raise ValueError(
                f"Invalid reaction name: '{reaction_name}' contains '(+M)' but "
                "this is only allowed for FallOff, CABR or Mixture Rule "
                "like reactions"
            )

        # Parse the reaction to get individual species tokens
        # Split by reaction arrows first
        reaction_temp = re.sub(r"(<=>|=>|=)", "+", reaction_name)

        # Split by + to get individual species tokens (including +M)
        species_tokens = [s.strip() for s in reaction_temp.split("+") if s.strip()]

        # Check if standalone "M" appears as a species
        # This will match "M" or "+M" but not "B2M2", "CH3M", etc.
        if "M" in species_tokens:
            raise ValueError(
                f"Invalid reaction name: '{reaction_name}' contains standalone 'M' but "
                "this is only allowed for Third Body, FallOff, CABR or Mixture Rule "
                "like reactions"
            )


def fort_float(s: str) -> float:
    """
    Convert Fortran-formatted floating-point string to Python float.

    This helper function handles numeric formats commonly found in CHEMKIN files that
    originate from Fortran code, including non-standard exponent notations.

    Parameters
    ----------
    s : str
        String representation of a floating-point number, possibly in Fortran format.
        Examples: "1.23E+02", "4.56D-03", "7.89E 10"

    Returns
    -------
    float
        Parsed floating-point value

    Notes
    -----
    The function performs the following transformations:

    1. **D exponent**: Converts Fortran double-precision exponent 'D' to 'E'
       (e.g., "1.5D+10" → "1.5E+10")
    2. **Space in exponent**: Adds '+' sign when space appears before exponent
       (e.g., "1.5E 10" → "1.5E+10")
    3. **Case normalization**: Converts to lowercase before parsing

    These transformations ensure compatibility with Python's float() parser.

    Examples
    --------
    >>> fort_float("1.23E+02")
    123.0

    >>> fort_float("4.56D-03")
    0.00456

    >>> fort_float("7.89E 10")
    78900000000.0

    References
    ----------
    Taken from Cantera CHEMKIN parser for handling legacy Fortran numeric formats.
    """
    return float(s.strip().lower().replace("d", "e").replace("e ", "e+"))
