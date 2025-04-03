from typing import Dict, Optional, Tuple


def extract_stoichiometry(reaction: str) -> Dict[str, Dict[str, float]]:
    """
    Extract species names and their corresponding stoichiometric coefficients from a chemical reaction.

    Args:
        reaction (str): A chemical reaction string (e.g., "2H2 + O2 = 2H2O").
            Can use "=", "=>", or "<=>" as reaction separators.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary with two keys: "reactants" and "products".
            Each key maps to a nested dictionary where keys are species names and
            values are their stoichiometric coefficients.

    Example:
        >>> extract_stoichiometry("2H2 + O2 = 2H2O")
        {"reactants": {"H2": 2.0, "O2": 1.0}, "products": {"H2O": 2.0}}
    """
    reaction = reaction.replace(" ", "")

    # 1. Split reactants and products
    reactants, products = split_reaction(reaction)

    # 2. Process each side of the reaction independently
    reactants = process_reaction_side(reactants)
    products = process_reaction_side(products)

    # 3. Assemble everything into a single dictionary
    return {"reactants": reactants, "products": products}


def split_reaction(reaction: str) -> Tuple[str, str]:
    """
    Split a chemical reaction string into reactants and products.

    Args:
        reaction (str): A chemical reaction string without spaces.
            The reaction can use "=", "=>", or "<=>" as separators.

    Returns:
        Tuple[str, str]: A tuple containing (reactants, products).

    Raises:
        ValueError: If no valid reaction separator is found.

    Example:
        >>> split_reaction("2CH3(+M)=C2H6(+M)")
        ("2CH3(+M)", "C2H6(+M)")
    """
    for separator in ["<=>", "=>", "="]:
        if separator in reaction:
            parts = reaction.split(separator, 1)  # Split only on first occurrence
            return parts[0].strip(), parts[1].strip()

    raise ValueError("No valid reaction separator found ('=', '<=>', or '=>')")


def process_reaction_side(reaction_side: str) -> Dict[str, float]:
    """
    Process one side of a chemical reaction to extract species and their stoichiometric coefficients.

    Args:
        reaction_side (str): One side of a chemical reaction (e.g., "2H2+O2").

    Returns:
        Dict[str, float]: A dictionary where keys are species names and
            values are their stoichiometric coefficients.

    Notes:
        - Removes third-body indicators like "(+M)".
        - If the same species appears multiple times, their coefficients are summed.

    Example:
        >>> process_reaction_side("2H2+O2+H2")
        {"H2": 3.0, "O2": 1.0}
    """
    reaction_side = reaction_side.replace("(+M)", "")
    reaction_side = reaction_side.replace("+M", "")
    species = reaction_side.split("+")
    stoichiometric_coefficients = {}
    for sp in species:
        species_name, stoichiometric_coefficient = extract_species_coefficient_pair(sp)
        if species_name in stoichiometric_coefficients:
            stoichiometric_coefficients[species_name] += stoichiometric_coefficient
        else:
            stoichiometric_coefficients[species_name] = stoichiometric_coefficient
    return stoichiometric_coefficients


def extract_species_coefficient_pair(species: str) -> Tuple[str, float]:
    """
    Extract the species name and its stoichiometric coefficient from a species term.

    Args:
        species (str): A species term from a chemical reaction (e.g., "2H2", "CH4").

    Returns:
        Tuple[str, float]: A tuple containing (species_name, stoichiometric_coefficient).
            If no coefficient is specified, 1.0 is returned.

    Example:
        >>> extract_species_coefficient_pair("2H2")
        ("H2", 2.0)
        >>> extract_species_coefficient_pair("CH4")
        ("CH4", 1.0)
    """
    i = 0
    while i < len(species) and species[i].isdigit():
        i += 1

    stoichiometric_coefficient = float(species[:i]) if i > 0 else 1.0
    species_name = species[i:]

    return species_name, stoichiometric_coefficient


def latexify_reaction(reaction_name: str, separator: Optional[str] = "=") -> str:
    """
    Convert a chemical reaction to LaTeX format using the mhchem package.

    Args:
        reaction_name: String representation of the chemical reaction
        separator: Reaction arrow symbol to use (default: "=")

    Returns:
        String containing the LaTeX representation of the reaction
    """
    # Determine special reaction types once
    is_falloff = "(+M)" in reaction_name
    is_third_body = "+M" in reaction_name and not is_falloff

    # Extract stoichiometry
    reaction = extract_stoichiometry(reaction_name)

    # Build reaction parts using join instead of string concatenation
    parts = ["\\ce{"]

    # Helper function to format species
    def format_species_list(species_dict):
        return " + ".join(f"{value}{key}" for key, value in species_dict.items())

    # Add reactants
    parts.append(format_species_list(reaction["reactants"]))

    # Add third body or falloff indicators for reactants side
    if is_third_body:
        parts.append("+ M")
    elif is_falloff:
        parts.append("( + M)")

    # Add separator
    parts.append(separator)

    # Add products
    parts.append(format_species_list(reaction["products"]))

    # Add third body or falloff indicators for products side
    if is_third_body:
        parts.append("+ M")
    elif is_falloff:
        parts.append("( + M)")

    # Close the LaTeX command
    parts.append("}")

    return " ".join(parts)
