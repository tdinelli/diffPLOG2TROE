import re
from typing import List, Dict
import numpy as np


def read_chemkin_extract_plog(kinetics: str):
    """
    """
    print("================================================================")
    print(" Reading the CHEMKIN kinetic file located in {}".format(kinetics))
    # Reading the file content
    with open(kinetics, "r") as file:
        raw_content = [line.rstrip('\n') for line in file]
    file.close()


    # Extractig only the reaction part and removing commented and empty or white lines
    reaction_start = raw_content.index("REACTIONS")
    reaction_end = np.where(np.asarray(raw_content) == "END")[0][-1]
    reactions_content = raw_content[reaction_start+1:reaction_end]  # Extracting reaction part
    reactions_content = remove_empty_lines(reactions_content)  # Removed empty lines
    reactions_content = remove_commented_lines(reactions_content)  # Removing commented lines

    raw_plog_reactions, indices_of_plog_reactions, indices_of_reactions, nr = identify_plog_reactions(reactions_content)
    print(" * Number of Reactions: {}".format(nr))
    print("    - Number of PLOG Reactions: {}".format(len(indices_of_plog_reactions)))

    plog_reactions = []
    for i in raw_plog_reactions:
        plog_reactions.extend(analyze_plog_reaction(i))

    print("================================================================")
    return plog_reactions, indices_of_plog_reactions, indices_of_reactions


def remove_commented_lines(content: list) -> list:
    """
    Function that remove the commented lines from the CHEMKIN file.
    Args:
        content (lsit): list of strings.
    Returns:
        (list): list of strings without the commented lines.
    """
    is_commented = lambda line: line.strip()[0] == "!"  # Check if the first element of a string is the char "!"
    indices_of_commented_lines = []
    for i, line in enumerate(content):
        if is_commented(line):
            indices_of_commented_lines.append(i)
    return [item for i, item in enumerate(content) if i not in indices_of_commented_lines]


def remove_empty_lines(content: list) -> list:
    """
    Function that removes empty strings from a list of strings.
    Args:
        content (lsit): list of strings.
    Returns:
        (list): list of strings without the commented lines.
    """
    is_empty = lambda line: line.strip() == ""
    indices_of_empty_lines = []
    for i, line in enumerate(content):
        if is_empty(line):
            indices_of_empty_lines.append(i)
    return [item for i, item in enumerate(content) if i not in indices_of_empty_lines]


def identify_plog_reactions(content: list) -> tuple:
    """
    Function needed to parse the entire reaction sets and extract only the PLOG reactions.

    Args:
        content (list): list of strings containing all the reactions whithin the mechanism.

    Returns:
        (tuple): a list of all the PLOG reactions, the indices of the plog reactions, the indices of all the reactions.
    """
    plog_reactions = []
    indices_of_reactions = []
    indices_of_plog_reactions = []
    for i, line in enumerate(content):
        if ("=" in line or "<=>" in line or "=>" in line):
            indices_of_reactions.append(i)
            is_a_plog = i != len(content) - 1 and "PLOG" in content[i+1]
            if is_a_plog:
                indices_of_plog_reactions.append(i)

    # Once I have the index of the PLOG reactions and the index of all the reactions available inside the mechanism i
    # extract the blocks
    n_reactions = len(indices_of_reactions)
    for i, idx in enumerate(indices_of_plog_reactions):
        idx_current = indices_of_reactions.index(idx)
        idx_next = indices_of_reactions[idx_current + 1] if idx_current + 1 < n_reactions else len(content)
        plog_reactions.append(content[idx:idx_next])

    return (plog_reactions, indices_of_plog_reactions, indices_of_reactions, n_reactions)


def analyze_plog_reaction(plog: list) -> List[Dict[str, any]]:
    """
    Analyzes a PLOG reaction from a list of strings, extracts the reaction name, pressure levels, and Arrhenius
    parameters, and identifies if the reaction is a duplicate.

    The function parses a PLOG reaction defined in a CHEMKIN file and returns a dictionary containing the reaction name
    and its associated parameters. If the reaction is declared explicitly or implicitly as a duplicate, the function
    will return two dictionaries representing the split PLOG reactions.

    Args:
        plog (list): A list of strings, where each string represents a line from a PLOG reaction section in a CHEMKIN
                     file. The first element contains the reaction name, and the subsequent elements contain the
                     pressure levels and Arrhenius parameters.

    Returns:
        dict or tuple: 
            - If no implicit or explicit duplicate is found, the function returns a dictionary with the following
              structure:
                {
                    "name": str,              # Reaction name (e.g., "NH3 + O2 = NH2 + OH")
                    "parameters": list,       # List of Arrhenius parameters at different pressure levels
                    "is_duplicate": bool      # Indicates whether the reaction is a duplicate
                }
            - If an implicit or explicit duplicate is found, the function returns two dictionaries (one for each
              reaction), with the same structure as above, each containing the respective parameters split for the
              duplicate.

    Raises:
        Exception: If an implicit duplicate has more than two entries for the same pressure level.

    Notes:
        - The function checks for duplicates explicitly through the presence of the keywords "DUP" or "DUPLICATE" in the
          reaction string.
        - Implicit duplicates are detected by comparing the pressure levels within the parameters.
        - If the reaction has more than two duplicate entries for the same pressure level, the function raises an
          exception, as this case is not handled at the moment.
    """
    is_number = re.compile(r'^[-+]?(?:\d*\.\d+|\d+\.?)(e[-+]?\d+)?$', re.IGNORECASE).match
    is_duplicate = lambda s: "DUP" in s or "DUPLICATE" in s
    is_implicitly_duplicate = lambda pressure_levels: len(pressure_levels) != len(set(pressure_levels))

    plog_reaction = {
        "name": "",
        "parameters": [],
        "is_duplicate": False
    }
    reaction_name_pattern = re.compile(r'^[A-Za-z0-9+\-()\s<=>]+(?=\s+[+-]?\d*\.?\d)')
    reaction_name = reaction_name_pattern.match(plog[0].strip()).group().strip()
    plog_reaction["name"] = reaction_name

    pressure_levels = []
    for line in plog[1:]:
        if is_duplicate(line):
            plog_reaction["is_duplicate"] = True
            break

        parameters = [float(part) for part in line.replace("/", " ").split() if is_number(part)]
        plog_reaction["parameters"].append(parameters)
        pressure_levels.append(parameters[0])

    reactions = [plog_reaction]
    if is_implicitly_duplicate(pressure_levels):
        counter = pressure_levels.count(pressure_levels[0])
        if counter > 2:
            raise Exception("PLOG duplicate with more than two duplicates not handled yet!")

        plog_reaction["is_duplicate"] = True
        plog_reaction_1 = plog_reaction.copy()
        plog_reaction_2 = plog_reaction.copy()
        plog_reaction_1["parameters"] = plog_reaction["parameters"][::2]
        plog_reaction_2["parameters"] = plog_reaction["parameters"][1::2]

        reactions = [plog_reaction_1, plog_reaction_2]

    return reactions
