import os
import numpy as np
from .interpreter import (
    remove_empty_lines,
    remove_commented_lines,
    identify_plog_reactions,
    analyze_plog_reaction,
)
from .utilities import plog_to_chemkin, comment_chemkin_string, arrheniusbase_to_chemkin


def write_chemkin(kinetics: str, output_folder: str, plog_converted: list, fitting_parameters: list = None):
    with open(kinetics, "r") as file:
        raw_content = [line.rstrip("\n") for line in file]
    file.close()

    # Extractig only the reaction part and removing commented and empty or white lines
    reaction_start = raw_content.index("REACTIONS")
    reaction_end = np.where(np.asarray(raw_content) == "END")[0][-1]
    elements_species_content = raw_content[:reaction_start]

    elements_species_content = [i.strip() for i in elements_species_content if i != ""]
    elements_species_content = [j for i in elements_species_content if i.strip() for j in i.split()]
    elements_species_content = format_elements_species_block(elements_species_content)

    reactions_content = raw_content[reaction_start + 1 : reaction_end]  # Extracting reaction part
    reactions_content = remove_empty_lines(reactions_content)  # Removed empty lines
    reactions_content = remove_commented_lines(reactions_content)  # Removing commented lines

    raw_plog_reactions, indices_plog_reactions, indices_of_reactions, _ = identify_plog_reactions(reactions_content)
    plog_reactions = []
    for i in raw_plog_reactions:
        plog_reactions.append(analyze_plog_reaction(i))

    # Generation of the string to be printed whithin the mechanism file
    converted_plog_string = []
    counter = 0
    for i, reaction_list in enumerate(plog_reactions):
        is_implicitly_dup = lambda l: len(l) == 2
        tmp = "\n!!!!!!!!!!!!!!!!!!!! PLOG reaction number {} !!!!!!!!!!!!!!!!!!!!\n".format(i + 1)
        if is_implicitly_dup(reaction_list):
            for j, reaction in enumerate(reaction_list):
                tmp += "! Original PLOG formulation (SPLITTED):\n"
                tmp += comment_chemkin_string(plog_to_chemkin(reaction))
                if fitting_parameters is not None:
                    tmp += "! * First guessed values for the fitting:\n"
                    tmp += "!    - A: {:.5E}, b: {:.5E}, Ea: {:.5E}\n".format(
                        fitting_parameters[j][0], fitting_parameters[j][1], fitting_parameters[j][2]
                    )
                    tmp += "! * Adjusted R2 value:\n"
                    tmp += "!    - R2adj: {:.5E}\n".format(fitting_parameters[j][3])
                tmp += arrheniusbase_to_chemkin(plog_converted[counter])
                tmp += "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n"
                counter += 1
        else:
            tmp += "! Original PLOG formulation:\n"
            tmp += comment_chemkin_string(plog_to_chemkin(plog_reactions[i][0]))
            if fitting_parameters is not None:
                tmp += "! * First guessed values for the fitting:\n"
                tmp += "!    - A: {:.5E}, b: {:.5E}, Ea: {:.5E}\n".format(
                    fitting_parameters[i][0], fitting_parameters[i][1], fitting_parameters[i][2]
                )
                tmp += "! * Adjusted R2 value:\n"
                tmp += "!    - R2adj: {:.5E}\n".format(fitting_parameters[i][3])
            tmp += arrheniusbase_to_chemkin(plog_converted[counter])
            tmp += "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n"
            counter += 1
        converted_plog_string.append(tmp)

    # substitution of the PLOG formalism with the refitted constant
    reactions_to_file = ""
    plog_idx = 0
    skip_until = None
    for i, line in enumerate(reactions_content):
        if skip_until is not None and i < skip_until:
            continue

        if i in indices_plog_reactions:
            reactions_to_file += converted_plog_string[plog_idx]
            idx_current = indices_of_reactions.index(indices_plog_reactions[plog_idx])
            idx_next = (
                indices_of_reactions[idx_current + 1]
                if idx_current + 1 < len(indices_of_reactions)
                else len(reactions_content)
            )
            skip_until = idx_next
            plog_idx += 1
        else:
            reactions_to_file += line + "\n"
            skip_until = None

    to_file = elements_species_content + ""
    to_file += reactions_to_file
    to_file += "END"

    file_name = os.path.join(output_folder, "PLOG_replaced.CKI")
    with open(file_name, "w") as file:
        file.write(to_file)


def format_elements_species_block(content: list) -> str:
    elements_end = content.index("END")
    block = content[0] + "\n"
    for i in content[1:elements_end]:
        block += i + " "
    block += "\n" + content[elements_end] + "\n\n" + content[elements_end + 1]

    n_col = 7
    col_width = 15

    for i, j in enumerate(content[elements_end + 2 : -1]):
        if i % n_col == 0:
            block += "\n"
        block += j.ljust(col_width)
    block += "\n" + content[-1] + "\n\n" + "REACTIONS\n"

    return block
