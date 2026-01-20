import os

import cantera as ct
import numpy as np

ct_version = ct.__version__
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

gas = ct.Solution(os.path.join(current_dir, "reference_mech.yaml"))

T_range = np.linspace(300, 3000, 300)
reaction_index = 5

cantera_rate = []

print(f"Extracting rate constants for (3body) reaction: {gas.reaction(reaction_index).equation}")
print(f" - Temperature range: {T_range[0]:.1f} - {T_range[-1]:.1f} K")
for t_idx, t in enumerate(T_range):
    gas.TPX = t, ct.one_atm, "N2:1"
    cantera_rate.append(gas.forward_rate_constants[reaction_index])

cantera_rate = np.array(cantera_rate)
combined_array = np.column_stack((T_range, cantera_rate))

data_directory = os.path.join(current_dir, "cantera_data", str(ct_version))
if not os.path.isdir(data_directory):
    os.makedirs(data_directory)

np.savetxt(os.path.join(data_directory, "3body.csv"), combined_array, delimiter=";", fmt="%.15e")
