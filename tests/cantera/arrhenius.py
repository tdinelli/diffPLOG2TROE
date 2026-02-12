import os

import cantera as ct
import numpy as np

ct_version = ct.__version__
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

cantera_rates = []

gas = ct.Solution(os.path.join(current_dir, "reference_mech.yaml"))

T_range = np.linspace(300, 3000, 300)

print(f"Extracting rate constants for (Arrhenius) reaction: {gas.reaction(0).equation}")
print(f" - Temperature range: {T_range[0]:.1f} - {T_range[-1]:.1f} K")
for t in T_range:
    gas.TP = t, ct.one_atm
    cantera_rates.append(gas.forward_rate_constants[0])

cantera_rates = np.array(cantera_rates)
combined_array = np.column_stack((T_range, cantera_rates))

data_directory = os.path.join(current_dir, "cantera_data", str(ct_version))
if not os.path.isdir(data_directory):
    os.makedirs(data_directory)

np.savetxt(os.path.join(data_directory, "arrhenius.csv"), combined_array, delimiter=";", fmt="%.10e")
