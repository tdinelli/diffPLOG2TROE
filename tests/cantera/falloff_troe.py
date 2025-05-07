import os

import cantera as ct
import numpy as np


ct_version = ct.__version__
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

gas = ct.Solution(os.path.join(current_dir, "reference_mech.yaml"))

T_range = np.linspace(300, 3000, 300)
P_range = np.logspace(np.log10(0.01), np.log10(100), 300)

k_matrix = np.zeros((len(P_range), len(T_range)))

for t_idx, t in enumerate(T_range):
    for p_idx, p in enumerate(P_range):
        gas.TPX = t, p, "AR:1"
        k_matrix[p_idx, t_idx] = gas.forward_rate_constants[1]


data_directory = os.path.join(current_dir, "cantera_data", str(ct_version))
if not os.path.isdir(data_directory):
    os.makedirs(data_directory)

np.savetxt(os.path.join(data_directory, "falloff_troe.csv"), k_matrix, delimiter=";", fmt="%.10e")
