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

print(f"Extracting rate constants for (FallOff TROE) reaction: {gas.reaction(1).equation}")
print(f" - Temperature range: {T_range[0]:.1f} - {T_range[-1]:.1f} K")
print(f" - Pressure range: {P_range[0]:.3f} - {P_range[-1]:.1f} atm")
print(" - Mixture 100% AR")
for t_idx, t in enumerate(T_range):
    for p_idx, p in enumerate(P_range):
        gas.TPX = t, p*ct.one_atm, "AR:1"
        k_matrix[p_idx, t_idx] = gas.forward_rate_constants[1]


data_directory = os.path.join(current_dir, "cantera_data", str(ct_version))
if not os.path.isdir(data_directory):
    os.makedirs(data_directory)

np.savetxt(os.path.join(data_directory, "falloff_troe_ar.csv"), k_matrix, delimiter=";", fmt="%.10e")

k_matrix = np.zeros((len(P_range), len(T_range)))

print(f"Extracting rate constants for (FallOff TROE) reaction: {gas.reaction(1).equation}")
print(f" - Temperature range: {T_range[0]:.1f} - {T_range[-1]:.1f} K")
print(f" - Pressure range: {P_range[0]:.3f} - {P_range[-1]:.1f} atm")
print(" - Mixture 50% AR, 50% H2O")
for t_idx, t in enumerate(T_range):
    for p_idx, p in enumerate(P_range):
        gas.TPX = t, p*ct.one_atm, "AR:0.5, H2O: 0.5"
        k_matrix[p_idx, t_idx] = gas.forward_rate_constants[1]


data_directory = os.path.join(current_dir, "cantera_data", str(ct_version))
if not os.path.isdir(data_directory):
    os.makedirs(data_directory)

np.savetxt(os.path.join(data_directory, "falloff_troe_ar_h2o.csv"), k_matrix, delimiter=";", fmt="%.10e")

k_matrix = np.zeros((len(P_range), len(T_range)))

print(f"Extracting rate constants for (FallOff TROE) reaction: {gas.reaction(1).equation}")
print(f" - Temperature range: {T_range[0]:.1f} - {T_range[-1]:.1f} K")
print(f" - Pressure range: {P_range[0]:.3f} - {P_range[-1]:.1f} atm")
print(" - Mixture 20% AR, 30% O2, 50% H2O")
for t_idx, t in enumerate(T_range):
    for p_idx, p in enumerate(P_range):
        gas.TPX = t, p*ct.one_atm, "AR:0.2, H2O: 0.5, O2: 0.3"
        k_matrix[p_idx, t_idx] = gas.forward_rate_constants[1]


data_directory = os.path.join(current_dir, "cantera_data", str(ct_version))
if not os.path.isdir(data_directory):
    os.makedirs(data_directory)

np.savetxt(os.path.join(data_directory, "falloff_troe_ar_o2_h2o.csv"), k_matrix, delimiter=";", fmt="%.10e")
