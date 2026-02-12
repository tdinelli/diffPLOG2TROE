import os

import cantera as ct
import numpy as np

ct_version = ct.__version__
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

# Load mechanism
gas = ct.Solution(os.path.join(current_dir, "reference_mech.yaml"))

# Temperature range for evaluation
T_range = np.linspace(300, 3000, 300)

# Get the first reaction (H2+O<=>H+OH)
reaction_index = 0
reaction = gas.reaction(reaction_index)

print(f"Extracting reverse rate constants for reaction: {reaction.equation}")
print(f" - Temperature range: {T_range[0]:.1f} - {T_range[-1]:.1f} K")

# Compute forward and reverse rate constants
cantera_forward_rates = []
cantera_reverse_rates = []
cantera_equilibrium_constants = []

for t in T_range:
    gas.TP = t, ct.one_atm

    # Forward rate constant
    k_f = gas.forward_rate_constants[reaction_index]
    cantera_forward_rates.append(k_f)

    # Reverse rate constant (computed by Cantera using thermodynamics)
    k_r = gas.reverse_rate_constants[reaction_index]
    cantera_reverse_rates.append(k_r)

    # Equilibrium constant K_c
    K_c = k_f / k_r if k_r > 0 else 0.0
    cantera_equilibrium_constants.append(K_c)

# Convert to arrays
cantera_forward_rates = np.array(cantera_forward_rates)
cantera_reverse_rates = np.array(cantera_reverse_rates)
cantera_equilibrium_constants = np.array(cantera_equilibrium_constants)

# Combine data: T, k_f, k_r, K_c
combined_array = np.column_stack((T_range, cantera_forward_rates, cantera_reverse_rates, cantera_equilibrium_constants))

# Save to CSV
data_directory = os.path.join(current_dir, "cantera_data", str(ct_version))
if not os.path.isdir(data_directory):
    os.makedirs(data_directory)

np.savetxt(
    os.path.join(data_directory, "reverse_arrhenius.csv"),
    combined_array,
    delimiter=";",
    fmt="%.10e",
    header="Temperature [K];Forward Rate [m3/kmol/s];Reverse Rate [m3/kmol/s];Equilibrium Constant K_c [dimensionless]",
    comments="",
)
