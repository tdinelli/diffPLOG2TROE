import os

import cantera as ct
import numpy as np


ct_version = ct.__version__
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

# Load gas phase with species
gas = ct.Solution(os.path.join(current_dir, "reference_mech.yaml"))

# Temperature range for testing
T_range = np.linspace(300, 3000, 300)

# Select a few representative species to test
test_species_names = ["CH4", "CH2O"]

print(f"Extracting thermodynamic data for species: {test_species_names}")
print(f" - Temperature range: {T_range[0]:.1f} - {T_range[-1]:.1f} K")

# Storage for all species data
all_species_data = []

for species_name in test_species_names:
    species_index = gas.species_index(species_name)
    species = gas.species(species_index)

    print(f"\nProcessing {species_name}...")

    # Storage for this species
    species_data = []

    for t in T_range:
        gas.TP = t, ct.one_atm

        # Get thermodynamic properties at this temperature
        # Units: Cp [J/kmol/K], H [J/kmol], S [J/kmol/K], G [J/kmol]
        cp = species.thermo.cp(t)  # J/kmol/K
        h = species.thermo.h(t)    # J/kmol
        s = species.thermo.s(t)    # J/kmol/K
        g = h - t * s              # G = H - TS [J/kmol]

        # Dimensionless properties (matching NASA polynomial definitions)
        R = ct.gas_constant  # J/kmol/K
        cp_R = cp / R
        h_RT = h / (R * t)
        s_R = s / R
        g_RT = g / (R * t)

        species_data.append([t, cp, h, s, g, cp_R, h_RT, s_R, g_RT])

    species_data = np.array(species_data)
    all_species_data.append(species_data)

# Save data for each species
data_directory = os.path.join(current_dir, "cantera_data", str(ct_version))
if not os.path.isdir(data_directory):
    os.makedirs(data_directory)

for species_name, species_data in zip(test_species_names, all_species_data):
    output_file = os.path.join(data_directory, f"species_{species_name.lower()}.csv")
    header = "T[K];Cp[J/kmol/K];H[J/kmol];S[J/kmol/K];G[J/kmol];Cp/R;H/(RT);S/R;G/(RT)"
    np.savetxt(output_file, species_data, delimiter=";", fmt="%.10e", header=header, comments="")
    print(f"Saved {species_name} data to {output_file}")
