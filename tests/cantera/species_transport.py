import os

import cantera as ct
import numpy as np

ct_version = ct.__version__
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

# Load gas phase with transport - use GRI-Mech 3.0 which has transport data
# This is a standard mechanism included with Cantera
gas = ct.Solution("gri30.yaml")

# Temperature range for testing
T_range = np.linspace(300, 3000, 300)

# Select representative species with different geometries
# CH4: nonlinear, H2O: nonlinear, H2: linear, AR: monatomic
test_species_names = ["CH4", "H2O", "H2", "AR"]

print(f"Extracting transport data for species: {test_species_names}")
print(f" - Temperature range: {T_range[0]:.1f} - {T_range[-1]:.1f} K")
print(f" - Cantera version: {ct_version}")

# Storage for all species data
all_species_data = []

for species_name in test_species_names:
    species_index = gas.species_index(species_name)
    species = gas.species(species_index)

    # Get transport parameters
    # Cantera stores epsilon in Joules, convert to epsilon/k_B in Kelvin
    k_B = 1.380649e-23  # Boltzmann constant [J/K]
    epsilon_over_k = species.transport.well_depth / k_B if species.transport.well_depth > 0 else 0.0
    sigma_angstrom = species.transport.diameter * 1e10  # Convert m to Angstrom

    print(f"\nProcessing {species_name}...")
    print(f"  Geometry: {species.transport.geometry}")
    print(f"  LJ epsilon/k_B: {epsilon_over_k:.3f} K")
    print(f"  LJ sigma: {sigma_angstrom:.3f} Angstrom")

    # Storage for this species
    species_data = []

    for t in T_range:
        gas.TP = t, ct.one_atm

        # Get transport properties at this temperature
        # viscosity [Pa·s], thermal_conductivity [W/(m·K)]
        mu = gas.viscosity  # Mixture viscosity - we'll use species-specific later
        lambda_cond = gas.thermal_conductivity  # Mixture thermal conductivity

        # For pure species, set mole fraction to 1.0 for this species only
        gas.X = {species_name: 1.0}

        # Get pure species transport properties
        mu_pure = gas.viscosity  # Pa·s
        lambda_pure = gas.thermal_conductivity  # W/(m·K)

        species_data.append([t, mu_pure, lambda_pure])

    species_data = np.array(species_data)
    all_species_data.append(species_data)

# Save data for each species
data_directory = os.path.join(current_dir, "cantera_data", str(ct_version))
if not os.path.isdir(data_directory):
    os.makedirs(data_directory)

for species_name, species_data in zip(test_species_names, all_species_data):
    output_file = os.path.join(data_directory, f"transport_{species_name.lower()}.csv")
    header = "T[K];viscosity[Pa*s];thermal_conductivity[W/(m*K)]"
    np.savetxt(output_file, species_data, delimiter=";", fmt="%.10e", header=header, comments="")
    print(f"Saved {species_name} transport data to {output_file}")

print("\n=== Transport Reference Data Generation Complete ===")
