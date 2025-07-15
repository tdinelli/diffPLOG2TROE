import cantera as ct
import matplotlib.pyplot as plt
import numpy as np


def test_reaction_equivalence(yaml_file_path):
    """
    Test if reactions 9 and 10 give the same rate values.
    Reaction 9: H + O2 (+M) <=> HO2 (+M) with AR efficiency = 1, others = 0
    Reaction 10: H + O2 (+AR) <=> HO2 (+AR) explicit AR as third body
    """

    # Load the gas mechanism
    gas = ct.Solution(yaml_file_path)

    # Find the indices of reactions 9 and 10
    # Note: Cantera uses 0-based indexing, so reaction 9 is index 8, reaction 10 is index 9
    reaction_9_idx = 8  # H + O2 (+M) <=> HO2 (+M) with AR efficiency
    reaction_10_idx = 9  # H + O2 (+AR) <=> HO2 (+AR)

    # Test conditions
    temperatures = np.linspace(300, 2000, 10)  # K
    pressures = np.logspace(-2, 2, 5)  # atm

    # Store results for comparison
    results = []

    for T in temperatures:
        for P in pressures:
            # Set gas state
            gas.TPX = T, P * ct.one_atm, "H:0.1, O2:0.1, AR:0.8"

            # Get forward rate constants for both reactions
            forward_rate_constants = gas.forward_rate_constants
            rate_9 = forward_rate_constants[reaction_9_idx]
            rate_10 = forward_rate_constants[reaction_10_idx]

            # Calculate relative difference
            rel_diff = abs(rate_9 - rate_10) / max(rate_9, rate_10) if max(rate_9, rate_10) > 0 else 0

            results.append(
                {
                    "T": T,
                    "P": P,
                    "rate_9": rate_9,
                    "rate_10": rate_10,
                    "rel_diff": rel_diff,
                    "abs_diff": abs(rate_9 - rate_10),
                }
            )

            print(f"T={T:6.1f}K, P={P:8.3f}atm: Rate_9={rate_9:.3e}, Rate_10={rate_10:.3e}, Rel_diff={rel_diff:.2e}")

    # Summary statistics
    rel_diffs = [r["rel_diff"] for r in results]
    abs_diffs = [r["abs_diff"] for r in results]

    print(f"\nSummary:")
    print(f"Maximum relative difference: {max(rel_diffs):.2e}")
    print(f"Average relative difference: {np.mean(rel_diffs):.2e}")
    print(f"Maximum absolute difference: {max(abs_diffs):.2e}")
    print(f"Average absolute difference: {np.mean(abs_diffs):.2e}")

    # Check if they're equivalent (within numerical precision)
    tolerance = 1e-10
    are_equivalent = max(rel_diffs) < tolerance
    print(f"\nAre reactions equivalent (rel_diff < {tolerance})? {are_equivalent}")

    return results


def plot_comparison(results):
    """Plot the rate constants to visualize the comparison"""

    # Extract data for plotting
    temperatures = [r["T"] for r in results]
    rate_9_values = [r["rate_9"] for r in results]
    rate_10_values = [r["rate_10"] for r in results]
    rel_diffs = [r["rel_diff"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot rate constants
    ax1.semilogy(temperatures, rate_9_values, "bo-", label="Reaction 9", alpha=0.7)
    ax1.semilogy(temperatures, rate_10_values, "ro-", label="Reaction 10", alpha=0.7)
    ax1.set_xlabel("Temperature (K)")
    ax1.set_ylabel("Forward Rate Constant")
    ax1.set_title("Comparison of Forward Rate Constants")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot relative differences
    ax2.semilogy(temperatures, rel_diffs, "go-", alpha=0.7)
    ax2.set_xlabel("Temperature (K)")
    ax2.set_ylabel("Relative Difference")
    ax2.set_title("Relative Difference Between Reactions")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def detailed_reaction_analysis(yaml_file_path):
    """Analyze the reaction mechanisms in detail"""

    gas = ct.Solution(yaml_file_path)

    # Get reactions 9 and 10
    rxn_9 = gas.reaction(8)  # 0-based indexing
    rxn_10 = gas.reaction(9)

    print("Detailed Analysis:")
    print("=" * 50)

    print(f"Reaction 9: {rxn_9}")
    print(f"Type: {type(rxn_9)}")
    if hasattr(rxn_9, "efficiencies"):
        print(f"Efficiencies: {rxn_9.efficiencies}")
    print()

    print(f"Reaction 10: {rxn_10}")
    print(f"Type: {type(rxn_10)}")
    if hasattr(rxn_10, "efficiencies"):
        print(f"Efficiencies: {rxn_10.efficiencies}")
    print()


# Example usage
if __name__ == "__main__":
    # Replace with your actual YAML file path
    yaml_file = "reference_mech.yaml"

    # Detailed analysis of the reactions
    detailed_reaction_analysis(yaml_file)

    # Test equivalence
    results = test_reaction_equivalence(yaml_file)

    # Plot results
    plot_comparison(results)
