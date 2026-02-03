# Quick Start

This guide will get you up and running with KiRATE in 5 minutes.

## Basic Arrhenius Rate Constant

The simplest rate constant type is Arrhenius:

$$
k(T) = A T^n \exp\left(-\frac{E_a}{RT}\right)
$$

```python
from KiRATE.kinetics import Arrhenius

# Create an Arrhenius rate constant
rate = Arrhenius(
    parameters={"A": 1.0e13, "n": 0.0, "Ea": 50000},  # Ea in cal/mol
    name="H + O2 = OH + O"
)

# Evaluate at a single temperature
k_1000K = rate.rate_constant(T=1000.0)  # K
print(f"k(1000 K) = {k_1000K:.3e} cm³/mol/s")
```

## Vectorized Evaluation

Evaluate over temperature ranges efficiently:

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Temperature range from 500 to 2500 K
T_range = jnp.linspace(500, 2500, 100)

# Vectorized evaluation
k_values = rate.rate_constant(T=T_range)

# Arrhenius plot
plt.semilogy(1000/T_range, k_values)
plt.xlabel('1000/T (K⁻¹)')
plt.ylabel('Rate constant (cm³/mol/s)')
plt.title('Arrhenius Plot')
plt.show()
```

## Pressure-Dependent Rates: PLOG

For pressure-dependent reactions:

```python
from KiRATE.kinetics import Plog

# PLOG rate constant with multiple pressure points
plog_rate = Plog(
    name="H + O2 (+Ar) = HO2 (+Ar)",
    parameters={
        0.01: {"A": 8.45e14, "n": -2.19, "Ea": 11.4},    # 0.01 atm
        1.00: {"A": 8.39e16, "n": -2.19, "Ea": 61.3},    # 1 atm
        100.: {"A": 1.06e21, "n": -2.82, "Ea": 1192.0},  # 100 atm
    }
)

# Evaluate at specific T and P
k = plog_rate.rate_constant(T=1000.0, P=1.0)
print(f"k(1000 K, 1 atm) = {k:.3e}")

# Create T-P grid
T_grid = jnp.linspace(500, 2500, 50)
P_grid = jnp.logspace(-2, 2, 40)  # 0.01 to 100 atm

k_grid = plog_rate.rate_constant(T=T_grid, P=P_grid)
print(f"Grid shape: {k_grid.shape}")  # (50, 40)
```

## FallOff Reactions

Lindemann-Troe falloff formalism:

```python
from KiRATE.kinetics import FallOff

falloff_rate = FallOff(
    name="H + O2 (+M) = HO2 (+M)",
    hpl_parameters={"A": 4.66e12, "n": 0.44, "Ea": 0.0},        # High-pressure limit
    lpl_parameters={"A": 4.07e19, "n": -1.4, "Ea": -180.5},     # Low-pressure limit
    falloff_parameters={"A": 0.5, "T3": 1.0, "T1": 1e10, "T2": 1e30},  # Troe parameters
    falloff_type="troe"
)

# Falloff curve at fixed temperature
P_range = jnp.logspace(-3, 2, 100)  # 0.001 to 100 atm
k_falloff = falloff_rate.rate_constant(T=1000.0, P=P_range)

plt.loglog(P_range, k_falloff)
plt.xlabel('Pressure (atm)')
plt.ylabel('Rate constant')
plt.title('Falloff Curve at 1000 K')
plt.show()
```

## Next Steps

Now that you've seen the basics, explore:

- [User Guide](../guide/overview.md) - Detailed tutorials for each rate constant type
- [Mixture Rules Guide](../guide/rate-constants/mixture-rule.md) - Advanced mixture rule usage
- [API Reference](../api/kinetics.md) - Complete API documentation
- [Examples](../examples/basic.md) - More practical examples
