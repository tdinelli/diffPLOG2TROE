# Quick Start

This guide will get you up and running with KiRATE in 5 minutes.

## Basic Arrhenius Rate Constant

The simplest rate constant type is Arrhenius:

\\[ k(T) = A T^n \\exp\\left(-\\frac{E_a}{RT}\\right) \\]

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

## Mixture Rules for Multi-Collider Systems

For reactions with different collision partners:

```python
from KiRATE.kinetics import MixtureRule

# Define collision efficiencies
h2_eff = Arrhenius(parameters={"A": 2.0, "n": 0, "Ea": 0})
h2o_eff = Arrhenius(parameters={"A": 17.6, "n": 0, "Ea": 0})

# Create mixture rule
mixture = MixtureRule(
    name="H + O2 (+M) = HO2 (+M)",
    default_rate_constant=falloff_rate,
    efficiencies={"H2": h2_eff, "H2O": h2o_eff},
    linear=True,
    reduced_pressure=False  # LMR-P formulation
)

# Evaluate for specific gas composition
composition = {"N2": 0.79, "O2": 0.19, "H2O": 0.02}
k_mixture = mixture.rate_constant(T=1000.0, P=1.0, composition=composition)
print(f"Mixed rate constant: {k_mixture:.3e}")
```

## Automatic Differentiation

Compute gradients for optimization:

```python
import jax

# Gradient with respect to temperature
def rate_func(T):
    return rate.rate_constant(T=T)

grad_func = jax.grad(rate_func)
dk_dT = grad_func(1000.0)
print(f"dk/dT at 1000 K = {dk_dT:.3e}")

# Gradient with respect to parameters
def param_rate(A):
    r = Arrhenius(parameters={"A": A, "n": 0.0, "Ea": 50000})
    return r.rate_constant(T=1000.0)

dkdA = jax.grad(param_rate)
gradient = dkdA(1.0e13)
print(f"dk/dA = {gradient:.3e}")
```

## Working with Experimental Data

Fitting rate constants to data:

```python
import optax

# Experimental data (T in K, k in cm³/mol/s)
T_exp = jnp.array([800, 1000, 1200, 1400, 1600])
k_exp = jnp.array([1.2e11, 4.5e11, 1.1e12, 2.3e12, 4.1e12])

# Define loss function
def loss(params):
    A, n, Ea = params
    rate = Arrhenius(parameters={"A": A, "n": n, "Ea": Ea})
    k_pred = rate.rate_constant(T=T_exp)
    return jnp.mean((jnp.log(k_pred) - jnp.log(k_exp))**2)

# Optimize using optax
optimizer = optax.adam(learning_rate=0.01)
params = jnp.array([1e13, 0.0, 50000.0])
opt_state = optimizer.init(params)

for i in range(1000):
    grads = jax.grad(loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    if i % 100 == 0:
        print(f"Step {i}: Loss = {loss(params):.6f}")

print(f"\nOptimized parameters:")
print(f"A = {params[0]:.3e}, n = {params[1]:.3f}, Ea = {params[2]:.1f}")
```

## Next Steps

Now that you've seen the basics, explore:

- [User Guide](../guide/overview.md) - Detailed tutorials for each rate constant type
- [Mixture Rules Guide](../guide/rate-constants/mixture-rule.md) - Advanced mixture rule usage
- [API Reference](../api/kinetics.md) - Complete API documentation
- [Examples](../examples/basic.md) - More practical examples
