# User Guide Overview

Welcome to the KiRATE user guide. This comprehensive guide covers all aspects of working with KiRATE.

## What You'll Learn

This guide is organized into the following sections:

### Rate Constants

Learn about each rate constant type supported by KiRATE:

- **[Arrhenius](rate-constants/arrhenius.md)** - Basic temperature-dependent rates
- **[PLOG](rate-constants/plog.md)** - Pressure-logarithmic interpolation
- **[FallOff](rate-constants/falloff.md)** - Lindemann, Troe, and SRI falloff
- **[CABR](rate-constants/cabr.md)** - Chemically Activated Bimolecular Reactions
- **[Chebyshev](rate-constants/chebyshev.md)** - Chebyshev polynomial representation
- **[Mixture Rules](rate-constants/mixture-rule.md)** - Multi-collider systems (LMR-P, LMR-R)

## Core Concepts

### Temperature and Pressure Units

KiRATE uses the following units consistently:

- **Temperature**: Kelvin (K)
- **Pressure**: Atmospheres (atm)
- **Energy**: Calories per mole (cal/mol) for activation energies
- **Rate constants**: cm³/mol/s (bimolecular) or s⁻¹ (unimolecular)

### Vectorized Evaluation

All rate constant types support vectorized evaluation using JAX arrays:

```python
import jax.numpy as jnp

# Single point
k = rate.rate_constant(T=1000.0, P=1.0)

# Temperature array
T_range = jnp.linspace(500, 2500, 100)
k_array = rate.rate_constant(T=T_range, P=1.0)

# T-P grid
P_range = jnp.logspace(-2, 2, 50)
k_grid = rate.rate_constant(T=T_range, P=P_range)  # Shape: (100, 50)
```

### Automatic Differentiation

All rate constants are fully differentiable:

```python
import jax

# Gradient with respect to temperature
dk_dT = jax.grad(lambda T: rate.rate_constant(T, P=1.0))(1000.0)

# Hessian
d2k_dT2 = jax.hessian(lambda T: rate.rate_constant(T, P=1.0))(1000.0)
```

## Common Workflows

### 1. Evaluating Mechanisms

Load and evaluate CHEMKIN mechanisms:

```python
# Coming soon: CHEMKIN parser integration
```

### 2. Parameter Fitting

Fit rate constants to experimental data:

```python
from KiRATE.refitter import ArrheniusRefitter

# Coming soon
```

### 3. Uncertainty Propagation

Propagate parameter uncertainties:

```python
# Coming soon
```

## Best Practices

### Performance Tips

1. **Use JAX arrays**: Always use `jax.numpy` instead of regular `numpy` for best performance
2. **JIT compilation**: First call will be slow (compilation), subsequent calls are fast
3. **Batch operations**: Evaluate multiple temperatures/pressures at once instead of looping
4. **GPU acceleration**: Install JAX with GPU support for large-scale computations

### Numerical Stability

1. **Temperature ranges**: Avoid extrapolating far outside fitted ranges
2. **Pressure interpolation**: PLOG uses log(P) interpolation for better accuracy
3. **Activation energies**: Check units (cal/mol vs J/mol)

## Next Steps

Choose a topic from the navigation menu or start with:

- [Arrhenius Rate Constants](rate-constants/arrhenius.md) - Start here if you're new
- [PLOG](rate-constants/plog.md) - For pressure-dependent reactions
- [Mixture Rules](rate-constants/mixture-rule.md) - For multi-collider systems
