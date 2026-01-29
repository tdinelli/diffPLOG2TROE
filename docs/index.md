# KiRATE Documentation

<!-- ```{image} _static/logo.png -->
<!-- :alt: KiRATE Logo -->
<!-- :width: 250px -->
<!-- :align: left -->
<!-- ``` -->


Welcome to **KiRATE** (**K**inetic **A**nalysis and **R**ate **T**uning **E**nvironment), a modern Python library for kinetic analysis and rate constant optimization built on JAX. KiRATE provides a comprehensive suite of tools for working with chemical reaction kinetics:

- **Pressure-dependent rate constants**: PLOG, FallOff (Lindemann, Troe, SRI), CABR, Chebyshev
- **Temperature-dependent rate constants**: Arrhenius, Reparametrized Arrhenius
- **Mixture rules**: LMR-P and LMR-R for multi-collider reactions
- **Fully differentiable**: Built on JAX for gradient-based optimization
- **CHEMKIN/Cantera compatible**: Read and evaluate mechanisms from standard formats

## Quick Example

```python
import jax.numpy as jnp
from KiRATE.kinetics import Arrhenius

# Create an Arrhenius rate constant
rate = Arrhenius(
    parameters={"A": 1.0e13, "n": 0.0, "Ea": 50000},
    name="H + O2 = OH + O"
)

# Evaluate at 1000 K
k = rate.rate_constant(T=1000.0)
print(f"k(1000 K) = {k:.3e} cm³/mol/s")
```

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting-started/installation
getting-started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

guide/overview
guide/rate-constants/arrhenius
guide/rate-constants/plog
guide/rate-constants/falloff
guide/rate-constants/cabr
guide/rate-constants/chebyshev
guide/rate-constants/mixture-rule
```

```{toctree}
:maxdepth: 2
:caption: Examples

examples/basic
examples/advanced
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/kinetics
api/species
api/utilities
```

```{toctree}
:maxdepth: 1
:caption: Development

contributing
```

## Indices

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
