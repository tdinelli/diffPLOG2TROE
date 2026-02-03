# KiRATE

**KiRATE** (**Ki**netic **R**ate **A**nalysis and **T**uning **E**nvironment) is a gradient processing and kinetic analysis library for JAX, designed to facilitate research in chemical reaction kinetics.

KiRATE is built on top of [JAX](https://jax.readthedocs.io/en/latest/). See the [JAX documentation](https://jax.readthedocs.io/en/latest/) for the details.

## Features

KiRATE provides a comprehensive suite of composable building blocks for working with chemical reaction kinetics:

- **Temperature-dependent rate constants**: Arrhenius, Reparametrized Arrhenius
- **Pressure-dependent rate constants**: PLOG, FallOff (Lindemann, Troe, SRI), CABR, Chebyshev
- **Mixture rules**: LMR-P and LMR-R for multi-collider reactions
- **Three-body reactions**: Configurable third-body efficiencies
- **Fully differentiable**: Built on JAX for automatic differentiation and gradient-based optimization
- **CHEMKIN/Cantera compatible**: Read and evaluate mechanisms from standard formats
- **Uncertainty quantification**: Propagate uncertainties through kinetic calculations
- **Rate constant refitting**: Optimize parameters to match experimental or computational data
- **Composable design**: Mix and match components to build custom kinetic models
<!-- - **GPU acceleration**: Leverage JAX's hardware acceleration for large-scale computations -->

## Installation

You can install the latest released version of KiRATE from PyPI via:

```bash
pip install KiRATE
```

or you can install the latest development version from GitHub:

```bash
pip install git+https://github.com/tdinelli/KiRATE.git
```

## Quick Example

```python
from KiRATE.kinetics import FallOff

# Create a rate constant object
reaction = FallOff(
    name="2CH3(+M)=C2H6(+M)",
    hpl_parameters={"A": 9.030E+16, "n": -1.180, "Ea": 654.00},
    lpl_parameters={"A": 3.180E+41, "n": -7.030, "Ea": 2762.00},
    falloff_parameters={"A": 0.6041, "T3": 6927, "T1": 132.00, "T2": 2762.00},
    falloff_type="troe",
    efficiencies={"H2": 2, "CO": 2, "CO2": 3, "H2O": 5},
)

# Evaluate the rate constant at 1000 K, 1 atm
k = rate.rate_constant(T=1000.0, P=1.00)
print(f"k(1000 K, 1 atm) = {k:.3e} cm³/mol/s")
```


## Citing KiRATE

If you use KiRATE in your research, please cite:

```bibtex
@software{kirate2025,
  author = {Dinelli, Timoteo},
  title = {KiRATE: Kinetic Rate Analysis and Tuning Environment},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/tdinelli/KiRATE}
}
```

**Find KiRATE useful?** [Star us on GitHub](https://github.com/tdinelli/KiRATE) to support the project!

## About

KiRATE was developed by [Timoteo Dinelli](https://github.com/tdinelli) at the [CRECK Modeling Lab](https://creckmodeling.polimi.it/), Politecnico di Milano.

**Want a cool logo like ours?** Contact [Alessia Contessi](https://alessiacontessi.framer.website/) for professional design services.

---

```{toctree}
:maxdepth: 1
:caption: 🚀 Getting Started

getting-started/installation
getting-started/quickstart
```

```{toctree}
:maxdepth: 1
:caption: 📖 User Guide

guide/overview
guide/rate-constants/arrhenius
guide/rate-constants/plog
guide/rate-constants/falloff
guide/rate-constants/cabr
guide/rate-constants/chebyshev
guide/rate-constants/mixture-rule
```

```{toctree}
:maxdepth: 1
:caption: 💡 Examples

examples/basic
examples/advanced
```

```{toctree}
:maxdepth: 1
:caption: 📚 API Reference

api/kinetics
api/species
api/utilities
```

```{toctree}
:maxdepth: 1
:caption: 🛠️ Development

contributing
```
