# KiRATE
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://kirate.readthedocs.io/)

**KiRATE** (**Ki**netic **R**ate **A**nalysis and **T**uning **E**nvironment) is a gradient processing and kinetic analysis library for JAX, designed to facilitate research in chemical reaction kinetics.

---

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

## Installation

### From PyPI
**TODO**

### From GitHub (latest development version)

```bash
pip install git+https://github.com/tdinelli/KiRATE.git
```

### For development

```bash
git clone https://github.com/tdinelli/KiRATE.git
cd KiRATE
pip install -e ".[dev]"
```

## Quick Start

### Fall-off Rate Constant with Third-Body Efficiencies

```python
from KiRATE.kinetics import FallOff

# Create a fall-off rate constant
reaction = FallOff(
    name="2CH3(+M)=C2H6(+M)",
    hpl_parameters={"A": 9.030E+16, "n": -1.180, "Ea": 654.00},
    lpl_parameters={"A": 3.180E+41, "n": -7.030, "Ea": 2762.00},
    falloff_parameters={"A": 0.6041, "T3": 6927, "T1": 132.00, "T2": 2762.00},
    falloff_type="troe",
    efficiencies={"H2": 2, "CO": 2, "CO2": 3, "H2O": 5},
)

# Evaluate at 1000 K, 1 atm
k = reaction.rate_constant(T=1000.0, P=1.0)
print(f"k(1000 K, 1 atm) = {k:.3e} cm³/mol/s")
```

### PLOG Rate Constants

```python
from KiRATE.kinetics import Plog

# Pressure-dependent rate constants using PLOG interpolation
plog = Plog(
    parameters={
        0.01: {"A": 5.02e21, "n": -4.24, "Ea": 898.9},
        1.0:  {"A": 3.09e23, "n": -4.17, "Ea": 1621.0},
        100.0: {"A": 7.29e22, "n": -3.41, "Ea": 2660.0},
    },
    name="OH+NO=HONO"
)

# Compute rate at specific T and P
k = plog.rate_constant(T=1500.0, P=10.0)
print(f"k(1500 K, 10 atm) = {k:.3e}")
```

## Documentation

Full documentation is available at: [https://kirate.readthedocs.io/](https://kirate.readthedocs.io/)

- [Installation Guide](https://kirate.readthedocs.io/getting-started/installation.html)
- [Quickstart Tutorial](https://kirate.readthedocs.io/getting-started/quickstart.html)
- [API Reference](https://kirate.readthedocs.io/api/kinetics.html)
- [Examples](https://kirate.readthedocs.io/examples/basic.html)

## Citation

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

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/contributing.md) for guidelines.

## About

KiRATE was developed by [Timoteo Dinelli](https://github.com/tdinelli) at the [CRECK Modeling Lab](https://creckmodeling.polimi.it/), Politecnico di Milano.

**Want a cool logo like ours?** Contact [Alessia Contessi](https://alessiacontessi.framer.website/) for professional design services.

## License

MIT License - see [LICENSE](LICENSE) file for details.
