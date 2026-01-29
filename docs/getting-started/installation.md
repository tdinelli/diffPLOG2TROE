# Installation

## Requirements

- Python 3.10 or higher
- pip or conda package manager

## Basic Installation

Install KiRATE from PyPI:

```bash
pip install KiRATE
```

This will install KiRATE and its core dependencies:
- `numpy`
- `jax` and `jaxlib`
- `jaxtyping`
- `chex`
- `optax`
- `optimistix`
- `equinox`
- `matplotlib`

## Development Installation

For contributing or development work:

```bash
git clone https://github.com/tdinelli/KiRATE
cd KiRATE
pip install -e ".[dev]"
```

This installs additional development dependencies:
- `pytest` and `pytest-cov` for testing
- MkDocs and plugins for documentation

## Optional Dependencies

### Documentation Building

To build documentation locally:

```bash
pip install -e ".[docs]"
```

### Testing

To run tests:

```bash
pip install -e ".[test]"
```

## Verify Installation

Test your installation:

```python
import KiRATE
from KiRATE.kinetics import Arrhenius

print(f"KiRATE version: {KiRATE.__version__}")

# Create a simple Arrhenius rate constant
rate = Arrhenius(parameters={"A": 1e13, "n": 0, "Ea": 0})
k = rate.rate_constant(T=1000.0)
print(f"Test rate constant: {k:.3e}")
```

Expected output:
```
KiRATE version: 1.0.0
Test rate constant: 1.000e+13
```

## GPU Support

KiRATE uses JAX which supports GPU acceleration. To use GPU:

### CUDA (NVIDIA GPUs)

```bash
pip install --upgrade "jax[cuda12]"
```

### Metal (Apple Silicon)

Metal support is experimental in JAX. Install with:

```bash
pip install --upgrade "jax[metal]"
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install --upgrade numpy jax jaxlib jaxtyping chex optax optimistix equinox matplotlib
```

### JAX Installation Issues

For platform-specific JAX installation, see the [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

### Version Conflicts

If you have dependency conflicts, try creating a fresh virtual environment:

```bash
python -m venv kirate_env
source kirate_env/bin/activate  # On Windows: kirate_env\Scripts\activate
pip install KiRATE
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Learn the basics
- [User Guide](../guide/overview.md) - Comprehensive tutorials
- [API Reference](../api/kinetics.md) - Detailed documentation
