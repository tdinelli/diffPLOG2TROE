# CLAUDE.md - AI Agent Instructions for KiRATE

This repository contains KiRATE (Kinetic Rate Analysis and Tuning Environment), a gradient-processing and kinetic analysis library built on JAX for chemical reaction kinetics research. The codebase emphasizes automatic differentiation, CHEMKIN compatibility, and scientifically validated implementations.

## Core Principles

1. **Numerical correctness is paramount** - Scientific accuracy and autodiff compatibility are non-negotiable
2. **CHEMKIN compatibility matters** - Industry-standard format support is critical for adoption
3. **JAX patterns must be respected** - Proper use of pytrees, gradients, and JIT compilation
4. **Validation tests are sacrosanct** - Cantera reference tests prove correctness
5. **Small, focused changes** - One concern per change; avoid bundling unrelated work
6. **Conservative approach** - When in doubt, ask the user before proceeding

## Critical Rules

### Protected Data Files (DO NOT MODIFY)
- **NEVER** modify validation test reference data without explicit user instruction
- **NEVER** modify `tests/cantera/cantera_data/reference_mech.yaml` - validated reference mechanism
- **NEVER** modify `KiRATE/species/atomic_weights_db.py` - NIST/CODATA atomic weight database
- **NEVER** modify NASA7 polynomial coefficients in test data
- **AVOID** changing any files in `tests/cantera/` without user confirmation

### JAX & Automatic Differentiation (CRITICAL)
- **ALWAYS** use `jnp` (JAX NumPy) instead of `np` for numeric operations that participate in gradients
- **ALWAYS** mark non-differentiable fields with `eqx.field(static=True)` (e.g., names, compositions)
- **NEVER** introduce operations that break gradient flow (e.g., in-place mutations, non-JAX libraries)
- **NEVER** use side effects inside `@jax.jit` decorated functions
- **VERIFY** that changes preserve autodiff compatibility by testing gradient computation
- **UNDERSTAND** pytree structure - `eqx.Module` classes must maintain pytree compatibility
- **BE CAREFUL** with array shapes - broadcasting and reshaping can affect gradient dimensions

### Algorithmic Changes (ASK FIRST)
- **ASK** before modifying any fitting algorithms in `refitter/`
- **ASK** before changing thermodynamic calculations (NASA7 evaluation, concentration calcs)
- **ASK** before modifying core rate constant implementations (`kinetics/`)
- **ASK** before changing convergence criteria, tolerances, or optimization strategies
- **ASK** before reordering operations that might affect numerical results
- **CALL OUT** any changes that might affect:
  - Floating-point arithmetic ordering
  - Gradient computation paths
  - Optimization convergence behavior
  - Temperature/pressure interpolation logic
  - Thermodynamic property calculations

### Code Quality (STRICT ENFORCEMENT)
- **ALWAYS** use strict type hints with `jaxtyping` (e.g., `Float64[Array, "n"]`)
- **ALWAYS** maintain `mypy` compliance - no type errors allowed
- **ALWAYS** follow `ruff` formatting (120 character line length)
- **AVOID** drive-by formatting or whitespace changes
- **AVOID** restructuring code for style alone
- **KEEP** diffs minimal and focused on the task at hand
- **PRESERVE** existing naming conventions and patterns
- **ADD** tests before or with changes - never commit untested code

### Testing Requirements (MANDATORY)
- **RUN** `pytest` before and after all changes
- **ENSURE** all Cantera validation tests pass
- **ADD** regression tests for bug fixes
- **ADD** validation tests for new features
- **VERIFY** that changes don't break existing functionality
- **TEST** gradient computation for any new rate constant operations
- **DOCUMENT** test cases clearly with docstrings

## Architecture & Layer Boundaries

```
KiRATE/
├── kinetics/           - Rate constant models (JAX-based, autodiff-enabled)
├── refitter/           - Parameter fitting & optimization
│   ├── core/          - Base classes, residuals, statistics, uncertainty
│   ├── methods/       - Linear and nonlinear optimization algorithms
│   └── fitters/       - High-level fitter implementations
├── species/           - Thermodynamic properties (NASA7 polynomials)
└── utilities/         - CHEMKIN parser, constants, concentration calcs

tests/
├── *_test.py          - Unit tests for each module
└── cantera/           - Validation against Cantera reference data

examples/              - Jupyter notebooks demonstrating features
docs/                  - Sphinx documentation
```

**Respect layer boundaries:**
- Do not mix rate constant changes with fitting algorithm changes
- Do not change scientific logic from utility layers
- Keep CHEMKIN parser separate from calculation logic
- Maintain clear separation between thermodynamics and kinetics

## Build System & Dependencies

**Build Tool**: Hatchling (defined in `pyproject.toml`)
**Python Version**: 3.10+
**Primary Language**: Python

**Core Dependencies:**
- **JAX** - Automatic differentiation and JIT compilation
- **Equinox** - JAX-based module system (`eqx.Module`)
- **jaxtyping** - Type hints for JAX arrays
- **optimistix** - Levenberg-Marquardt and advanced optimizers
- **NumPy** - Array operations
- **matplotlib** - Plotting

**Development Tools:**
- **ruff** - Fast linter & formatter (120 char lines)
- **mypy** - Static type checking
- **pytest** - Testing framework
- **pytest-cov** - Code coverage
- **sphinx** - Documentation (with nbsphinx for notebooks)

## Testing

**Run all tests:**
```bash
pytest
```

**Run with coverage:**
```bash
pytest --cov=KiRATE --cov-report=term-missing
```

**Test organization:**
- Unit tests: `tests/*_test.py`
- Cantera validation: `tests/cantera/*.py`
- Reference data: `tests/cantera/cantera_data/`

**CI/CD Pipeline** (`.github/workflows/ci.yml`):
- Runs on: Ubuntu, macOS, Windows
- Python versions: 3.10, 3.11, 3.12, 3.13
- Includes coverage upload to Codecov

## Common Scenarios

### Bug Fixes
1. **Identify root cause** before fixing (ask user if unclear)
2. **Add regression test** to prevent recurrence
3. **Keep fix minimal** and focused
4. **Verify** all tests pass, including Cantera validation
5. **Document** what was broken and how it's fixed
6. **Test gradients** if the bug affects autodiff operations

### Adding New Rate Constant Models
1. **Discuss with user** first - this is a major change
2. **Inherit from** `eqx.Module`
3. **Use JAX arrays** for all numeric parameters
4. **Mark static fields** appropriately (e.g., `name: str = eqx.field(static=True)`)
5. **Implement** `rate_constant(T, P=None)` method
6. **Add validation** against Cantera or literature data
7. **Write unit tests** and gradient tests
8. **Update documentation** and add example notebook

### Refitter/Optimization Changes
1. **ASK USER FIRST** - optimization is sensitive
2. **Understand two-stage fitting**: Linear (log-space initial guess) → Nonlinear (refined)
3. **Preserve** existing convergence behavior unless explicitly improving
4. **Validate** uncertainty propagation if touching Jacobian calculations
5. **Test** with multiple datasets (well-conditioned and ill-conditioned)
6. **Document** any tolerance or convergence changes

### Thermodynamic Property Changes
1. **ASK USER FIRST** - NASA7 evaluation is foundational
2. **Maintain** temperature range handling (Tmin, Tmid, Tmax)
3. **Preserve** existing polynomial evaluation logic
4. **Test** against Cantera species data
5. **Verify** gradient flow through thermodynamic calculations

### CHEMKIN Parser Changes
1. **Discuss with user** first - compatibility is critical
2. **Test** with real-world CHEMKIN mechanism files
3. **Preserve** existing parsing behavior
4. **Handle** edge cases (comments, whitespace, format variants)
5. **Update** tests with new examples if needed

### Documentation & Examples
When adding new features:
- **Update** relevant API documentation in docstrings
- **Add or update** example notebooks in `examples/` if the feature is user-facing
- **Keep notebooks** simple and focused on demonstrating the feature
- **Run notebooks** to ensure they execute without errors

## JAX-Specific Guidelines

### Static vs. Dynamic Fields
```python
import equinox as eqx
from jaxtyping import Float, Array

class RateConstant(eqx.Module):
    # Static fields (don't participate in gradients)
    name: str = eqx.field(static=True)
    reaction_type: str = eqx.field(static=True)

    # Dynamic fields (JAX arrays, participate in gradients)
    A: Float[Array, ""]  # Pre-exponential factor
    n: Float[Array, ""]  # Temperature exponent
    Ea: Float[Array, ""] # Activation energy
```

### JIT Compilation
```python
import jax

# Good: Pure function, no side effects
@jax.jit
def compute_rate(T, A, n, Ea):
    return A * T**n * jnp.exp(-Ea / (R * T))

# Bad: Side effects inside JIT
@jax.jit
def compute_rate_bad(T, A, n, Ea):
    print(f"Computing at T={T}")  # Side effect!
    return A * T**n * jnp.exp(-Ea / (R * T))
```

### Array Shape Handling
```python
from jaxtyping import Float, Array

# Be explicit about shapes
def rate_constant(
    self,
    T: Float[Array, " *batch"],  # Supports scalar or batched
    P: Float[Array, " *batch"] | None = None
) -> Float[Array, " *batch"]:
    # Ensure operations preserve batch dimensions
    return self.A * T**self.n * jnp.exp(-self.Ea / (R * T))
```

### Testing Gradients
```python
import jax

# Always test that gradients work
def test_arrhenius_gradient():
    k = Arrhenius(A=1e14, n=0.0, Ea=5000.0)
    T = jnp.array(1000.0)

    # Test gradient computation
    grad_fn = jax.grad(lambda t: k.rate_constant(t))
    grad_val = grad_fn(T)

    assert jnp.isfinite(grad_val), "Gradient must be finite"
```

## What to Call Out

**Always explicitly mention if your change involves:**
- Breaking gradient flow or autodiff compatibility
- Modifying JAX array operations or shapes
- Changing optimization algorithms or convergence criteria
- Reordering floating-point operations
- Modifying thermodynamic calculations (NASA7, concentrations)
- Changing CHEMKIN parser behavior
- Updating validation test expectations
- Changes that might affect numerical reproducibility
- Modifications to static/dynamic field marking in `eqx.Module` classes

## Common Pitfalls to Avoid

1. **Using NumPy instead of JAX NumPy** in gradient-dependent code
2. **Forgetting to mark static fields** in `eqx.Module` classes
3. **Modifying test reference data** without understanding validation strategy
4. **Changing optimization algorithms** without validating convergence behavior
5. **Breaking CHEMKIN compatibility** with parser changes
6. **Introducing circular imports** (see recent fixes in commit history)
7. **Side effects in JIT-compiled functions** (print, file I/O, etc.)
8. **Inconsistent array shapes** in batched operations
9. **Missing type hints** or using `Any` types
10. **Drive-by refactoring** unrelated to the task at hand

## Dependencies Update Policy

- **Core dependencies** (JAX, Equinox, NumPy): Discuss with user before upgrading
- **Development tools** (ruff, mypy, pytest): Can update but verify all checks pass
- **Documentation tools** (sphinx, nbsphinx): Safe to update
- **New dependencies**: Must justify and discuss with user first

## Summary

**Priority order:** Numerical correctness = Autodiff compatibility = CHEMKIN compatibility > Test validation > Performance > Style

**Conservative approach:** When uncertain about a change, **ASK THE USER FIRST**.

**Testing is mandatory:** All changes must pass existing tests and include new tests where appropriate.

**JAX patterns are critical:** Understand pytrees, static fields, JIT compilation, and gradient flow.

---

For detailed contribution guidelines, build instructions, and usage examples, refer to:
- [README.md](README.md) - Project overview and installation
- [CONTRIBUTING.md](CONTRIBUTING.md) - Detailed contribution guidelines (if available)
- [docs/](docs/) - Full documentation and API reference
- [examples/](examples/) - Jupyter notebooks demonstrating features
