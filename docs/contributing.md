# Contributing to KiRATE

Thank you for your interest in contributing to KiRATE!

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR-USERNAME/KiRATE.git
cd KiRATE
```

3. Create a virtual environment and install in development mode:

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -e ".[dev]"
```

## Running Tests

Run the test suite:

```bash
pytest tests/
```

With coverage:

```bash
pytest --cov=KiRATE tests/
```

## Code Style

We follow Python best practices:

- **Type hints** for all function signatures
- **NumPy-style docstrings** with full RST formatting (Parameters, Returns, Raises, Notes, Examples sections)
- **Black** for code formatting (applied automatically)
- **Maximum line length**: 120 characters
- **Copyright headers** in all new files:
  ```python
  """
  Copyright (c) 2024-2026 Timoteo Dinelli
  Licensed under the MIT License - see LICENSE file for details
  """
  ```

## Documentation

KiRATE uses Sphinx for documentation. Build documentation locally:

```bash
cd docs
make html
```

Then open `docs/_build/html/index.html` in your browser.

For live reload during development:

```bash
cd docs
make livehtml
```

This will automatically rebuild the docs when files change and open your browser at http://127.0.0.1:8000.

### Documentation Guidelines

- All public functions and classes must have comprehensive NumPy-style docstrings
- Include mathematical formulas using LaTeX notation (e.g., `:math:` role or `.. math::` directive)
- Add code examples in docstrings where appropriate
- Reference scientific papers using the References section

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure:
   - All tests pass (`pytest tests/`)
   - Code is formatted (`black KiRATE/`)
   - Type hints are present
   - Documentation is updated

3. Commit your changes with descriptive messages:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Open a Pull Request on GitHub with:
   - Clear description of changes
   - Link to relevant issues
   - Screenshots/examples if applicable

## Reporting Issues

Use the [GitHub issue tracker](https://github.com/tdinelli/KiRATE/issues) to report bugs or request features.

When reporting bugs, please include:
- KiRATE version
- Python version
- JAX version
- Minimal reproducible example
- Expected vs actual behavior

## Questions?

Feel free to reach out:
- [GitHub Discussions](https://github.com/tdinelli/KiRATE/discussions)
- Email: timoteo.dinelli@polimi.it
