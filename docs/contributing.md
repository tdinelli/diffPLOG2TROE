# Contributing to KiRATE

Thank you for your interest in contributing to KiRATE!

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR-USERNAME/KiRATE
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

- Type hints for all functions
- NumPy-style docstrings
- Black for code formatting
- Maximum line length: 120 characters

## Documentation

Build documentation locally:

```bash
mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser.

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request on GitHub

## Reporting Issues

Use the [GitHub issue tracker](https://github.com/tdinelli/KiRATE/issues) to report bugs or request features.

## Questions?

Feel free to reach out:
- GitHub Discussions
- Email: timoteo.dinelli@polimi.it
