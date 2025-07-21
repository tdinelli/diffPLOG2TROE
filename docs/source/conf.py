import os
import sys


sys.path.insert(0, os.path.abspath("../../"))

# Project information
project = "diffPLOG2TROE"
copyright = "2024, Timoteo Dinelli"
author = "Timoteo Dinelli"
release = "1.0.0"

# Extensions
extensions = [
    "sphinx.ext.autodoc",  # Auto-generate docs from docstrings
    "sphinx.ext.autosummary",  # Generate summary tables
    "sphinx.ext.viewcode",  # Add source code links
    "sphinx.ext.napoleon",  # Support for NumPy/Google style docstrings
    "sphinx.ext.intersphinx",  # Link to other projects' docs
    "sphinx.ext.mathjax",  # Math support
    "sphinx_autodoc_typehints",  # Type hints support
    "myst_parser",  # Markdown support
    "sphinx_copybutton",  # Copy buttons for code blocks
]

# Templates path
templates_path = ["_templates"]

# List of patterns to ignore when looking for source files
exclude_patterns = []

# HTML theme
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Napoleon settings for docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Autosummary settings
autosummary_generate = True

# Type hints
autodoc_typehints = "description"
always_document_param_types = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# Math support
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
