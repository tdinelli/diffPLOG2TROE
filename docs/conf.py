# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "KiRATE"
copyright = "2026, Timoteo Dinelli"
author = "Timoteo Dinelli"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",  # Markdown support
    "nbsphinx",  # Jupyter notebook support
    "nbsphinx_link",  # Link to notebooks outside docs directory
]

# MyST parser configuration (Markdown support)
myst_enable_extensions = [
    "dollarmath",  # $inline$ and $$display$$ math
    "amsmath",  # Advanced math
    "deflist",  # Definition lists
    "colon_fence",  # ::: fences
]

# Source file types
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    ".ipynb": "nbsphinx",
}

# Templates
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress specific warnings
suppress_warnings = [
    "ref.citation",  # Suppress duplicate citation reference warnings
    "toc.no_title",  # Suppress missing notebook title warnings (handled by content)
]

# Sidebar settings - collapsible navigation
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]

# Logo and favicon
# html_logo = "_static/logo.png"
html_logo = "_static/logokirate_white.png"
html_favicon = "_static/favicon.png"

# Custom CSS
html_css_files = [
    "custom.css",
]

# Furo theme options
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2962ff",
        "color-brand-content": "#2962ff",
    },
    "dark_css_variables": {
        "color-brand-primary": "#448aff",
        "color-brand-content": "#448aff",
    },
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/tdinelli/KiRATE",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings (NumPy/Google docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "exclude-members": "__weakref__",
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Autosummary
autosummary_generate = True

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# MathJax
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

# Copybutton
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# nbsphinx
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True

# Allow nbsphinx to find notebooks outside the docs directory
exclude_patterns.extend([
    os.path.join('..', 'examples', '_stuff'),  # Exclude work-in-progress notebooks
    os.path.join('..', 'examples', '.ipynb_checkpoints'),
    os.path.join('..', 'examples', 'rate_constant', '.ipynb_checkpoints'),
])

# Add link to notebooks using nbsphinx-link
import glob
import json

# Create .nblink files for each notebook we want to include
# Paths are relative to the .nblink file location
nblink_notebooks = [
    ('examples/rate_constant/Arrhenius.nblink', '../../../examples/rate_constant/Arrhenius.ipynb'),
    ('examples/rate_constant/reparameter.nblink', '../../../examples/rate_constant/reparameter.ipynb'),
    ('examples/species_thermodynamics.nblink', '../../examples/species_thermodynamics.ipynb'),
    ('examples/rate_constant/PLOG.nblink', '../../../examples/rate_constant/PLOG.ipynb'),
    ('examples/rate_constant/FallOff.nblink', '../../../examples/rate_constant/FallOff.ipynb'),
    ('examples/rate_constant/CABR.nblink', '../../../examples/rate_constant/CABR.ipynb'),
    ('examples/rate_constant/Threebody.nblink', '../../../examples/rate_constant/Threebody.ipynb'),
    ('examples/rate_constant/MixtureRule.nblink', '../../../examples/rate_constant/MixtureRule.ipynb'),
]

# Create nblink files during configuration
for nblink_path, notebook_path in nblink_notebooks:
    full_nblink_path = os.path.join(os.path.dirname(__file__), nblink_path)
    os.makedirs(os.path.dirname(full_nblink_path), exist_ok=True)
    with open(full_nblink_path, 'w') as f:
        json.dump({"path": notebook_path}, f)
