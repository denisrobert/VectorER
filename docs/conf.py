"""Sphinx configuration for the vector-er documentation.

Build locally with::

    pip install -e ".[docs]"
    sphinx-build -b html docs docs/_build/html
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the package is importable when building from a source checkout
# without installing (docs are built from the repo root).
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

project = "vector-er"
copyright = "2026, vector-er contributors"
author = "vector-er contributors"

# Import the package version dynamically (matches pyproject.toml).
from vectorer import __version__

version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

# Napoleon: parse NumPy/Google-style docstrings.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Intersphinx: link to Python's stdlib docs.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML theme.
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "vector-er"
html_short_title = "vector-er"

# Branding: the vector-er logo (glyph "ER" with a vector macron overline).
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"

# autosummary
autosummary_generate = True

# Optional: point at the README for the index if preferred.  Kept explicit.
master_doc = "index"