# -- Project info -----------------------------------------------------
project = "TiRex Python API"
author = "NXAI"
extensions = [
    "myst_parser",  # allow Markdown sources if needed
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # built-in, no pip install
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",  # from sphinx-autodoc-typehints
    "sphinx_copybutton",  # from sphinx-copybutton
    "sphinx_design",  # from sphinx-design
    "sphinx_markdown_builder",  # enable Markdown output
]
autosummary_generate = True
autodoc_default_options = {}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_preserve_defaults = True
autodoc_mock_imports = [
    "sklearn",  # classification/regression extra (train_test_split, RF, etc.)
    "joblib",  # RandomForest serialization dependency
    "lightgbm",  # GBM classifier/regressor dependency
]  # mock optional deps so autodoc works without installing extras
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_examples = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Make sure "src" is on sys.path so `import tirex` resolves
import os
import sys
from textwrap import dedent

ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

# Sphinx general config
templates_path = ["_templates"]
exclude_patterns = []

# Output (HTML theme irrelevant for Markdown builder)
html_theme = "alabaster"
