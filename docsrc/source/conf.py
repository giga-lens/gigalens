# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------

project = "GIGA-Lens"
copyright = "2021, Andi Gu"
author = "Andi Gu"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx_toolbox.collapse",
]

bibtex_bibfiles = ["./refs.bib"]
bibtex_default_style = "unsrtalpha"
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "tf": (
        "https://www.tensorflow.org/api_docs/python",
        "https://github.com/GPflow/tensorflow-intersphinx/raw/master/tf2_py_objects.inv",
    ),
    "tfp": (
        "https://www.tensorflow.org/probability/api_docs/python",
        "https://github.com/GPflow/tensorflow-intersphinx/raw/master/tfp_py_objects.inv",
    ),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Extension configuration -------------------------------------------------
