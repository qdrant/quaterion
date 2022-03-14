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
# import os
# import sys
#
# sys.path.insert(0, os.path.abspath("../quaterion"))

# -- Project information -----------------------------------------------------

project = "Quaterion"
copyright = "2022, Quaterion Authors"
author = "Quaterion Authors"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]

# prevents sphinx from adding full path to type hints
autodoc_typehints_format = "short"

# order members by type and not alphabetically, it prevents mixing of class attributes
# and methods
autodoc_member_order = "groupwise"

# moves ``Return type`` to ``Returns``
napoleon_use_rtype = False

# If true, suppress the module name of the python reference if it can be resolved.
# Experimental feature:
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-python_use_unqualified_type_names
python_use_unqualified_type_names = True

# prevents sphinx to add full module path in titles
add_module_names = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# prevents unfolding type hints
autodoc_type_aliases = {
    "KeyExtractorType": "KeyExtractorType",
    "TensorInterchange": "TensorInterchange",
    "CacheCollateReturnType": "CacheCollateReturnType",
    "CollateFnType": "CollateFnType",
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "*cache_mixin*",
    "*cache_encoder*",
    "*cache_model*",
    "*cache_train_collater*",
    "*indexing_dataset*",
]
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "quaterion.css",
]
