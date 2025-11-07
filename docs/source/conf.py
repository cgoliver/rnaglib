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
from unittest.mock import MagicMock

class MockModule(type):
    """A mock module class that properly handles introspection."""
    def __new__(cls, name):
        # Create a new class that looks like a module
        class MockModuleInstance:
            __module__ = name
            __name__ = name.split('.')[-1]
            __qualname__ = name.split('.')[-1]
            
            def __getattr__(self, attr):
                # Return a mock that has __bases__ attribute
                mock_obj = MagicMock()
                # Ensure __bases__ is accessible
                if not hasattr(mock_obj, '__bases__'):
                    mock_obj.__bases__ = (object,)
                return mock_obj
            
            def __repr__(self):
                return f"<mock module '{name}'>"
        
        instance = MockModuleInstance()
        # Set __bases__ as an attribute on the instance
        instance.__bases__ = (object,)
        return instance

class Mock(MagicMock):
    """Enhanced Mock class that handles introspection better."""
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()
    
    def __getattr__(self, name):
        return MagicMock()
    
    # Add attributes needed for introspection
    __bases__ = (object,)
    __module__ = 'mock'
    __name__ = 'Mock'
    __qualname__ = 'Mock'

# Mock modules that might not be available during doc build
# These must be mocked BEFORE importing rnaglib
# Note: Do NOT mock modules that Sphinx itself needs (like requests, networkx, numpy, etc.)
# Only mock truly optional dependencies that rnaglib can work without
# Do NOT mock modules that are already installed (like rna_fm if it's available)
MOCK_MODULES = [
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.utils',
    'torch.utils.data',
    'torch_geometric',
    'torch_geometric.nn',
    'torch_geometric.utils',
    'torch_scatter',
    'torch_scatter.scatter',
    'dgl',
    'fr3d',
    'fr3d.classifiers',
    'fr3d.classifiers.NA_pairwise_interactions',
    # Don't mock rna_fm if it's installed - let it use real torch.nn
    # 'rna_fm',
    # 'rna_fm.model',
    # 'rna_fm.tokenizer',
]

# Mock all modules before importing rnaglib
# Only mock if the module doesn't already exist
# Check if torch is installed - if so, don't mock it
try:
    import torch
    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False

# Create a new Mock instance for each module to avoid shared state
for mod_name in MOCK_MODULES:
    # Skip mocking torch-related modules if torch is installed
    if TORCH_INSTALLED and mod_name.startswith('torch'):
        continue
        
    if mod_name not in sys.modules:
        # Create a mock module instance that properly handles introspection
        mock_mod = MockModule(mod_name)
        sys.modules[mod_name] = mock_mod
        
        # Special handling for torch.nn.Module - it needs to be a class-like object
        if mod_name == 'torch.nn':
            # Create a mock class that has __bases__ as a class attribute
            class MockModuleClass(object):
                __module__ = 'torch.nn'
                __name__ = 'Module'
                __qualname__ = 'torch.nn.Module'
                
                def __init__(self, *args, **kwargs):
                    pass
                
                def __getattr__(self, attr):
                    return MagicMock()
            
            # Ensure __bases__ is accessible on the class
            MockModuleClass.__bases__ = (object,)
            mock_mod.Module = MockModuleClass

# Add the src directory to sys.path so we can import rnaglib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import rnaglib


# -- Project information -----------------------------------------------------

project = "rnaglib"
copyright = "2021, Vincent Mallet et al."
author = "Vincent Mallet, Carlos Oliver, Jonathan Broadbent, William L. Hamilton, Jerome Waldispuhl"

# The full version, including alpha/beta/rc tags
release = "0.0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

# html_favicon = "images/favicon.png"

myst_enable_extensions = [
    "substitution",
]
extensions += ["sphinx-prompt", "sphinx_substitution_extensions"]
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


autoclass_content = "both"
autodoc_member_order = "bysource"

# Configure autosummary to skip modules that fail to import
autosummary_generate = True
autosummary_imported_members = False

# Suppress warnings for modules that might not import correctly
suppress_warnings = ['autosummary.import']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"  # Modern, clean theme as base

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#00d4aa",
        "color-brand-content": "#00d4aa",
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f8f9fa",
    },
    "dark_css_variables": {
        "color-brand-primary": "#00ffc8",
        "color-brand-content": "#00ffc8",
        "color-background-primary": "#0a0e1a",
        "color-background-secondary": "#151b2e",
        "color-background-hover": "#1a2233",
        "color-foreground-primary": "#e0e6f0",
        "color-foreground-secondary": "#a0b0c8",
        "color-api-name": "#00d4aa",
        "color-api-pre-name": "#00ffc8",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['custom.js']

html_title = f"{project} {release}"
html_short_title = project

# Logo and favicon (using existing logo URL as fallback)
_logo_path = os.path.join(os.path.dirname(__file__), "_static", "logo.svg")
_favicon_path = os.path.join(os.path.dirname(__file__), "_static", "favicon.ico")
html_logo = "_static/logo.svg" if os.path.exists(_logo_path) else "https://jwgitlab.cs.mcgill.ca/cgoliver/rnaglib/-/raw/zenodo/images/rgl.png"
if os.path.exists(_favicon_path):
    html_favicon = "_static/favicon.ico"


source_suffix = {".rst": "restructuredtext", ".txt": "markdown", ".md": "markdown"}
