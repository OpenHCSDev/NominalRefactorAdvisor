from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, os.fspath(ROOT))
sys.path.insert(0, os.fspath(SOURCE / "_ext"))

from catalog_generation import generate_api_reference_pages
from metaclass_registry import AutoRegisterMeta


generate_api_reference_pages(SOURCE)
AutoRegisterMeta.__doc__ = (
    "Metaclass from ``metaclass-registry`` used for class-time auto-registration."
)

project = "Nominal Refactor Advisor"
author = "OpenHCSDev"
copyright = "2026, OpenHCSDev"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "api/_generated"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "mixed"
autodoc_inherit_docstrings = False
add_module_names = False

html_theme = "alabaster"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
