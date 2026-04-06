from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(ROOT))

project = "Nominal Refactor Advisor"
author = "OpenHCSDev"
copyright = "2026, OpenHCSDev"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "mixed"
add_module_names = False

html_theme = "alabaster"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
