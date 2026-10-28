"""Modules outside the scan, read from their installed source and never run."""

from __future__ import annotations

import ast
from functools import cache
from importlib.machinery import PathFinder
from pathlib import Path

from .ast_tools import ParsedModule, SourceModule
from .python_module_identity import PythonModulePathIdentity
from .source_identity import PACKAGE_INIT_FILE_NAME, source_path_is_package_init


class InstalledModule:
    """A module outside the scan, read from its installed source and never run.

    The analysing interpreter's import path locates it (`PathFinder`, which
    executes no package); only a `.py` source is read.
    """

    @staticmethod
    def source_path(module_name: str) -> Path | None:
        """The `.py` file a module name selects: the top package through `PathFinder`, each submodule under its package's directories."""

        top, *rest = module_name.split(".")
        spec = PathFinder.find_spec(top)
        if spec is None or spec.origin is None and not spec.submodule_search_locations:
            return None
        found = None if spec.origin is None else Path(spec.origin)
        locations = [
            Path(location) for location in spec.submodule_search_locations or ()
        ]
        for part in rest:
            found = next(
                (
                    candidate
                    for location in locations
                    for candidate in (
                        location / part / PACKAGE_INIT_FILE_NAME,
                        location / f"{part}.py",
                    )
                    if candidate.is_file()
                ),
                None,
            )
            if found is None:
                return None
            locations = [found.parent] if source_path_is_package_init(found) else []
        if found is None or found.suffix != ".py":
            return None
        return found

    @staticmethod
    @cache
    def parsed(module_name: str) -> ParsedModule | None:
        """The module parsed from its installed source, or None when it has none."""

        path = InstalledModule.source_path(module_name)
        if path is None:
            return None
        source = path.read_text()
        try:
            module = ast.parse(source, filename=str(path))
        except SyntaxError:
            return None
        return SourceModule.from_path_identity(
            PythonModulePathIdentity(path=path, import_name=module_name),
            source,
        ).parsed_module(module)
