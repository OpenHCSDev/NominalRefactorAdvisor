"""Modules outside the scan, read from their installed source and never run."""

from __future__ import annotations

import ast
from functools import cache
from importlib.machinery import PathFinder
from pathlib import Path
from typing import TypeVar

from .ast_tools import (
    CollectedFamily,
    CollectedFamilyCacheContext,
    ParsedModule,
    SourceModule,
)
from .cache_paths import ActiveScanCache
from .python_module_identity import PythonModulePathIdentity
from .source_identity import PACKAGE_INIT_FILE_NAME, source_path_is_package_init


ItemT = TypeVar("ItemT")


class InstalledModule:
    """A module outside the scan, read from its installed source and never run.

    The analysing interpreter's import path locates it (`PathFinder`, which
    executes no package); only a `.py` source is read. Its collected families
    are kept in the running scan's family cache, keyed like a scanned module's
    by path, source signature and family schema, so a warm scan reads them
    instead of parsing the installed source again.
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
    def located_source(module_name: str) -> tuple[Path, str] | None:
        """The installed `.py` file and its text, or None when it has none."""

        path = InstalledModule.source_path(module_name)
        return None if path is None else (path, path.read_text())

    @classmethod
    def family_items(
        cls, module_name: str, family: type[CollectedFamily[ItemT]]
    ) -> tuple[ItemT, ...] | None:
        """One family's items for the module: the scan's family cache holds them,
        or they are collected from the parsed source and stored there."""

        located = cls.located_source(module_name)
        if located is None:
            return None
        path, source = located
        context = CollectedFamilyCacheContext.from_source(
            path=path,
            module_name=module_name,
            source=source,
            family_cache_dir=ActiveScanCache.directory(),
        )
        cached = context.load_items(family)
        if cached is not None:
            return cached
        parsed = cls.parsed(module_name)
        if parsed is None:
            return None
        items = tuple(family.collect(parsed))
        context.store_items(family, items)
        return items

    @staticmethod
    @cache
    def parsed(module_name: str) -> ParsedModule | None:
        """The module parsed from its installed source, or None when it has none."""

        located = InstalledModule.located_source(module_name)
        if located is None:
            return None
        path, source = located
        try:
            module = ast.parse(source, filename=str(path))
        except SyntaxError:
            return None
        return SourceModule.from_path_identity(
            PythonModulePathIdentity(path=path, import_name=module_name),
            source,
        ).parsed_module(module)
