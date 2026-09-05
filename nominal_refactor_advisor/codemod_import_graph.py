"""Cycle-safe source-module import graph for codemod planning."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import (
    dataclass,
    field,
)
from functools import cached_property

from .codemod_paths import (
    SourcePathCandidateSet,
    SourcePathResolutionAuthority,
)
from .python_module_identity import python_module_name_is_importable
from .source_index import (
    SourceFileDigest,
    SourceIndex,
)


@dataclass(frozen=True)
class SourceModuleImportGraph:
    """Source-index-local import graph for cycle-safe generated imports."""

    source_index: SourceIndex
    module_nodes_by_file_path: Mapping[str, ast.Module] = field(default_factory=dict)
    imported_modules_by_module: Mapping[str, frozenset[str]] | None = None

    @cached_property
    def source_file_by_path(self) -> dict[str, SourceFileDigest]:
        return {
            source_file.file_path: source_file
            for source_file in self.source_index.files
        }

    @cached_property
    def source_path_candidates(self) -> SourcePathCandidateSet:
        return SourcePathCandidateSet.from_paths(tuple(self.source_file_by_path))

    @cached_property
    def known_module_names(self) -> frozenset[str]:
        return frozenset(
            source_file.module_name for source_file in self.source_index.files
        )

    @cached_property
    def source_files_by_module_name(
        self,
    ) -> dict[str, tuple[SourceFileDigest, ...]]:
        """Derive every parsed file claiming each canonical module identity."""

        files_by_module_name: dict[str, list[SourceFileDigest]] = {}
        for source_file in self.source_index.files:
            files_by_module_name.setdefault(source_file.module_name, []).append(
                source_file
            )
        return {
            module_name: tuple(source_files)
            for module_name, source_files in files_by_module_name.items()
        }

    def unique_source_file_for_module_name(
        self,
        module_name: str,
    ) -> SourceFileDigest | None:
        """Return the unique parsed owner of a module identity or fail closed."""

        candidates = self.source_files_by_module_name.get(module_name, ())
        if len(candidates) > 1:
            raise ValueError(
                f"Module identity {module_name!r} has multiple source authorities: "
                f"{tuple(candidate.file_path for candidate in candidates)!r}"
            )
        if not candidates:
            return None
        return candidates[0]

    @cached_property
    def import_edges_by_module(self) -> dict[str, frozenset[str]]:
        if self.imported_modules_by_module is not None:
            return dict(self.imported_modules_by_module)
        return {
            source_file.module_name: self.import_edges_for_source_file(source_file)
            for source_file in self.source_index.files
        }

    def import_edges_for_source_file(
        self,
        source_file: SourceFileDigest,
    ) -> frozenset[str]:
        module_node = self.module_nodes_by_file_path.get(source_file.file_path)
        if module_node is None:
            return frozenset()
        edges: set[str] = set()
        for statement in module_node.body:
            edges.update(self.statement_edges(source_file, statement))
        return frozenset(edges)

    def statement_edges(
        self,
        source_file: SourceFileDigest,
        statement: ast.stmt,
    ) -> frozenset[str]:
        if isinstance(statement, ast.Import):
            return frozenset(
                edge
                for alias in statement.names
                for edge in self.known_import_targets(alias.name)
            )
        if isinstance(statement, ast.ImportFrom):
            resolved_module = (
                source_file.module_path_identity.resolve_import_from_module(
                    imported_module=statement.module,
                    level=statement.level,
                )
            )
            if resolved_module is None:
                return frozenset()
            edges = set(self.known_import_targets(resolved_module))
            for alias in statement.names:
                if alias.name == "*":
                    continue
                edges.update(
                    self.known_import_targets(f"{resolved_module}.{alias.name}")
                )
            return frozenset(edges)
        return frozenset()

    def known_import_targets(self, module_name: str) -> frozenset[str]:
        if module_name in self.known_module_names:
            return frozenset((module_name,))
        return frozenset()

    def import_would_create_cycle(
        self,
        *,
        importing_file_path: str,
        imported_file_path: str,
    ) -> bool:
        importing_module = self.module_name_for_file_path(importing_file_path)
        imported_module = self.module_name_for_file_path(imported_file_path)
        if importing_module is None or imported_module is None:
            return True
        if importing_module == imported_module:
            return False
        return self.module_reaches(imported_module, importing_module)

    def module_name_for_file_path(self, file_path: str) -> str | None:
        source_file = self.source_file_for_path(file_path)
        if source_file is None:
            return None
        return source_file.module_name

    def source_file_for_path(self, file_path: str) -> SourceFileDigest | None:
        exact_match = self.source_file_by_path.get(file_path)
        if exact_match is not None:
            return exact_match
        resolved_path = SourcePathResolutionAuthority(
            requested_path=file_path,
            candidate_set=self.source_path_candidates,
        ).optional_path()
        if resolved_path is None:
            return None
        return self.source_file_by_path.get(resolved_path)

    @cached_property
    def package_module_names(self) -> frozenset[str]:
        return frozenset(
            source_file.module_name
            for source_file in self.source_index.files
            if source_file.is_package_init
        )

    def import_source(
        self,
        *,
        importing_file_path: str,
        imported_file_path: str,
        imported_name: str,
    ) -> str | None:
        """Render an import only from canonical parsed-module identities."""

        module_reference = self.import_module_reference(
            importing_file_path=importing_file_path,
            imported_file_path=imported_file_path,
            imported_name=imported_name,
        )
        if module_reference is None:
            return None
        return f"from {module_reference} import {imported_name}\n"

    def import_module_reference(
        self,
        *,
        importing_file_path: str,
        imported_file_path: str,
        imported_name: str,
    ) -> str | None:
        """Resolve the canonical module reference for one import binding."""

        if not python_module_name_is_importable(imported_name):
            return None
        importing_file = self.source_file_for_path(importing_file_path)
        imported_file = self.source_file_for_path(imported_file_path)
        if importing_file is None or imported_file is None:
            return None
        if not python_module_name_is_importable(imported_file.module_name):
            return None
        return (
            self.relative_module_reference(
                importing_file,
                imported_file,
            )
            or imported_file.module_name
        )

    def required_import_source(
        self,
        *,
        importing_file_path: str,
        imported_file_path: str,
        imported_name: str,
    ) -> str:
        """Return one canonical acyclic import or fail closed."""

        module_reference = self.required_import_module_reference(
            importing_file_path=importing_file_path,
            imported_file_path=imported_file_path,
            imported_name=imported_name,
        )
        return f"from {module_reference} import {imported_name}\n"

    def required_reexport_source(
        self,
        *,
        importing_file_path: str,
        imported_file_path: str,
        imported_name: str,
    ) -> str:
        """Return one canonical acyclic explicit re-export or fail closed."""

        module_reference = self.required_import_module_reference(
            importing_file_path=importing_file_path,
            imported_file_path=imported_file_path,
            imported_name=imported_name,
        )
        return (
            f"from {module_reference} import {imported_name} as {imported_name}\n"
        )

    def required_import_module_reference(
        self,
        *,
        importing_file_path: str,
        imported_file_path: str,
        imported_name: str,
    ) -> str:
        """Prove and return the canonical module reference for one import."""

        if self.import_would_create_cycle(
            importing_file_path=importing_file_path,
            imported_file_path=imported_file_path,
        ):
            raise ValueError(
                f"Importing {imported_name!r} from {imported_file_path!r} into "
                f"{importing_file_path!r} would create a module cycle"
            )
        module_reference = self.import_module_reference(
            importing_file_path=importing_file_path,
            imported_file_path=imported_file_path,
            imported_name=imported_name,
        )
        if module_reference is None:
            raise ValueError(
                f"No canonical import exists for {imported_name!r} from "
                f"{imported_file_path!r} into {importing_file_path!r}"
            )
        return module_reference

    def relative_module_reference(
        self,
        importing_file: SourceFileDigest,
        imported_file: SourceFileDigest,
    ) -> str | None:
        importing_parts = importing_file.module_name.split(".")
        imported_parts = imported_file.module_name.split(".")
        importing_package = (
            tuple(importing_parts)
            if importing_file.is_package_init
            else tuple(importing_parts[:-1])
        )
        imported_package = (
            tuple(imported_parts)
            if imported_file.is_package_init
            else tuple(imported_parts[:-1])
        )
        if not importing_package:
            return None
        common_length = 0
        for importing_part, imported_part in zip(
            importing_package,
            imported_parts,
            strict=False,
        ):
            if importing_part != imported_part:
                break
            common_length += 1
        if common_length == 0:
            return None
        if not self.declared_package_chain(importing_package):
            return None
        if not self.declared_package_chain(imported_package):
            return None
        dots = "." * (len(importing_package) - common_length + 1)
        remainder = ".".join(imported_parts[common_length:])
        return f"{dots}{remainder}"

    def declared_package_chain(self, package_parts: tuple[str, ...]) -> bool:
        return all(
            ".".join(package_parts[:length]) in self.package_module_names
            for length in range(1, len(package_parts) + 1)
        )

    def module_reaches(self, start_module: str, target_module: str) -> bool:
        visited: set[str] = set()
        stack = [start_module]
        while stack:
            module_name = stack.pop()
            if module_name in visited:
                continue
            visited.add(module_name)
            for imported_module in self.import_edges_by_module.get(module_name, ()):
                if imported_module == target_module:
                    return True
                stack.append(imported_module)
        return False
