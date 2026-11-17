"""The checkouts that import a repository's modules, kept true when a name moves.

A move redirects every importer it scans; a consumer in another checkout is not
scanned, so a moved name it imports from the old module breaks it while the
move's own plan stays clean. A repository declares those checkouts
(`DeclaredRoot.downstream_roots_for`), and a move of names out of a module
rewrites each `from <module> import <name>` there to the name's new home, in
the same plan. A use it cannot rewrite (the module object read as
`module.Name`, a star import, the module spelled in a string) refuses, naming
the file and line: the consumer has to change first.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from nominal_refactor_advisor.codemod_source_edits import (
    SourceTextGeometry,
    SourceTextSpanReplacement,
    TextFileEdit,
)
from nominal_refactor_advisor.declared_root import DeclaredRoot
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput

SKIPPED_DIRECTORIES = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist"})


@dataclass(frozen=True)
class DownstreamImport:
    """One `from <source> import ...` in a consumer that names a moved name."""

    statement: ast.ImportFrom
    homes: Mapping[str, str]

    def replacement(self, source_module: str, line_length: int) -> str:
        """The statement split by owner: the names that stay on the source, then
        one statement per new home, each in the source's layout."""

        node = self.statement
        indent = " " * node.col_offset
        kept = tuple(alias for alias in node.names if alias.name not in self.homes)
        moved: dict[str, list[ast.alias]] = {}
        for alias in node.names:
            if alias.name in self.homes:
                moved.setdefault(self.homes[alias.name], []).append(alias)
        groups = ((source_module, kept),) if kept else ()
        groups += tuple((home, tuple(aliases)) for home, aliases in sorted(moved.items()))
        return ("\n" + indent).join(
            self.spelled(module, aliases, indent, line_length, node.lineno == node.end_lineno)
            for module, aliases in groups
        )

    @staticmethod
    def spelled(module: str, aliases: tuple[ast.alias, ...], indent: str, line_length: int, one_line: bool) -> str:
        names = [alias.name if alias.asname is None else f"{alias.name} as {alias.asname}" for alias in aliases]
        line = f"from {module} import {', '.join(names)}"
        if one_line and len(indent) + len(line) <= line_length:
            return line
        return f"from {module} import (\n" + "".join(f"{indent}    {name},\n" for name in names) + f"{indent})"


@dataclass(frozen=True)
class DownstreamConsumers:
    """The declared checkouts importing the repository of `anchor`."""

    roots: tuple[Path, ...]

    @classmethod
    def of(cls, anchor: Path) -> DownstreamConsumers:
        return cls(DeclaredRoot.downstream_roots_for(anchor))

    def files(self) -> Iterator[Path]:
        for root in self.roots:
            if not root.is_dir():
                raise NeedsInput(
                    f"`{root}` is declared a downstream root but is not checked out there; "
                    "a move cannot keep its imports true",
                    NeededInput.SOURCE_CHANGE,
                )
            for path in sorted(root.rglob("*.py")):
                if not SKIPPED_DIRECTORIES.intersection(path.relative_to(root).parts):
                    yield path

    def edits(self, operation_type: type, source_module: str, homes: Mapping[str, str]) -> tuple[TextFileEdit, ...]:
        """Each consumer's imports of a moved name, pointed at its home; refuses,
        naming every use it cannot rewrite, before planning any."""

        if not self.roots or not homes:
            return ()
        package, _, leaf = source_module.rpartition(".")
        edits: list[TextFileEdit] = []
        refusals: list[str] = []
        for path in self.files():
            source = path.read_text(encoding="utf-8")
            if leaf not in source:
                continue
            tree = ast.parse(source)
            imports, blocked = self.uses(tree, source, source_module, package, leaf, homes)
            refusals.extend(f"{path}:{line}: {why}" for line, why in blocked)
            if not imports or blocked:
                continue
            geometry = SourceTextGeometry(source)
            line_length = DeclaredRoot.line_length_for(path)
            edits.append(
                TextFileEdit(
                    operation_type=operation_type,
                    file_path=path.as_posix(),
                    source=source,
                    replacements=tuple(
                        SourceTextSpanReplacement.from_offsets(
                            start_offset=(span := geometry.required_node_offsets(found.statement))[0],
                            end_offset=span[1],
                            replacement_source=found.replacement(source_module, line_length),
                        )
                        for found in imports
                    ),
                    rationale=f"A consumer's imports of names leaving {source_module!r} follow them.",
                )
            )
        if refusals:
            raise NeedsInput(
                "needs the consumer's change: " + "; ".join(refusals),
                NeededInput.SOURCE_CHANGE,
            )
        return tuple(edits)

    @staticmethod
    def uses(
        tree: ast.Module, source: str, source_module: str, package: str, leaf: str, homes: Mapping[str, str]
    ) -> tuple[tuple[DownstreamImport, ...], tuple[tuple[int, str], ...]]:
        """The consumer's rewritable imports of moved names, and each use it
        cannot rewrite with its line."""

        imports: list[DownstreamImport] = []
        blocked: list[tuple[int, str]] = []
        aliases: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == source_module:
                if any(alias.name == "*" for alias in node.names):
                    blocked.append((node.lineno, f"star-imports `{source_module}`"))
                elif any(alias.name in homes for alias in node.names):
                    imports.append(DownstreamImport(node, homes))
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == package:
                aliases.update(alias.asname or alias.name for alias in node.names if alias.name == leaf)
            elif isinstance(node, ast.Import):
                aliases.update(alias.asname for alias in node.names if alias.name == source_module and alias.asname)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in homes
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases
            ):
                blocked.append((node.lineno, f"reads `{node.attr}` off `{source_module}` itself"))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                mentioned = sorted(name for name in homes if f"{source_module}.{name}" in node.value)
                if mentioned:
                    blocked.append((node.lineno, f"spells `{source_module}.{mentioned[0]}` in a string"))
        del source
        return tuple(imports), tuple(blocked)


__all__ = ("DownstreamConsumers", "DownstreamImport")
