"""The checkouts that import a repository's modules, kept true when a name moves.

A move redirects every importer it scans; a consumer in another checkout is not
scanned, so a moved name it imports from the old module breaks it while the
move's own plan stays clean. Which checkouts consume the repository is the
consumers' own fact: each declares the dependency in its `pyproject.toml`. So
the roots are derived, never restated here: a sibling checkout of the project
root, or a path its uv workspace or sources name, is downstream when its
`[project]` dependencies, optional dependencies or dependency groups name this
project's distribution (PEP 503 normalized). A move of names out of a module
rewrites each `from <module> import <name>` there to the name's new home, in
the same plan. A use it cannot rewrite (the module object read as
`module.Name`, a star import, the module spelled in a string), or a declared
path that is not checked out, refuses, naming the file: the consumer has to
change first.
"""

from __future__ import annotations

import ast
import re
import tomllib
from collections.abc import Iterable, Iterator, Mapping
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


def distribution(name: str) -> str:
    """A distribution name as PEP 503 compares it."""

    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_names(project: Mapping[str, object]) -> frozenset[str]:
    """The distributions a `pyproject.toml` depends on: `[project]` dependencies
    and optional dependencies, and its dependency groups."""

    declared = project.get("project")
    table = declared if isinstance(declared, dict) else {}
    requirements: list[object] = list(table.get("dependencies", ()))
    for group in (table.get("optional-dependencies") or {}).values():
        requirements.extend(group)
    for group in (project.get("dependency-groups") or {}).values():
        requirements.extend(group)
    return frozenset(
        distribution(found[0])
        for requirement in requirements
        if isinstance(requirement, str) and (found := re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement.strip()))
    )


@dataclass(frozen=True)
class DownstreamConsumers:
    """The checkouts whose `pyproject.toml` depends on the project holding `anchor`."""

    project: str
    roots: tuple[Path, ...]

    @classmethod
    def of(cls, anchor: Path) -> DownstreamConsumers:
        pyproject = DeclaredRoot.nearest_pyproject(anchor)
        if pyproject is None:
            return cls("", ())
        declared = tomllib.loads(pyproject.read_text())
        name = (declared.get("project") or {}).get("name")
        if not isinstance(name, str):
            return cls("", ())
        project = distribution(name)
        named = tuple(cls.uv_paths(pyproject.parent, declared))
        missing = sorted(str(path) for path in named if not (path / "pyproject.toml").is_file())
        if missing:
            raise NeedsInput(
                "needs the consumer's checkout: " + ", ".join(f"`{path}`" for path in missing)
                + f" is named by `{pyproject}` but not checked out, so a move cannot keep its imports true",
                NeededInput.SOURCE_CHANGE,
            )
        siblings = (path for path in sorted(pyproject.parent.parent.iterdir()) if (path / "pyproject.toml").is_file())
        roots = tuple(
            dict.fromkeys(
                root.resolve()
                for root in (*siblings, *named)
                if root.resolve() != pyproject.parent.resolve()
                and project in requirement_names(tomllib.loads((root / "pyproject.toml").read_text()))
            )
        )
        return cls(project, roots)

    @staticmethod
    def uv_paths(root: Path, declared: Mapping[str, object]) -> Iterable[Path]:
        """The checkouts the project's uv workspace members and path sources name."""

        uv = ((declared.get("tool") or {}).get("uv")) or {}
        for member in (uv.get("workspace") or {}).get("members", ()):
            yield from (path for path in root.glob(member) if path.is_dir()) if "*" in member else ((root / member).resolve(),)
        for source in (uv.get("sources") or {}).values():
            if isinstance(source, dict) and isinstance(source.get("path"), str):
                yield (root / source["path"]).resolve()

    def files(self, root: Path) -> Iterator[Path]:
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
        for root, path in ((root, path) for root in self.roots for path in self.files(root)):
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
                    rationale=(
                        f"Consumer `{root}` depends on {self.project!r}: its imports of names "
                        f"leaving {source_module!r} follow them."
                    ),
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
