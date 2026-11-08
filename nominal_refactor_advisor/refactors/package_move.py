"""Move flat modules into the packages a destination table declares for them.

A destination table is Markdown: one row per package, the package in backticks
in the first cell and, in the second, what belongs there. A backticked
`name.py` names a flat module; `` `name` moves to `package/module` `` names one
with a new module name; any other word that is a class declared at the top of
exactly one flat module names that module. Words that name nothing are prose.

Each flat module the table sends somewhere is one finding. Its recipe is read
from the live sources of the state it runs in: a module still declaring names
relocates every one of them, in source order, into its destination; a module
that declares nothing dissolves with the destination as its successor. Stages
run top layer first: a module waits while another flat module of the same
table still declares names and imports it, because relocating a lower layer
first leaves the upper flat module and the new one importing each other.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import PurePosixPath

from ..class_index import CompactModuleClassHeader
from ..codemod_module_declarations import ModuleSymbolTable
from ..codemod_runtime import FindingRecipeEvaluation, RefactorRecipe
from ..codemod_selection_context import CodemodSelectorContext
from ..codemod_selector_models import SourceRewriteTarget
from ..models import PackageDestinationMetrics, RefactorFinding, SourceLocation
from .module_move import (
    DissolveModuleOperation,
    RelocateSymbolsToNewModuleOperation,
)
from ..refactor_concepts import PackageMoveConcept
from .finding_recipes import (
    PrimaryEvidenceActionKeysMixin,
    SelfDeclaredRecipeSynthesizer,
)

_ROW = re.compile(r"^\s*\|\s*`(?P<package>[A-Za-z_][\w./]*)`\s*\|(?P<owners>[^|]*)\|")
_MODULE_FILE = re.compile(r"`(?P<name>[A-Za-z_]\w*)\.py`")
_RENAMED_MODULE = re.compile(
    r"`(?P<name>[A-Za-z_]\w*)`\s+moves\s+to\s+`(?P<destination>[A-Za-z_][\w/]*)`"
)
_WORD = re.compile(r"[A-Za-z_]\w*")


@dataclass(frozen=True)
class DestinationRow:
    """One package row: the package and the text naming what belongs there."""

    package: str
    owners: str

    def module_destinations(self) -> Iterator[tuple[str, str]]:
        """(flat module name, destination module path under the root package)."""

        renamed = {
            match["name"]: match["destination"]
            for match in _RENAMED_MODULE.finditer(self.owners)
        }
        yield from renamed.items()
        for match in _MODULE_FILE.finditer(self.owners):
            if match["name"] not in renamed:
                yield match["name"], f"{self.package}/{match['name']}"

    def owner_words(self) -> frozenset[str]:
        return frozenset(_WORD.findall(self.owners))


@dataclass(frozen=True)
class DestinationTable:
    """Every package row of the destination tables a run declares."""

    rows: tuple[DestinationRow, ...]

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> DestinationTable:
        return cls(
            tuple(
                DestinationRow(match["package"], match["owners"])
                for text in texts
                for line in text.splitlines()
                if (match := _ROW.match(line)) is not None
            )
        )

    def destinations(
        self, modules: Iterable[CompactModuleClassHeader]
    ) -> tuple[FlatModuleDestination, ...]:
        """Each flat module the table sends somewhere, per root package."""

        flat_by_root: dict[str, list[CompactModuleClassHeader]] = {}
        packaged_by_root: dict[str, list[CompactModuleClassHeader]] = {}
        for module in modules:
            parts = module.module_name.split(".")
            if PurePosixPath(module.file_path).name == "__init__.py" or len(parts) < 2:
                continue
            by_root = flat_by_root if len(parts) == 2 else packaged_by_root
            by_root.setdefault(parts[0], []).append(module)
        return tuple(
            destination
            for root, root_modules in flat_by_root.items()
            for destination in self.root_destinations(
                tuple(root_modules), tuple(packaged_by_root.get(root, ()))
            )
        )

    def root_destinations(
        self,
        flat_modules: tuple[CompactModuleClassHeader, ...],
        package_modules: tuple[CompactModuleClassHeader, ...],
    ) -> tuple[FlatModuleDestination, ...]:
        """A class names the flat module declaring it, or, once relocated into
        its row's package, the flat module of the same name it left behind."""

        by_stem = {
            PurePosixPath(module.file_path).stem: module for module in flat_modules
        }
        declaring: dict[str, list[tuple[str, CompactModuleClassHeader]]] = {}
        for module in (*flat_modules, *package_modules):
            package = ".".join(module.module_name.split(".")[1:-1])
            for declared in module.classes:
                if "." not in declared.qualname:
                    declaring.setdefault(declared.qualname, []).append((package, module))
        chosen: dict[str, str] = {}
        for row in self.rows:
            for name, destination in row.module_destinations():
                if name in by_stem:
                    chosen.setdefault(by_stem[name].file_path, destination)
            row_package = row.package.replace("/", ".")
            for word in sorted(row.owner_words()):
                owners = [
                    module
                    for package, module in declaring.get(word, ())
                    if package in ("", row_package)
                ]
                stem = PurePosixPath(owners[0].file_path).stem if len(owners) == 1 else ""
                if stem in by_stem:
                    chosen.setdefault(by_stem[stem].file_path, f"{row.package}/{stem}")
        cluster = tuple(sorted(chosen))
        return tuple(
            FlatModuleDestination(
                source_path=source_path,
                destination_path=(
                    PurePosixPath(source_path).parent / f"{destination}.py"
                ).as_posix(),
                cluster_paths=cluster,
            )
            for source_path, destination in sorted(chosen.items())
        )


@dataclass(frozen=True)
class FlatModuleDestination:
    source_path: str
    destination_path: str
    cluster_paths: tuple[str, ...]

    def metrics(self) -> PackageDestinationMetrics:
        return PackageDestinationMetrics(
            destination_path=self.destination_path,
            cluster_paths=self.cluster_paths,
        )


@dataclass(frozen=True)
class ClusterLayers:
    """The table's flat modules in one source state, read for which moves next."""

    context: CodemodSelectorContext
    cluster_paths: tuple[str, ...]

    @cached_property
    def present_paths(self) -> tuple[str, ...]:
        return tuple(
            path
            for path in self.cluster_paths
            if path in self.context.module_nodes_by_file_path
        )

    def declared_names(self, path: str) -> tuple[str, ...]:
        """The names a module declares itself, in source order. `__all__` moves
        with them: a whole-module move takes every name it lists."""

        table = ModuleSymbolTable(
            file_path=path,
            source=self.context.sources_by_file_path[path],
            module=self.context.module_nodes_by_file_path[path],
        )
        names: dict[str, None] = {}
        for statement in table.module.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                continue
            for name in table.bound_names_for_statement(statement):
                names.setdefault(name)
        return tuple(names)

    def emptied(self, path: str) -> bool:
        return not set(self.declared_names(path)) - {"__all__"}

    def upper_importers(self, path: str) -> tuple[str, ...]:
        """Flat modules of the cluster that still declare names and import `path`."""

        graph = self.context.module_import_graph
        module_name = graph.module_name_for_file_path(path)
        return tuple(
            other
            for other in self.present_paths
            if other != path
            and not self.emptied(other)
            and module_name
            in graph.import_edges_by_module.get(
                graph.module_name_for_file_path(other) or "", frozenset()
            )
        )

    @cached_property
    def next_path(self) -> str | None:
        """The one module that moves next: an emptied module dissolves before
        anything else relocates; otherwise the first top-layer module by path."""

        emptied = [path for path in self.present_paths if self.emptied(path)]
        if emptied:
            return emptied[0]
        tops = [path for path in self.present_paths if not self.upper_importers(path)]
        return tops[0] if tops else None


class PackageMoveFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin,
    SelfDeclaredRecipeSynthesizer,
    PackageMoveConcept,
):
    """Relocate a flat module's names into its declared package, then dissolve it."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        metrics = finding.metrics
        evidence = finding.primary_evidence
        if (
            context is None
            or evidence is None
            or not isinstance(metrics, PackageDestinationMetrics)
        ):
            return self.rejected_evaluation(
                "a package move needs source context and its destination metrics"
            )
        source_path = evidence.file_path
        layers = ClusterLayers(context, metrics.cluster_paths)
        if source_path not in layers.present_paths:
            return self.rejected_evaluation(f"{source_path!r} is already gone")
        if layers.next_path is None:
            return self.rejected_evaluation(
                "every flat module of the table is imported by another one that "
                "still declares names, so their imports form a cycle no move order breaks"
            )
        if layers.next_path != source_path:
            waits_for = layers.upper_importers(source_path)
            reason = (
                f"waits for {', '.join(waits_for)}, which import it and move first"
                if waits_for
                else f"waits its turn behind {layers.next_path}"
            )
            return self.rejected_evaluation(reason)
        target = SourceRewriteTarget(file_path=source_path)
        stem = PurePosixPath(source_path).stem
        if layers.emptied(source_path):
            return self.executable_evaluation(
                RefactorRecipe(
                    recipe_id=f"dissolve-{stem}",
                    reason="Dissolve the emptied flat module into its package module.",
                ).with_operation(
                    DissolveModuleOperation(
                        target=target,
                        successor_path=metrics.destination_path,
                    )
                )
            )
        return self.executable_evaluation(
            RefactorRecipe(
                recipe_id=f"relocate-{stem}",
                reason="Relocate every name the flat module declares into its package.",
            ).with_operation(
                RelocateSymbolsToNewModuleOperation(
                    target=target,
                    symbol_qualnames=layers.declared_names(source_path),
                    destination_path=metrics.destination_path,
                )
            )
        )

    def destination_findings(
        self, modules: Iterable[CompactModuleClassHeader], tables: Iterable[str]
    ) -> list[RefactorFinding]:
        return [
            self.build_finding(
                f"`{destination.source_path}` belongs in "
                f"`{destination.destination_path}`, the package its destination "
                "table declares.",
                (SourceLocation(destination.source_path, 1, "<module>"),),
                metrics=destination.metrics(),
            )
            for destination in DestinationTable.from_texts(tables).destinations(modules)
        ]
