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
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import PurePosixPath

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import CompactModuleClassHeader
from nominal_refactor_advisor.codemod_import_scopes import ModuleImportScope
from nominal_refactor_advisor.codemod_module_declarations import ModuleSymbolTable
from nominal_refactor_advisor.codemod_runtime import (
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.models import PackageDestinationMetrics, RefactorFinding
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.refactor_concepts import PackageMoveConcept
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingRecipeSynthesizer,
    PrimaryEvidenceActionKeysMixin,
)
from nominal_refactor_advisor.refactors.module_move import (
    BatchModuleMove,
    DissolveModuleOperation,
    RelocateSymbolsToModuleOperation,
    RelocateSymbolsToNewModuleOperation,
)

_ROW = re.compile(r"^\s*\|\s*`(?P<package>[A-Za-z_][\w./]*)`\s*\|(?P<owners>[^|]*)\|")
_MODULE_FILE = re.compile(r"`(?P<name>[A-Za-z_]\w*)\.py`")
_RENAMED_MODULE = re.compile(
    r"`(?P<name>[A-Za-z_]\w*)`\s+moves\s+to\s+`(?P<destination>[A-Za-z_][\w/]*)`"
)
_MERGED_MODULE = re.compile(
    r"`(?P<name>[A-Za-z_]\w*)\.py`\s+merges\s+into\s+`(?P<target>[A-Za-z_]\w*)\.py`"
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
        merged = frozenset(name for name, _target in self.module_merges())
        yield from renamed.items()
        for match in _MODULE_FILE.finditer(self.owners):
            if match["name"] not in renamed and match["name"] not in merged:
                yield match["name"], f"{self.package}/{match['name']}"

    def module_merges(self) -> Iterator[tuple[str, str]]:
        """(flat module name, flat module whose destination it joins)."""

        for match in _MERGED_MODULE.finditer(self.owners):
            yield match["name"], match["target"]

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
        """Each flat module the table sends somewhere. A table describes the one root
        package holding the most of its row packages; every other root (a consumer
        scanned beside it, `tools`, a test package) keeps every module where it is,
        even one whose name a row lists, and only its imports follow. Two roots
        holding equally many are refused by name rather than guessed between. A
        flat module named like a row package holds that name too, so the root it
        shadows a package in is still the subject and the move reports the shadow."""

        flat_by_root: dict[str, list[CompactModuleClassHeader]] = {}
        packaged_by_root: dict[str, list[CompactModuleClassHeader]] = {}
        packages_by_root: dict[str, set[str]] = {}
        for module in modules:
            parts = module.module_name.split(".")
            if len(parts) >= 2:
                packages_by_root.setdefault(parts[0], set()).add(parts[1])
            if PurePosixPath(module.file_path).name == "__init__.py" or len(parts) < 2:
                continue
            by_root = flat_by_root if len(parts) == 2 else packaged_by_root
            by_root.setdefault(parts[0], []).append(module)
        table_packages = {row.package.split("/")[0] for row in self.rows}
        held = {
            root: len(table_packages & packages)
            for root, packages in packages_by_root.items()
            if table_packages & packages
        }
        if not table_packages:
            return ()
        if not held:
            raise ValueError(
                "no scanned root holds any of the table's packages: "
                f"{', '.join(sorted(table_packages))}; create one of them in the root "
                "the table describes"
            )
        most = max(held.values())
        described = sorted(root for root, count in held.items() if count == most)
        if len(described) > 1:
            raise ValueError(
                f"the table's packages are split evenly between {' and '.join(described)}; "
                "add one of the table's packages to the root it describes"
            )
        (root,) = described
        return self.root_destinations(
            tuple(flat_by_root.get(root, ())), tuple(packaged_by_root.get(root, ()))
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
        declared = {
            name: destination
            for row in self.rows
            for name, destination in row.module_destinations()
        }
        packaged = {
            ".".join(module.module_name.split(".")[1:]).replace(".", "/")
            for module in package_modules
        }
        merge_sources: set[str] = set()
        for row in self.rows:
            for name, target in row.module_merges():
                if name not in by_stem:
                    continue
                merge_sources.add(by_stem[name].file_path)
                if target in by_stem:
                    target_path = by_stem[target].file_path
                    chosen[by_stem[name].file_path] = chosen.get(target_path, target)
                elif declared.get(target) in packaged:
                    chosen[by_stem[name].file_path] = declared[target]
        destination_paths = {
            source_path: (PurePosixPath(source_path).parent / f"{destination}.py").as_posix()
            for source_path, destination in chosen.items()
        }
        cluster = tuple(sorted(chosen))
        cluster_destinations = tuple(destination_paths[path] for path in cluster)
        cluster_merge_sources = tuple(path for path in cluster if path in merge_sources)
        return tuple(
            FlatModuleDestination(
                source_path=source_path,
                destination_path=destination_paths[source_path],
                cluster_paths=cluster,
                cluster_destination_paths=cluster_destinations,
                cluster_merge_source_paths=cluster_merge_sources,
            )
            for source_path in cluster
        )


@dataclass(frozen=True)
class FlatModuleDestination:
    source_path: str
    destination_path: str
    cluster_paths: tuple[str, ...]
    cluster_destination_paths: tuple[str, ...]
    cluster_merge_source_paths: tuple[str, ...]

    def metrics(self) -> PackageDestinationMetrics:
        return PackageDestinationMetrics(
            destination_path=self.destination_path,
            cluster_paths=self.cluster_paths,
            cluster_destination_paths=self.cluster_destination_paths,
            cluster_merge_source_paths=self.cluster_merge_source_paths,
        )


@dataclass(frozen=True)
class ClusterLayers:
    """The table's flat modules in one source state, read for which moves next."""

    context: CodemodSelectorContext
    cluster_paths: tuple[str, ...]
    cluster_destinations: tuple[str, ...]
    merge_sources: tuple[str, ...]

    @cached_property
    def destinations(self) -> dict[str, str]:
        return dict(zip(self.cluster_paths, self.cluster_destinations, strict=True))

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

    def imported_module_names(self, path: str) -> frozenset[str]:
        """The modules `path` imports when it loads: module scope, unguarded.
        Only these order the moves; a guarded or function-local import is
        rewritten where it stands whichever module moves first."""

        return self._edges(
            path,
            lambda parsed: ModuleImportScope.RUNTIME.import_statements(parsed.module),
        )

    def any_scope_module_names(self, path: str) -> frozenset[str]:
        """The modules `path` imports at any scope, guarded and local included."""

        return self._edges(path, lambda parsed: parsed.fact(ModuleNodeIndex).imports)

    def _edges(
        self, path: str, statements: Callable[[ParsedModule], Iterable[ast.stmt]]
    ) -> frozenset[str]:
        graph = self.context.module_import_graph
        source_file = graph.source_file_for_path(path)
        if source_file is None:
            return frozenset()
        return frozenset(
            edge
            for node in statements(self.context.parsed_module_for_source_path(path))
            for edge in graph.statement_edges(source_file, node)
        )

    def joins(self, path: str, other: str) -> bool:
        """Whether `path`'s code moves into the module `other` moves to."""

        return self.destinations[path] == self.destinations[other]

    def imports(self, importer: str, imported: str) -> bool:
        module_name = self.context.module_import_graph.module_name_for_file_path(imported)
        return module_name in self.imported_module_names(importer)

    def upper_importers(self, path: str) -> tuple[str, ...]:
        """Flat modules of the cluster that still declare names, import `path`
        and keep a module of their own."""

        return tuple(
            other
            for other in self.present_paths
            if other != path
            and not self.emptied(other)
            and not self.joins(other, path)
            and self.imports(other, path)
        )

    @cached_property
    def mutual_imports(self) -> tuple[tuple[str, str], ...]:
        """Pairs of declaring flat modules that import each other when they load
        and keep separate modules: no move order separates them."""

        return self._mutual_pairs(self.imports)

    def deferred_cycles(self, path: str) -> tuple[tuple[str, str], ...]:
        """Pairs with `path` that import each other only through a guard or a
        function-local import: the move carries that cycle, it does not make it."""

        return tuple(
            pair
            for pair in self._mutual_pairs(self.imports_at_any_scope)
            if path in pair and pair not in self.mutual_imports
        )

    def imports_at_any_scope(self, importer: str, imported: str) -> bool:
        module_name = self.context.module_import_graph.module_name_for_file_path(imported)
        return module_name in self.any_scope_module_names(importer)

    def _mutual_pairs(
        self, imports: Callable[[str, str], bool]
    ) -> tuple[tuple[str, str], ...]:
        declaring = [path for path in self.present_paths if not self.emptied(path)]
        return tuple(
            (path, other)
            for index, path in enumerate(declaring)
            for other in declaring[index + 1 :]
            if not self.joins(path, other) and imports(path, other) and imports(other, path)
        )

    def ready(self, path: str) -> bool:
        """It can move now: its destination exists, it is not a module a row
        merges into another (the merge target creates the destination, its
        sources join it), or no other present module will."""

        destination = self.destinations[path]
        return (
            destination in self.context.module_nodes_by_file_path
            or path not in self.merge_sources
            or not any(
                other != path and self.destinations[other] == destination
                for other in self.present_paths
            )
        )

    @cached_property
    def batch_paths(self) -> tuple[str, ...]:
        """The modules that relocate together in one stage: every present module
        while each still declares names, keeps a module of its own and moves into
        a destination of its own. Each move then derives from the same starting
        state and names its peers' new homes (`BatchModuleMove`). Fewer than two,
        or a shared destination, moves one module per stage."""

        declaring = tuple(path for path in self.present_paths if not self.emptied(path))
        destinations = [self.destinations[path] for path in declaring]
        if (
            len(declaring) < 2
            or len(declaring) != len(self.present_paths)
            or len(set(destinations)) != len(destinations)
        ):
            return ()
        return declaring

    def module_name_at(self, destination_path: str, source_path: str) -> str:
        """The module a destination path names, beside the source's package."""

        source_module = self.context.module_import_graph.module_name_for_file_path(source_path)
        if source_module is None:
            raise ValueError(f"{source_path!r} has no module name")
        package = source_module.rpartition(".")[0]
        relative = PurePosixPath(destination_path).relative_to(PurePosixPath(source_path).parent)
        return ".".join((*filter(None, (package,)), *relative.with_suffix("").parts))

    def batch_recipe(self, context: CodemodSelectorContext) -> RefactorRecipe:
        import_graph = context.module_import_graph
        moves = {
            path: BatchModuleMove(
                source_module=import_graph.module_name_for_file_path(path) or "",
                destination_module=self.module_name_at(self.destinations[path], path),
            )
            for path in self.batch_paths
        }
        recipe = RefactorRecipe(
            recipe_id="relocate-batch",
            reason="Relocate every name each flat module of the table declares into "
            "its package, in one stage.",
        )
        for path, own in moves.items():
            destination = self.destinations[path]
            recipe = recipe.with_operation(
                (
                    RelocateSymbolsToModuleOperation
                    if destination in context.module_nodes_by_file_path
                    else RelocateSymbolsToNewModuleOperation
                )(
                    target=SourceRewriteTarget(file_path=path),
                    symbol_qualnames=self.declared_names(path),
                    destination_path=destination,
                    batch=tuple(move for move in moves.values() if move is not own),
                )
            )
        return recipe

    @cached_property
    def next_path(self) -> str | None:
        """The one module that moves next: an emptied module dissolves before
        anything else relocates; otherwise the first top-layer module by path
        whose destination it can move into."""

        emptied = [path for path in self.present_paths if self.emptied(path)]
        if emptied:
            return emptied[0]
        tops = [
            path
            for path in self.present_paths
            if not self.upper_importers(path) and self.ready(path)
        ]
        return tops[0] if tops else None


class PackageMoveFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
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
        layers = ClusterLayers(
            context,
            metrics.cluster_paths,
            metrics.cluster_destination_paths,
            metrics.cluster_merge_source_paths,
        )
        if source_path not in layers.present_paths:
            return self.rejected_evaluation(f"{source_path!r} is already gone")
        if layers.mutual_imports:
            return self.rejected_evaluation(
                "; ".join(
                    f"{left} and {right} import each other"
                    for left, right in layers.mutual_imports
                )
                + ", so no move order keeps either importable: declare one "
                "merging into the other (`a.py` merges into `b.py`)"
            )
        batch = layers.batch_paths
        if batch:
            if source_path != batch[0]:
                return self.rejected_evaluation(f"moves in {batch[0]}'s batch")
            return self.executable_evaluation(layers.batch_recipe(context))
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
                reason="Relocate every name the flat module declares into its package."
                + "".join(
                    f" Debt carried: {left} and {right} import each other through a "
                    "guarded or function-local import; break that cycle at its owner."
                    for left, right in layers.deferred_cycles(source_path)
                ),
            ).with_operation(
                (
                    RelocateSymbolsToModuleOperation
                    if metrics.destination_path in context.module_nodes_by_file_path
                    else RelocateSymbolsToNewModuleOperation
                )(
                    target=target,
                    symbol_qualnames=layers.declared_names(source_path),
                    destination_path=metrics.destination_path,
                )
            )
        )
