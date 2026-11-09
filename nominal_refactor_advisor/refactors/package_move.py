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
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path, PurePosixPath

from nominal_refactor_advisor.ast_tools import (
    CompactModuleIdentity,
    DomainSourcePathPolicy,
    ParsedModule,
    PythonSourcePathDiscovery,
    PythonSourcePathPolicy,
)
from nominal_refactor_advisor.class_index import CompactModuleClassHeader
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.codemod_import_bindings import DirectModuleImportBindingIdentity
from nominal_refactor_advisor.codemod_import_scopes import ModuleImportScope
from nominal_refactor_advisor.codemod_module_declarations import ModuleSymbolTable
from nominal_refactor_advisor.codemod_preflight import CodemodOperationPreflightError
from nominal_refactor_advisor.codemod_runtime import (
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.models import PackageDestinationMetrics, RefactorFinding
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.python_module_identity import PythonModulePathIdentity
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
        """The words that can name a class: the row's text without the spans
        naming modules (`name.py`, a rename, a merge), whose stems name files."""

        prose = self.owners
        for pattern in (_RENAMED_MODULE, _MERGED_MODULE, _MODULE_FILE):
            prose = pattern.sub(" ", prose)
        return frozenset(_WORD.findall(prose))


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

    @property
    def reads_classes(self) -> bool:
        """Whether a row can name a module by a class it declares: only then does
        resolving the table read any module's class headers."""

        return any(row.owner_words() for row in self.rows)

    def module_headers(
        self,
        modules: Sequence[ParsedModule],
        class_headers: Callable[
            [Sequence[ParsedModule]], Sequence[CompactModuleClassHeader]
        ],
    ) -> Sequence[CompactModuleIdentity]:
        """What resolving the table reads of each module: its class headers when a
        row names classes, else its name and path, which no cached payload holds."""

        if self.reads_classes:
            return class_headers(modules)
        return tuple(
            CompactModuleIdentity(module.module_name, module.file_path) for module in modules
        )

    @staticmethod
    def class_declarations(
        modules: Iterable[CompactModuleClassHeader],
    ) -> dict[str, list[tuple[str, CompactModuleClassHeader]]]:
        """Each top-level class name, with the package and module declaring it."""

        declaring: dict[str, list[tuple[str, CompactModuleClassHeader]]] = {}
        for module in modules:
            package = ".".join(module.module_name.split(".")[1:-1])
            for declared in module.classes:
                if "." not in declared.qualname:
                    declaring.setdefault(declared.qualname, []).append((package, module))
        return declaring

    def scan_domain(
        self, policy: PythonSourcePathPolicy, roots: tuple[Path, ...]
    ) -> PythonSourcePathPolicy:
        """The files a move by this table can read, found from raw text before
        anything parses: `TableDomain`. A table with no rows moves nothing, and its
        scan keeps every file."""

        if not self.rows:
            return policy
        return TableDomain(self, policy, roots).policy()

    def destinations(
        self, modules: Iterable[CompactModuleIdentity]
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
        declaring = (
            self.class_declarations((*flat_modules, *package_modules))
            if self.reads_classes
            else {}
        )
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
class TableDomain:
    """The files a move by one destination table reads, from raw text alone.

    - Seeds: every file a row names by stem, as a renamed module or a merge, and
      every file whose text declares a row's capitalised word as a top-level
      class (a module a row names by its class has to spell `class Word`).
    - Spellers: every file whose text holds a seed's stem. A file that imports,
      binds, reads or names a moved module spells its leaf (`FilesReachingModule`).
    - The seeds' import closure, transitively: a moved name's declaring module,
      through any chain of re-exports, which spells no moved leaf.
    - Each seed's package `__init__.py` files and every file in a row's package.

    The seeds are a superset of the modules the table moves, and each part is a
    superset of what its readers need, so a move over the domain writes the tree a
    move over every file writes."""

    table: DestinationTable
    policy_base: PythonSourcePathPolicy
    roots: tuple[Path, ...]

    @cached_property
    def discovered(self) -> tuple[tuple[Path, Path], ...]:
        """(file, its scan root's analysis root), discovered once."""

        return tuple(
            (path.resolve(), PythonModulePathIdentity.analysis_root_for_scan_root(root))
            for root in self.roots
            for path in PythonSourcePathDiscovery(root, self.policy_base).paths()
        )

    @cached_property
    def candidates(self) -> dict[Path, bytes]:
        return {path: path.read_bytes() for path, _root in self.discovered}

    @cached_property
    def names(self) -> dict[Path, str]:
        """Each candidate's module name, from its package `__init__.py` chain,
        walked once per directory."""

        import_roots: dict[tuple[Path, Path], Path] = {}
        names: dict[Path, str] = {}
        for path, analysis_root in self.discovered:
            key = (path.parent, analysis_root)
            if key not in import_roots:
                import_roots[key] = PythonModulePathIdentity.import_root(path, analysis_root)
            names[path] = PythonModulePathIdentity.from_import_root(
                path, import_roots[key]
            ).import_name
        return names

    @cached_property
    def identities(self) -> dict[str, Path]:
        return {name: path for path, name in self.names.items()}

    @cached_property
    def seeds(self) -> frozenset[Path]:
        stems = {
            name
            for row in self.table.rows
            for named in (row.module_destinations(), row.module_merges())
            for pair in named
            for name in (pair[0], PurePosixPath(pair[1]).name)
        }
        declared = [
            re.compile(rb"^class\s+" + re.escape(word.encode()) + rb"\b", re.MULTILINE)
            for row in self.table.rows
            for word in sorted(row.owner_words())
            if word[:1].isupper()
        ]
        return frozenset(
            path
            for path, text in self.candidates.items()
            if path.stem in stems or any(pattern.search(text) for pattern in declared)
        )

    @cached_property
    def import_closure(self) -> frozenset[Path]:
        closure: set[Path] = set()
        pending = list(self.seeds)
        while pending:
            path = pending.pop()
            if path in closure:
                continue
            closure.add(path)
            identity = PythonModulePathIdentity(path=path, import_name=self.names[path])
            for node in self.statements(ast.parse(self.candidates[path]).body):
                for imported in self.imported_module_names(identity, node):
                    owner = self.identities.get(imported)
                    if owner is not None and owner not in closure:
                        pending.append(owner)
        return frozenset(closure)

    @classmethod
    def statements(cls, body: Iterable[ast.stmt]) -> Iterator[ast.stmt]:
        """Every statement at any depth: imports are statements, so the walk
        follows statement bodies and never enters an expression."""

        for statement in body:
            yield statement
            yield from cls.statements(cls.nested(statement))

    @staticmethod
    def nested(statement: ast.stmt) -> tuple[ast.stmt, ...]:
        """The statements one statement holds, by the statement kinds that hold any."""

        match statement:
            case ast.FunctionDef(body=body) | ast.AsyncFunctionDef(body=body) | ast.ClassDef(
                body=body
            ) | ast.With(body=body) | ast.AsyncWith(body=body):
                return tuple(body)
            case ast.If(body=body, orelse=orelse) | ast.For(
                body=body, orelse=orelse
            ) | ast.AsyncFor(body=body, orelse=orelse) | ast.While(body=body, orelse=orelse):
                return (*body, *orelse)
            case ast.Try(body=body, handlers=handlers, orelse=orelse, finalbody=finalbody) | (
                ast.TryStar(body=body, handlers=handlers, orelse=orelse, finalbody=finalbody)
            ):
                return (
                    *body,
                    *(statement for handler in handlers for statement in handler.body),
                    *orelse,
                    *finalbody,
                )
            case ast.Match(cases=cases):
                return tuple(statement for case in cases for statement in case.body)
        return ()

    @staticmethod
    def imported_module_names(
        identity: PythonModulePathIdentity, node: ast.AST
    ) -> tuple[str, ...]:
        match node:
            case ast.Import(names=aliases):
                return tuple(
                    ".".join(alias.name.split(".")[:end])
                    for alias in aliases
                    for end in range(1, alias.name.count(".") + 2)
                )
            case ast.ImportFrom(module=module, level=level, names=aliases):
                resolved = identity.resolve_import_from_module(
                    imported_module=module, level=level
                )
                if resolved is None:
                    return ()
                return (resolved, *(f"{resolved}.{alias.name}" for alias in aliases))
        return ()

    @cached_property
    def packages(self) -> frozenset[Path]:
        row_packages = tuple(tuple(row.package.split("/")) for row in self.table.rows)
        return frozenset(
            path
            for path in self.candidates
            if any(path.parent.parts[-len(package) :] == package for package in row_packages)
            or (
                path.name == "__init__.py"
                and any(seed.is_relative_to(path.parent) for seed in self.seeds)
            )
        )

    def policy(self) -> DomainSourcePathPolicy:
        stems = tuple(seed.stem.encode() for seed in self.seeds)
        spellers = frozenset(
            path for path, text in self.candidates.items() if any(stem in text for stem in stems)
        )
        return DomainSourcePathPolicy(
            include_tests=self.policy_base.include_tests,
            admitted=spellers | self.import_closure | self.packages,
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
        with them: a whole-module move takes every name it lists. Retained on the
        source state: every finding's layers ask it for every other module."""

        return self.context.retained_operation_derivation(
            (ClusterLayers.declared_names, path), lambda: self._declared_names(path)
        )

    def _declared_names(self, path: str) -> tuple[str, ...]:
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

        return self._runtime_edges[path]

    def any_scope_module_names(self, path: str) -> frozenset[str]:
        """The modules `path` imports at any scope, guarded and local included."""

        return self._any_scope_edges[path]

    @cached_property
    def _runtime_edges(self) -> dict[str, frozenset[str]]:
        """Each present module's load-time imports. The layers are rebuilt per finding,
        so the edges are retained on the source state, read once for all of them."""

        return {
            path: self.context.retained_operation_derivation(
                (ClusterLayers.imported_module_names, path),
                lambda path=path: self._edges(
                    path,
                    lambda parsed: ModuleImportScope.RUNTIME.import_statements(parsed.module),
                ),
            )
            for path in self.present_paths
        }

    @cached_property
    def _any_scope_edges(self) -> dict[str, frozenset[str]]:
        """Each present module's imports at any scope, retained on the source state."""

        return {
            path: self.context.retained_operation_derivation(
                (ClusterLayers.any_scope_module_names, path),
                lambda path=path: self._edges(
                    path, lambda parsed: parsed.fact(ModuleNodeIndex).imports
                ),
            )
            for path in self.present_paths
        }

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

        return self._mutual_pairs(self._runtime_edges)

    @cached_property
    def _any_scope_mutual_imports(self) -> tuple[tuple[str, str], ...]:
        return self._mutual_pairs(self._any_scope_edges)

    def deferred_cycles(self, path: str) -> tuple[tuple[str, str], ...]:
        """Pairs with `path` that import each other only through a guard or a
        function-local import: the move carries that cycle, it does not make it."""

        return tuple(
            pair
            for pair in self._any_scope_mutual_imports
            if path in pair and pair not in self.mutual_imports
        )

    def carried_debt(self, paths: tuple[str, ...]) -> str:
        """What a move of `paths` carries and names instead of making: each guarded
        or function-local import cycle through one of them."""

        return "".join(
            f" Debt carried: {left} and {right} import each other through a "
            "guarded or function-local import; break that cycle at its owner."
            for left, right in dict.fromkeys(
                pair for path in paths for pair in self.deferred_cycles(path)
            )
        )

    def _mutual_pairs(
        self, edges: dict[str, frozenset[str]]
    ) -> tuple[tuple[str, str], ...]:
        """Declaring modules that import each other, in declaring order. Read from
        each module's own import edges, so the work is the edges, not every pair."""

        graph = self.context.module_import_graph
        declaring = [path for path in self.present_paths if not self.emptied(path)]
        order = {path: index for index, path in enumerate(declaring)}
        by_module = {graph.module_name_for_file_path(path): path for path in declaring}
        pairs = {
            (path, other)
            for path in declaring
            for module in edges[path]
            if (other := by_module.get(module)) is not None
            and order[other] > order[path]
            and graph.module_name_for_file_path(path) in edges[other]
            and not self.joins(path, other)
        }
        return tuple(sorted(pairs, key=lambda pair: (order[pair[0]], order[pair[1]])))

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
    def live_paths(self) -> tuple[str, ...]:
        """The present modules that still take part: an emptied module whose own
        dissolve refuses stays where it is and orders nothing."""

        return tuple(path for path in self.present_paths if path not in self.dissolve_refusals)

    @cached_property
    def dissolve_candidates(self) -> tuple[str, ...]:
        return tuple(path for path in self.present_paths if self.emptied(path))

    @cached_property
    def dissolve_refusals(self) -> dict[str, str]:
        return self.refusals(self.dissolve_candidates, self.dissolve_recipe_of)

    @cached_property
    def dissolving_paths(self) -> tuple[str, ...]:
        """Every emptied module whose own dissolve does not refuse: they dissolve
        together in one stage, before anything else relocates, each consumer
        rewritten to the modules that declare what it imports."""

        return tuple(
            path for path in self.dissolve_candidates if path not in self.dissolve_refusals
        )

    def dissolve_recipe(self) -> RefactorRecipe:
        return self.dissolve_recipe_of(self.dissolving_paths)

    def dissolve_recipe_of(self, paths: tuple[str, ...]) -> RefactorRecipe:
        recipe = RefactorRecipe(
            recipe_id="dissolve-emptied",
            reason="Dissolve every emptied flat module into its package module, in one stage.",
        )
        for path in paths:
            recipe = recipe.with_operation(
                DissolveModuleOperation(
                    target=SourceRewriteTarget(file_path=path),
                    successor_path=self.destinations[path],
                )
            )
        return recipe

    @cached_property
    def batch_candidates(self) -> tuple[str, ...]:
        """The modules that could relocate together in one stage: every present
        module while each still declares names, keeps a module of its own and
        moves into a destination of its own. Each move then derives from the same
        starting state and names its peers' new homes (`BatchModuleMove`). Fewer
        than two, or a shared destination, moves one module per stage, as does a
        module importing a peer bare (`import pkg.peer`): its readers spell the
        old path."""

        declaring = tuple(path for path in self.live_paths if not self.emptied(path))
        destinations = [self.destinations[path] for path in declaring]
        if (
            len(declaring) < 2
            or len(declaring) != len(self.live_paths)
            or len(set(destinations)) != len(destinations)
            or self.imports_a_peer_bare(declaring)
        ):
            return ()
        return declaring

    @cached_property
    def batch_refusals(self) -> dict[str, str]:
        return self.refusals(self.batch_candidates, self.recipe_of)

    def refusals(
        self,
        candidates: tuple[str, ...],
        recipe_of: Callable[[tuple[str, ...]], RefactorRecipe],
    ) -> dict[str, str]:
        """The candidates of one stage whose own row refuses, each with its
        preflight's reason. Each row is checked against the state where every
        file the stage creates already exists (`rewrite_context`), so a refusal
        is the row's own and never "my destination is not there yet". A refused
        row leaves the stage and the rest re-form without it, until none
        refuses: the refused module stays where it is."""

        refused: dict[str, str] = {}
        while paths := tuple(path for path in candidates if path not in refused):
            found = self.row_refusals(recipe_of(paths))
            if not found:
                return refused
            refused.update(found)
        return refused

    def row_refusals(self, recipe: RefactorRecipe) -> dict[str, str]:
        """Each finding of the table re-reads the same batch, so the state keeps
        the answer: one projection and one preflight per row, not per finding."""

        return self.context.retained_operation_derivation(
            (recipe, ClusterLayers.row_refusals),
            lambda: self._row_refusals(recipe),
        )

    def _row_refusals(self, recipe: RefactorRecipe) -> dict[str, str]:
        paths = {
            operation.operation_key(): operation.target.file_path
            for operation in recipe.operations
        }
        try:
            projected = recipe.rewrite_context(self.context)
        except CodemodOperationPreflightError as error:
            return {paths[error.report.operation]: error.report.message}
        return {
            paths[report.operation]: report.message
            for operation in recipe.operations
            for report in operation.preflight_reports(projected)
            if not report.status.is_passed
        }

    @cached_property
    def batch_paths(self) -> tuple[str, ...]:
        """The candidates that relocate together: every one whose own move does
        not refuse, while two or more remain."""

        paths = tuple(
            path for path in self.batch_candidates if path not in self.batch_refusals
        )
        return paths if len(paths) >= 2 else ()

    def imports_a_peer_bare(self, paths: tuple[str, ...]) -> bool:
        graph = self.context.module_import_graph
        peers = frozenset(graph.module_name_for_file_path(path) for path in paths)
        return any(
            alias.name in peers
            and not DirectModuleImportBindingIdentity(module_name=alias.name).follows_move(
                alias.asname or alias.name.partition(".")[0]
            )
            for path in paths
            for node in self.context.parsed_module_for_source_path(path).fact(
                ModuleNodeIndex
            ).imports
            if isinstance(node, ast.Import)
            for alias in node.names
        )

    def batch_recipe(self) -> RefactorRecipe:
        return self.recipe_of(self.batch_paths)

    def recipe_of(self, paths: tuple[str, ...]) -> RefactorRecipe:
        moves = {
            path: BatchModuleMove(source_path=path, destination_path=self.destinations[path])
            for path in paths
        }
        recipe = RefactorRecipe(
            recipe_id="relocate-batch",
            reason="Relocate every name each flat module of the table declares into "
            "its package, in one stage." + self.carried_debt(paths),
        )
        for path, own in moves.items():
            destination = self.destinations[path]
            recipe = recipe.with_operation(
                (
                    RelocateSymbolsToModuleOperation
                    if destination in self.context.module_nodes_by_file_path
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

        if self.dissolving_paths:
            return self.dissolving_paths[0]
        tops = [
            path
            for path in self.live_paths
            if not self.upper_importers(path) and self.ready(path)
        ]
        return tops[0] if tops else None


class PackageMoveFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    PackageMoveConcept,
):
    """Relocate a flat module's names into its declared package, then dissolve it."""

    @classmethod
    def scan_policy(
        cls,
        policy: PythonSourcePathPolicy,
        roots: tuple[Path, ...],
        config: DetectorConfig,
    ) -> PythonSourcePathPolicy:
        return DestinationTable.from_texts(config.package_destination_tables).scan_domain(
            policy, roots
        )

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
        if source_path in layers.dissolve_refusals:
            return self.rejected_evaluation(
                f"stays: its dissolve refused: {layers.dissolve_refusals[source_path]}"
            )
        if source_path in layers.dissolving_paths[1:]:
            return self.rejected_evaluation(
                f"dissolves in {layers.dissolving_paths[0]}'s stage"
            )
        if source_path in layers.dissolving_paths:
            return self.executable_evaluation(layers.dissolve_recipe())
        batch = layers.batch_paths
        if batch and source_path in layers.batch_refusals:
            return self.rejected_evaluation(
                f"left {batch[0]}'s batch, refused: {layers.batch_refusals[source_path]}"
            )
        if batch:
            if source_path != batch[0]:
                return self.rejected_evaluation(f"moves in {batch[0]}'s batch")
            return self.executable_evaluation(layers.batch_recipe())
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
        return self.executable_evaluation(
            RefactorRecipe(
                recipe_id=f"relocate-{stem}",
                reason="Relocate every name the flat module declares into its package."
                + layers.carried_debt((source_path,)),
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
