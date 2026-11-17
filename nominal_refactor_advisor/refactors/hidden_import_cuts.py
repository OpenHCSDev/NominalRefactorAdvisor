"""Cut the cycles hidden imports keep, one hub at a time.

A hidden import that `HiddenImportPlan` refuses for a cycle, `A` importing names
from `B` while `B` already loads `A`, says the names `A` reads are declared too
high: in a module that sits above `A` in the load order. A hub is such a `B`.
Its cut moves every declaration the refused readers take from it, with what
those read in `B`, down to a module below `B` and its importers that reaches no
reader (`ModuleImportGraph.shared_lower_module`). Each reader then imports that
module with no cycle, so promoting the import again finds it free. A hub whose
moved code reads a module that reaches a reader cannot go down; the need names
the declaration that reads back.
"""

from __future__ import annotations

import ast
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass

from nominal_refactor_advisor.audit.family import snake_case
from nominal_refactor_advisor.codemod_module_declarations import SourceTopLevelDeclarationIndex
from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.hidden_imports import ClosesCycle, HiddenImportPlan
from nominal_refactor_advisor.models import RefactorFinding
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.owner_candidates import ModuleImportGraph, common_package
from nominal_refactor_advisor.refactor_concepts import HiddenImportCutConcept
from nominal_refactor_advisor.refactors.class_overlap import (
    MovedClosure,
    annotation_modes,
    relocation_to,
)
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingRecipeSynthesizer,
    PrimaryEvidenceActionKeysMixin,
)
from nominal_refactor_advisor.refactors.module_move import (
    ImportedNameOwner,
    RelocateSymbolsToModuleOperation,
    RelocateSymbolsToNewModuleOperation,
)


@dataclass(frozen=True)
class HubEdge:
    """One import refused for a cycle, as a hub's cut sees it: the names it reads."""

    refusal: ClosesCycle
    names: tuple[str, ...]

    @property
    def reader(self) -> str:
        return self.refusal.hidden.module.module_name

    @staticmethod
    def hub_of(refusal: ClosesCycle) -> str | None:
        """The module the import reads names from; none for `import m`."""

        statement = refusal.hidden.statement
        return refusal.hidden.bindings[0].module if isinstance(statement, ast.ImportFrom) else None


@dataclass(frozen=True)
class HubPlan(ABC):
    """What cutting one hub does: the edges it dissolves, and what each edge it
    leaves needs."""

    hub: str
    dissolved: tuple[HubEdge, ...]
    needs: tuple[tuple[ClosesCycle, NeedsInput], ...]


@dataclass(frozen=True)
class HubCut(HubPlan):
    """Move `moved` from the hub to `destination`; every dissolved edge's reader
    then imports `destination` with no cycle."""

    file_path: str
    moved: tuple[str, ...]
    destination: str

    @property
    def yield_per_declaration(self) -> float:
        return len(self.dissolved) / len(self.moved)

    def operation(
        self, snapshot: CodemodSourceSnapshot
    ) -> RelocateSymbolsToModuleOperation | RelocateSymbolsToNewModuleOperation:
        return relocation_to(
            snapshot,
            SourceRewriteTarget(file_path=self.file_path, qualname=self.moved[0]),
            self.destination,
            self.moved,
        )


@dataclass(frozen=True)
class HubStays(HubPlan):
    """No cut of the hub dissolves an edge; every edge's need says why."""


@dataclass(frozen=True)
class HiddenImportCuts:
    """Each hub of one state's cycle-refused hidden imports, planned against the
    load-time graph with every accepted promotion in place."""

    snapshot: CodemodSourceSnapshot
    plan: HiddenImportPlan
    imports: ModuleImportGraph

    @classmethod
    def of(cls, snapshot: CodemodSourceSnapshot) -> HiddenImportCuts:
        plan = snapshot.module_set.fact(HiddenImportPlan)
        graph = snapshot.module_import_graph
        for hidden in plan.accepted:
            graph = graph.with_edges(hidden.module.module_name, hidden.targets)
        return cls(snapshot, plan, ModuleImportGraph(graph.import_edges_by_module))

    @property
    def edges_by_hub(self) -> dict[str, list[ClosesCycle]]:
        hubs: dict[str, list[ClosesCycle]] = defaultdict(list)
        for refusal in self.plan.refused:
            if isinstance(refusal, ClosesCycle):
                hubs[HubEdge.hub_of(refusal) or refusal.reaching].append(refusal)
        return hubs

    def hubs(self) -> tuple[HubPlan, ...]:
        """Every hub's plans, cuts first by edges dissolved per declaration moved."""

        plans = tuple(
            plan for hub, refusals in sorted(self.edges_by_hub.items()) for plan in self.hub(hub, refusals)
        )
        return tuple(
            sorted(
                plans,
                key=lambda plan: (
                    not isinstance(plan, HubCut),
                    -plan.yield_per_declaration if isinstance(plan, HubCut) else 0,
                    -len(plan.dissolved),
                    plan.hub,
                ),
            )
        )

    def hub(self, hub: str, refusals: list[ClosesCycle]) -> tuple[HubPlan, ...]:
        """One cut of everything the hub's readers take, when it frees them all;
        else one cut per reader that goes down alone, and what every other edge
        needs."""

        needs: list[tuple[ClosesCycle, NeedsInput]] = []
        edges: list[HubEdge] = []
        for refusal in refusals:
            try:
                edges.append(self.edge(hub, refusal))
            except NeedsInput as need:
                needs.append((refusal, need))
        source = self.snapshot.module_import_graph.unique_source_file_for_module_name(hub)
        if not edges or source is None:
            return (HubStays(hub, (), tuple(needs)),)
        whole = self.cut(hub, source.file_path, tuple(edges))
        readers = sorted({edge.reader for edge in edges})
        if (isinstance(whole, HubCut) and not whole.needs) or len(readers) == 1:
            return (whole, HubStays(hub, (), tuple(needs))) if needs else (whole,)
        alone = tuple(
            self.cut(hub, source.file_path, tuple(edge for edge in edges if edge.reader == reader))
            for reader in readers
        )
        cuts = tuple(plan for plan in alone if isinstance(plan, HubCut))
        left = (*needs, *(need for plan in alone for need in plan.needs))
        return (*cuts, HubStays(hub, (), left)) if left else cuts

    def cut(self, hub: str, file_path: str, edges: tuple[HubEdge, ...]) -> HubPlan:
        """Moving every name `edges` read, with its closure, down to one module."""

        closure = MovedClosure.of_roots(self.snapshot, file_path, tuple(sorted({n for e in edges for n in e.names})))
        readers = tuple(sorted({edge.reader for edge in edges}))
        users = (hub, *self.importers(hub, frozenset(closure.qualnames)))
        try:
            destination = self.destination(hub, users, readers, closure)
        except NeedsInput as need:
            return HubStays(hub, (), tuple((edge.refusal, need) for edge in edges))
        reached = self.reached_by(destination, closure)
        dissolved = tuple(edge for edge in edges if not any(self.imports.reaches(m, edge.reader) for m in reached))
        needs = tuple(
            (
                edge.refusal,
                NeedsInput(
                    f"`{destination}` would load `{back}`, which loads `{edge.reader}`",
                    NeededInput.SOURCE_CHANGE,
                ),
            )
            for edge in edges
            if edge not in dissolved
            for back in (next(m for m in reached if self.imports.reaches(m, edge.reader)),)
        )
        if not dissolved:
            return HubStays(hub, (), needs)
        return HubCut(hub, dissolved, needs, file_path, closure.qualnames, destination)

    def edge(self, hub: str, refusal: ClosesCycle) -> HubEdge:
        """The names the refused import reads from the hub, each declared there."""

        statement = refusal.hidden.statement
        if not isinstance(statement, ast.ImportFrom):
            raise NeedsInput(
                f"`import {statement.names[0].name}` reads attributes; a cut moves names an import names",
                NeededInput.OPERATION,
            )
        names = tuple(alias.name for alias in statement.names)
        source = self.snapshot.module_import_graph.unique_source_file_for_module_name(hub)
        declarations = (
            None
            if source is None
            else SourceTopLevelDeclarationIndex(
                source_path=source.file_path, module=self.snapshot.module_nodes_by_file_path[source.file_path]
            )
        )
        for name in names:
            owner = ImportedNameOwner.declaring_module(self.snapshot, hub, name)
            if owner != hub:
                raise NeedsInput(
                    f"`{name}` is declared in `{owner}`, not `{hub}`: import it from its owner first",
                    NeededInput.SOURCE_CHANGE,
                )
            if declarations is None or declarations.declaration_if_unambiguous(name) is None:
                raise NeedsInput(
                    f"`{hub}` binds `{name}` by more than one movable top-level declaration",
                    NeededInput.OPERATION,
                )
        return HubEdge(refusal, names)

    def importers(self, hub: str, moved: frozenset[str]) -> tuple[str, ...]:
        """The modules importing a moved name from the hub at load time."""

        return tuple(
            sorted(
                {
                    module.module_name
                    for module in self.snapshot.module_set
                    for statement in module.module.body
                    if isinstance(statement, ast.ImportFrom)
                    and module.module_name != hub
                    and any(alias.name in moved for alias in statement.names)
                    and module.module_path_identity.resolve_import_from_module(
                        imported_module=statement.module, level=statement.level
                    )
                    == hub
                }
            )
        )

    def destination(
        self, hub: str, users: tuple[str, ...], readers: tuple[str, ...], closure: MovedClosure
    ) -> str:
        """A module below the hub and every importer of the moved names that
        reaches no reader; else a new module of their package, named for the
        first moved declaration. Needs a source change, naming each declaration
        that reads back, when the moved code reads a module that reaches them."""

        lower = self.imports.shared_lower_module(
            users, closure.reads, above=readers, modes=annotation_modes(self.snapshot), source=hub
        )
        if lower is not None:
            return lower.module
        pinned = frozenset(self.imports.pinning_reads((*users, *readers), closure.reads))
        if pinned:
            raise NeedsInput(
                "; ".join(
                    f"`{hub}.{read.declaration}` reads `{read.name}` from `{read.module}`, so it cannot "
                    "sit below "
                    + ", ".join(
                        f"`{module}`"
                        for module in (*users, *readers)
                        if module == read.module or self.imports.reaches(read.module, module)
                    )
                    for read in closure.read_names
                    if read.module in pinned
                ),
                NeededInput.SOURCE_CHANGE,
            )
        package = common_package(users)
        home = f"{package}.{snake_case(closure.qualnames[0])}"
        if not package or home in self.imports.imports:
            raise NeedsInput(
                f"no module below `{hub}` and its importers avoids the cycle, and no new one can hold it",
                NeededInput.SOURCE_CHANGE,
            )
        return home

    def reached_by(self, destination: str, closure: MovedClosure) -> tuple[str, ...]:
        """What loading `destination` runs once the cut moves in: the module, its
        packages' `__init__`s, and every module the moved code reads."""

        parts = destination.split(".")
        packages = tuple(".".join(parts[:length]) for length in range(1, len(parts)))
        return tuple(
            module
            for module in (destination, *packages, *sorted(closure.reads))
            if module in self.imports.imports
        )


class HiddenImportCutFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    HiddenImportCutConcept,
):
    """One relocation of the declarations the finding's hub holds too high."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        evidence = finding.primary_evidence
        if context is None or evidence is None:
            return self.rejected_evaluation("cutting a hidden cycle needs source context and its hub")
        snapshot = context.execution_snapshot()
        cuts = HiddenImportCuts.of(snapshot)
        refusals = cuts.edges_by_hub.get(evidence.symbol)
        if not refusals:
            return self.rejected_evaluation(f"`{evidence.symbol}` keeps no cycle a hidden import closes")
        plans = cuts.hub(evidence.symbol, refusals)
        plan = next((plan for plan in plans if isinstance(plan, HubCut)), plans[-1])
        if not isinstance(plan, HubCut):
            need = NeedsInput(
                "; ".join(f"{refusal.hidden.location}: {need}" for refusal, need in plan.needs),
                *(kind for _refusal, need in plan.needs for kind in need.needs),
            )
            return self.stopped_evaluation(need)
        operation = plan.operation(snapshot)
        try:
            operation.source_edits(context)
        except ValueError as error:
            return self.stopped_evaluation(error)
        return self.executable_evaluation(
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason=(
                    f"Moving {len(plan.moved)} declaration(s) to `{plan.destination}` lets "
                    f"{len(plan.dissolved)} hidden import(s) load with no cycle."
                ),
            ).with_operation(operation)
        )


__all__ = (
    "HiddenImportCutFindingRecipeSynthesizer",
    "HiddenImportCuts",
    "HubCut",
    "HubEdge",
    "HubPlan",
    "HubStays",
)
