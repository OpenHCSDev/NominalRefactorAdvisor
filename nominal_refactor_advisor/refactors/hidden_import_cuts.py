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
from nominal_refactor_advisor.codemod_payload import (
    RequiredStringPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_source_edits import NominalSourceEdit
from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.composition_layering import CompositionRootLayering
from nominal_refactor_advisor.hidden_imports import ClosesCycle, HiddenImportPlan
from nominal_refactor_advisor.models import RefactorFinding
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.refactors.type_only_readers import (
    RetypeTypeOnlyReadsOperation,
    TypeOnlyRead,
    TypeOnlyReadRefused,
)
from nominal_refactor_advisor.owner_candidates import ModuleImportGraph, common_package
from nominal_refactor_advisor.refactor_concepts import HiddenImportCutConcept, LayeringViolationConcept
from nominal_refactor_advisor.refactors.class_overlap import (
    MovedClosure,
    RelocationStep,
    annotation_modes,
    in_order_edits,
    in_order_states,
    relocation_to,
)
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingRecipeSynthesizer,
    PrimaryEvidenceActionKeysMixin,
)
from nominal_refactor_advisor.refactors.hidden_imports import PromoteHiddenImportsOperation
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


class ReadsBack(NeedsInput):
    """The moved code reads a module that reaches a reader, so no module below
    the readers can hold it. Each kind is one shape of that read, and names the
    input that would close it."""

    shape = ""

    def __init__(self, hub: str, closure: MovedClosure, below: tuple[str, ...], pins: str) -> None:
        super().__init__(f"{self.head(hub, closure, below)}: {pins}", NeededInput.SOURCE_CHANGE)

    def head(self, hub: str, closure: MovedClosure, below: tuple[str, ...]) -> str:
        return self.shape.format(
            hub=hub,
            roots=", ".join(f"`{root}`" for root in closure.roots),
            below=", ".join(f"`{module}`" for module in below),
        )


class DraggedReadsBack(ReadsBack):
    """Only declarations the roots need, which move with them, read back: the
    roots go down once they stop needing those, or those stop reading back."""

    shape = "`{hub}`'s {roots} read nothing back themselves; declarations they drag along do"


class SoleReaderKept(ReadsBack):
    """The moved code reads back only from the one reader, which cannot take it."""

    shape = "{roots} read back only from {below}, which cannot take them"


class OneReaderReadsBack(ReadsBack):
    """The moved code reads back toward one module that must load it: the
    modules between them are the cycle to cut, one inversion."""

    shape = "{roots} read back toward {below} alone"


class CentralClassReadsBack(ReadsBack):
    """The moved code reads back through more than `limit` modules: the class is
    central, so no move or contract below its readers holds what it reads, and
    its readers' layering is the input."""

    limit = 10

    def __init__(self, hub: str, closure: MovedClosure, through: tuple[str, ...], pins: str) -> None:
        self.through = through
        super().__init__(hub, closure, through, pins)

    def head(self, hub: str, closure: MovedClosure, below: tuple[str, ...]) -> str:
        roots = ", ".join(f"`{root}`" for root in closure.roots)
        return (
            f"{roots} read back through {len(self.through)} modules, more than {self.limit}: "
            "a central class, its readers' layering is the input"
        )


class HubFanReadsBack(ReadsBack):
    """The moved code reads back toward several modules that must load it:
    inverting what it reads is a decision about their layering, not a move."""

    shape = "{roots} read back toward {below}: a hub fan, inverting it is a design input"


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

    def operation(self) -> CutHiddenCycleOperation:
        return CutHiddenCycleOperation(
            target=SourceRewriteTarget(file_path=self.file_path, qualname=self.moved[0]),
            moved=self.moved,
            destination=self.destination,
        )


@dataclass(frozen=True, kw_only=True)
class CutHiddenCycleOperation(RelocationStep, RepositorySourceReprovedOperation):
    """Move `moved` from the target hub to `destination`, then import at the top
    every hidden import the moved state no longer needs to hide.

    The relocation redirects each importer of a moved name, the hidden ones
    included, in place; those it frees would stay guarded or function-local,
    a new hidden import the cut wrote. So the cut is the relocation followed by
    the promotion of every module whose hidden imports the plan, derived again
    on the moved source, now accepts, each step proved on the source the
    previous one left."""

    moved: tuple[str, ...] = codemod_payload_field(StringArrayPayloadValueCodec())
    destination: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def relocation(
        self, snapshot: CodemodSourceSnapshot
    ) -> RelocateSymbolsToModuleOperation | RelocateSymbolsToNewModuleOperation:
        return relocation_to(snapshot, self.target, self.destination, self.moved)

    def source_edits_from_snapshot(self, snapshot: CodemodSourceSnapshot) -> tuple[NominalSourceEdit, ...]:
        relocation = self.relocation(snapshot)
        (_, moved) = in_order_states(snapshot, (relocation,), f"cut-{self.moved[0]}")
        freed = moved.module_set.fact(HiddenImportPlan).freed_by(snapshot.module_set.fact(HiddenImportPlan))
        by_file: dict[str, list[str]] = defaultdict(list)
        for hidden in freed:
            by_file[hidden.module.file_path].append(hidden.identity)
        return in_order_edits(
            snapshot,
            (
                relocation,
                *(
                    PromoteHiddenImportsOperation(target=SourceRewriteTarget(file_path=path), only=tuple(identities))
                    for path, identities in sorted(by_file.items())
                ),
            ),
            f"cut-{self.moved[0]}",
            self.rationale_text(
                f"Move {', '.join(f'`{name}`' for name in self.moved)} to `{self.destination}` "
                f"and import the {len(freed)} hidden import(s) that move frees at their modules' tops."
            ),
        )


@dataclass(frozen=True)
class HubStays(HubPlan):
    """No cut of the hub dissolves an edge; every edge's need says why."""


@dataclass(frozen=True)
class HubRetype(HubPlan):
    """Readers that import the hub's classes only to annotate with them, each
    retyped to a base of the class below it: one batch, so a reader and the
    callee it hands the value to change together."""

    reads: tuple[TypeOnlyRead, ...]

    def operation(self) -> RetypeTypeOnlyReadsOperation:
        return RetypeTypeOnlyReadsOperation.of(self.reads)


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
            plan
            for hub, refusals in sorted(self.edges_by_hub.items())
            for plan in self.retyped(self.hub(hub, refusals))
        )
        return tuple(
            sorted(
                plans,
                key=lambda plan: (
                    not isinstance(plan, HubCut),
                    not isinstance(plan, HubRetype),
                    -plan.yield_per_declaration if isinstance(plan, HubCut) else 0,
                    -len(plan.dissolved),
                    plan.hub,
                ),
            )
        )

    def retyped(self, plans: tuple[HubPlan, ...]) -> tuple[HubPlan, ...]:
        """Each edge no cut dissolves whose reader imports the hub's classes only
        to annotate with them, retyped to a base below the reader instead,
        whatever kept the cut from it; a reader with no such base needs one."""

        reads: list[TypeOnlyRead] = []
        dissolved: list[HubEdge] = []
        out: list[HubPlan] = []
        for plan in plans:
            if isinstance(plan, HubCut | HubRetype):
                out.append(plan)
                continue
            left: list[tuple[ClosesCycle, NeedsInput]] = []
            for refusal, need in plan.needs:
                try:
                    found = TypeOnlyRead.of(self.snapshot, self.imports, refusal)
                except TypeOnlyReadRefused as missing:
                    left.append((refusal, missing))
                    continue
                if found is None:
                    left.append((refusal, need))
                    continue
                reads.extend(found)
                dissolved.append(HubEdge(refusal, tuple(read.name for read in found)))
            if left:
                out.append(HubStays(plan.hub, (), tuple(left)))
        if reads:
            out.append(HubRetype(plans[0].hub, tuple(dissolved), (), tuple(reads)))
        return tuple(out)

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
        cycle = tuple(sorted({refusal.hidden.module.module_name for refusal in refusals}))
        whole = self.cut(hub, source.file_path, tuple(edges), cycle)
        readers = sorted({edge.reader for edge in edges})
        if (isinstance(whole, HubCut) and not whole.needs) or len(readers) == 1:
            return (whole, HubStays(hub, (), tuple(needs))) if needs else (whole,)
        alone = tuple(
            self.cut(hub, source.file_path, tuple(edge for edge in edges if edge.reader == reader), cycle)
            for reader in readers
        )
        cuts = tuple(plan for plan in alone if isinstance(plan, HubCut))
        left = (*needs, *(need for plan in alone for need in plan.needs))
        return (*cuts, HubStays(hub, (), left)) if left else cuts

    def cut(
        self, hub: str, file_path: str, edges: tuple[HubEdge, ...], cycle: tuple[str, ...]
    ) -> HubPlan:
        """Moving every name `edges` read, with its closure, down to one module.
        The destination sits below every module `cycle` holds, each a reader of the
        hub through a refused import, not only the readers this cut frees: a
        declaration moved into another reader stays inside the cycle it leaves.
        Into the sole reader it reads back from, it leaves the cycle: that
        reader's import of it becomes its own name."""

        closure = MovedClosure.of_roots(self.snapshot, file_path, tuple(sorted({n for e in edges for n in e.names})))
        readers = tuple(sorted({edge.reader for edge in edges}))
        users = (hub, *self.importers(hub, frozenset(closure.qualnames)))
        try:
            destination = self.destination(hub, users, tuple(sorted({*readers, *cycle})), closure)
        except NeedsInput as need:
            return HubStays(hub, (), tuple((edge.refusal, need) for edge in edges))
        reached = self.reached_by(destination, closure)
        dissolved = tuple(
            edge
            for edge in edges
            if not any(m != edge.reader and self.imports.reaches(m, edge.reader) for m in reached)
            and (edge.reader == destination or edge.reader not in reached)
        )
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
        reaches no reader; else a new module of the hub's package, named for the
        first declaration the move is for. Needs a source change, naming each declaration
        that reads back, when the moved code reads a module that reaches them.

        When the moved names are misplaced in a composition root's layering, the
        module is one that layer loads, or the new module reads only modules it
        loads: a cut never moves a declaration to another place above its readers."""

        layer = self.snapshot.module_set.fact(CompositionRootLayering).layer_of(hub, closure.qualnames)
        lower = self.imports.shared_lower_module(
            users, closure.reads, above=readers, modes=annotation_modes(self.snapshot), source=hub, layer=layer
        )
        if lower is not None and (not layer or lower.module in layer):
            return lower.module
        pinned = frozenset(self.imports.pinning_reads((*users, *readers), closure.reads))
        if pinned:
            home = self.reader_home(hub, users, readers, closure, layer, pinned)
            if home is not None:
                return home
            backs = {
                read: tuple(
                    module
                    for module in (*users, *readers)
                    if module == read.module or self.imports.reaches(read.module, module)
                )
                for read in closure.read_names
                if read.module in pinned
            }
            pins = "; ".join(
                f"`{hub}.{read.declaration}` reads `{read.name}` from `{read.module}`, so it cannot "
                "sit below " + ", ".join(f"`{module}`" for module in back)
                for read, back in backs.items()
            )
            below = tuple(sorted({module for back in backs.values() for module in back}))
            if not {read.declaration for read in backs} & set(closure.roots):
                raise DraggedReadsBack(hub, closure, below, pins)
            if len(pinned) > CentralClassReadsBack.limit:
                raise CentralClassReadsBack(hub, closure, tuple(sorted(pinned)), pins)
            if len(below) == 1:
                raise OneReaderReadsBack(hub, closure, below, pins)
            raise HubFanReadsBack(hub, closure, below, pins)
        above_layer = sorted(closure.reads - layer) if layer else ()
        if above_layer:
            raise NeedsInput(
                f"`{hub}.{closure.roots[0]}` belongs to a composition-root layer that loads none of "
                + ", ".join(f"`{module}`" for module in above_layer)
                + ", which the moved code reads",
                NeededInput.SOURCE_CHANGE,
            )
        package = common_package((hub,))
        home = f"{package}.{snake_case(closure.roots[0])}"
        if not package or home in self.imports.imports:
            raise NeedsInput(
                f"no module below `{hub}` and its importers avoids the cycle, and no new one can hold it",
                NeededInput.SOURCE_CHANGE,
            )
        return home

    def reader_home(
        self,
        hub: str,
        users: tuple[str, ...],
        readers: tuple[str, ...],
        closure: MovedClosure,
        layer: frozenset[str],
        pinned: frozenset[str],
    ) -> str | None:
        """The sole reader, when the moved code reads back only from it: in the
        reader, what it reads is its own, and the cycle the two modules closed
        is gone. Only when nothing left in the hub reads a moved name (the hub
        would then import the reader), the reader loads no other importer of
        the moved names and can bind every other module the code reads, and,
        under a composition root, the reader is in the declarations' layer.
        A sole reader that fails one of those refuses, naming which."""

        if len(readers) != 1 or pinned != {readers[0]}:
            return None
        reader = readers[0]

        def kept(reason: str) -> SoleReaderKept:
            return SoleReaderKept(hub, closure, (reader,), reason)

        if layer and reader not in layer:
            raise kept(f"`{reader}` is outside the composition-root layer of the moved declarations")
        others = tuple(user for user in users if user not in (hub, reader))
        if loaded := next((user for user in others if self.imports.reaches(reader, user)), None):
            raise kept(f"`{reader}` loads `{loaded}`, another importer of the moved names")
        if unbound := next(
            (read for read in sorted(closure.reads - {reader}) if not self.imports.can_bind_at(reader, read, others)),
            None,
        ):
            raise kept(f"`{reader}` cannot import `{unbound}`, which the moved code reads, without a cycle")
        source = self.snapshot.module_import_graph.unique_source_file_for_module_name(hub)
        if source is None:
            return None
        moved = frozenset(closure.qualnames)
        staying = (
            statement
            for statement in self.snapshot.module_nodes_by_file_path[source.file_path].body
            if not (isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and statement.name in moved)
            and not isinstance(statement, ast.Import | ast.ImportFrom)
        )
        if read := next(
            (node.id for statement in staying for node in ast.walk(statement) if isinstance(node, ast.Name) and node.id in moved),
            None,
        ):
            raise kept(f"`{hub}` keeps reading `{read}`, so it would import `{reader}`")
        return reader

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
        plans = cuts.retyped(cuts.hub(evidence.symbol, refusals))
        plan = next((plan for plan in plans if isinstance(plan, HubCut | HubRetype)), plans[-1])
        if isinstance(plan, HubRetype):
            operation = plan.operation()
            try:
                recipe = RefactorRecipe(
                    recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                    reason=(
                        f"Annotating with a base below each reader lets {len(plan.dissolved)} guarded "
                        f"import(s) load with no cycle: {operation.described}."
                    ),
                ).with_proved_operation(operation, context)
            except ValueError as error:
                return self.stopped_evaluation(error)
            return self.executable_evaluation(recipe)
        if not isinstance(plan, HubCut):
            need = NeedsInput(
                "; ".join(f"{refusal.hidden.location}: {need}" for refusal, need in plan.needs),
                *(kind for _refusal, need in plan.needs for kind in need.needs),
            )
            return self.stopped_evaluation(need)
        operation = plan.operation()
        try:
            recipe = RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason=(
                    f"Moving {len(plan.moved)} declaration(s) to `{plan.destination}` lets "
                    f"{len(plan.dissolved)} hidden import(s) load with no cycle."
                ),
            ).with_proved_operation(operation, context)
        except ValueError as error:
            return self.stopped_evaluation(error)
        return self.executable_evaluation(recipe)


class LayeringViolationFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    LayeringViolationConcept,
):
    """The cut of one misplaced declaration, with what its module's own
    declarations need of it, into its home layer."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        evidence = finding.primary_evidence
        if context is None or evidence is None:
            return self.rejected_evaluation("moving a misplaced declaration needs source context and its declaration")
        snapshot = context.execution_snapshot()
        layering = snapshot.module_set.fact(CompositionRootLayering)
        module = snapshot.parsed_module_for_source_path(evidence.file_path).module_name
        if evidence.symbol == module:
            return self.rejected_evaluation(
                f"code below its layer imports the module `{module}` itself; a move takes named declarations",
                (NeededInput.OPERATION,),
            )
        reads = layering.by_declaration.get((module, evidence.symbol))
        if not reads:
            return self.rejected_evaluation(f"`{module}.{evidence.symbol}` is read from no layer below it")
        if not layering.layer_of(module, (evidence.symbol,)):
            return self.rejected_evaluation(
                f"no service at or below {', '.join(sorted({f'`{read.lower}`' for read in reads}))} "
                f"loads every module reading `{evidence.symbol}`",
                (NeededInput.SOURCE_CHANGE,),
            )
        cuts = HiddenImportCuts.of(snapshot)
        closure = MovedClosure.of_roots(snapshot, evidence.file_path, (evidence.symbol,))
        users = (module, *cuts.importers(module, frozenset(closure.qualnames)))
        try:
            destination = cuts.destination(module, users, tuple(sorted({read.reader for read in reads})), closure)
            operation = HubCut(module, (), (), evidence.file_path, closure.qualnames, destination).operation()
            recipe = RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason=(
                    f"Moving {len(closure.qualnames)} declaration(s) to `{destination}` puts "
                    f"`{evidence.symbol}` in the layer of the {len(reads)} read(s) of it."
                ),
            ).with_proved_operation(operation, context)
        except (NeedsInput, ValueError) as error:
            return self.stopped_evaluation(error)
        return self.executable_evaluation(recipe)


__all__ = (
    "CutHiddenCycleOperation",
    "HiddenImportCutFindingRecipeSynthesizer",
    "HiddenImportCuts",
    "HubCut",
    "HubEdge",
    "HubPlan",
    "HubStays",
    "LayeringViolationFindingRecipeSynthesizer",
)
