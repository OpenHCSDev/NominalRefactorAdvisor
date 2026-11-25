"""Drop a base another base of the same class already supplies.

`class N(Mixin, ABC)` where `Mixin` is an `ABC` lists `ABC` twice: once by name
and once through `Mixin`. When dropping the name leaves the class's C3 order
exactly as it is (`DeclaredC3.restated_bases`), every lookup is unchanged; what
can still see the drop is code that reads `__bases__` itself. Class creation
code is read by `ClassCreationProof`; every other read in the repository must
either test a named class other than the dropped one, or keep only bases that
are subclasses of a named class the dropped one is not.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from nominal_refactor_advisor.class_symbols import _referenced_bases
from nominal_refactor_advisor.codemod_declaration_source import (
    ClassHeaderSpanSourceAuthority,
)
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_source_edits import NominalSourceEdit
from nominal_refactor_advisor.models import RefactorFinding
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.refactor_concepts import RestatedBaseConcept
from nominal_refactor_advisor.refactors.class_attribute_reads import BasesReaders
from nominal_refactor_advisor.refactors.class_creation import ClassCreationProof
from nominal_refactor_advisor.refactors.class_overlap import (
    OverlapClass,
    obsolete_import_edits,
    require_static_c3,
    source_rewrite_target,
)
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingRecipeSynthesizer,
    PrimaryEvidenceActionKeysMixin,
)


@dataclass(frozen=True)
class RestatedBaseDrop:
    """One class's restated bases, proved droppable on the current source."""

    overlap: OverlapClass
    dropped: tuple[tuple[ast.expr, str], ...]
    """Each dropped base expression, with the base that supplies it."""

    @classmethod
    def proved(cls, snapshot: CodemodSourceSnapshot, overlap: OverlapClass) -> RestatedBaseDrop:
        """The class's restated bases, each proved: its order closes statically,
        its creation and every `__bases__` reader see nothing of the drop."""

        node = overlap.node
        referenced = tuple(_referenced_bases(node))
        if len(referenced) != len(node.bases):
            raise NeedsInput(f"`{overlap.qualname}` has a computed base", NeededInput.SOURCE_CHANGE)
        verdicts = snapshot.class_overlap_verdicts
        restated = verdicts.restated_bases(
            overlap.symbol,
            tuple(isinstance(base, (ast.Name, ast.Attribute)) for base, _parts in referenced),
        )
        if not restated:
            raise NeedsInput(f"`{overlap.qualname}` lists no base another base supplies", NeededInput.SOURCE_CHANGE)
        ancestry = require_static_c3(snapshot, overlap.facts)
        listed = verdicts.bases(overlap.symbol)
        dropped_symbols = tuple(listed[index] for index, _supplier in restated)
        header = ClassHeaderSpanSourceAuthority(node, snapshot.sources_by_file_path[overlap.file_path])
        if not header.can_rewrite:
            raise NeedsInput(f"`{overlap.qualname}` has an unrewritable header", NeededInput.OPERATION)
        creation = ClassCreationProof.dropping(snapshot, overlap.facts, ancestry, dropped_symbols)
        creation.require_unchanged()
        BasesReaders(snapshot).require_unread(overlap.facts, dropped_symbols, creation.installed_packages())
        return cls(overlap, tuple((referenced[index][0], supplier) for index, supplier in restated))

    def header_edits(self, snapshot: CodemodSourceSnapshot, rationale: str) -> tuple[NominalSourceEdit, ...]:
        node = self.overlap.node
        header = ClassHeaderSpanSourceAuthority(node, snapshot.sources_by_file_path[self.overlap.file_path])
        dropped = frozenset(id(base) for base, _supplier in self.dropped)
        return header.source_edits(
            header.with_base_items(
                tuple(header.item_source(base) for base in node.bases if id(base) not in dropped)
            ),
            file_path=self.overlap.file_path,
            rationale=rationale,
        )

    def rationale(self) -> str:
        return " ".join(
            f"Drops `{ast.unparse(base)}` from `{self.overlap.qualname}`'s bases: "
            f"`{supplier.rpartition('.')[2]}` already supplies it."
            for base, supplier in self.dropped
        )


@dataclass(frozen=True, kw_only=True)
class DropRestatedBasesOperation(RepositorySourceReprovedOperation):
    """Drop every base the target class, and each class beside it in its module,
    lists that another of its bases already supplies; then drop each import only
    those bases read, judged once for the whole module."""

    def source_edits_from_snapshot(self, snapshot: CodemodSourceSnapshot) -> tuple[NominalSourceEdit, ...]:
        target = OverlapClass.require(snapshot, self.target)
        drops = [RestatedBaseDrop.proved(snapshot, target)]
        for indexed in snapshot.class_overlap_verdicts.class_index.classes_by_symbol.values():
            if indexed.module_name != target.facts.module_name or indexed.symbol == target.symbol:
                continue
            try:
                drops.append(
                    RestatedBaseDrop.proved(snapshot, OverlapClass.for_symbol(snapshot, indexed.symbol))
                )
            except (NeedsInput, ValueError):
                continue
        rationale = self.rationale_text(" ".join(drop.rationale() for drop in drops))
        return (
            *(edit for drop in drops for edit in drop.header_edits(snapshot, rationale)),
            *obsolete_import_edits(
                snapshot,
                target.file_path,
                tuple(base for drop in drops for base, _supplier in drop.dropped),
                rationale,
            ),
        )


class RestatedBaseFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    RestatedBaseConcept,
):
    """One drop of the restated bases the finding's class lists."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        evidence = finding.primary_evidence
        if context is None or evidence is None:
            return self.rejected_evaluation("dropping a restated base needs source context and its class")
        snapshot = context.execution_snapshot()
        indexed = next(
            (
                indexed
                for indexed in snapshot.class_overlap_verdicts.class_index.classes_by_symbol.values()
                if indexed.qualname == evidence.symbol and indexed.file_path == evidence.file_path
            ),
            None,
        )
        if indexed is None:
            return self.rejected_evaluation(f"`{evidence.symbol}` left the source")
        operation = DropRestatedBasesOperation(target=source_rewrite_target(indexed))
        try:
            recipe = RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason="Another base already supplies it: the class order and every lookup stay the same.",
            ).with_proved_operation(operation, context)
        except ValueError as error:
            return self.stopped_evaluation(error)
        return self.executable_evaluation(recipe)


__all__ = (
    "DropRestatedBasesOperation",
    "RestatedBaseDrop",
    "RestatedBaseFindingRecipeSynthesizer",
)
