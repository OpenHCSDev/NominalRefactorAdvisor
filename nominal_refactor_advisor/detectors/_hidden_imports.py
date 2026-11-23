"""Imports of scanned modules the load-time graph does not see."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from nominal_refactor_advisor.detectors._base import (
    CompactContextCandidateDetector,
    DetectorConfig,
    finding_spec_template,
)
from nominal_refactor_advisor.hidden_imports import (
    CompactHiddenImportPlan,
    HiddenImportModuleFacts,
    HiddenImportModuleFamily,
)
from nominal_refactor_advisor.models import RefactorFinding, SourceLocation
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.refactors.hidden_imports import (
    HiddenImportFindingRecipeSynthesizer,
)
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag


class CompactHiddenImportCandidateDetector[HiddenImportCandidateT](
    CompactContextCandidateDetector[
        HiddenImportModuleFacts,
        CompactHiddenImportPlan,
        HiddenImportCandidateT,
    ],
):
    """Candidate detector sharing one compact hidden-import plan across detectors."""

    module_projection_family = HiddenImportModuleFamily
    compact_shared_context_builder = staticmethod(
        CompactHiddenImportPlan.from_projections
    )

    @classmethod
    def _compact_context_from_projections(
        cls,
        projections: Sequence[HiddenImportModuleFacts],
        config: DetectorConfig,
    ) -> CompactHiddenImportPlan:
        return CompactHiddenImportPlan.from_projections(projections, config)

    @classmethod
    def _compact_context_from_shared(
        cls,
        context: object | None,
    ) -> CompactHiddenImportPlan:
        return CompactHiddenImportPlan.require(context)


@dataclass(frozen=True)
class HiddenImportCandidate:
    """One module's hidden imports, as the plan judged them."""

    file_path: str
    module_name: str
    line: int
    accepted: int
    reaching: tuple[str, ...]
    """The module each import refused for a cycle would load the importer through."""
    unpromotable: int


class HiddenImportDetector(
    HiddenImportFindingRecipeSynthesizer,
    CompactHiddenImportCandidateDetector[HiddenImportCandidate],
):
    """A module whose guarded or function-local imports close no load-time cycle."""

    candidate_type = HiddenImportCandidate
    finding_spec = finding_spec_template(
        PatternId.HIDDEN_IMPORT,
        "A module hides imports that close no cycle",
        "The module imports a scanned module under `TYPE_CHECKING` or inside a function, and importing it at the top closes no load-time cycle, even with every other accepted promotion in place: the hidden edge guards against a cycle that is not there.",
        "the imports at the module's top",
        "a guarded or function-local import of a scanned module whose targets reach no module that loads the importer, judged against the graph with every earlier accepted import added",
        (CapabilityTag.PROVENANCE,),
        (ObservationTag.PARTIAL_VIEW,),
    )

    def _candidates_from_compact_context(
        self,
        context: CompactHiddenImportPlan,
        config: DetectorConfig,
    ) -> tuple[HiddenImportCandidate, ...]:
        del config
        return tuple(
            HiddenImportCandidate(
                summary.file_path,
                summary.module_name,
                summary.first_accepted_line,
                summary.accepted_count,
                summary.cycle_reaching,
                summary.unpromotable_count,
            )
            for summary in context.module_summaries
        )

    def _finding_for_candidate(self, candidate: HiddenImportCandidate) -> RefactorFinding:
        kept = (
            f"; {len(candidate.reaching)} stay, closing a cycle through "
            + ", ".join(f"`{module}`" for module in sorted(set(candidate.reaching)))
            if candidate.reaching
            else ""
        )
        unpromotable = f"; {candidate.unpromotable} need a source change first" if candidate.unpromotable else ""
        return self.build_finding(
            f"`{candidate.module_name}` hides {candidate.accepted} import(s) that close no cycle{kept}{unpromotable}.",
            (SourceLocation(candidate.file_path, candidate.line, candidate.module_name),),
        )


__all__ = (
    "CompactHiddenImportCandidateDetector",
    "HiddenImportCandidate",
    "HiddenImportDetector",
)
