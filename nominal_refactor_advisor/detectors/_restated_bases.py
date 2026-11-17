"""A base another base of the same class already supplies."""

from __future__ import annotations

from dataclasses import dataclass

from nominal_refactor_advisor.class_index import CompactIndexedClass
from nominal_refactor_advisor.class_overlap import ClassOverlapVerdicts
from nominal_refactor_advisor.detectors._base import (
    CompactClassRepositoryCandidateDetector,
    CompactClassRepositoryContext,
    DetectorConfig,
    finding_spec_template,
)
from nominal_refactor_advisor.models import RefactorFinding, SourceLocation
from nominal_refactor_advisor.needed_input import NeedsInput
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.refactors.restated_bases import (
    RestatedBaseFindingRecipeSynthesizer,
)
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag


@dataclass(frozen=True)
class RestatedBaseCandidate:
    """One class and the bases it lists that another of its bases supplies."""

    indexed: CompactIndexedClass
    restated: tuple[tuple[str, str], ...]
    """Each restated base, with the base whose order already holds it."""

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (SourceLocation(self.indexed.file_path, self.indexed.line, self.indexed.qualname),)


class RestatedBaseDetector(
    CompactClassRepositoryCandidateDetector[RestatedBaseCandidate],
    RestatedBaseFindingRecipeSynthesizer,
):
    """A class listing a base its other bases already supply."""

    finding_spec = finding_spec_template(
        PatternId.RESTATED_BASE,
        "A class lists a base another of its bases supplies",
        "The class names a base that another of its bases already has in its order, and its order is the same without the name: the base list restates an ancestry the other base declares, a second copy that has to be kept in step with it.",
        "the base list without the restated base",
        "a plain-name base that another direct base's C3 order holds, whose removal leaves the class's C3 order unchanged",
        (CapabilityTag.MRO_ORDERING,),
        (ObservationTag.CLASS_FAMILY,),
    )

    def _candidates_from_compact_context(
        self,
        context: CompactClassRepositoryContext,
        config: DetectorConfig,
    ) -> tuple[RestatedBaseCandidate, ...]:
        del config
        verdicts = context.cached(
            ClassOverlapVerdicts,
            lambda: ClassOverlapVerdicts(tuple(context.projections), context.class_index),
        )
        candidates = []
        for indexed in context.class_index.classes_by_symbol.values():
            if not indexed.base_references or len(indexed.base_argument_sources) != len(
                indexed.base_references
            ):
                continue
            try:
                restated = verdicts.restated_bases(
                    indexed.symbol,
                    tuple(not arguments for arguments in indexed.base_argument_sources),
                )
                listed = verdicts.bases(indexed.symbol)
            except NeedsInput:
                continue
            if restated:
                candidates.append(
                    RestatedBaseCandidate(
                        indexed,
                        tuple((listed[index], supplier) for index, supplier in restated),
                    )
                )
        return tuple(sorted(candidates, key=lambda candidate: (candidate.indexed.file_path, candidate.indexed.line)))

    def _finding_for_candidate(self, candidate: RestatedBaseCandidate) -> RefactorFinding:
        return self.build_finding(
            f"`{candidate.indexed.qualname}` lists "
            + ", ".join(
                f"`{base.rpartition('.')[2]}`, which `{supplier.rpartition('.')[2]}` already supplies"
                for base, supplier in candidate.restated
            )
            + ": its order is the same without it.",
            candidate.evidence,
        )


__all__ = ("RestatedBaseCandidate", "RestatedBaseDetector")
