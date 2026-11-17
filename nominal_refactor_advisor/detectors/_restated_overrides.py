"""A member its class declares exactly as the class it would inherit it from."""

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
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.refactors.restated_overrides import (
    RestatedOverrideFindingRecipeSynthesizer,
    inherited_restatements,
)
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag


@dataclass(frozen=True)
class RestatedOverrideCandidate:
    """One class and the members it restates, each with the class it would inherit it from."""

    indexed: CompactIndexedClass
    restated: tuple[tuple[str, str], ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (SourceLocation(self.indexed.file_path, self.indexed.line, self.indexed.qualname),)


class RestatedOverrideDetector(
    CompactClassRepositoryCandidateDetector[RestatedOverrideCandidate],
    RestatedOverrideFindingRecipeSynthesizer,
):
    """A class restating a member it would inherit unchanged."""

    finding_spec = finding_spec_template(
        PatternId.RESTATED_OVERRIDE,
        "A class restates a member it would inherit",
        "The class declares a member exactly as the next class in its order that declares it: the same declaration, bound to the same names, reading nothing of the class it sits in. Without the copy every lookup reaches the inherited one, so the copy is a second statement of one answer that has to be kept in step with it.",
        "the class without the restated member",
        "a member whose class-body normal form equals the next declaring class's, with neither reading its class",
        (CapabilityTag.MRO_ORDERING,),
        (ObservationTag.CLASS_FAMILY,),
    )

    def _candidates_from_compact_context(
        self,
        context: CompactClassRepositoryContext,
        config: DetectorConfig,
    ) -> tuple[RestatedOverrideCandidate, ...]:
        del config
        verdicts = context.cached(
            ClassOverlapVerdicts,
            lambda: ClassOverlapVerdicts(tuple(context.projections), context.class_index),
        )
        candidates = [
            RestatedOverrideCandidate(indexed, restated)
            for indexed in context.class_index.classes_by_symbol.values()
            if (restated := inherited_restatements(verdicts, indexed.symbol))
        ]
        return tuple(sorted(candidates, key=lambda candidate: (candidate.indexed.file_path, candidate.indexed.line)))

    def _finding_for_candidate(self, candidate: RestatedOverrideCandidate) -> RefactorFinding:
        return self.build_finding(
            f"`{candidate.indexed.qualname}` declares "
            + ", ".join(
                f"`{name}` exactly as `{inherited.rpartition('.')[2]}` does"
                for name, inherited in candidate.restated
            )
            + ": lookup reaches the same declaration without it.",
            candidate.evidence,
        )


__all__ = ("RestatedOverrideCandidate", "RestatedOverrideDetector")
