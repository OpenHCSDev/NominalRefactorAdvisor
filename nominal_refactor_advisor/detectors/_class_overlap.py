"""Duplicate classes and missing mixins (#68, shapes 5 and 6)."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

from ..class_overlap import (
    ClassOverlapCandidate,
    ClassOverlapMetrics,
    ClassOverlapShape,
    ClassOverlapVerdicts,
)
from ..models import RefactorFinding
from ..patterns import PatternId
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import (
    CompactClassRepositoryCandidateDetector,
    CompactClassRepositoryContext,
    DetectorConfig,
    finding_spec_template,
)


class ClassOverlapDetectorABC(
    CompactClassRepositoryCandidateDetector[ClassOverlapCandidate], ABC
):
    """One #68 shape read off the scan's shared class-overlap verdicts."""

    shape: ClassVar[ClassOverlapShape]

    def _candidates_from_compact_context(
        self,
        context: CompactClassRepositoryContext,
        config: DetectorConfig,
    ) -> tuple[ClassOverlapCandidate, ...]:
        del config
        verdicts = context.cached(
            ClassOverlapVerdicts,
            lambda: ClassOverlapVerdicts(tuple(context.projections), context.class_index),
        )
        return tuple(
            candidate
            for candidate in verdicts.candidates
            if candidate.shape is type(self).shape
        )

    def _finding_for_candidate(self, candidate: ClassOverlapCandidate) -> RefactorFinding:
        receipts = "; ".join(
            f"{receipt.weight:+.1f} {receipt.detail}" for receipt in candidate.receipts
        )
        return self.build_finding(
            (
                f"`{candidate.restating.qualname}` restates "
                f"{len(candidate.overlap.shared)} declaration(s) of "
                f"`{candidate.owner.qualname}` ({', '.join(candidate.shared_names)}); "
                f"move: {candidate.move.value}; probability "
                f"{candidate.probability:.2f} from receipts: {receipts}."
            ),
            candidate.evidence,
            metrics=ClassOverlapMetrics.of(candidate),
        )


class DuplicateClassDetector(ClassOverlapDetectorABC):
    """A class whose every declaration another class already makes."""

    shape = ClassOverlapShape.DUPLICATE_CLASS
    finding_spec = finding_spec_template(
        PatternId.DUPLICATE_CLASS,
        "A class restates a class that already exists",
        "Every declaration of this class is one another class already makes, with the same syntax and bindings. Either it is that class, or it composes it; keeping both bodies keeps two copies that drift apart.",
        "one declaration of the body, composed or referenced by every class that needs it",
        "a class body contained in another class's body, after normalizing positions and docstrings and resolving every free name",
        (CapabilityTag.NOMINAL_IDENTITY, CapabilityTag.SHARED_ALGORITHM_AUTHORITY),
        (ObservationTag.CLASS_FAMILY, ObservationTag.NORMALIZED_AST),
    )


class MissingMixinDetector(ClassOverlapDetectorABC):
    """Two classes sharing a block of declarations, each keeping its own."""

    shape = ClassOverlapShape.MISSING_MIXIN
    finding_spec = finding_spec_template(
        PatternId.MISSING_MIXIN,
        "Two classes declare the same block",
        "Two classes make the same block of declarations and each keeps declarations of its own. The block is one capability declared twice; it belongs in a mixin, or in the base the two alone share.",
        "one mixin or shared base declaring the block, composed by both classes",
        "three or more identical declarations shared by two classes that each keep differences",
        (CapabilityTag.SHARED_ALGORITHM_AUTHORITY, CapabilityTag.MRO_ORDERING),
        (ObservationTag.CLASS_FAMILY, ObservationTag.NORMALIZED_AST),
    )


__all__ = (
    "ClassOverlapDetectorABC",
    "DuplicateClassDetector",
    "MissingMixinDetector",
)
