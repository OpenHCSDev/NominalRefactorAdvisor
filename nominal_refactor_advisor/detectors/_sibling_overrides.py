"""Classes that answer one inherited member with the same declaration."""

from __future__ import annotations

from nominal_refactor_advisor.class_overlap import ClassOverlapVerdicts
from nominal_refactor_advisor.detectors._base import (
    CompactClassRepositoryCandidateDetector,
    CompactClassRepositoryContext,
    DetectorConfig,
    finding_spec_template,
)
from nominal_refactor_advisor.models import RefactorFinding, SiblingOverrideMetrics
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.refactors.sibling_overrides import (
    SiblingOverrideCandidate,
    SiblingOverrideFindingRecipeSynthesizer,
    sibling_override_candidates,
)
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag


class SiblingOverrideDetector(
    CompactClassRepositoryCandidateDetector[SiblingOverrideCandidate],
    SiblingOverrideFindingRecipeSynthesizer,
):
    """Two or more classes overriding one inherited member with the same declaration.

    `RestatedOverrideDetector` finds a class restating what it would inherit, and
    `MissingMixinDetector` two classes declaring a block neither inherits. This is the
    case between them: the member is inherited, each class overrides it, and the
    overrides are one declaration. It is how two frontends of one client drift (Toad's
    Textual and Qt conversations each implemented `thread_scope`, `status_thread` and
    `history_loading` identically under one abstract parent, 2026-11-20).

    A class is grouped under every scanned ancestor that declares the member, so two
    classes are found whichever of several declarations each reaches first, and a class
    whose order is not proved (a widget over a C-extension base) is still found, with
    that said: the frontends this is for are exactly such classes.

    A value several classes share (`bold = True`) is a row of the family's value table,
    not a second copy of behaviour (`class_overlap`'s rule), unless a class already
    exists to hold it: then the row is written once per class for no reason.
    """

    finding_spec = finding_spec_template(
        PatternId.SIBLING_OVERRIDE,
        "Classes override one inherited member with the same declaration",
        "Each class answers the member its ancestor declares, and the answers are one declaration: the same syntax, bound to the same names, "
        "reading no class body. The answer exists once per class, so a change to it is made in every copy or the classes drift.",
        "the answer declared once, on the ancestor or on a class these alone share beneath it",
        "members whose class-body normal forms are equal, each overriding a declaration of one shared ancestor, none reading its class",
        (CapabilityTag.MRO_ORDERING,),
        (ObservationTag.CLASS_FAMILY,),
    )

    def _candidates_from_compact_context(
        self,
        context: CompactClassRepositoryContext,
        config: DetectorConfig,
    ) -> tuple[SiblingOverrideCandidate, ...]:
        del config
        verdicts = context.cached(
            ClassOverlapVerdicts,
            lambda: ClassOverlapVerdicts(tuple(context.projections), context.class_index),
        )
        return sibling_override_candidates(verdicts)

    def _finding_for_candidate(self, candidate: SiblingOverrideCandidate) -> RefactorFinding:
        classes = ", ".join(f"`{indexed.qualname}`" for indexed, _ in candidate.answering)
        declaring = candidate.declaring.qualname
        owner = (
            f"`{declaring}.{candidate.name}` is an abstract hook, so this is the only answer they could inherit"
            if candidate.unanswered
            else f"`{declaring}.{candidate.name}` answers otherwise"
        )
        others = (
            f"; {candidate.answering_otherwise} other class(es) override it with another answer"
            if candidate.answering_otherwise
            else ""
        )
        unordered = (
            "; the lookup order of "
            + ", ".join(f"`{name}`" for name in candidate.unordered)
            + " is not proved (a base has no source the scan reads), so a base outside the scan may declare the member first"
            if candidate.unordered
            else ""
        )
        home = (
            ""
            if candidate.home is None
            else f"; `{candidate.home.qualname}` already exists to hold it: every one of them inherits it"
            + (" and no class answering otherwise does" if candidate.answering_otherwise else "")
        )
        return self.build_finding(
            f"{classes} each override `{declaring}.{candidate.name}` with the same declaration: {owner}{others}{home}{unordered}.",
            candidate.evidence,
            metrics=SiblingOverrideMetrics(
                member_name=candidate.name,
                declaring_qualname=declaring,
                declaring_symbol=candidate.declaring.symbol,
                sibling_count=len(candidate.answering),
            ),
        )


__all__ = ("SiblingOverrideCandidate", "SiblingOverrideDetector")
