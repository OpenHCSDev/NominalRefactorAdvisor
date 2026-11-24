"""Classes that override one inherited method differently and take the same step in it."""

from __future__ import annotations

from nominal_refactor_advisor.class_overlap import ClassOverlapVerdicts
from nominal_refactor_advisor.detectors._base import (
    CompactClassRepositoryCandidateDetector,
    CompactClassRepositoryContext,
    DetectorConfig,
    finding_spec_template,
)
from nominal_refactor_advisor.models import RefactorFinding
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.refactors.sibling_override_steps import (
    OmittedStep,
    SiblingOverrideStepCandidate,
    SiblingOverrideStepFindingRecipeSynthesizer,
    sibling_override_step_candidates,
)
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag


class SiblingOverrideStepDetector(
    CompactClassRepositoryCandidateDetector[SiblingOverrideStepCandidate],
    SiblingOverrideStepFindingRecipeSynthesizer,
):
    """Two or more classes overriding one inherited method differently, with a step in common.

    `SiblingOverrideDetector` finds overrides that are one declaration. Most overrides of a
    hook are not: each class does its own thing around steps every one of them takes. Those
    steps are the ancestor's sequence, written in its leaves, and a leaf can leave one out
    with nothing to say so. Toad's Textual conversation answered `history_painted` by
    retiring its splash, telling the application a history was displayed (which froze the
    heap) and settling submissions; its Qt thread view answered it by settling submissions.
    The freeze was missing from one frontend for as long as nobody compared the two
    (2026-11-22).

    Two facts are reported for one overridden method:

    - a *restated* step: a call through the receiver (`self.submissions.settle()`) that every
      override makes. It belongs in the ancestor's method, with what differs left to a hook.
    - an *omitted* step: a call made on a member of the receiver (`self.app.f()`) by every
      override but one, which calls nothing on that member, while its class reads the
      member in some method of its own or of an ancestor: the collaborator is there and
      this override does not go to it. The method called is one some scanned class
      declares: `self.document.setVisible()` in one frontend's override is a call into
      its toolkit, which the other frontend has no reason to make (on Toad, of 40 hooks
      left across its frontends after two rounds of hoists, the next four read were all
      this: a window opened, a widget queried). Two overrides that each call something else on one
      member (`self.progress.setText()` and `self.progress.update()`) are each class's
      own spelling and are not reported: Toad's two frontends gave 107 such "omissions"
      in 88 hooks before this was required, nearly all a widget under one name in both.

    What is compared is the member path a call names on the receiver, so two classes that
    reach one collaborator under different attribute names are not matched, and a call made
    inside a nested function counts as the method's. Whether the classes hold the same
    object under a shared member name is not proved here: a member a common ancestor
    declares is, one each class assigns for itself is the reader's to check.
    """

    finding_spec = finding_spec_template(
        PatternId.SIBLING_OVERRIDE_STEP,
        "Classes override one inherited method differently and take the same step in it",
        "Each class answers the method its ancestor declares in its own way, and a call through the receiver is in every "
        "answer, or in every answer but one whose class reads the same member. The ancestor's sequence is written once per "
        "class, so a step added to it is added in every override or one class goes without, silently.",
        "the shared steps in the ancestor's method, calling a hook for what differs",
        "overrides of one declaration of a shared ancestor, not one declaration themselves, compared by the member paths each calls on its receiver",
        (CapabilityTag.MRO_ORDERING,),
        (ObservationTag.CLASS_FAMILY,),
    )

    def _candidates_from_compact_context(
        self,
        context: CompactClassRepositoryContext,
        config: DetectorConfig,
    ) -> tuple[SiblingOverrideStepCandidate, ...]:
        del config
        verdicts = context.cached(
            ClassOverlapVerdicts,
            lambda: ClassOverlapVerdicts(tuple(context.projections), context.class_index),
        )
        return sibling_override_step_candidates(verdicts, context.projections)

    def _finding_for_candidate(self, candidate: SiblingOverrideStepCandidate) -> RefactorFinding:
        classes = ", ".join(f"`{indexed.qualname}`" for indexed, _ in candidate.overriding)
        declaring = candidate.declaring.qualname
        owner = (
            f"`{declaring}.{candidate.name}` is an abstract hook, so the whole sequence is theirs"
            if candidate.unanswered
            else f"`{declaring}.{candidate.name}` has an answer of its own they replace"
        )
        restated = (
            ""
            if not candidate.restated
            else "; every one of them calls "
            + ", ".join(f"`self.{call}()`" for call in candidate.restated)
            + f": {len(candidate.restated)} step(s) of the sequence written once per class"
        )
        omitted = "".join(
            f"; `{step.omitting.qualname}` alone does not call `self.{step.call}()`, and it reads `self.{step.call.split('.', 1)[0]}`"
            for step in candidate.omitted
        )
        return self.build_finding(
            f"{classes} each override `{declaring}.{candidate.name}` differently: {owner}{restated}{omitted}.",
            candidate.evidence,
        )


__all__ = ("OmittedStep", "SiblingOverrideStepCandidate", "SiblingOverrideStepDetector")
