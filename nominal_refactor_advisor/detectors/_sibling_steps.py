"""Classes that override one inherited method differently and take the same step in it."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from nominal_refactor_advisor.class_index import (
    ClassBodyDeclarationKind,
    CompactIndexedClass,
)
from nominal_refactor_advisor.class_method_facets import CompactClassMethod
from nominal_refactor_advisor.class_overlap import ClassOverlapVerdicts
from nominal_refactor_advisor.detectors._base import (
    CompactClassRepositoryCandidateDetector,
    CompactClassRepositoryContext,
    DetectorConfig,
    finding_spec_template,
)
from nominal_refactor_advisor.models import RefactorFinding, SourceLocation
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.refactors.restated_overrides import PER_CLASS_NAMES
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag


@dataclass(frozen=True)
class OmittedStep:
    """A call every overriding class makes but one, which reads the member it is made on."""

    call: str
    omitting: CompactIndexedClass


@dataclass(frozen=True)
class SiblingOverrideStepCandidate:
    """One inherited method, the classes overriding it differently, and the steps they share."""

    declaring: CompactIndexedClass
    """The nearest class every one of them inherits from that declares the method."""
    name: str
    overriding: tuple[tuple[CompactIndexedClass, int], ...]
    """Each overriding class, with the line of its override."""
    restated: tuple[str, ...]
    """The calls through the receiver that every override makes, by member path."""
    omitted: tuple[OmittedStep, ...]

    @property
    def unanswered(self) -> bool:
        """Whether the overridden declaration is an abstract hook: the ancestor runs no
        sequence of its own, so every step of it is written in the overriding classes."""

        return self.name in self.declaring.abstract_method_names

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(
            SourceLocation(indexed.file_path, line, f"{indexed.qualname}.{self.name}")
            for indexed, line in self.overriding
        )


class SiblingOverrideStepDetector(CompactClassRepositoryCandidateDetector[SiblingOverrideStepCandidate]):
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
        index, bodies = verdicts.class_index, verdicts.bodies
        methods: dict[str, dict[str, CompactClassMethod]] = defaultdict(dict)
        for projection in context.projections:
            for method in projection.class_methods:
                methods[method.class_symbol][method.method_name] = method
        declared = frozenset(name for owned in methods.values() for name in owned)
        """Every method name some scanned class declares: what a call must name to be a step
        of the scanned code and not a call into a library the classes sit on."""
        overriding: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
        for symbol in bodies.declarations_by_symbol:
            mine = methods.get(symbol)
            if not mine:
                continue
            abstract = index.class_for(symbol).abstract_method_names
            for name, declaration in bodies.declarations_of(symbol).items():
                if (
                    name in PER_CLASS_NAMES
                    or name in abstract
                    or name not in mine
                    or declaration.kind is not ClassBodyDeclarationKind.METHOD
                ):
                    continue
                for ancestor in index.ancestor_symbols(symbol):
                    if ancestor in bodies.declarations_by_symbol and name in bodies.declarations_of(ancestor):
                        overriding[(ancestor, name)].append(symbol)
        nearest: dict[tuple[str, frozenset[str]], SiblingOverrideStepCandidate] = {}
        read: dict[str, frozenset[str]] = {}
        for (declaring, name), symbols in overriding.items():
            independent = [
                symbol for symbol in symbols if not any(other in index.ancestor_symbols(symbol) for other in symbols)
            ]
            if len(independent) < 2:
                continue
            declarations = [bodies.declarations_of(symbol)[name] for symbol in independent]
            if all(declaration.same_declaration_as(declarations[0]) for declaration in declarations[1:]):
                continue  # one declaration: `SiblingOverrideDetector`'s finding
            steps = {symbol: frozenset(methods[symbol][name].receiver_steps.calls) - {name} for symbol in independent}
            restated = frozenset.intersection(*steps.values())
            omitted = tuple(
                OmittedStep(call, index.class_for(symbol))
                for symbol in sorted(independent)
                for call in sorted(self._made_by_every_other(steps, symbol))
                if "." in call
                and call.rsplit(".", 1)[1] in declared
                and (member := call.split(".", 1)[0]) not in {made.split(".", 1)[0] for made in steps[symbol]}
                and member in self._members_read(index, methods, symbol, read)
            )
            if not restated and not omitted:
                continue
            key = (name, frozenset(independent))
            held = nearest.get(key)
            # Several ancestors can declare the method: the sequence belongs on the nearest
            # one, which every other one is an ancestor of.
            if held is None or held.declaring.symbol in index.ancestor_symbols(declaring):
                nearest[key] = SiblingOverrideStepCandidate(
                    declaring=index.class_for(declaring),
                    name=name,
                    overriding=tuple(
                        sorted(
                            ((index.class_for(symbol), methods[symbol][name].line) for symbol in independent),
                            key=lambda found: (found[0].file_path, found[1]),
                        )
                    ),
                    restated=tuple(sorted(restated)),
                    omitted=omitted,
                )
        return tuple(
            sorted(
                nearest.values(),
                key=lambda candidate: (candidate.evidence[0].file_path, candidate.evidence[0].line, candidate.name),
            )
        )

    @staticmethod
    def _made_by_every_other(steps: dict[str, frozenset[str]], symbol: str) -> frozenset[str]:
        """The calls every override but `symbol`'s makes and `symbol`'s does not."""

        others = [calls for other, calls in steps.items() if other != symbol]
        return frozenset.intersection(*others) - steps[symbol]

    @staticmethod
    def _members_read(
        index, methods: dict[str, dict[str, CompactClassMethod]], symbol: str, read: dict[str, frozenset[str]]
    ) -> frozenset[str]:
        """The members of the receiver that some method of the class, or of a scanned
        ancestor, reads: what the class has to call a step on. Kept in `read` by class."""

        if symbol not in read:
            read[symbol] = frozenset(
                member
                for owner in (symbol, *index.ancestor_symbols(symbol))
                for method in methods.get(owner, {}).values()
                for member in method.receiver_steps.members
            )
        return read[symbol]

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
