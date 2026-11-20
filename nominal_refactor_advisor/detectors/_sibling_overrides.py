"""Classes that answer one inherited member with the same declaration."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from nominal_refactor_advisor.class_body import ClassBodyDeclaration
from nominal_refactor_advisor.class_index import (
    ClassBodyDeclarationKind,
    CompactIndexedClass,
)
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
from nominal_refactor_advisor.refactors.restated_overrides import (
    PER_CLASS_NAMES,
    next_binder,
)
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag


@dataclass(frozen=True)
class SiblingOverrideCandidate:
    """One inherited member and the classes that override it with one declaration."""

    declaring: CompactIndexedClass
    """The nearest class every one of them inherits from that declares the member."""
    name: str
    answering: tuple[tuple[CompactIndexedClass, int], ...]
    """Each class giving the shared answer, with the line of its copy."""
    answering_otherwise: int
    """How many other classes override the same declaration with another answer."""
    unordered: tuple[str, ...]
    """The answering classes whose lookup order is not proved (a base has no source the
    scan reads): a base outside the scan may declare the member ahead of `declaring`."""
    home: CompactIndexedClass | None
    """The class that already exists to hold the answer: the nearest class beneath
    `declaring` that every answering class inherits and no class answering otherwise
    does, or `declaring` itself when it has no answer and nothing answers otherwise."""

    @property
    def unanswered(self) -> bool:
        """Whether the overridden declaration is an abstract hook: it has no answer of
        its own, so the shared one is the only answer these classes could inherit."""

        return self.name in self.declaring.abstract_method_names

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(
            SourceLocation(indexed.file_path, line, f"{indexed.qualname}.{self.name}")
            for indexed, line in self.answering
        )


@dataclass(frozen=True)
class Override:
    """One class's own declaration of a member some scanned ancestor declares."""

    symbol: str
    declaration: ClassBodyDeclaration
    ordered: bool
    """Whether the class's C3 order is proved."""


class SiblingOverrideDetector(CompactClassRepositoryCandidateDetector[SiblingOverrideCandidate]):
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
        index, bodies = verdicts.class_index, verdicts.bodies
        overriding: defaultdict[tuple[str, str], list[Override]] = defaultdict(list)
        for symbol in bodies.declarations_by_symbol:
            ancestors = tuple(
                ancestor for ancestor in index.ancestor_symbols(symbol) if ancestor in bodies.declarations_by_symbol
            )
            if not ancestors:
                continue
            try:
                order: tuple[str, ...] | None = verdicts.linearized(symbol)
            except NeedsInput:
                order = None
            for name, mine in bodies.declarations_of(symbol).items():
                if name in PER_CLASS_NAMES or mine.declaration.reads_its_class or mine.kind.is_field:
                    continue
                if order is not None and self._restates_next_binder(verdicts, order, name, mine):
                    continue
                for ancestor in ancestors:
                    theirs = bodies.declarations_of(ancestor).get(name)
                    if theirs is not None and not theirs.declaration.reads_its_class:
                        overriding[(ancestor, name)].append(Override(symbol, mine, order is not None))
        nearest: dict[tuple[str, frozenset[str]], SiblingOverrideCandidate] = {}
        for (declaring, name), overrides in overriding.items():
            theirs = bodies.declarations_of(declaring)[name]
            unanswered = name in index.class_for(declaring).abstract_method_names or theirs.kind is ClassBodyDeclarationKind.FIELD
            for answer in self._shared_answers(index, overrides):
                answering = frozenset(override.symbol for override in answer)
                otherwise = frozenset(override.symbol for override in overrides) - answering
                home = self._home(index, declaring, name, answering, otherwise, unanswered, bodies)
                if home is None and answer[0].declaration.kind is not ClassBodyDeclarationKind.METHOD:
                    continue
                key = (name, answering)
                held = nearest.get(key)
                # Several ancestors can declare the member: the answer belongs beneath the
                # nearest one, which every other one is an ancestor of.
                if held is None or held.declaring.symbol in index.ancestor_symbols(declaring):
                    nearest[key] = SiblingOverrideCandidate(
                        home=None if home is None else index.class_for(home),
                        declaring=index.class_for(declaring),
                        name=name,
                        answering=tuple(
                            sorted(
                                ((index.class_for(override.symbol), override.declaration.declaration.line) for override in answer),
                                key=lambda found: (found[0].file_path, found[1]),
                            )
                        ),
                        answering_otherwise=len(overrides) - len(answer),
                        unordered=tuple(
                            sorted(index.class_for(override.symbol).qualname for override in answer if not override.ordered)
                        ),
                    )
        return tuple(
            sorted(
                nearest.values(),
                key=lambda candidate: (candidate.evidence[0].file_path, candidate.evidence[0].line, candidate.name),
            )
        )

    @staticmethod
    def _home(index, declaring: str, name: str, answering: frozenset[str], otherwise: frozenset[str],
              unanswered: bool, bodies) -> str | None:
        """The existing class the shared answer belongs on, if one exists."""

        shared = frozenset.intersection(*(frozenset(index.ancestor_symbols(symbol)) for symbol in answering))
        beneath = [
            symbol
            for symbol in shared
            if symbol in bodies.declarations_by_symbol
            and declaring in index.ancestor_symbols(symbol)
            and name not in bodies.declarations_of(symbol)
            and not any(symbol in index.ancestor_symbols(other) for other in otherwise)
        ]
        if beneath:
            return max(beneath, key=lambda symbol: (len(index.ancestor_symbols(symbol)), symbol))
        return declaring if unanswered and not otherwise else None

    @staticmethod
    def _restates_next_binder(
        verdicts: ClassOverlapVerdicts, order: tuple[str, ...], name: str, mine: ClassBodyDeclaration
    ) -> bool:
        """Whether lookup without the member reaches the same declaration: that is
        `RestatedOverrideDetector`'s finding, one class's and not siblings'."""

        inherited = next_binder(verdicts, order[1:], name)
        return inherited is not None and mine.same_declaration_as(verdicts.bodies.declarations_of(inherited)[name])

    @staticmethod
    def _shared_answers(index, overrides: list[Override]) -> list[list[Override]]:
        """The overrides grouped by declaration, keeping each one two or more classes give
        that do not inherit it from one another."""

        shared: list[list[Override]] = []
        for override in overrides:
            for answer in shared:
                if override.declaration.same_declaration_as(answer[0].declaration):
                    answer.append(override)
                    break
            else:
                shared.append([override])
        independent = [
            [
                override
                for override in answer
                if not any(other.symbol in index.ancestor_symbols(override.symbol) for other in answer)
            ]
            for answer in shared
        ]
        return [answer for answer in independent if len(answer) > 1]

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
        )


__all__ = ("SiblingOverrideCandidate", "SiblingOverrideDetector")
