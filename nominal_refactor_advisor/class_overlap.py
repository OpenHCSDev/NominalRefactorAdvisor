"""Classes that restate another class's body (#68, shapes 5 and 6).

`ClassBodyIndex` says which declarations two classes share. This module says
what that means and how sure NRA is: a ranked verdict with the receipts that
produced it, in the #67 form.

**The score.** Every receipt carries a fixed log-odds weight, stated on its
kind below; the confidence score is `1 / (1 + exp(-sum))` over the receipts,
from even odds. Nothing else enters it, so a reader can re-add it from the
receipts. It is not a calibrated probability: the weights are set by hand and
checked, not fitted, against the labeled verdicts in
`tests/fixtures/class_overlap_labels.json`, whose test pins the score and its
calibration error on that set so a weight cannot change unnoticed.

Shared declarations raise it, weighted by how much each says (a method more
than a valued declaration, a valued declaration more than a bare annotation).
A value two siblings share while a third sibling answers differently is a row
of the family's value table, not a shared block, and counts for nothing. Few
shared declarations, different class construction, and any use that tells the
classes apart by type lower it; no consumer using both and no common ancestor
are recorded but, on the labeled set, did not separate concepts from
coincidences, so they weigh nothing. Identity is the guard (#68): a class used
as a type test, a class pattern, a type key, or registered or tagged by its
class creation is a declaration of one fact even when its body is another
class's. NRA never drops such a pair silently; it lowers the score and names
the use.

**Moves.** When one body lies inside the other, the contained class owns the
block: the other composes it (inherits it and drops the restated
declarations), or, when the two bodies, bases and construction are the same,
is deleted and its uses point at the owner. When both keep declarations of
their own, the shared block moves into the base they alone share, or is
extracted into a mixin both compose. Enum members are identities of their
enum and never move.
"""

from __future__ import annotations

import ast
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import NamedTuple, Self
from enum import StrEnum
from functools import cached_property

from nominal_refactor_advisor.ast_projection import ExpressionReference
from nominal_refactor_advisor.case_root import Case
from nominal_refactor_advisor.class_body import (
    ClassBodyDeclaration,
    ResolvedClassBodyDeclaration,
    SharedClassBodyDeclaration,
    ClassBodyIndex,
    ClassBodyOverlap,
)
from nominal_refactor_advisor.class_index import (
    ClassBodyDeclarationKind,
    CompactClassFamilyIndex,
    CompactClassReferenceResolver,
    CompactIndexedClass,
    CompactModuleClassProjection,
    CompactRepositoryPublicExposureIndex,
    InstalledClassDeclarations,
    followed_reexports,
)
from nominal_refactor_advisor.class_references import (
    ClassReferenceRole,
    CompactClassReference,
)
from nominal_refactor_advisor.overlap_receipts import OverlapReceiptKind
from nominal_refactor_advisor.receipts import KeywordCuredReceipt, Receipt, ScoredByReceipts
from nominal_refactor_advisor.owner_candidates import (
    MINIMUM_SHARED_WEIGHT,
    BlockOwnerQuestion,
    ModuleImportGraph,
    OwnerCandidate,
    PickedOwner,
)
from nominal_refactor_advisor.models import FindingMetrics, ImpactDelta, SourceLocation


class ClassOverlapShape(StrEnum):
    """Which #68 shape one overlap is."""

    DUPLICATE_CLASS = "duplicate_class"
    MISSING_MIXIN = "missing_mixin"


class SubclassAnswers(ABC):
    """Which identity reads of a class a move changes the answer of.

    A read whose answer decides by the class's identity alone (an `isinstance`
    of a value, a class pattern, a type key) changes with every move that
    changes the identity. An `issubclass(C, tested)` whose `C` is a scanned
    class is decided by `C`'s ancestry, so it changes only if the move changes
    whether `C` descends from the tested class (paper1 Prop 3.3, Thm 4.8).
    """

    @abstractmethod
    def changes(self, tested_subclass: str | None) -> bool:
        """Whether a read testing the class `tested_subclass` (none when the
        read tests a value, or no scanned class) answers differently."""

        raise NotImplementedError


class EveryAnswer(SubclassAnswers):
    """The move changes the class's identity itself: every read may answer otherwise."""

    def changes(self, tested_subclass: str | None) -> bool:
        return True


@dataclass(frozen=True)
class AnswersFor(SubclassAnswers):
    """The move gives the tested class new descendants, `classes`: only a read
    asking about one of them, or about a value, answers otherwise."""

    classes: frozenset[str]

    @classmethod
    def gaining(
        cls, tested: str, composers: Iterable[str], class_index: CompactClassFamilyIndex
    ) -> Self:
        """Each composer, and each descendant of one, that does not descend
        from `tested` already becomes one when the composers inherit it."""

        already = {tested, *class_index.descendant_symbols(tested)}
        return cls(
            frozenset(
                symbol
                for composer in composers
                for symbol in (composer, *class_index.descendant_symbols(composer))
                if symbol not in already
            )
        )

    def changes(self, tested_subclass: str | None) -> bool:
        return tested_subclass is None or tested_subclass in self.classes


class ClassOverlapMove(Case):
    """The factoring an overlap calls for, and whose identities it changes.

    `reorders_fields` says whether the move puts moved dataclass fields ahead of
    the class's own, so a call passing fields by position would bind them to
    other parameters.
    """

    reorders_fields: bool

    @classmethod
    def obligation_classes(
        cls, owner_symbol: str, restating_symbols: Sequence[str]
    ) -> tuple[str, ...]:
        """The classes that name this obligation, in an order the move fixes.

        The owner keeps its place: the move keeps it and redirects or composes
        the others, so each choice of owner is its own obligation.
        """

        return (owner_symbol, *sorted(restating_symbols))

    @classmethod
    @abstractmethod
    def guarded_classes(
        cls, owner: CompactIndexedClass, restating: CompactIndexedClass
    ) -> tuple[CompactIndexedClass, ...]:
        """The classes whose type identity this move changes."""

        raise NotImplementedError

    @classmethod
    def changed_answers(
        cls,
        owner: CompactIndexedClass,
        restating: CompactIndexedClass,
        class_index: CompactClassFamilyIndex,
    ) -> SubclassAnswers:
        """Which identity reads of a guarded class the move answers otherwise."""

        return EveryAnswer()

    @classmethod
    def receiving_classes(
        cls, restating: Sequence[CompactIndexedClass]
    ) -> tuple[CompactIndexedClass, ...]:
        """The classes that come to have everything the owner has, its ancestry
        included: none, for a move that keeps no class as owner."""

        return ()


class DeleteAndRedirect(ClassOverlapMove):
    """Delete the restating class and point its uses at the owner: its identity is gone."""

    value = "delete_and_redirect"
    reorders_fields = False

    @classmethod
    def guarded_classes(
        cls, owner: CompactIndexedClass, restating: CompactIndexedClass
    ) -> tuple[CompactIndexedClass, ...]:
        return (restating, owner)

    @classmethod
    def receiving_classes(
        cls, restating: Sequence[CompactIndexedClass]
    ) -> tuple[CompactIndexedClass, ...]:
        """Every use of a restating class gets an owner instance instead."""

        return tuple(restating)


class ComposeExisting(ClassOverlapMove):
    """The restating class inherits the owner: the owner gains a descendant.

    Only a type test on the owner can answer differently afterwards, since the
    restating class's instances become owner instances. The restating class
    keeps its own type, so `isinstance(x, Restating)`, `type(x) is Restating`
    and a dispatch keyed by it answer as before.
    """

    value = "compose_existing"
    reorders_fields = False

    @classmethod
    def guarded_classes(
        cls, owner: CompactIndexedClass, restating: CompactIndexedClass
    ) -> tuple[CompactIndexedClass, ...]:
        return (owner,)

    @classmethod
    def changed_answers(
        cls,
        owner: CompactIndexedClass,
        restating: CompactIndexedClass,
        class_index: CompactClassFamilyIndex,
    ) -> SubclassAnswers:
        return AnswersFor.gaining(owner.symbol, (restating.symbol,), class_index)

    @classmethod
    def receiving_classes(
        cls, restating: Sequence[CompactIndexedClass]
    ) -> tuple[CompactIndexedClass, ...]:
        """Each restating class inherits the owner, and with it the owner's ancestry."""

        return tuple(restating)


class PromoteToSharedBase(ClassOverlapMove):
    """The block moves into the base the two alone share: no identity changes."""

    value = "promote_to_shared_base"
    reorders_fields = True

    @classmethod
    def obligation_classes(
        cls, owner_symbol: str, restating_symbols: Sequence[str]
    ) -> tuple[str, ...]:
        """No class is kept as owner: the block moves out of all of them."""

        return tuple(sorted((owner_symbol, *restating_symbols)))

    @classmethod
    def guarded_classes(
        cls, owner: CompactIndexedClass, restating: CompactIndexedClass
    ) -> tuple[CompactIndexedClass, ...]:
        return ()


class ExtractMixin(ClassOverlapMove):
    """Both compose a new mixin nobody tests yet: no identity changes."""

    value = "extract_mixin"
    reorders_fields = True

    @classmethod
    def obligation_classes(
        cls, owner_symbol: str, restating_symbols: Sequence[str]
    ) -> tuple[str, ...]:
        """No class is kept as owner: the block moves out of all of them."""

        return tuple(sorted((owner_symbol, *restating_symbols)))

    @classmethod
    def guarded_classes(
        cls, owner: CompactIndexedClass, restating: CompactIndexedClass
    ) -> tuple[CompactIndexedClass, ...]:
        return ()


FEW_SHARED_DECLARATIONS = 3
"""Below this many shared declarations the overlap is weak evidence of one concept."""


@dataclass(frozen=True)
class ReportingFloor:
    """Which overlaps become verdicts: enough shared weight, and enough shared
    declarations to read as a missing mixin. A scan reports above the declared
    floor; a move asking about one pair judges it with none."""

    weight: float = MINIMUM_SHARED_WEIGHT
    declarations: int = FEW_SHARED_DECLARATIONS

    def judged_repair(
        self, reported: PairRepair, first: CompactIndexedClass, second: CompactIndexedClass
    ) -> PairRepair:
        """The repair the verdict judges: the one the overlap's shape reports."""

        return reported


class PairRepair(NamedTuple):
    """The move a verdict judges, the class it keeps as owner and the classes
    that restate the owner's block."""

    move: type[ClassOverlapMove]
    owner: CompactIndexedClass
    restating: CompactIndexedClass
    co_restating: CompactIndexedClass | None


@dataclass(frozen=True)
class ExtractionRequest(ReportingFloor):
    """A step that declares the block once on a new mixin both classes compose
    asks about the pair as that move, whatever repair the overlap's shape
    reports. A duplicate class, or a block a third class already declares, can
    still be extracted; the step proves the extraction it makes, so the verdict
    judges that and not the other repair."""

    weight: float = 0.0
    declarations: int = 1

    def judged_repair(
        self, reported: PairRepair, first: CompactIndexedClass, second: CompactIndexedClass
    ) -> PairRepair:
        return PairRepair(ExtractMixin, first, second, None)


REPORTED = ReportingFloor()
NO_FLOOR = ReportingFloor(weight=0.0, declarations=1)
EXTRACTION = ExtractionRequest()



@dataclass(frozen=True)
class ClassUse:
    """One module's reference resolved to a scanned class, with the role it plays there."""

    file_path: str
    reference: CompactClassReference
    role: ClassReferenceRole
    tested_subclass: str | None
    """For an `issubclass` test, the scanned class it asks about."""

    @property
    def consumer(self) -> str:
        return self.reference.consumer

    @property
    def location(self) -> SourceLocation:
        return SourceLocation(
            self.file_path, self.reference.line, self.consumer or "<module>"
        )


@dataclass(frozen=True)
class ClassUseIndex:
    """Which consumers use each scanned class, and how.

    A reference is resolved only when a class it may name is asked about. Every
    answer the resolver gives ends in the same name as one of the reference's
    parts, or as the last part of an imported prefix's target once it follows
    every re-export rename (`followed_reexports`), so a reference is filed under
    those names and resolved as a whole, longest prefix first, exactly as when
    every reference is resolved up front."""

    resolver: CompactClassReferenceResolver
    references_by_terminal_name: Mapping[str, tuple[tuple[str, str, CompactClassReference], ...]]
    """Each reference with the file holding it and the module scope it resolves in."""

    @classmethod
    def from_projections(
        cls,
        projections: Sequence[CompactModuleClassProjection],
        class_index: CompactClassFamilyIndex,
    ) -> ClassUseIndex:
        scoped = CompactClassReferenceResolver.from_index(projections, class_index)
        resolver = replace(
            scoped,
            import_aliases_by_module_name={
                **scoped.import_aliases_by_module_name,
                **{
                    script.module_name: dict(script.import_aliases)
                    for projection in projections
                    for script in projection.script_scopes
                },
            },
        )
        scopes = tuple(
            (projection.file_path, scope_name, references)
            for projection in projections
            for scope_name, references in (
                (projection.module_name, projection.class_references),
                *((script.module_name, script.class_references) for script in projection.script_scopes),
            )
        )
        filed: dict[str, dict[tuple[int, int], tuple[str, str, CompactClassReference]]] = defaultdict(dict)
        for module_order, (file_path, module_name, references) in enumerate(scopes):
            aliases = resolver.import_aliases_by_module_name.get(module_name, {})
            for reference_order, reference in enumerate(references):
                parts = reference.reference_parts
                names = {*parts}
                if parts[0] in aliases:
                    for length in range(1, len(parts) + 1):
                        names.add(
                            followed_reexports(
                                ".".join((aliases[parts[0]], *parts[1:length])),
                                resolver.import_aliases_by_module_name,
                            ).rpartition(".")[2]
                        )
                for name in names:
                    filed[name][module_order, reference_order] = (file_path, module_name, reference)
        return cls(
            resolver,
            {name: tuple(found[key] for key in sorted(found)) for name, found in filed.items()},
        )

    def resolved(
        self, module_name: str, reference: CompactClassReference
    ) -> tuple[str, ClassReferenceRole] | None:
        """The class the reference's longest resolving prefix names, and its role."""

        for length in range(len(reference.reference_parts), 0, -1):
            symbol = self.resolver.symbol_for(
                module_name=module_name,
                reference_parts=reference.reference_parts[:length],
                allow_unique_unqualified=False,
            )
            if symbol is not None:
                return symbol, (
                    reference.role
                    if length == len(reference.reference_parts)
                    else ClassReferenceRole.OTHER
                )
        return None

    def uses_for(self, symbol: str) -> tuple[ClassUse, ...]:
        """Every reference to the class, in module and reference order."""

        found = self.__dict__.setdefault("_uses", {})
        if symbol not in found:
            found[symbol] = tuple(
                ClassUse(
                    file_path,
                    reference,
                    role,
                    self.resolver.symbol_for(
                        module_name=module_name,
                        reference_parts=reference.tested_subclass,
                        allow_unique_unqualified=False,
                    )
                    if reference.tested_subclass
                    else None,
                )
                for file_path, module_name, reference in self.references_by_terminal_name.get(
                    symbol.rpartition(".")[2], ()
                )
                for resolved in [self.resolved(module_name, reference)]
                if resolved is not None and resolved[0] == symbol
                for role in [resolved[1]]
            )
        return found[symbol]

    def constructions_of(
        self, indexed_class: CompactIndexedClass, *, positional: bool = False
    ) -> tuple[ClassUse, ...]:
        """Every call that constructs the class, its own methods' included: a
        `cls(...)` in its classmethod binds fields exactly as an outside call
        does, so a change to its fields breaks both. With `positional`, only
        the calls that bind by position or unpacking."""

        return tuple(
            use
            for use in self.uses_for(indexed_class.symbol)
            if use.role is ClassReferenceRole.CONSTRUCTION
            and (
                not positional
                or use.reference.positional_arguments
                or use.reference.unpacked_arguments
            )
        )

    def uses_of(self, indexed_class: CompactIndexedClass) -> tuple[ClassUse, ...]:
        """Uses outside the class's own body."""

        own_prefix = f"{indexed_class.qualname}."
        return tuple(
            use
            for use in self.uses_for(indexed_class.symbol)
            if not (
                use.file_path == indexed_class.file_path
                and (
                    use.consumer == indexed_class.qualname
                    or use.consumer.startswith(own_prefix)
                )
            )
        )

    def consumer_locations(
        self, indexed_class: CompactIndexedClass
    ) -> Mapping[tuple[str, str], SourceLocation]:
        """Each consumer of the class with the first place it uses it."""

        locations: dict[tuple[str, str], SourceLocation] = {}
        for use in self.uses_of(indexed_class):
            if use.consumer:
                locations.setdefault((use.file_path, use.consumer), use.location)
        return locations

    def consumers_of(self, indexed_class: CompactIndexedClass) -> frozenset[tuple[str, str]]:
        """Functions and classes that use the class; the module scope is a declaration site, not a consumer."""

        return frozenset(self.consumer_locations(indexed_class))


def _location(indexed_class: CompactIndexedClass, line: int | None = None, member: str | None = None) -> SourceLocation:
    symbol = indexed_class.qualname if member is None else f"{indexed_class.qualname}.{member}"
    return SourceLocation(indexed_class.file_path, indexed_class.line if line is None else line, symbol)


def _shared_locations(shared: SharedClassBodyDeclaration) -> tuple[SourceLocation, SourceLocation]:
    """Where each class's copy of one shared declaration is made."""

    return (
        _location(shared.first_owner, shared.first_line, shared.name),
        _location(shared.second_owner, shared.second_line, shared.name),
    )


def _construction(indexed_class: CompactIndexedClass) -> tuple[object, ...]:
    return (
        indexed_class.dataclass_declaration.runtime_declaration,
        indexed_class.metaclass_names,
        indexed_class.class_keyword_names,
        indexed_class.class_decorators_are_promotion_safe,
    )


LITERAL_FORMS = frozenset(("typing.Literal", "typing_extensions.Literal"))


def _literal_tags(
    indexed_class: CompactIndexedClass, import_aliases: Mapping[str, str]
) -> tuple[tuple[str, int, str], ...]:
    """Fields typed as one literal and defaulting to it: a tag that names the class."""

    tags = []
    for member in indexed_class.direct_member_declarations:
        if member.annotation_expression is None or member.value.constant_string is None:
            continue
        annotation = ast.parse(member.annotation_expression, mode="eval").body
        if not (
            isinstance(annotation, ast.Subscript)
            and isinstance(annotation.slice, ast.Constant)
            and annotation.slice.value == member.value.constant_string
        ):
            continue
        parts = ExpressionReference.decoded(annotation.value).attribute_chain
        if parts is None:
            continue
        qualified = ".".join((import_aliases.get(parts[0], parts[0]), *parts[1:]))
        if qualified in LITERAL_FORMS:
            tags.append((member.name, member.line, member.value.constant_string))
    return tuple(tags)


def _declared_identity_reasons(indexed_class: CompactIndexedClass) -> tuple[str, ...]:
    """Every way class creation itself makes this class an identity."""

    reasons = list(indexed_class.enum_declaration.identity_reasons)
    if indexed_class.class_keyword_names:
        reasons.append(
            f"class keywords {', '.join(indexed_class.class_keyword_names)} run at creation"
        )
    if indexed_class.metaclass_names:
        reasons.append(
            f"metaclass {', '.join(indexed_class.metaclass_names)} runs at creation"
        )
    if indexed_class.declares_autoregister_meta or indexed_class.is_registration_authority:
        reasons.append("it registers its subclasses")
    return tuple(reasons)


@dataclass(frozen=True)
class ClassOverlapCandidate(ScoredByReceipts):
    """One verdict: which class owns the block, which restates it, the move, the receipts."""

    shape: ClassOverlapShape
    move: type[ClassOverlapMove]
    owner: CompactIndexedClass
    restating: CompactIndexedClass
    overlap: ClassBodyOverlap
    receipts: tuple[Receipt, ...]
    co_restating: CompactIndexedClass | None = None
    owner_candidates: tuple[OwnerCandidate, ...] = ()
    """For a move that needs an owner decision, where the block could live (#67)."""

    @property
    def confidence(self) -> float:
        """The uncalibrated concept score: are these one concept? Guards weigh nothing here."""

        return 1.0 / (1.0 + math.exp(-self.log_odds))

    @property
    def shared_names(self) -> tuple[str, ...]:
        return tuple(shared.name for shared in self.overlap.shared)

    @property
    def inherited_names(self) -> frozenset[str]:
        """Names from the shared block that one class inherits instead of declaring."""
        return frozenset(shared.name for shared in self.overlap.inherited)

    @property
    def owned_shared_names(self) -> tuple[str, ...]:
        """Shared names both classes declare themselves (not inherited)."""
        inherited = self.inherited_names
        return tuple(name for name in self.shared_names if name not in inherited)

    @property
    def restating_classes(self) -> tuple[CompactIndexedClass, ...]:
        return tuple(
            indexed_class
            for indexed_class in (self.restating, self.co_restating)
            if indexed_class is not None
        )

    @property
    def restating_symbols(self) -> frozenset[str]:
        return frozenset(indexed_class.symbol for indexed_class in self.restating_classes)

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        first, second = self.overlap.first, self.overlap.second
        return (
            _location(self.owner),
            _location(self.restating),
            *(() if self.co_restating is None else (_location(self.co_restating),)),
            *(
                location
                for shared in self.overlap.shared
                for location in _shared_locations(shared)
            ),
        )


@dataclass(frozen=True)
class ClassOverlapMetrics(FindingMetrics, ScoredByReceipts):
    """One verdict as finding metrics: the move, the confidence score and every receipt."""

    shape: ClassOverlapShape
    move: type[ClassOverlapMove]
    owner_symbol: str
    restating_symbols: tuple[str, ...]
    shared_declaration_names: tuple[str, ...]
    confidence_score: float
    receipts: tuple[Receipt, ...]
    owner_candidates: tuple[OwnerCandidate, ...] = ()
    owner_pick: PickedOwner | None = None
    """The picker's choice among `owner_candidates`, from a picks file (#67)."""

    @classmethod
    def of(cls, candidate: ClassOverlapCandidate) -> ClassOverlapMetrics:
        return cls(
            shape=candidate.shape,
            move=candidate.move,
            owner_symbol=candidate.owner.symbol,
            restating_symbols=tuple(
                indexed_class.symbol for indexed_class in candidate.restating_classes
            ),
            shared_declaration_names=candidate.shared_names,
            confidence_score=round(candidate.confidence, 4),
            receipts=candidate.receipts,
            owner_candidates=candidate.owner_candidates,
        )

    @property
    def impact_delta(self) -> ImpactDelta:
        count = len(self.shared_declaration_names)
        return ImpactDelta(
            lower_bound_removable_loc=count,
            upper_bound_removable_loc=count,
            loci_of_change_before=2,
            loci_of_change_after=1,
        )

    @property
    def plan_class_names(self) -> tuple[str, ...]:
        return (self.owner_symbol, *self.restating_symbols)

    def finding_identity(
        self, summary: str, evidence: tuple[SourceLocation, ...]
    ) -> tuple[str, ...]:
        """The obligation: the move, the classes that restate the block, and the
        block.

        Scores, receipts and owner candidates are derived from it and stay out,
        so a pick keyed by this id survives rescoring and line moves.
        """

        return (
            self.move.value,
            ",".join(self.move.obligation_classes(self.owner_symbol, self.restating_symbols)),
            ",".join(sorted(self.shared_declaration_names)),
        )

    @property
    def plan_field_names(self) -> tuple[str, ...]:
        return self.shared_declaration_names


@dataclass(frozen=True)
class ClassOverlapVerdicts:
    """Every overlap worth a verdict in one scan, with its receipts."""

    projections: tuple[CompactModuleClassProjection, ...]
    class_index: CompactClassFamilyIndex

    @cached_property
    def bodies(self) -> ClassBodyIndex:
        return ClassBodyIndex.from_projections(self.projections)

    @cached_property
    def exposure(self) -> CompactRepositoryPublicExposureIndex:
        """What the scanned modules export, for reading a class outside them."""

        return CompactRepositoryPublicExposureIndex(self.projections)

    def ancestor(self, qualified_name: str) -> CompactIndexedClass | None:
        """The class a base names: the scan's, or else one an installed module
        declares in its `.py` source, read without running it, re-exports followed."""

        return self.class_index.class_for(qualified_name) or InstalledClassDeclarations.declaration(
            qualified_name, self.exposure
        )

    @cached_property
    def uses(self) -> ClassUseIndex:
        return ClassUseIndex.from_projections(self.projections, self.class_index)

    @cached_property
    def import_aliases_by_module(self) -> Mapping[str, Mapping[str, str]]:
        return {
            projection.module_name: dict(projection.import_aliases)
            for projection in self.projections
        }

    @cached_property
    def candidates(self) -> tuple[ClassOverlapCandidate, ...]:
        """A verdict per pair of classes: on their own declarations, or, when
        those are too few, on the declarations they have through their MROs."""

        return self.judged(
            self.bodies.overlaps,
            self._inheriting_pairs_of(self.bodies.classes_by_symbol),
        )

    def candidates_touching(self, symbols: frozenset[str]) -> tuple[ClassOverlapCandidate, ...]:
        """The verdicts `candidates` holds whose owner or restating classes include
        one of `symbols`, in the same order, judging only the pairs such a verdict
        can come from. A pair's verdict is owned by one of its classes or by a
        class declaring its shared declarations itself, so each such pair shares
        a declaration one of `symbols` makes, or is an inheriting pair starting
        from a class that inherits one. A verdict reads only its own overlap and
        the scan's indexes, so a pair judged alone is judged as it is among all
        of them; a move proving one composition asks this instead of judging
        every overlap in the scan on each virtual source."""

        if "candidates" in self.__dict__:
            found = self.candidates
        else:
            touching = self.__dict__.setdefault("_candidates_touching", {})
            if symbols not in touching:
                touching[symbols] = self.judged(
                    self.bodies.overlaps_sharing_with(symbols),
                    self._inheriting_pairs_of(self._inheritors_pairing_with(symbols)),
                )
            found = touching[symbols]
        return tuple(
            candidate
            for candidate in found
            if not symbols.isdisjoint((candidate.owner.symbol, *candidate.restating_symbols))
        )

    def requested_verdict(self, first: str, second: str) -> ClassOverlapCandidate | None:
        """The verdict on one pair a move asks about, judged without the reporting
        floor. `REPORTED` decides which overlaps a scan reports; a
        composition a move asks for, such as a new mixin's composer, is judged on
        its own pair with every other receipt and guard."""

        return self.verdict(self.bodies.overlap(first, second), NO_FLOOR)

    def requested_extractions_restating(
        self, anchor: str, names: Iterable[str]
    ) -> Iterator[ClassOverlapCandidate]:
        """The extraction of `names` out of `anchor` into a new mixin, judged on
        `anchor` and each class that declares every one of them itself: the pairs
        a step extracting those names can be asking about."""

        holders = [
            frozenset(
                self.bodies.holders.get(
                    self.declaration(anchor, name).require_proved(f"`{anchor}.{name}`").identity,
                    (),
                )
            )
            for name in names
        ]
        restaters = frozenset.intersection(*holders) - {anchor} if holders else frozenset()
        for other in sorted(restaters):
            if (verdict := self.verdict(self.bodies.overlap(anchor, other), EXTRACTION)) is not None:
                yield verdict

    def judged(
        self,
        overlaps: Iterable[ClassBodyOverlap],
        inheriting_pairs: frozenset[tuple[str, str]],
    ) -> tuple[ClassOverlapCandidate, ...]:
        """The verdicts on these overlaps, then on these inheriting pairs that
        share nothing of their own."""

        candidates: list[ClassOverlapCandidate] = []
        judged: set[tuple[str, str]] = set()
        for overlap in overlaps:
            candidate = self.verdict(overlap)
            if candidate is not None:
                candidates.append(candidate)
                judged.add((overlap.first.symbol, overlap.second.symbol))
        for first, second in sorted(inheriting_pairs - judged):
            overlap = self.bodies.overlap(first, second).with_inherited(
                self.resolved_declarations(first), self.resolved_declarations(second)
            )
            if (
                overlap.inherited
                and not self._restated_through_an_ancestor(overlap)
                and (candidate := self.verdict(overlap)) is not None
            ):
                candidates.append(candidate)
        return tuple(candidates)

    def resolved_declarations(
        self, symbol: str
    ) -> Mapping[str, ResolvedClassBodyDeclaration]:
        """A scanned class's declarations through the known prefix of its MRO;
        nothing for a class outside the scan."""

        resolved = self.__dict__.setdefault("_resolved_declarations", {})
        if symbol not in resolved:
            resolved[symbol] = (
                self.bodies.resolved_declarations_of(
                    self.class_index.mro_authority.lookup_prefix(symbol)
                )
                if symbol in self.bodies.classes_by_symbol
                else {}
            )
        return resolved[symbol]

    def declared_through_ancestry(self, symbol: str) -> Mapping[str, ResolvedClassBodyDeclaration]:
        """Every name the class or any scanned ancestor declares, with the first
        class along its ancestry to declare it. Whether a class has a member does
        not depend on which ancestor wins its lookup, so this reads past a base the
        lookup prefix cannot order."""

        found: dict[str, ResolvedClassBodyDeclaration] = {}
        for declaring in (symbol, *self.class_index.ancestor_symbols(symbol)):
            indexed = self.bodies.classes_by_symbol.get(declaring)
            if indexed is None:
                continue
            for name, declaration in self.bodies.declarations_of(declaring).items():
                found.setdefault(name, ResolvedClassBodyDeclaration(declaration, indexed))
        return found

    def _restated_through_an_ancestor(self, overlap: ClassBodyOverlap) -> bool:
        """One class has every shared copy exactly as one of its ancestors has
        it: that ancestor and the other class are the overlap, judged as their
        own pair, and every descendant would only repeat it."""

        for indexed_class, owners in (
            (overlap.first, {shared.name: shared.first_owner.symbol for shared in overlap.shared}),
            (overlap.second, {shared.name: shared.second_owner.symbol for shared in overlap.shared}),
        ):
            for ancestor in self.class_index.ancestor_symbols(indexed_class.symbol):
                resolved = self.resolved_declarations(ancestor)
                if all(
                    name in resolved and resolved[name].owner.symbol == owner
                    for name, owner in owners.items()
                ):
                    return True
        return False

    def declaration(self, symbol: str, name: str) -> ClassBodyDeclaration:
        """The declaration of `name` the class answers with, own or inherited."""

        return self.resolved_declarations(symbol)[name].declaration

    def block_fields(self, symbol: str, block: Iterable[str]) -> frozenset[str]:
        """The names of the block that the class declares as fields, own or
        inherited: only a block with one needs a dataclass to hold it."""

        return frozenset(
            name for name in block if self.declaration(symbol, name).kind.is_field
        )

    def _inheriting_pairs_of(self, symbols: Iterable[str]) -> frozenset[tuple[str, str]]:
        """Pairs where one of `symbols` inherits a declaration another class makes itself.

        The other is not the ancestor it inherits from, and neither is an
        ancestor of the other. Declarations more than `holder_limit` classes
        make do not pair, as for own declarations.
        """

        pairs: set[tuple[str, str]] = set()
        for symbol in symbols:
            family = {symbol, *self.class_index.ancestor_symbols(symbol)}
            for resolved in self.resolved_declarations(symbol).values():
                if resolved.owner.symbol == symbol:
                    continue
                for identity in resolved.declaration.identities:
                    holders = self.bodies.holders.get(identity, ())
                    if len(holders) > self.bodies.holder_limit:
                        continue
                    for holder in holders:
                        if holder in family or symbol in self.class_index.ancestor_symbols(
                            holder
                        ):
                            continue
                        pairs.add((min(holder, symbol), max(holder, symbol)))
        return frozenset(pairs)

    def _inheritors_pairing_with(self, symbols: frozenset[str]) -> tuple[str, ...]:
        """Every class an inheriting pair with one of `symbols` can start from:
        the symbols themselves, and each class inheriting a declaration whose
        identity one of them makes. The owner it inherits from holds that
        identity too, so the class is that owner or one of its descendants."""

        found: dict[str, None] = dict.fromkeys(
            symbol for symbol in symbols if symbol in self.bodies.classes_by_symbol
        )
        for symbol in tuple(found):
            for declaration in self.bodies.declarations_of(symbol).values():
                for identity in declaration.identities:
                    holders = self.bodies.holders[identity]
                    if len(holders) > self.bodies.holder_limit:
                        continue
                    for owner in holders:
                        found.update(
                            dict.fromkeys(
                                descendant
                                for descendant in self.class_index.descendant_symbols(owner)
                                if descendant in self.bodies.classes_by_symbol
                            )
                        )
        return tuple(found)

    def resolved(self, indexed_class: CompactIndexedClass) -> CompactIndexedClass:
        """The class as the family index resolved it: its bases are symbols there."""

        return self.class_index.class_for(indexed_class.symbol) or indexed_class

    def verdict(
        self, overlap: ClassBodyOverlap, floor: ReportingFloor = REPORTED
    ) -> ClassOverlapCandidate | None:
        first, second = self.resolved(overlap.first), self.resolved(overlap.second)
        if second.symbol in self.class_index.ancestor_symbols(
            first.symbol
        ) or first.symbol in self.class_index.ancestor_symbols(second.symbol):
            return None
        members = frozenset(
            (*first.direct_enum_member_names, *second.direct_enum_member_names)
        )
        axis = self.family_axis_names(first, second, overlap)
        extractable = tuple(
            shared
            for shared in overlap.shared
            if shared.name not in members and shared.name not in axis
        )
        shared_receipts = tuple(
            Receipt(
                shared.kind.shared_receipt,
                shared.receipt_text,
                _shared_locations(shared),
            )
            for shared in extractable
        )
        differing_receipts = tuple(
            Receipt(
                OverlapReceiptKind.DECLARATION_DIFFERS,
                f"`{shared.name}` is declared differently in each class: one "
                "declaration cannot serve both, so the repair keeps each declaration",
                _shared_locations(shared),
            )
            for shared in extractable
            if shared.declaration_differs
        )
        axis_receipts = tuple(
            Receipt(
                OverlapReceiptKind.FAMILY_AXIS_VALUE,
                f"`{shared.name}` is equal here but `{axis[shared.name]}` answers "
                "it differently: a row of the family's value table, not a shared block",
                _shared_locations(shared),
            )
            for shared in overlap.shared
            if shared.name in axis
        )
        if sum(receipt.weight for receipt in shared_receipts) < floor.weight:
            return None
        if not members and (overlap.first_is_contained or overlap.second_is_contained):
            shape = ClassOverlapShape.DUPLICATE_CLASS
            owner, restating = self._owner_and_restating(overlap)
            move = (
                DeleteAndRedirect
                if overlap.first_is_contained
                and overlap.second_is_contained
                and owner.base_resolution_is_complete
                and restating.base_resolution_is_complete
                and set(restating.resolved_base_symbols)
                <= set(owner.resolved_base_symbols)
                and _construction(owner) == _construction(restating)
                else ComposeExisting
            )
            co_restating = None
        elif len(extractable) >= floor.declarations:
            shape = ClassOverlapShape.MISSING_MIXIN
            existing = self.existing_owner(first, second, extractable)
            if existing is not None:
                owner, restating, co_restating = existing, first, second
                move = ComposeExisting
            else:
                owner, restating, co_restating = first, second, None
                move = (
                    PromoteToSharedBase
                    if self.sole_shared_base(first, second) is not None
                    else ExtractMixin
                )
        else:
            return None
        repair = PairRepair(move, owner, restating, co_restating)
        if self._supplied_member_receipts(
            move, owner, tuple(item for item in (restating, co_restating) if item is not None)
        ):
            # Keeping a class as owner would hand a composer members nobody
            # requires of it: the block needs a mixin of its own, so that is the
            # repair the verdict judges, and its owner is the picker's question.
            repair = PairRepair(ExtractMixin, first, second, None)
        move, owner, restating, co_restating = floor.judged_repair(repair, first, second)
        return ClassOverlapCandidate(
            shape=shape,
            move=move,
            owner=owner,
            restating=restating,
            overlap=overlap,
            co_restating=co_restating,
            owner_candidates=(
                self.owner_question(
                    first,
                    second,
                    tuple(shared.name for shared in extractable),
                    {shared.name: shared.kind.shared_receipt.weight for shared in extractable},
                ).candidates()
                if move is ExtractMixin
                else ()
            ),
            receipts=(
                *shared_receipts,
                *(
                    ()
                    if co_restating is None
                    else (
                        Receipt(
                            OverlapReceiptKind.EXISTING_OWNER,
                            f"`{owner.qualname}` already declares the block; both "
                            "classes restate it",
                            (_location(owner),),
                        ),
                    )
                ),
                *axis_receipts,
                *differing_receipts,
                *(
                    self._inherited_receipts(overlap)
                    if move is not ExtractMixin
                    else ()
                ),
                *self._size_receipts(extractable, first, second),
                *self._member_receipts(overlap, members),
                *self._kept_difference_receipts(overlap),
                *self._consumer_receipts(first, second),
                *self._family_receipts(first, second),
                *self._construction_receipts(
                    first, second, move, owner, tuple(shared.name for shared in extractable)
                ),
                *self._identity_receipts(owner, restating, move),
                *self._supplied_member_receipts(
                    move,
                    owner,
                    tuple(composer for composer in (restating, co_restating) if composer is not None),
                ),
                *(
                    self._positional_construction_receipts(
                        move, (first, second)
                    )
                    if move is not ExtractMixin
                    else ()
                ),
            ),
        )

    @cached_property
    def module_imports(self) -> ModuleImportGraph:
        return ModuleImportGraph.from_aliases(
            {
                module_name: tuple(aliases.values())
                for module_name, aliases in self.import_aliases_by_module.items()
            }
        )

    def owner_question(
        self,
        first: CompactIndexedClass,
        second: CompactIndexedClass,
        block: tuple[str, ...],
        weights: Mapping[str, float],
    ) -> BlockOwnerQuestion:
        """Where the block `first` and `second` share could live; `first`'s
        declarations are the block's identities."""

        block_declarations = {
            name: self.declaration(first.symbol, name).require_proved(
                f"`{first.qualname}.{name}`"
            )
            for name in block
        }
        identities = {
            name: declaration.identity for name, declaration in block_declarations.items()
        }

        def own_holders_of(name: str) -> frozenset[str]:
            return frozenset(self.bodies.holders.get(identities[name], ()))

        def holders_of(name: str) -> frozenset[str]:
            holders = self.bodies.holders.get(identities[name], ())
            return frozenset(
                symbol
                for holder in holders
                for symbol in (holder, *self.class_index.descendant_symbols(holder))
            )

        def declares_differently(symbol: str, name: str) -> bool:
            for declaring in (symbol, *self.class_index.ancestor_symbols(symbol)):
                if declaring not in self.bodies.classes_by_symbol:
                    continue
                declaration = self.bodies.declarations_of(declaring).get(name)
                if declaration is not None:
                    return not declaration.same_declaration_as(block_declarations[name])
            return False

        def declared_names(symbol: str) -> frozenset[str]:
            return frozenset(
                name
                for declaring in (symbol, *self.class_index.ancestor_symbols(symbol))
                if declaring in self.bodies.classes_by_symbol
                for name in self.bodies.declarations_of(declaring)
            )

        def constructions_of(symbol: str, positional: bool = False) -> tuple[SourceLocation, ...]:
            indexed_class = self.class_index.class_for(symbol)
            if indexed_class is None:
                return ()
            return tuple(
                use.location
                for use in self.uses.constructions_of(indexed_class, positional=positional)
            )

        def copies_of(symbol: str) -> tuple[tuple[SourceLocation, tuple[str, ...]], ...]:
            indexed_class = self.class_index.class_for(symbol)
            if indexed_class is None:
                return ()
            return tuple(
                (use.location, use.reference.copied_fields)
                for use in self.uses.uses_for(indexed_class.symbol)
                if use.role is ClassReferenceRole.CONSTRUCTION
                and use.reference.copied_fields
            )

        return BlockOwnerQuestion(
            first=first,
            second=second,
            block=block,
            weights=weights,
            copies_of=copies_of,
            holders_of=holders_of,
            own_holders_of=own_holders_of,
            declared_names=declared_names,
            required_fields=frozenset(
                name
                for name in block
                if block_declarations[name].kind is ClassBodyDeclarationKind.FIELD
            ),
            block_fields=self.block_fields(first.symbol, block),
            constructions_of=constructions_of,
            positional_constructions_of=lambda symbol: constructions_of(
                symbol, positional=True
            ),
            same_construction=lambda one, other: self.construction_kept(one, other, block),
            declares_differently=declares_differently,
            class_for=self.class_index.class_for,
            ancestors_of=self.class_index.ancestor_symbols,
            descendants_of=self.class_index.descendant_symbols,
            consumers_of=self.uses.consumers_of,
            use_locations=self.uses.consumer_locations,
            imports=self.module_imports,
        )

    def existing_owner(
        self,
        first: CompactIndexedClass,
        second: CompactIndexedClass,
        extractable: Sequence[SharedClassBodyDeclaration],
    ) -> CompactIndexedClass | None:
        """The smallest existing owner of the block: a scanned class other classes
        already inherit, whose own or inherited declarations include the block.

        A class that only restates the block too is not its owner; one that
        other classes compose for it already is. The owner is never one of the
        two classes, nor an ancestor or descendant of either, nor a class whose
        creation makes it an identity (a registry, class keywords, a
        metaclass): composing it would enroll both classes in what it
        identifies, so the block stays a mixin to extract, with that class one
        more declaration of it. Between owners, the one with the fewest
        effective declarations owns least beyond the block.
        """

        candidates: set[str] | None = None
        for shared in extractable:
            holders = {
                holder
                for identity in self.declaration(first.symbol, shared.name).identities
                for holder in self.bodies.holders.get(identity, ())
            }
            having = holders | {
                descendant
                for holder in holders
                for descendant in self.class_index.descendant_symbols(holder)
            }
            candidates = having if candidates is None else candidates & having
        excluded = {
            first.symbol,
            second.symbol,
            *self.class_index.ancestor_symbols(first.symbol),
            *self.class_index.ancestor_symbols(second.symbol),
            *self.class_index.descendant_symbols(first.symbol),
            *self.class_index.descendant_symbols(second.symbol),
        }
        options = sorted(
            (
                sum(
                    len(self.bodies.declarations_of(symbol))
                    for symbol in (candidate, *self.class_index.ancestor_symbols(candidate))
                    if symbol in self.bodies.classes_by_symbol
                ),
                candidate,
            )
            for candidate in (candidates or set()) - excluded
            if (indexed := self.class_index.class_for(candidate)) is not None
            and self.class_index.children_by_symbol.get(candidate)
            and not _declared_identity_reasons(indexed)
        )
        return self.class_index.class_for(options[0][1]) if options else None

    def _positional_construction_receipts(
        self,
        move: type[ClassOverlapMove],
        classes: Sequence[CompactIndexedClass],
    ) -> tuple[Receipt, ...]:
        """Calls that bind fields by position, for a move that reorders them."""

        if not move.reorders_fields:
            return ()
        receipts = []
        for indexed_class in classes:
            positional = self.uses.constructions_of(indexed_class, positional=True)
            if positional:
                receipts.append(
                    KeywordCuredReceipt(
                        OverlapReceiptKind.POSITIONAL_CONSTRUCTION,
                        f"`{indexed_class.qualname}` is constructed by position or "
                        f"unpacking in {len(positional)} place(s); moving its fields "
                        "ahead of its own would rebind them",
                        tuple(use.location for use in positional[:4]),
                        constructed=(indexed_class.symbol,),
                    )
                )
        return tuple(receipts)

    def family_axis_names(
        self,
        first: CompactIndexedClass,
        second: CompactIndexedClass,
        overlap: ClassBodyOverlap,
    ) -> Mapping[str, str]:
        """Shared names a third sibling of both declares differently, with that sibling."""

        siblings = {
            child
            for base in set(first.resolved_base_symbols)
            & set(second.resolved_base_symbols)
            for child in self.class_index.children_by_symbol.get(base, ())
        } - {first.symbol, second.symbol}
        first_declarations = self.bodies.declarations_of(first.symbol)
        axis: dict[str, str] = {}
        for sibling in sorted(siblings):
            if sibling not in self.bodies.classes_by_symbol:
                continue
            for name, declaration in self.bodies.declarations_of(sibling).items():
                if (
                    name not in axis
                    and name in first_declarations
                    and any(shared.name == name for shared in overlap.shared)
                    and not declaration.same_declaration_as(first_declarations[name])
                ):
                    axis[name] = sibling.rsplit(".", 1)[-1]
        return axis

    def sole_shared_base(
        self, first: CompactIndexedClass, second: CompactIndexedClass
    ) -> CompactIndexedClass | None:
        """A scanned direct base of both whose only direct children are these two."""

        common = set(first.resolved_base_symbols) & set(second.resolved_base_symbols)
        for symbol in sorted(common):
            base = self.class_index.class_for(symbol)
            if base is None:
                continue
            children = set(self.class_index.children_by_symbol.get(symbol, ()))
            if children == {first.symbol, second.symbol}:
                return base
        return None

    def _owner_and_restating(
        self, overlap: ClassBodyOverlap
    ) -> tuple[CompactIndexedClass, CompactIndexedClass]:
        """The contained body owns the block; between equal bodies, the one with fewer bases."""

        first, second = self.resolved(overlap.first), self.resolved(overlap.second)
        if overlap.first_is_contained and not overlap.second_is_contained:
            return first, second
        if overlap.second_is_contained and not overlap.first_is_contained:
            return second, first
        return tuple(
            sorted(
                (first, second),
                key=lambda item: (
                    len(item.resolved_base_symbols),
                    -len(self.uses.consumers_of(item)),
                    item.symbol,
                ),
            )
        )

    @staticmethod
    def _inherited_receipts(overlap: ClassBodyOverlap) -> tuple[Receipt, ...]:
        """A move deletes a class's own copies; an inherited copy is its
        ancestor's, which the move does not touch."""

        return tuple(
            Receipt(
                OverlapReceiptKind.INHERITED_DECLARATION,
                f"`{indexed_class.qualname}` inherits `{shared.name}` from "
                f"`{owner.qualname}`: the move deletes only declarations a class makes itself",
                _shared_locations(shared),
            )
            for shared in overlap.inherited
            for indexed_class, owner in (
                (overlap.first, shared.first_owner),
                (overlap.second, shared.second_owner),
            )
            if owner.symbol != indexed_class.symbol
        )

    @staticmethod
    def _size_receipts(
        extractable: Sequence[object],
        first: CompactIndexedClass,
        second: CompactIndexedClass,
    ) -> tuple[Receipt, ...]:
        if len(extractable) >= FEW_SHARED_DECLARATIONS:
            return ()
        return (
            Receipt(
                OverlapReceiptKind.FEW_DECLARATIONS,
                f"only {len(extractable)} shared declaration(s)",
                (_location(first), _location(second)),
            ),
        )

    @staticmethod
    def _member_receipts(
        overlap: ClassBodyOverlap, members: frozenset[str]
    ) -> tuple[Receipt, ...]:
        shared_members = tuple(
            shared for shared in overlap.shared if shared.name in members
        )
        if not shared_members:
            return ()
        return (
            Receipt(
                OverlapReceiptKind.IDENTITY_DECLARED,
                f"{len(shared_members)} shared enum member(s) "
                f"({', '.join(shared.name for shared in shared_members[:4])}) are "
                "identities of each enum and stay where they are",
                tuple(
                    _location(shared.first_owner, shared.first_line, shared.name)
                    for shared in shared_members[:4]
                ),
            ),
        )

    @staticmethod
    def _kept_difference_receipts(
        overlap: ClassBodyOverlap,
    ) -> tuple[Receipt, ...]:
        return tuple(
            Receipt(
                OverlapReceiptKind.KEPT_DIFFERENCE,
                f"`{divergent.name}` answers differently and stays in each class",
                (
                    _location(overlap.first, divergent.first.line, divergent.name),
                    _location(overlap.second, divergent.second.line, divergent.name),
                ),
            )
            for divergent in overlap.divergent
        )

    def _consumer_receipts(
        self, first: CompactIndexedClass, second: CompactIndexedClass
    ) -> tuple[Receipt, ...]:
        shared = sorted(self.uses.consumers_of(first) & self.uses.consumers_of(second))
        if not shared:
            return (
                Receipt(
                    OverlapReceiptKind.NO_SHARED_CONSUMER,
                    f"no function or class uses both (`{first.qualname}`: "
                    f"{len(self.uses.consumers_of(first))} consumer(s), "
                    f"`{second.qualname}`: {len(self.uses.consumers_of(second))})",
                    (_location(first), _location(second)),
                ),
            )
        uses = {
            (use.file_path, use.consumer): use
            for use in (*self.uses.uses_of(first), *self.uses.uses_of(second))
        }
        return (
            Receipt(
                OverlapReceiptKind.SHARED_CONSUMER,
                f"{len(shared)} consumer(s) use both: "
                + ", ".join(f"`{consumer}`" for _file_path, consumer in shared[:4]),
                tuple(uses[key].location for key in shared[:4]),
            ),
        )

    def _family_receipts(
        self, first: CompactIndexedClass, second: CompactIndexedClass
    ) -> tuple[Receipt, ...]:
        common = sorted(
            set(self.class_index.ancestor_symbols(first.symbol))
            & set(self.class_index.ancestor_symbols(second.symbol))
        )
        if common:
            ancestor = self.class_index.class_for(common[0])
            return (
                Receipt(
                    OverlapReceiptKind.SAME_FAMILY,
                    f"both descend from scanned `{common[0]}`",
                    () if ancestor is None else (_location(ancestor),),
                ),
            )
        return (
            Receipt(
                OverlapReceiptKind.SEPARATE_FAMILIES,
                "no scanned ancestor in common",
                (_location(first), _location(second)),
            ),
        )

    def _construction_receipts(
        self,
        first: CompactIndexedClass,
        second: CompactIndexedClass,
        move: type[ClassOverlapMove] | None = None,
        owner: CompactIndexedClass | None = None,
        block: Sequence[str] = (),
    ) -> tuple[Receipt, ...]:
        if _construction(first) == _construction(second):
            return (
                Receipt(
                    OverlapReceiptKind.SAME_CONSTRUCTION,
                    "same dataclass declaration, metaclass and class keywords",
                    (_location(first), _location(second)),
                ),
            )
        differences: list[str] = []
        plain_base = self._plain_base(first, second, move, owner, block)
        if (
            first.dataclass_declaration.runtime_declaration
            != second.dataclass_declaration.runtime_declaration
        ) and not plain_base:
            differences.append(
                f"dataclass declaration (`{first.qualname}` has "
                f"{first.dataclass_declaration.declared_by}, `{second.qualname}` has "
                f"{second.dataclass_declaration.declared_by})"
            )
        if first.metaclass_names != second.metaclass_names:
            differences.append(
                f"metaclass ({first.metaclass_names!r} vs {second.metaclass_names!r})"
            )
        if first.class_keyword_names != second.class_keyword_names:
            differences.append(
                f"class keywords ({first.class_keyword_names!r} vs "
                f"{second.class_keyword_names!r})"
            )
        if (
            first.class_decorators_are_promotion_safe
            != second.class_decorators_are_promotion_safe
        ):
            differences.append("decorator safety")
        if not differences and plain_base:
            return (
                Receipt(
                    OverlapReceiptKind.SAME_CONSTRUCTION,
                    "dataclass declarations differ, but the block declares no field and its "
                    "owner generates no construction: each composer keeps its own",
                    (_location(first), _location(second)),
                ),
            )
        if not differences:
            differences.append("class creation")
        if (
            all(d.startswith("class keywords") for d in differences)
            and (
                move is ExtractMixin
                or (
                    move is ComposeExisting
                    and not (first.class_keyword_names and second.class_keyword_names)
                )
            )
        ):
            return (
                Receipt(
                    OverlapReceiptKind.SAME_CONSTRUCTION,
                    "class keywords differ, but the owner gets none: a composer keeps "
                    "its own, and what its creation reads is proved at the move",
                    (_location(first), _location(second)),
                ),
            )
        detail = ", ".join(differences)
        return (
            Receipt(
                OverlapReceiptKind.DIFFERENT_CONSTRUCTION,
                f"`{first.qualname}` and `{second.qualname}` differ in {detail}",
                (_location(first), _location(second)),
            ),
        )

    def _plain_base(
        self,
        first: CompactIndexedClass,
        second: CompactIndexedClass,
        move: type[ClassOverlapMove] | None,
        owner: CompactIndexedClass | None,
        block: Sequence[str],
    ) -> bool:
        """The base each composer gains is a plain class taking no field from the
        block: a new mixin of a fieldless block, or an existing owner no dataclass
        decorator declares."""

        if move is ExtractMixin:
            return self.takes_no_field(None, block, first, second)
        return move is ComposeExisting and owner is not None and self.takes_no_field(
            owner, block, first, second
        )

    def takes_no_field(
        self, base: CompactIndexedClass | None, block: Sequence[str], *composers: CompactIndexedClass
    ) -> bool:
        """Composing `base` (a new mixin when `None`) over `block` changes no
        composer's construction because of a dataclass declaration: no dataclass
        decorator declares the base, and no class declares a field in the block.
        A dataclass collects fields only from dataclass bases, so each composer
        keeps its own fields, frozenness and generated methods."""

        holders = (*composers, *(() if base is None else (base,)))
        return (
            bool(block)
            and (base is None or not base.dataclass_declaration.generates_construction)
            and not any(self.block_fields(indexed.symbol, block) for indexed in holders)
        )

    def construction_kept(
        self, base: CompactIndexedClass, composer: CompactIndexedClass, block: Sequence[str]
    ) -> bool:
        """Composing `base` builds `composer` as before: both are created alike, or
        they differ only in a dataclass declaration a plain fieldless base leaves alone."""

        return _construction(base) == _construction(composer) or (
            _construction(base)[1:] == _construction(composer)[1:]
            and self.takes_no_field(base, block, composer)
        )

    def _supplied_member_receipts(
        self,
        move: type[ClassOverlapMove],
        owner: CompactIndexedClass,
        restating: Sequence[CompactIndexedClass],
    ) -> tuple[Receipt, ...]:
        """Members a class would come to have from the owner and its ancestors
        without declaring or inheriting them now: supply nobody required, which
        no later addition cancels (a dataclass field among them also changes
        how the class is built)."""

        owned = self.declared_through_ancestry(owner.symbol)
        receipts: list[Receipt] = []
        for receiver in move.receiving_classes(restating):
            had = self.declared_through_ancestry(receiver.symbol)
            gained = sorted(name for name in owned if name not in had)
            if gained:
                receipts.append(
                    Receipt(
                        OverlapReceiptKind.UNWANTED_SUPPLY,
                        f"`{receiver.qualname}` would gain "
                        + ", ".join(
                            f"`{owned[name].owner.qualname}.{name}`" for name in gained[:6]
                        )
                        + (f" and {len(gained) - 6} more" if len(gained) > 6 else "")
                        + f" through `{owner.qualname}`, which it neither declares "
                        "nor inherits: the block needs a mixin of its own",
                        tuple(_location(owned[name].owner, None, name) for name in gained[:4]),
                        count=len(gained),
                    )
                )
        return tuple(receipts)

    def _identity_receipts(
        self,
        owner: CompactIndexedClass,
        restating: CompactIndexedClass,
        move: type[ClassOverlapMove],
    ) -> tuple[Receipt, ...]:
        """Uses that tell apart, by type, a class whose identity the move changes."""

        guarded = move.guarded_classes(owner, restating)
        answers = move.changed_answers(owner, restating, self.class_index)
        receipts: list[Receipt] = []
        for indexed_class in guarded:
            identity_uses = tuple(
                use
                for use in self.uses.uses_of(indexed_class)
                if use.role.distinguishes_identity and answers.changes(use.tested_subclass)
            )
            if identity_uses:
                receipts.append(
                    Receipt(
                        OverlapReceiptKind.IDENTITY_USED,
                        f"`{indexed_class.qualname}` is told apart by type in "
                        f"{len(identity_uses)} place(s) "
                        f"({', '.join(sorted({use.role.value for use in identity_uses}))})",
                        tuple(use.location for use in identity_uses[:4]),
                    )
                )
            if reasons := _declared_identity_reasons(indexed_class):
                receipts.append(
                    Receipt(
                        OverlapReceiptKind.IDENTITY_DECLARED,
                        f"`{indexed_class.qualname}`: {'; '.join(reasons)}",
                        (_location(indexed_class),),
                    )
                )
            for name, line, tag in _literal_tags(
                indexed_class, self.import_aliases_by_module[indexed_class.module_name]
            ):
                receipts.append(
                    Receipt(
                        OverlapReceiptKind.IDENTITY_DECLARED,
                        f"`{indexed_class.qualname}.{name}` is the literal tag "
                        f"{tag!r}, a wire identity of this class",
                        (_location(indexed_class, line, name),),
                    )
                )
        return tuple(receipts)


def class_overlap_candidates(
    projections: Iterable[CompactModuleClassProjection],
    class_index: CompactClassFamilyIndex,
) -> tuple[ClassOverlapCandidate, ...]:
    return ClassOverlapVerdicts(tuple(projections), class_index).candidates


__all__ = (
    "ClassOverlapCandidate",
    "ClassOverlapMove",
    "ComposeExisting",
    "DeleteAndRedirect",
    "ExtractMixin",
    "PromoteToSharedBase",
    "ClassOverlapShape",
    "ClassOverlapVerdicts",
    "ClassUseIndex",
    "Receipt",
    "OverlapReceiptKind",
    "class_overlap_candidates",
)
