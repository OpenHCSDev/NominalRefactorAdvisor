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
from collections.abc import Iterable, Mapping, Sequence
from abc import abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

from .ast_projection import ExpressionReference
from .case_root import Case
from .class_body import (
    ClassBodyDeclaration,
    SharedClassBodyDeclaration,
    ClassBodyIndex,
    ClassBodyOverlap,
)
from .class_index import (
    ClassBodyDeclarationKind,
    ClassReferenceRole,
    CompactClassFamilyIndex,
    CompactClassReference,
    CompactClassReferenceResolver,
    CompactIndexedClass,
    CompactModuleClassProjection,
)
from .receipts import Receipt, ReceiptKind, ScoredByReceipts
from .owner_candidates import (
    BlockOwnerQuestion,
    ModuleImportGraph,
    OwnerCandidate,
    OwnerPick,
)
from .models import FindingMetrics, ImpactDelta, SourceLocation


class ClassOverlapShape(StrEnum):
    """Which #68 shape one overlap is."""

    DUPLICATE_CLASS = "duplicate_class"
    MISSING_MIXIN = "missing_mixin"


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


class DeleteAndRedirect(ClassOverlapMove):
    """Delete the restating class and point its uses at the owner: its identity is gone."""

    value = "delete_and_redirect"
    reorders_fields = False

    @classmethod
    def guarded_classes(
        cls, owner: CompactIndexedClass, restating: CompactIndexedClass
    ) -> tuple[CompactIndexedClass, ...]:
        return (restating, owner)


class ComposeExisting(ClassOverlapMove):
    """The restating class inherits the owner: the owner gains a descendant."""

    value = "compose_existing"
    reorders_fields = False

    @classmethod
    def guarded_classes(
        cls, owner: CompactIndexedClass, restating: CompactIndexedClass
    ) -> tuple[CompactIndexedClass, ...]:
        return (restating, owner)


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


class OverlapReceiptKind(ReceiptKind):
    """One kind of evidence: a concept receipt with its fixed log-odds weight, or a
    move guard that refuses the move and weighs nothing in the concept score."""

    SHARED_METHOD = "shared_method", 1.5, False
    SHARED_VALUE = "shared_value", 1.0, False
    SHARED_FIELD = "shared_field", 0.5, False
    EXISTING_OWNER = "existing_owner", 1.0, False
    FAMILY_AXIS_VALUE = "family_axis_value", 0.0, False
    FEW_DECLARATIONS = "few_declarations", -1.5, False
    KEPT_DIFFERENCE = "kept_difference", 0.0, False
    SHARED_CONSUMER = "shared_consumer", 1.0, False
    NO_SHARED_CONSUMER = "no_shared_consumer", 0.0, False
    SAME_FAMILY = "same_family", 0.5, False
    SEPARATE_FAMILIES = "separate_families", 0.0, False
    SAME_CONSTRUCTION = "same_construction", 0.0, False
    DIFFERENT_CONSTRUCTION = "different_construction", 0.0, True
    IDENTITY_USED = "identity_used", 0.0, True
    IDENTITY_DECLARED = "identity_declared", 0.0, True
    POSITIONAL_CONSTRUCTION = "positional_construction", 0.0, True


def _shared_receipt_kind(declaration: ClassBodyDeclaration) -> OverlapReceiptKind:
    if declaration.kind is ClassBodyDeclarationKind.METHOD:
        return OverlapReceiptKind.SHARED_METHOD
    if declaration.declaration.assigns_value:
        return OverlapReceiptKind.SHARED_VALUE
    return OverlapReceiptKind.SHARED_FIELD

FEW_SHARED_DECLARATIONS = 3
"""Below this many shared declarations the overlap is weak evidence of one concept."""

MINIMUM_SHARED_WEIGHT = 1.5
"""Declared bound on what is worth a verdict: the shared declarations' own receipts
must reach this weight (one method, a valued declaration and an annotation, or
three annotations). A lone shared field is not reported at all."""


@dataclass(frozen=True)
class ClassUse:
    """One module's reference resolved to a scanned class, with the role it plays there."""

    file_path: str
    reference: CompactClassReference
    role: ClassReferenceRole

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
    """Which consumers use each scanned class, and how."""

    uses_by_symbol: Mapping[str, tuple[ClassUse, ...]]

    @classmethod
    def from_projections(
        cls,
        projections: Sequence[CompactModuleClassProjection],
        class_index: CompactClassFamilyIndex,
    ) -> ClassUseIndex:
        resolver = CompactClassReferenceResolver.from_index(projections, class_index)
        uses: dict[str, list[ClassUse]] = defaultdict(list)
        for projection in projections:
            for reference in projection.class_references:
                for length in range(len(reference.reference_parts), 0, -1):
                    symbol = resolver.symbol_for(
                        module_name=projection.module_name,
                        reference_parts=reference.reference_parts[:length],
                        allow_unique_unqualified=False,
                    )
                    if symbol is None:
                        continue
                    role = (
                        reference.role
                        if length == len(reference.reference_parts)
                        else ClassReferenceRole.OTHER
                    )
                    uses[symbol].append(ClassUse(projection.file_path, reference, role))
                    break
        return cls({symbol: tuple(found) for symbol, found in uses.items()})

    def uses_of(self, indexed_class: CompactIndexedClass) -> tuple[ClassUse, ...]:
        """Uses outside the class's own body."""

        own_prefix = f"{indexed_class.qualname}."
        return tuple(
            use
            for use in self.uses_by_symbol.get(indexed_class.symbol, ())
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


def _construction(indexed_class: CompactIndexedClass) -> tuple[object, ...]:
    declaration = indexed_class.dataclass_declaration
    return (
        None if declaration is None else declaration.runtime_declaration,
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
        if member.annotation_expression is None or member.constant_string is None:
            continue
        annotation = ast.parse(member.annotation_expression, mode="eval").body
        if not (
            isinstance(annotation, ast.Subscript)
            and isinstance(annotation.slice, ast.Constant)
            and annotation.slice.value == member.constant_string
        ):
            continue
        parts = ExpressionReference.decoded(annotation.value).attribute_chain
        if parts is None:
            continue
        qualified = ".".join((import_aliases.get(parts[0], parts[0]), *parts[1:]))
        if qualified in LITERAL_FORMS:
            tags.append((member.name, member.line, member.constant_string))
    return tuple(tags)


def _declared_identity(indexed_class: CompactIndexedClass) -> str | None:
    """Why class creation itself makes this class an identity, if it does."""

    if indexed_class.enum_declaration is not None:
        return "its members are enum identities"
    if indexed_class.class_keyword_names:
        return f"class keywords {', '.join(indexed_class.class_keyword_names)} run at creation"
    if indexed_class.metaclass_names:
        return f"metaclass {', '.join(indexed_class.metaclass_names)} runs at creation"
    if indexed_class.declares_autoregister_meta or indexed_class.is_registration_authority:
        return "it registers its subclasses"
    return None


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
                for location in (
                    _location(first, shared.first_line, shared.name),
                    _location(second, shared.second_line, shared.name),
                )
            ),
        )


@dataclass(frozen=True)
class ClassOverlapMetrics(FindingMetrics):
    """One verdict as finding metrics: the move, the confidence score and every receipt."""

    shape: ClassOverlapShape
    move: type[ClassOverlapMove]
    owner_symbol: str
    restating_symbols: tuple[str, ...]
    shared_declaration_names: tuple[str, ...]
    confidence_score: float
    move_permitted: bool
    receipts: tuple[Receipt, ...]
    owner_candidates: tuple[OwnerCandidate, ...] = ()
    owner_pick: OwnerPick | None = None
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
            move_permitted=candidate.permitted,
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
        return tuple(
            candidate
            for overlap in self.bodies.overlaps
            if (candidate := self.verdict(overlap)) is not None
        )

    def resolved(self, indexed_class: CompactIndexedClass) -> CompactIndexedClass:
        """The class as the family index resolved it: its bases are symbols there."""

        return self.class_index.class_for(indexed_class.symbol) or indexed_class

    def verdict(self, overlap: ClassBodyOverlap) -> ClassOverlapCandidate | None:
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
                _shared_receipt_kind(
                    self.bodies.declarations_of(first.symbol)[shared.name]
                ),
                f"`{shared.name}` is declared identically",
                (
                    _location(first, shared.first_line, shared.name),
                    _location(second, shared.second_line, shared.name),
                ),
            )
            for shared in extractable
        )
        axis_receipts = tuple(
            Receipt(
                OverlapReceiptKind.FAMILY_AXIS_VALUE,
                f"`{shared.name}` is equal here but `{axis[shared.name]}` answers "
                "it differently: a row of the family's value table, not a shared block",
                (
                    _location(first, shared.first_line, shared.name),
                    _location(second, shared.second_line, shared.name),
                ),
            )
            for shared in overlap.shared
            if shared.name in axis
        )
        if sum(receipt.weight for receipt in shared_receipts) < MINIMUM_SHARED_WEIGHT:
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
        elif len(extractable) >= FEW_SHARED_DECLARATIONS:
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
        return ClassOverlapCandidate(
            shape=shape,
            move=move,
            owner=owner,
            restating=restating,
            overlap=overlap,
            co_restating=co_restating,
            owner_candidates=(
                self.owner_question(
                    first, second, tuple(shared.name for shared in extractable)
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
                *self._size_receipts(extractable, first, second),
                *self._member_receipts(overlap, members),
                *self._kept_difference_receipts(overlap),
                *self._consumer_receipts(first, second),
                *self._family_receipts(first, second),
                *self._construction_receipts(first, second),
                *self._identity_receipts(owner, restating, move),
                *self._positional_construction_receipts(
                    move, (first, second)
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
    ) -> BlockOwnerQuestion:
        """Where the block `first` and `second` share could live; `first`'s
        declarations are the block's identities."""

        identities = {
            name: self.bodies.declarations_of(first.symbol)[name].identity
            for name in block
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
                    return declaration.identity != identities[name]
            return False

        def declared_names(symbol: str) -> frozenset[str]:
            return frozenset(
                name
                for declaring in (symbol, *self.class_index.ancestor_symbols(symbol))
                if declaring in self.bodies.classes_by_symbol
                for name in self.bodies.declarations_of(declaring)
            )

        first_declarations = self.bodies.declarations_of(first.symbol)

        def constructions_of(symbol: str, positional: bool = False) -> tuple[SourceLocation, ...]:
            indexed_class = self.class_index.class_for(symbol)
            if indexed_class is None:
                return ()
            return tuple(
                use.location
                for use in self.uses.uses_of(indexed_class)
                if use.role is ClassReferenceRole.CONSTRUCTION
                and (
                    not positional
                    or use.reference.positional_arguments
                    or use.reference.unpacked_arguments
                )
            )

        return BlockOwnerQuestion(
            first=first,
            second=second,
            block=block,
            holders_of=holders_of,
            own_holders_of=own_holders_of,
            declared_names=declared_names,
            required_fields=frozenset(
                name
                for name in block
                if first_declarations[name].kind is ClassBodyDeclarationKind.FIELD
                and not first_declarations[name].declaration.assigns_value
            ),
            constructions_of=constructions_of,
            positional_constructions_of=lambda symbol: constructions_of(
                symbol, positional=True
            ),
            same_construction=lambda one, other: _construction(one) == _construction(other),
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
        two classes, nor an ancestor or descendant of either. Between owners,
        the one with the fewest effective declarations owns least beyond the
        block.
        """

        first_declarations = self.bodies.declarations_of(first.symbol)
        candidates: set[str] | None = None
        for shared in extractable:
            holders = set(
                self.bodies.holders.get(first_declarations[shared.name].identity, ())
            )
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
            if self.class_index.class_for(candidate) is not None
            and self.class_index.children_by_symbol.get(candidate)
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
            positional = tuple(
                use
                for use in self.uses.uses_of(indexed_class)
                if use.role is ClassReferenceRole.CONSTRUCTION
                and (
                    use.reference.positional_arguments
                    or use.reference.unpacked_arguments
                )
            )
            if positional:
                receipts.append(
                    Receipt(
                        OverlapReceiptKind.POSITIONAL_CONSTRUCTION,
                        f"`{indexed_class.qualname}` is constructed by position or "
                        f"unpacking in {len(positional)} place(s); moving its fields "
                        "ahead of its own would rebind them",
                        tuple(use.location for use in positional[:4]),
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
                    and declaration.identity != first_declarations[name].identity
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
                    _location(overlap.first, shared.first_line, shared.name)
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

    @staticmethod
    def _construction_receipts(
        first: CompactIndexedClass, second: CompactIndexedClass
    ) -> tuple[Receipt, ...]:
        if _construction(first) == _construction(second):
            return (
                Receipt(
                    OverlapReceiptKind.SAME_CONSTRUCTION,
                    "same dataclass declaration, metaclass and class keywords",
                    (_location(first), _location(second)),
                ),
            )
        return (
            Receipt(
                OverlapReceiptKind.DIFFERENT_CONSTRUCTION,
                "class creation differs (dataclass declaration, metaclass, class "
                "keywords or decorators), so construction and field semantics differ",
                (_location(first), _location(second)),
            ),
        )

    def _identity_receipts(
        self,
        owner: CompactIndexedClass,
        restating: CompactIndexedClass,
        move: type[ClassOverlapMove],
    ) -> tuple[Receipt, ...]:
        """Uses that tell apart, by type, a class whose identity the move changes."""

        guarded = move.guarded_classes(owner, restating)
        receipts: list[Receipt] = []
        for indexed_class in guarded:
            identity_uses = tuple(
                use
                for use in self.uses.uses_of(indexed_class)
                if use.role.distinguishes_identity
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
            if (reason := _declared_identity(indexed_class)) is not None:
                receipts.append(
                    Receipt(
                        OverlapReceiptKind.IDENTITY_DECLARED,
                        f"`{indexed_class.qualname}`: {reason}",
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
