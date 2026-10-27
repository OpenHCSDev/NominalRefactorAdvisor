"""Classes that restate another class's body (#68, shapes 5 and 6).

`ClassBodyIndex` says which declarations two classes share. This module says
what that means and how sure NRA is: a ranked verdict with the receipts that
produced it, in the #67 form.

**The model.** Every receipt carries a fixed log-odds weight, stated on its
kind below; the probability that the two classes are one concept is
`1 / (1 + exp(-sum))` over the receipts, from even odds. Nothing else enters
the score, so a reader can re-add it from the receipts.

Shared declarations raise it, weighted by how much each says (a method more
than a valued declaration, a valued declaration more than a bare annotation).
Few shared declarations, a name both declare with different answers, no
consumer that uses both classes, different class construction, and any use
that tells the classes apart by type lower it. Identity is the guard (#68):
a class used as a type test, a class pattern, a type key, or registered or
tagged by its class creation is a declaration of one fact even when its body
is another class's. NRA never drops such a pair silently; it lowers the
probability and names the use.

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
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from typing import Self

from .ast_projection import ExpressionReference
from .class_body import (
    ClassBodyDeclaration,
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
from .json_reports import SemanticRecord
from .models import FindingMetrics, ImpactDelta, SourceLocation


class ClassOverlapShape(StrEnum):
    """Which #68 shape one overlap is."""

    DUPLICATE_CLASS = "duplicate_class"
    MISSING_MIXIN = "missing_mixin"


class ClassOverlapMove(StrEnum):
    """The factoring an overlap calls for."""

    DELETE_AND_REDIRECT = "delete_and_redirect"
    COMPOSE_EXISTING = "compose_existing"
    PROMOTE_TO_SHARED_BASE = "promote_to_shared_base"
    EXTRACT_MIXIN = "extract_mixin"


class OverlapReceiptKind(StrEnum):
    """One kind of evidence and its fixed log-odds weight."""

    SHARED_METHOD = "shared_method", 1.5
    SHARED_VALUE = "shared_value", 1.0
    SHARED_FIELD = "shared_field", 0.5
    FEW_DECLARATIONS = "few_declarations", -1.5
    KEPT_DIFFERENCE = "kept_difference", 0.0
    SHARED_CONSUMER = "shared_consumer", 1.0
    NO_SHARED_CONSUMER = "no_shared_consumer", -1.0
    SAME_FAMILY = "same_family", 0.5
    SEPARATE_FAMILIES = "separate_families", -0.5
    SAME_CONSTRUCTION = "same_construction", 0.5
    DIFFERENT_CONSTRUCTION = "different_construction", -1.5
    IDENTITY_USED = "identity_used", -1.5
    IDENTITY_DECLARED = "identity_declared", -1.5

    weight: float

    def __new__(cls, value: str, weight: float) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.weight = weight
        return member


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
class OverlapReceipt(SemanticRecord):
    """One fact behind a verdict, its weight, and where it is."""

    kind: OverlapReceiptKind
    detail: str
    evidence: tuple[SourceLocation, ...]

    @property
    def weight(self) -> float:
        return self.kind.weight


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

    def consumers_of(self, indexed_class: CompactIndexedClass) -> frozenset[tuple[str, str]]:
        """Functions and classes that use the class; the module scope is a declaration site, not a consumer."""

        return frozenset(
            (use.file_path, use.consumer)
            for use in self.uses_of(indexed_class)
            if use.consumer
        )


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
class ClassOverlapCandidate:
    """One verdict: which class owns the block, which restates it, the move, the receipts."""

    shape: ClassOverlapShape
    move: ClassOverlapMove
    owner: CompactIndexedClass
    restating: CompactIndexedClass
    overlap: ClassBodyOverlap
    receipts: tuple[OverlapReceipt, ...]

    @property
    def log_odds(self) -> float:
        return sum(receipt.weight for receipt in self.receipts)

    @property
    def probability(self) -> float:
        return 1.0 / (1.0 + math.exp(-self.log_odds))

    @property
    def shared_names(self) -> tuple[str, ...]:
        return tuple(shared.name for shared in self.overlap.shared)

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        first, second = self.overlap.first, self.overlap.second
        return (
            _location(self.owner),
            _location(self.restating),
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
    """One verdict as finding metrics: the move, the probability and every receipt."""

    shape: ClassOverlapShape
    move: ClassOverlapMove
    owner_symbol: str
    restating_symbol: str
    shared_declaration_names: tuple[str, ...]
    probability: float
    receipts: tuple[OverlapReceipt, ...]

    @classmethod
    def of(cls, candidate: ClassOverlapCandidate) -> ClassOverlapMetrics:
        return cls(
            shape=candidate.shape,
            move=candidate.move,
            owner_symbol=candidate.owner.symbol,
            restating_symbol=candidate.restating.symbol,
            shared_declaration_names=candidate.shared_names,
            probability=round(candidate.probability, 4),
            receipts=candidate.receipts,
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
        return (self.owner_symbol, self.restating_symbol)

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
        extractable = tuple(
            shared for shared in overlap.shared if shared.name not in members
        )
        shared_receipts = tuple(
            OverlapReceipt(
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
        if sum(receipt.weight for receipt in shared_receipts) < MINIMUM_SHARED_WEIGHT:
            return None
        if not members and (overlap.first_is_contained or overlap.second_is_contained):
            shape = ClassOverlapShape.DUPLICATE_CLASS
            owner, restating = self._owner_and_restating(overlap)
            move = (
                ClassOverlapMove.DELETE_AND_REDIRECT
                if overlap.first_is_contained
                and overlap.second_is_contained
                and owner.base_resolution_is_complete
                and restating.base_resolution_is_complete
                and set(restating.resolved_base_symbols)
                <= set(owner.resolved_base_symbols)
                and _construction(owner) == _construction(restating)
                else ClassOverlapMove.COMPOSE_EXISTING
            )
        elif len(extractable) >= FEW_SHARED_DECLARATIONS:
            shape = ClassOverlapShape.MISSING_MIXIN
            owner, restating = first, second
            move = (
                ClassOverlapMove.PROMOTE_TO_SHARED_BASE
                if self.sole_shared_base(first, second) is not None
                else ClassOverlapMove.EXTRACT_MIXIN
            )
        else:
            return None
        return ClassOverlapCandidate(
            shape=shape,
            move=move,
            owner=owner,
            restating=restating,
            overlap=overlap,
            receipts=(
                *shared_receipts,
                *self._size_receipts(extractable, first, second),
                *self._member_receipts(overlap, members),
                *self._kept_difference_receipts(overlap),
                *self._consumer_receipts(first, second),
                *self._family_receipts(first, second),
                *self._construction_receipts(first, second),
                *self._identity_receipts(owner, restating, move),
            ),
        )

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
    ) -> tuple[OverlapReceipt, ...]:
        if len(extractable) >= FEW_SHARED_DECLARATIONS:
            return ()
        return (
            OverlapReceipt(
                OverlapReceiptKind.FEW_DECLARATIONS,
                f"only {len(extractable)} shared declaration(s)",
                (_location(first), _location(second)),
            ),
        )

    @staticmethod
    def _member_receipts(
        overlap: ClassBodyOverlap, members: frozenset[str]
    ) -> tuple[OverlapReceipt, ...]:
        shared_members = tuple(
            shared for shared in overlap.shared if shared.name in members
        )
        if not shared_members:
            return ()
        return (
            OverlapReceipt(
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
    ) -> tuple[OverlapReceipt, ...]:
        return tuple(
            OverlapReceipt(
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
    ) -> tuple[OverlapReceipt, ...]:
        shared = sorted(self.uses.consumers_of(first) & self.uses.consumers_of(second))
        if not shared:
            return (
                OverlapReceipt(
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
            OverlapReceipt(
                OverlapReceiptKind.SHARED_CONSUMER,
                f"{len(shared)} consumer(s) use both: "
                + ", ".join(f"`{consumer}`" for _file_path, consumer in shared[:4]),
                tuple(uses[key].location for key in shared[:4]),
            ),
        )

    def _family_receipts(
        self, first: CompactIndexedClass, second: CompactIndexedClass
    ) -> tuple[OverlapReceipt, ...]:
        common = sorted(
            set(self.class_index.ancestor_symbols(first.symbol))
            & set(self.class_index.ancestor_symbols(second.symbol))
        )
        if common:
            ancestor = self.class_index.class_for(common[0])
            return (
                OverlapReceipt(
                    OverlapReceiptKind.SAME_FAMILY,
                    f"both descend from scanned `{common[0]}`",
                    () if ancestor is None else (_location(ancestor),),
                ),
            )
        return (
            OverlapReceipt(
                OverlapReceiptKind.SEPARATE_FAMILIES,
                "no scanned ancestor in common",
                (_location(first), _location(second)),
            ),
        )

    @staticmethod
    def _construction_receipts(
        first: CompactIndexedClass, second: CompactIndexedClass
    ) -> tuple[OverlapReceipt, ...]:
        if _construction(first) == _construction(second):
            return (
                OverlapReceipt(
                    OverlapReceiptKind.SAME_CONSTRUCTION,
                    "same dataclass declaration, metaclass and class keywords",
                    (_location(first), _location(second)),
                ),
            )
        return (
            OverlapReceipt(
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
        move: ClassOverlapMove,
    ) -> tuple[OverlapReceipt, ...]:
        """Uses that tell either class apart by type.

        Deleting the restating class loses its identity, and composing makes the
        owner an ancestor of the restating class, so a type test, class pattern,
        type key or creation-time registration of either one is a fact the
        merge would change. Extracting a mixin leaves both identities as they
        are and adds an ancestor nobody tests yet, so it guards nothing.
        """

        if move in (
            ClassOverlapMove.EXTRACT_MIXIN,
            ClassOverlapMove.PROMOTE_TO_SHARED_BASE,
        ):
            return ()
        guarded = (restating, owner)
        receipts: list[OverlapReceipt] = []
        for indexed_class in guarded:
            identity_uses = tuple(
                use
                for use in self.uses.uses_of(indexed_class)
                if use.role.distinguishes_identity
            )
            if identity_uses:
                receipts.append(
                    OverlapReceipt(
                        OverlapReceiptKind.IDENTITY_USED,
                        f"`{indexed_class.qualname}` is told apart by type in "
                        f"{len(identity_uses)} place(s) "
                        f"({', '.join(sorted({use.role.value for use in identity_uses}))})",
                        tuple(use.location for use in identity_uses[:4]),
                    )
                )
            if (reason := _declared_identity(indexed_class)) is not None:
                receipts.append(
                    OverlapReceipt(
                        OverlapReceiptKind.IDENTITY_DECLARED,
                        f"`{indexed_class.qualname}`: {reason}",
                        (_location(indexed_class),),
                    )
                )
            for name, line, tag in _literal_tags(
                indexed_class, self.import_aliases_by_module[indexed_class.module_name]
            ):
                receipts.append(
                    OverlapReceipt(
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
    "ClassOverlapShape",
    "ClassOverlapVerdicts",
    "ClassUseIndex",
    "OverlapReceipt",
    "OverlapReceiptKind",
    "class_overlap_candidates",
)
