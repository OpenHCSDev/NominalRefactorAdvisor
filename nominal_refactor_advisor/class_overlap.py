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
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from itertools import combinations
from typing import NamedTuple, Self
from enum import StrEnum
from functools import cached_property

from nominal_refactor_advisor.ast_projection import ExpressionReference
from nominal_refactor_advisor.case_root import Case
from nominal_refactor_advisor.class_body import (
    ClassBodyDeclaration,
    ClassBodyDeclarationIdentity,
    ResolvedClassBodyDeclaration,
    SharedClassBodyDeclaration,
    ClassBodyIndex,
    ClassBodyOverlap,
)
from nominal_refactor_advisor.scan_cache import ScanCache
from nominal_refactor_advisor.field_order_reads import FieldOrderReads
from nominal_refactor_advisor.class_index import (
    ClassBodyDeclarationKind,
    ClassCreationOrderProof,
    CompactClassFamilyIndex,
    CompactDataclassDeclaration,
    CompactClassReferenceResolver,
    CompactIndexedClass,
    CompactModuleClassProjection,
    CompactRepositoryPublicExposureIndex,
    InstalledClassDeclarations,
    followed_reexports,
)
from nominal_refactor_advisor.declared_c3 import DeclaredC3
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


@dataclass(frozen=True, kw_only=True)
class MoveRequest(ReportingFloor):
    """A step asks about the pair as the move it makes, whatever repair the
    overlap's shape reports. A duplicate class, or a block a third class already
    declares, can still be extracted, and a pair whose merge is refused can still
    fold its block into the base the two alone share; the step proves the move it
    makes, so the verdict judges that and not the other repair."""

    move: type[ClassOverlapMove] = ExtractMixin
    weight: float = 0.0
    declarations: int = 1

    def judged_repair(
        self, reported: PairRepair, first: CompactIndexedClass, second: CompactIndexedClass
    ) -> PairRepair:
        return PairRepair(self.move, first, second, None)


REPORTED = ReportingFloor()
NO_FLOOR = ReportingFloor(weight=0.0, declarations=1)
PART_DECLARATIONS = 2
"""The fewest names a part of a block shares: across agent-comms censuses 17-19 every
one-name part offered (`close`, `available`, `parts`, `detail`, `updates`, `display`,
`line`) was two classes' coincidence, and the two-name part was a key."""
EXTRACTION = MoveRequest(move=ExtractMixin)



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


@ScanCache.cached
def _base_class_use_index_slot() -> list[
    tuple[
        tuple[CompactModuleClassProjection, ...],
        ClassUseIndex,
        frozenset[str],
    ]
    | None
]:
    """One baseline ``ClassUseIndex`` plus the set of modules whose aliases can
    participate in re-export chains, for O(changed) projection."""

    return [None]


@dataclass(frozen=True)
class ClassUseIndex:
    """Which consumers use each scanned class, and how.

    A reference is resolved only when a class it may name is asked about. Every
    answer the resolver gives ends in the same name as one of the reference's
    parts, or as the last part of an imported prefix's target once it follows
    every re-export rename (`followed_reexports`). Each immutable projection
    indexes its own references by part name once; this index records only the
    few renamed aliases whose resolved tail differs from their local spelling,
    and reads matching references on demand."""

    resolver: CompactClassReferenceResolver
    projections: tuple[CompactModuleClassProjection, ...]
    extra_part_names_by_tail: Mapping[str, tuple[str, ...]]

    @classmethod
    def from_projections(
        cls,
        projections: Sequence[CompactModuleClassProjection],
        class_index: CompactClassFamilyIndex,
    ) -> ClassUseIndex:
        proj_tuple = projections if isinstance(projections, tuple) else tuple(projections)
        if ScanCache.active() is not None:
            slot = _base_class_use_index_slot()
            base_entry = slot[0]
            if base_entry is not None:
                base_projs, base_index, base_reexport_modules = base_entry
                base_by_path = {p.file_path: p for p in base_projs}
                cur_by_path = {p.file_path: p for p in proj_tuple}
                changed = tuple(
                    p for p in proj_tuple if base_by_path.get(p.file_path) is not p
                )
                removed = tuple(
                    p for path, p in base_by_path.items() if path not in cur_by_path
                )
                if len(changed) + len(removed) <= 64:
                    old_changed = tuple(
                        base_by_path[p.file_path]
                        for p in changed
                        if p.file_path in base_by_path
                    )
                    aliases_identical = all(
                        (old_p := base_by_path.get(p.file_path)) is not None
                        and old_p.import_aliases == p.import_aliases
                        and old_p.script_scopes == p.script_scopes
                        for p in changed
                    ) and not removed
                    if aliases_identical:
                        resolver = replace(
                            base_index.resolver,
                            known_symbols=class_index.classes_by_symbol,
                            unique_symbols_by_name=class_index.unique_symbols_by_name,
                            unique_symbols_by_suffix=class_index.unique_symbols_by_suffix,
                        )
                        built_inc = cls(
                            resolver,
                            proj_tuple,
                            base_index.extra_part_names_by_tail,
                        )
                        projected_parts = cls._project_part_index(
                            base_index, proj_tuple, changed, old_changed, removed
                        )
                        if projected_parts is not None:
                            built_inc.__dict__["_projections_by_part"] = projected_parts
                        return built_inc
                    # Check if any changed/removed module changes a renamed alias or
                    # touches a module referenced as a re-export chain source.
                    def _has_renamed_alias(p: CompactModuleClassProjection) -> bool:
                        for local_name, target in p.import_aliases:
                            if local_name != target.rpartition(".")[2]:
                                return True
                        for script in p.script_scopes:
                            for local_name, target in script.import_aliases:
                                if local_name != target.rpartition(".")[2]:
                                    return True
                        return False

                    touched_modules = {
                        p.module_name for p in (*changed, *old_changed, *removed)
                    }
                    if (
                        touched_modules.isdisjoint(base_reexport_modules)
                        and not any(_has_renamed_alias(p) for p in (*changed, *old_changed, *removed))
                    ):
                        updated_aliases = dict(
                            base_index.resolver.import_aliases_by_module_name
                        )
                        for p in (*old_changed, *removed):
                            updated_aliases.pop(p.module_name, None)
                            for script in p.script_scopes:
                                updated_aliases.pop(script.module_name, None)
                        for p in changed:
                            updated_aliases[p.module_name] = dict(p.import_aliases)
                            for script in p.script_scopes:
                                updated_aliases[script.module_name] = dict(
                                    script.import_aliases
                                )
                        resolver = replace(
                            base_index.resolver,
                            known_symbols=class_index.classes_by_symbol,
                            unique_symbols_by_name=class_index.unique_symbols_by_name,
                            unique_symbols_by_suffix=class_index.unique_symbols_by_suffix,
                            import_aliases_by_module_name=updated_aliases,
                        )
                        built_inc = cls(
                            resolver,
                            proj_tuple,
                            base_index.extra_part_names_by_tail,
                        )
                        projected_parts = cls._project_part_index(
                            base_index, proj_tuple, changed, old_changed, removed
                        )
                        if projected_parts is not None:
                            built_inc.__dict__["_projections_by_part"] = projected_parts
                        return built_inc

        scoped = CompactClassReferenceResolver.from_index(proj_tuple, class_index)
        script_aliases = {
            script.module_name: dict(script.import_aliases)
            for projection in proj_tuple
            for script in projection.script_scopes
        }
        resolver = (
            replace(
                scoped,
                import_aliases_by_module_name={
                    **scoped.import_aliases_by_module_name,
                    **script_aliases,
                },
            )
            if script_aliases
            else scoped
        )
        import_aliases_by_module = resolver.import_aliases_by_module_name
        reexport_tail: dict[str, str] = {}
        extra_by_tail: dict[str, list[str]] = defaultdict(list)
        reexport_modules: set[str] = set()
        for module_name, aliases in import_aliases_by_module.items():
            for local_name, alias_target in aliases.items():
                tail = reexport_tail.get(alias_target)
                if tail is None:
                    resolved = followed_reexports(
                        alias_target,
                        import_aliases_by_module,
                    )
                    if resolved != alias_target:
                        reexport_modules.add(alias_target.rpartition(".")[0])
                        reexport_modules.add(module_name)
                    tail = resolved.rpartition(".")[2]
                    reexport_tail[alias_target] = tail
                if tail != local_name:
                    extra_by_tail[tail].append(local_name)
                    reexport_modules.add(module_name)
        built = cls(
            resolver,
            proj_tuple,
            {
                tail: tuple(dict.fromkeys(names))
                for tail, names in extra_by_tail.items()
            },
        )
        if ScanCache.active() is not None:
            slot = _base_class_use_index_slot()
            if slot[0] is None:
                slot[0] = (proj_tuple, built, frozenset(reexport_modules))
        return built

    @cached_property
    def _projections_by_part(
        self,
    ) -> Mapping[str, tuple[CompactModuleClassProjection, ...]]:
        by_part: dict[str, list[CompactModuleClassProjection]] = defaultdict(list)
        for projection in self.projections:
            for part in projection.references_by_part:
                by_part[part].append(projection)
        return {part: tuple(projs) for part, projs in by_part.items()}

    @staticmethod
    def _project_part_index(
        base_index: ClassUseIndex,
        proj_tuple: tuple[CompactModuleClassProjection, ...],
        changed: tuple[CompactModuleClassProjection, ...],
        old_changed: tuple[CompactModuleClassProjection, ...],
        removed: tuple[CompactModuleClassProjection, ...],
    ) -> dict[str, tuple[CompactModuleClassProjection, ...]] | None:
        base_by_part = base_index.__dict__.get("_projections_by_part")
        if base_by_part is None:
            return None
        if not changed and not removed:
            return base_by_part
        old_by_path = {p.file_path: p for p in (*old_changed, *removed)}
        new_by_path = {p.file_path: p for p in changed}
        order_by_path = {p.file_path: idx for idx, p in enumerate(proj_tuple)}
        touched_parts: set[str] = set()
        for p in (*old_changed, *removed, *changed):
            touched_parts.update(p.references_by_part)
        if not touched_parts:
            return base_by_part
        updated = dict(base_by_part)
        for part in touched_parts:
            cur_list = [
                p for p in updated.get(part, ()) if p.file_path not in old_by_path
            ]
            for p in changed:
                if part in p.references_by_part:
                    cur_list.append(p)
            if cur_list:
                cur_list.sort(key=lambda p: order_by_path[p.file_path])
                updated[part] = tuple(cur_list)
            else:
                updated.pop(part, None)
        return updated

    def _candidate_references(
        self, terminal_name: str
    ) -> Iterator[tuple[str, str, CompactClassReference]]:
        extra_names = self.extra_part_names_by_tail.get(terminal_name, ())
        by_part_index = self._projections_by_part
        if not extra_names:
            for projection in by_part_index.get(terminal_name, ()):
                entries = projection.references_by_part.get(terminal_name)
                if entries:
                    file_path = projection.file_path
                    for scope_name, reference in entries:
                        yield file_path, scope_name, reference
            return
        wanted_names = (terminal_name, *extra_names)
        candidate_projs: dict[int, CompactModuleClassProjection] = {}
        for name in wanted_names:
            for projection in by_part_index.get(name, ()):
                candidate_projs[id(projection)] = projection
        if not candidate_projs:
            return
        if len(candidate_projs) > 1:
            order_by_id = {id(p): i for i, p in enumerate(self.projections)}
            ordered_projs = sorted(
                candidate_projs.values(), key=lambda p: order_by_id[id(p)]
            )
        else:
            ordered_projs = list(candidate_projs.values())
        for projection in ordered_projs:
            by_part = projection.references_by_part
            matched: list[tuple[str, CompactClassReference]] = []
            for name in wanted_names:
                entries = by_part.get(name)
                if entries:
                    matched.extend(entries)
            if not matched:
                continue
            file_path = projection.file_path
            if len(matched) == 1:
                scope_name, reference = matched[0]
                yield file_path, scope_name, reference
                continue
            seen: set[tuple[str, int]] = set()
            for scope_name, references in (
                (projection.module_name, projection.class_references),
                *(
                    (script.module_name, script.class_references)
                    for script in projection.script_scopes
                ),
            ):
                matched_ids = {
                    id(reference) for scope, reference in matched if scope == scope_name
                }
                if not matched_ids:
                    continue
                for reference in references:
                    reference_id = id(reference)
                    if reference_id in matched_ids and (scope_name, reference_id) not in seen:
                        seen.add((scope_name, reference_id))
                        yield file_path, scope_name, reference

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
                for file_path, module_name, reference in self._candidate_references(
                    symbol.rpartition(".")[2]
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
    offered: ClassOverlapCandidate | None = None
    """The rung below a move only its own guards refuse: the extraction that
    shares the block on a mixin and keeps both identities (paper1 Thm 4.8). The
    pair's finding is that rung; a move that asks for this one still finds it."""

    @property
    def rungs(self) -> tuple[ClassOverlapCandidate, ...]:
        """This verdict, then the repair it offers below its own move."""

        return (self,) if self.offered is None else (self, *self.offered.rungs)

    @property
    def repairs(self) -> bool:
        """Whether this rung is a repair. No owner moves a block one of whose
        names the classes declare differently; otherwise a move with owners to
        choose from repairs when one of them is permitted, possibly once built
        by keyword, and any other move when the verdict is."""

        if any(receipt.kind is OverlapReceiptKind.DECLARATION_DIFFERS for receipt in self.refusals):
            return False
        if self.owner_candidates:
            return any(owner.permitted or owner.keyword_cure for owner in self.owner_candidates)
        return self.permitted

    def ending_with(self, rung: ClassOverlapCandidate) -> ClassOverlapCandidate:
        """This ladder with `rung` below its lowest rung."""

        return replace(self, offered=rung if self.offered is None else self.offered.ending_with(rung))

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


@ScanCache.cached
def _base_class_overlap_verdicts_slots() -> dict[
    bool,
    tuple[
        ClassOverlapVerdicts,
        tuple[tuple[str, str, bool], ...],
        dict[tuple[str, str], tuple[bool, tuple[ClassOverlapCandidate, ...]]],
    ],
]:
    """Baseline full-scan ``ClassOverlapVerdicts`` and candidate pairs keyed by
    whether field declarations are settled, so settled detector scans and
    unsettled snapshot verdicts each project incrementally without colliding."""

    return {}


@dataclass(frozen=True)
class ClassOverlapVerdicts(DeclaredC3):
    """Every overlap worth a verdict in one scan, with its receipts, over the
    scan's classes and the installed ones they name, in their C3 orders."""

    projections: tuple[CompactModuleClassProjection, ...]
    class_index: CompactClassFamilyIndex

    @cached_property
    def _is_settled_projections(self) -> bool:
        return not any(
            p.has_unsettled_field_declarations for p in self.projections
        )

    @cached_property
    def bodies(self) -> ClassBodyIndex:
        index = ClassBodyIndex.from_projections(self.projections)
        if ScanCache.active() is not None:
            base_entry = _base_class_overlap_verdicts_slots().get(
                self._is_settled_projections
            )
            if base_entry is not None:
                base_verdicts = base_entry[0]
                if base_verdicts is not self and "holders" in base_verdicts.bodies.__dict__:
                    self._try_project_holders(index, base_verdicts)
        return index

    def _try_project_holders(
        self,
        index: ClassBodyIndex,
        base_verdicts: ClassOverlapVerdicts,
    ) -> tuple[
        tuple[CompactModuleClassProjection, ...],
        tuple[CompactModuleClassProjection, ...],
        tuple[CompactModuleClassProjection, ...],
        set[ClassBodyDeclarationIdentity],
        bool,
    ] | None:
        """Project ``holders`` incrementally from ``base_verdicts`` when only a
        small set of modules changed without star imports or wrapper edits."""

        cached_delta = self.__dict__.get("_projected_holders_delta")
        if cached_delta is not None:
            return cached_delta
        if any(p.star_import_origins for p in self.projections) or any(
            p.star_import_origins for p in base_verdicts.projections
        ):
            return None
        base_by_path = {p.file_path: p for p in base_verdicts.projections}
        cur_by_path = {p.file_path: p for p in self.projections}
        changed_projs = tuple(
            p for p in self.projections if base_by_path.get(p.file_path) is not p
        )
        removed_projs = tuple(
            p for path, p in base_by_path.items() if path not in cur_by_path
        )
        if len(changed_projs) + len(removed_projs) > 64:
            return None
        old_changed = tuple(
            base_by_path[p.file_path]
            for p in changed_projs
            if p.file_path in base_by_path
        )
        old_wrappers = tuple(
            w for p in (*old_changed, *removed_projs) for w in p.field_declaration_wrappers
        )
        new_wrappers = tuple(
            w for p in changed_projs for w in p.field_declaration_wrappers
        )
        if old_wrappers != new_wrappers:
            return None
        if not changed_projs and not removed_projs:
            index.__dict__["holders"] = base_verdicts.bodies.holders
            delta = (changed_projs, old_changed, removed_projs, set(), False)
            self.__dict__["_projected_holders_delta"] = delta
            return delta

        old_symbols = {
            c.symbol for p in (*old_changed, *removed_projs) for c in p.classes
        }
        new_symbols = {c.symbol for p in changed_projs for c in p.classes}
        base_holders = base_verdicts.bodies.holders
        holders_dict: dict[ClassBodyDeclarationIdentity, tuple[str, ...]] = dict(
            base_holders
        )
        touched_idents: dict[ClassBodyDeclarationIdentity, list[str]] = {}
        for sym in old_symbols | new_symbols:
            old_idents: set[ClassBodyDeclarationIdentity] = set()
            if sym in old_symbols and sym in base_verdicts.bodies.classes_by_symbol:
                for decl in base_verdicts.bodies.declarations_of(sym).values():
                    old_idents.update(decl.identities)
            new_idents: set[ClassBodyDeclarationIdentity] = set()
            if sym in new_symbols and sym in index.classes_by_symbol:
                for decl in index.declarations_of(sym).values():
                    new_idents.update(decl.identities)
            if old_idents == new_idents:
                continue
            for ident in old_idents - new_idents:
                bucket = touched_idents.get(ident)
                if bucket is None:
                    bucket = list(holders_dict.get(ident, ()))
                    touched_idents[ident] = bucket
                if sym in bucket:
                    bucket.remove(sym)
            for ident in new_idents - old_idents:
                bucket = touched_idents.get(ident)
                if bucket is None:
                    bucket = list(holders_dict.get(ident, ()))
                    touched_idents[ident] = bucket
                if sym not in bucket:
                    bucket.append(sym)
        limit = index.holder_limit
        any_crossed_limit = False
        for ident, bucket in touched_idents.items():
            old_len = len(base_holders.get(ident, ()))
            new_len = len(bucket)
            if (old_len <= limit) != (new_len <= limit):
                any_crossed_limit = True
            if bucket:
                holders_dict[ident] = tuple(sorted(bucket))
            else:
                holders_dict.pop(ident, None)
        index.__dict__["holders"] = holders_dict
        delta = (
            changed_projs,
            old_changed,
            removed_projs,
            set(touched_idents),
            any_crossed_limit,
        )
        self.__dict__["_projected_holders_delta"] = delta
        return delta

    @cached_property
    def exposure(self) -> CompactRepositoryPublicExposureIndex:
        """What the scanned modules export, for reading a class outside them."""

        return CompactRepositoryPublicExposureIndex(self.projections)

    @property
    def creation(self) -> ClassCreationOrderProof:
        return self.class_index.creation

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
    def field_order_reads(self) -> FieldOrderReads:
        """The scan's ordered field reads: an owner whose composition reorders a
        class they read is refused here and at its apply, by one proof."""

        return FieldOrderReads(self.projections, self.class_index)

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

        if ScanCache.active() is not None:
            slots = _base_class_overlap_verdicts_slots()
            settled_key = self._is_settled_projections
            base_entry = slots.get(settled_key)
            if base_entry is None:
                candidates, pairs, by_pair = self._judged_with_pairs(
                    self.bodies.overlaps,
                    self._inheriting_pairs_of(self.bodies.classes_by_symbol),
                )
                slots[settled_key] = (self, pairs, by_pair)
                return candidates
            base_verdicts, base_pairs, base_by_pair = base_entry
            projected = self._projected_candidates(base_verdicts, base_pairs, base_by_pair)
            if projected is not None:
                return projected
        return self.judged(
            self.bodies.overlaps,
            self._inheriting_pairs_of(self.bodies.classes_by_symbol),
        )

    def _judged_with_pairs(
        self,
        overlaps: Iterable[ClassBodyOverlap],
        inheriting_pairs: frozenset[tuple[str, str]],
    ) -> tuple[
        tuple[ClassOverlapCandidate, ...],
        tuple[tuple[str, str, bool], ...],
        dict[tuple[str, str], tuple[bool, tuple[ClassOverlapCandidate, ...]]],
    ]:
        """Run ``judged`` while recording each surviving pair and its verdicts."""

        candidates: list[ClassOverlapCandidate] = []
        judged_direct: set[tuple[str, str]] = set()
        recorded_pairs: list[tuple[str, str, bool]] = []
        by_pair: dict[tuple[str, str], tuple[bool, tuple[ClassOverlapCandidate, ...]]] = {}
        for overlap in overlaps:
            candidate = self.verdict(overlap)
            if candidate is not None:
                rungs = candidate.rungs
                candidates.extend(rungs)
                pair = (overlap.first.symbol, overlap.second.symbol)
                judged_direct.add(pair)
                recorded_pairs.append((pair[0], pair[1], True))
                by_pair[pair] = (True, rungs)
        floor_declarations = REPORTED.declarations
        for first, second in sorted(inheriting_pairs - judged_direct):
            first_resolved = self.resolved_declarations(first)
            second_resolved = self.resolved_declarations(second)
            shared_names = first_resolved.keys() & second_resolved.keys()
            if not shared_names:
                continue
            first_own = self.bodies.declarations_of(first)
            second_own = self.bodies.declarations_of(second)
            if (
                len(shared_names) < floor_declarations
                and not (first_own.keys() <= shared_names)
                and not (second_own.keys() <= shared_names)
            ):
                continue
            has_candidate_inherited = False
            for name in shared_names:
                first_owner = first_resolved[name].owner.symbol
                second_owner = second_resolved[name].owner.symbol
                if first_owner == second_owner:
                    continue
                if first_owner != first and second_owner != second:
                    continue
                if first_owner == first and second_owner == second:
                    continue
                has_candidate_inherited = True
                break
            if not has_candidate_inherited:
                continue
            overlap = self.bodies.overlap(first, second).with_inherited(
                first_resolved, second_resolved
            )
            if (
                overlap.inherited
                and (
                    overlap.first_is_contained
                    or overlap.second_is_contained
                    or len(overlap.shared) >= floor_declarations
                )
                and not self._restated_through_an_ancestor(overlap)
                and (candidate := self.verdict(overlap)) is not None
            ):
                rungs = candidate.rungs
                candidates.extend(rungs)
                recorded_pairs.append((first, second, False))
                by_pair[(first, second)] = (False, rungs)
        return tuple(candidates), tuple(recorded_pairs), by_pair

    def _projected_candidates(
        self,
        base_verdicts: ClassOverlapVerdicts,
        base_pairs: tuple[tuple[str, str, bool], ...],
        base_by_pair: Mapping[tuple[str, str], tuple[bool, tuple[ClassOverlapCandidate, ...]]],
    ) -> tuple[ClassOverlapCandidate, ...] | None:
        """Exact incremental projection of ``candidates`` from ``base_verdicts``."""

        bodies = self.bodies
        delta = self._try_project_holders(bodies, base_verdicts)
        if delta is None:
            return None
        changed_projs, old_changed, removed_projs, changed_identities, any_crossed_limit = (
            delta
        )
        if not changed_projs and not removed_projs:
            return base_verdicts.candidates
        if any_crossed_limit:
            return None

        old_symbols = {
            c.symbol for p in (*old_changed, *removed_projs) for c in p.classes
        }
        new_symbols = {c.symbol for p in changed_projs for c in p.classes}
        classes = bodies.classes_by_symbol
        base_bodies = base_verdicts.bodies
        base_index = base_verdicts.class_index
        cur_index = self.class_index

        def _class_semantics_equal(sym: str) -> bool:
            old_cls = base_index.class_for(sym)
            cur_cls = cur_index.class_for(sym)
            if old_cls is None or cur_cls is None:
                return False
            if old_cls == cur_cls:
                return True
            if (
                old_cls.resolved_base_symbols != cur_cls.resolved_base_symbols
                or old_cls.base_references != cur_cls.base_references
                or old_cls.direct_base_count != cur_cls.direct_base_count
                or old_cls.base_references_are_complete != cur_cls.base_references_are_complete
                or old_cls.mro_bases_are_static != cur_cls.mro_bases_are_static
                or old_cls.metaclass_names != cur_cls.metaclass_names
                or old_cls.class_keyword_names != cur_cls.class_keyword_names
                or old_cls.class_header_is_reconstructible != cur_cls.class_header_is_reconstructible
                or old_cls.declares_autoregister_meta != cur_cls.declares_autoregister_meta
                or old_cls.is_final != cur_cls.is_final
                or old_cls.is_abstract != cur_cls.is_abstract
                or old_cls.abstract_method_names != cur_cls.abstract_method_names
                or old_cls.method_names != cur_cls.method_names
                or old_cls.type_parameter_names != cur_cls.type_parameter_names
                or old_cls.enum_declaration != cur_cls.enum_declaration
                or old_cls.base_argument_sources != cur_cls.base_argument_sources
                or old_cls.declared_base_names != cur_cls.declared_base_names
                or old_cls.registry_facts != cur_cls.registry_facts
                or old_cls.star_resolution != cur_cls.star_resolution
                or old_cls.class_decorators_are_promotion_safe
                != cur_cls.class_decorators_are_promotion_safe
            ):
                return False
            old_dc = old_cls.dataclass_declaration
            cur_dc = cur_cls.dataclass_declaration
            if type(old_dc) is not type(cur_dc):
                return False
            if isinstance(old_dc, CompactDataclassDeclaration) and isinstance(
                cur_dc, CompactDataclassDeclaration
            ):
                if (
                    old_dc.runtime_declaration != cur_dc.runtime_declaration
                    or len(old_dc.failures) != len(cur_dc.failures)
                    or any(
                        a.violation != b.violation
                        for a, b in zip(old_dc.failures, cur_dc.failures, strict=True)
                    )
                    or old_dc.frozenness != cur_dc.frozenness
                    or len(old_dc.fields) != len(cur_dc.fields)
                    or any(
                        (a.name, a.role, a.binding, a.declaration_call)
                        != (b.name, b.role, b.binding, b.declaration_call)
                        for a, b in zip(old_dc.fields, cur_dc.fields, strict=True)
                    )
                ):
                    return False
            old_in = sym in base_bodies.classes_by_symbol
            cur_in = sym in classes
            if old_in != cur_in:
                return False
            if old_in:
                old_decls = base_bodies.declarations_of(sym)
                cur_decls = bodies.declarations_of(sym)
                if old_decls.keys() != cur_decls.keys():
                    return False
                for name, old_d in old_decls.items():
                    cur_d = cur_decls[name]
                    if (
                        not old_d.same_declaration_as(cur_d)
                        or old_d.identities != cur_d.identities
                        or old_d.kind is not cur_d.kind
                    ):
                        return False
            return True

        direct_changed: set[str] = set()
        line_only_changed: set[str] = set()
        for sym in old_symbols | new_symbols:
            if sym not in old_symbols or sym not in new_symbols:
                direct_changed.add(sym)
            elif not _class_semantics_equal(sym):
                direct_changed.add(sym)
            elif base_index.class_for(sym) != cur_index.class_for(sym):
                line_only_changed.add(sym)

        mro_changed: set[str] = set(direct_changed)
        for sym in direct_changed:
            mro_changed.update(base_index.descendant_symbols(sym))
            mro_changed.update(cur_index.descendant_symbols(sym))

        # Only track exact sibling-dependent effects instead of invalidating all
        # children of hub bases like DeclaredFamily/ABC:
        # 1. sole_shared_base changes only when a base's child count crosses <=2.
        # 2. existing_owner changes when a base gains its first or loses its last child.
        # 3. family_axis_names changes only for sibling pairs sharing a base AND a
        #    member name whose direct declaration on a child of that base changed.
        affected_bases: set[str] = set()
        for sym in mro_changed:
            if (old_cls := base_index.class_for(sym)) is not None:
                affected_bases.update(old_cls.resolved_base_symbols)
            if (cur_cls := cur_index.class_for(sym)) is not None:
                affected_bases.update(cur_cls.resolved_base_symbols)

        sole_base_changed: set[str] = set()
        owner_identities_changed: set[tuple[str, ...]] = set(changed_identities)
        for base_sym in affected_bases:
            old_children = base_index.children_by_symbol.get(base_sym, ())
            cur_children = cur_index.children_by_symbol.get(base_sym, ())
            if old_children != cur_children:
                if len(old_children) <= 2 or len(cur_children) <= 2:
                    sole_base_changed.update(old_children)
                    sole_base_changed.update(cur_children)
                if bool(old_children) != bool(cur_children):
                    for sym in (base_sym, *cur_index.ancestor_symbols(base_sym)):
                        if sym in classes:
                            for decl in bodies.declarations_of(sym).values():
                                owner_identities_changed.update(decl.identities)
                    for sym in (base_sym, *base_index.ancestor_symbols(base_sym)):
                        if sym in base_verdicts.bodies.classes_by_symbol:
                            for decl in base_verdicts.bodies.declarations_of(sym).values():
                                owner_identities_changed.update(decl.identities)

        changed_axis_names_by_base: dict[str, set[str]] = {}
        base_bodies = base_verdicts.bodies
        for sym in direct_changed:
            old_cls = base_index.class_for(sym)
            cur_cls = cur_index.class_for(sym)
            old_bases = set(old_cls.resolved_base_symbols) if old_cls is not None else set()
            cur_bases = set(cur_cls.resolved_base_symbols) if cur_cls is not None else set()
            old_decls = (
                base_bodies.declarations_of(sym)
                if sym in base_bodies.classes_by_symbol
                else {}
            )
            cur_decls = bodies.declarations_of(sym) if sym in classes else {}
            all_names = set(old_decls) | set(cur_decls)
            for base_sym in old_bases ^ cur_bases:
                if all_names:
                    changed_axis_names_by_base.setdefault(base_sym, set()).update(all_names)
            shared_bases = old_bases & cur_bases
            if shared_bases:
                diff_names = {
                    name
                    for name in all_names
                    if (old_d := old_decls.get(name)) is None
                    or (cur_d := cur_decls.get(name)) is None
                    or not old_d.same_declaration_as(cur_d)
                }
                if diff_names:
                    for base_sym in shared_bases:
                        changed_axis_names_by_base.setdefault(base_sym, set()).update(
                            diff_names
                        )

        # Track every class whose ClassUseIndex, FieldOrderReads, or import
        # graph context could have changed across the edited/removed modules.
        old_by_path = {p.file_path: p for p in (*old_changed, *removed_projs)}
        new_by_path = {p.file_path: p for p in changed_projs}
        graph_changed_modules: set[str] = set()
        order_changed = False
        dirty_order_classes: set[str] = set()
        changed_ref_names: set[str] = set()
        base_aliases = base_verdicts.import_aliases_by_module
        cur_aliases = self.import_aliases_by_module
        all_module_names = frozenset(base_aliases) | frozenset(cur_aliases)

        def _scanned_imports_of(aliases: Iterable[tuple[str, str]], own_mod: str) -> frozenset[str]:
            found: set[str] = set()
            for _local, target in aliases:
                parts = target.split(".")
                for length in range(1, len(parts) + 1):
                    cand = ".".join(parts[:length])
                    if cand in all_module_names and cand != own_mod:
                        found.add(cand)
            return frozenset(found)

        def _record_ref_names(refs: Iterable[CompactClassReference]) -> None:
            for ref in refs:
                changed_ref_names.update(ref.reference_parts)
                if ref.tested_subclass:
                    changed_ref_names.update(ref.tested_subclass)

        for path in old_by_path.keys() | new_by_path.keys():
            old_p = old_by_path.get(path)
            new_p = new_by_path.get(path)
            if old_p is None or new_p is None:
                p = new_p if new_p is not None else old_p
                assert p is not None
                graph_changed_modules.add(p.module_name)
                if p.field_order_summaries:
                    order_changed = True
                _record_ref_names(p.class_references)
                for script in p.script_scopes:
                    _record_ref_names(script.class_references)
                for local_name, target in p.import_aliases:
                    changed_ref_names.add(local_name)
                    changed_ref_names.add(target.rpartition(".")[2])
                continue
            if old_p.import_aliases != new_p.import_aliases:
                if _scanned_imports_of(old_p.import_aliases, old_p.module_name) != _scanned_imports_of(
                    new_p.import_aliases, new_p.module_name
                ):
                    graph_changed_modules.add(new_p.module_name)
                old_alias_map = dict(old_p.import_aliases)
                new_alias_map = dict(new_p.import_aliases)
                changed_local_names: set[str] = set()
                for local_name in old_alias_map.keys() | new_alias_map.keys():
                    old_t = old_alias_map.get(local_name)
                    new_t = new_alias_map.get(local_name)
                    if old_t != new_t:
                        changed_local_names.add(local_name)
                        changed_ref_names.add(local_name)
                        if old_t is not None:
                            changed_ref_names.add(old_t.rpartition(".")[2])
                            changed_ref_names.add(
                                followed_reexports(old_t, base_aliases).rpartition(".")[2]
                            )
                        if new_t is not None:
                            changed_ref_names.add(new_t.rpartition(".")[2])
                            changed_ref_names.add(
                                followed_reexports(new_t, cur_aliases).rpartition(".")[2]
                            )
                # Only references whose leading name changed alias, or references
                # added/removed in this module, can change ClassUseIndex resolution.
                if old_p.class_references != new_p.class_references:
                    old_counts = Counter(old_p.class_references)
                    new_counts = Counter(new_p.class_references)
                    _record_ref_names((old_counts - new_counts).elements())
                    _record_ref_names((new_counts - old_counts).elements())
                if changed_local_names:
                    for ref in (*old_p.class_references, *new_p.class_references):
                        if (ref.reference_parts and ref.reference_parts[0] in changed_local_names) or (
                            ref.tested_subclass and ref.tested_subclass[0] in changed_local_names
                        ):
                            changed_ref_names.update(ref.reference_parts)
                            if ref.tested_subclass:
                                changed_ref_names.update(ref.tested_subclass)
            elif old_p.class_references != new_p.class_references:
                old_counts = Counter(old_p.class_references)
                new_counts = Counter(new_p.class_references)
                _record_ref_names((old_counts - new_counts).elements())
                _record_ref_names((new_counts - old_counts).elements())
            if old_p.script_scopes != new_p.script_scopes:
                graph_changed_modules.add(new_p.module_name)
                for script in (*old_p.script_scopes, *new_p.script_scopes):
                    _record_ref_names(script.class_references)
                    for local_name, target in script.import_aliases:
                        changed_ref_names.add(local_name)
                        changed_ref_names.add(target.rpartition(".")[2])
            if old_p.field_order_summaries != new_p.field_order_summaries:
                old_active = {
                    summary.symbol: summary
                    for summary in old_p.field_order_summaries
                    if summary.reads or summary.handed
                }
                new_active = {
                    summary.symbol: summary
                    for summary in new_p.field_order_summaries
                    if summary.reads or summary.handed
                }
                if old_active != new_active:
                    changed_order_syms: list[tuple[str, FunctionOrderSummary | None, FunctionOrderSummary | None]] = []
                    for sym in old_active.keys() | new_active.keys():
                        os_ = old_active.get(sym)
                        ns_ = new_active.get(sym)
                        if os_ != ns_:
                            changed_order_syms.append((sym, os_, ns_))
                    for sym, os_, ns_ in changed_order_syms:
                        if os_ is not None and ns_ is not None and replace(os_, line=0) == replace(ns_, line=0):
                            if ns_.class_symbol is None:
                                # Top-level functions never contribute their line number to
                                # `FieldOrderReads.first_read` (only the class method's line
                                # appears in its message).
                                continue
                            if base_verdicts.field_order_reads.first_read(ns_.class_symbol) is not None:
                                dirty_order_classes.add(ns_.class_symbol)
                            continue
                        cls_sym = (ns_.class_symbol if ns_ is not None else None) or (
                            os_.class_symbol if os_ is not None else None
                        )
                        if cls_sym is None:
                            # A top-level function only affects `FieldOrderReads` when some
                            # function hands a parameter to it by qualified callee name.
                            referenced_externally = any(
                                any(h.callee == sym for h in other_summary.handed)
                                for other_summary in base_verdicts.field_order_reads.by_symbol.values()
                            ) or any(
                                any(h.callee == sym for h in s.handed)
                                for p_ch in changed_projs
                                for s in p_ch.field_order_summaries
                            )
                            if referenced_externally:
                                order_changed = True
                                break
                            continue
                        # If a class method summary changed, it affects only `cls_sym`
                        # and its descendants unless another class hands a parameter to
                        # it by qualified callee name or receiver method name.
                        method_name = ns_.name if ns_ is not None else os_.name  # type: ignore[union-attr]
                        referenced_externally = False
                        for other_summary in base_verdicts.field_order_reads.by_symbol.values():
                            if other_summary.class_symbol == cls_sym:
                                continue
                            if any(
                                h.callee == sym or h.receiver_method == method_name
                                for h in other_summary.handed
                            ):
                                referenced_externally = True
                                break
                        if referenced_externally:
                            order_changed = True
                            break
                        dirty_order_classes.add(cls_sym)

        # Expand renamed-alias parts back to their target tails so renamed
        # references (`from m import Foo as Bar`) invalidate `Foo`.
        if changed_ref_names:
            for tail, extras in base_verdicts.uses.extra_part_names_by_tail.items():
                if not changed_ref_names.isdisjoint(extras):
                    changed_ref_names.add(tail)
            for p in changed_projs:
                for local_name, target in p.import_aliases:
                    tail = target.rpartition(".")[2]
                    if local_name != tail and local_name in changed_ref_names:
                        changed_ref_names.add(tail)

        dirty_symbols: set[str] = (
            set(mro_changed) | sole_base_changed | dirty_order_classes | line_only_changed
        )
        for sym in dirty_order_classes:
            dirty_symbols.update(base_index.descendant_symbols(sym))
            dirty_symbols.update(cur_index.descendant_symbols(sym))
        for name in changed_ref_names:
            dirty_symbols.update(base_index.symbols_by_simple_name.get(name, ()))
            dirty_symbols.update(cur_index.symbols_by_simple_name.get(name, ()))

        graph_affected_cache: dict[str, bool] = {}
        graph_topology_changed = False
        changed_reach_targets: frozenset[str] = frozenset()
        if graph_changed_modules:
            base_mi = base_verdicts.module_imports
            cur_mi = self.module_imports
            if (
                base_mi.package_modules != cur_mi.package_modules
                or base_mi.imports.keys() != cur_mi.imports.keys()
            ):
                graph_topology_changed = True
            else:
                reach_diffs: set[str] = set()
                for mod in graph_changed_modules:
                    diff = base_mi.reachable(mod) ^ cur_mi.reachable(mod)
                    if diff:
                        reach_diffs.update(diff)
                changed_reach_targets = frozenset(reach_diffs)

        def _module_sees_graph_change(module_name: str) -> bool:
            if not graph_changed_modules:
                return False
            if module_name in graph_changed_modules:
                return True
            if graph_topology_changed:
                return True
            if not changed_reach_targets:
                return False
            cached_hit = graph_affected_cache.get(module_name)
            if cached_hit is None:
                base_reach = base_verdicts.module_imports.reachable(module_name)
                cur_reach = self.module_imports.reachable(module_name)
                cached_hit = (
                    module_name in changed_reach_targets
                    or base_reach != cur_reach
                    or not cur_reach.isdisjoint(changed_reach_targets)
                )
                graph_affected_cache[module_name] = cached_hit
            return cached_hit

        live_mro = frozenset(sym for sym in mro_changed if sym in classes)
        live_affected = frozenset(sym for sym in (mro_changed | sole_base_changed) if sym in classes)

        holders = bodies.holders
        limit = bodies.holder_limit
        direct_pairs: set[tuple[str, str]] = set()
        for ident in changed_identities:
            group = holders.get(ident, ())
            if 2 <= len(group) <= limit:
                for sym in group:
                    if sym in new_symbols or sym in live_affected:
                        for other in group:
                            if other != sym:
                                direct_pairs.add(
                                    (sym, other) if sym < other else (other, sym)
                                )
        for sym in live_affected:
            for decl in bodies.declarations_of(sym).values():
                for ident in decl.identities:
                    group = holders.get(ident, ())
                    if 2 <= len(group) <= limit:
                        for other in group:
                            if other != sym:
                                direct_pairs.add(
                                    (sym, other) if sym < other else (other, sym)
                                )
        # Any unchanged sibling pair sharing a base whose axis-name declarations
        # changed may flip from filtered-out to reported (or vice versa).
        axis_dirty_pairs: set[tuple[str, str]] = set()
        for base_sym, changed_names in changed_axis_names_by_base.items():
            children = [
                child
                for child in cur_index.children_by_symbol.get(base_sym, ())
                if child in classes
            ]
            by_name: dict[str, list[str]] = {}
            for child in children:
                child_decls = bodies.declarations_of(child)
                for name in changed_names:
                    if name in child_decls:
                        by_name.setdefault(name, []).append(child)
            for members in by_name.values():
                if len(members) < 2:
                    continue
                for i, first_sym in enumerate(members):
                    for second_sym in members[i + 1 :]:
                        pair = (
                            (first_sym, second_sym)
                            if first_sym < second_sym
                            else (second_sym, first_sym)
                        )
                        axis_dirty_pairs.add(pair)
                        first_decls = bodies.declarations_of(first_sym)
                        second_decls = bodies.declarations_of(second_sym)
                        if any(
                            ident in holders
                            and len(holders[ident]) <= limit
                            and first_sym in holders[ident]
                            and second_sym in holders[ident]
                            for shared_name in first_decls.keys() & second_decls.keys()
                            for ident in first_decls[shared_name].identities
                        ):
                            direct_pairs.add(pair)

        # Any new or changed inheriting pair (holder, inheritor) must touch
        # live_affected on either side: either `inheritor in live_affected`
        # (recomputed via _inheriting_pairs_of), or `holder in live_affected`
        # paired with descendants of co-holders of its own declaration identities.
        inh_pairs: set[tuple[str, str]] = set(self._inheriting_pairs_of(live_affected))
        for holder in live_affected:
            holder_family = {
                holder,
                *cur_index.ancestor_symbols(holder),
                *cur_index.descendant_symbols(holder),
            }
            for decl in bodies.declarations_of(holder).values():
                for ident in decl.identities:
                    group = holders.get(ident, ())
                    if len(group) > limit:
                        continue
                    for owner in group:
                        if owner == holder:
                            continue
                        for desc in cur_index.descendant_symbols(owner):
                            if desc in classes and desc not in holder_family:
                                inh_pairs.add(
                                    (holder, desc) if holder < desc else (desc, holder)
                                )

        for first, second, was_direct in base_pairs:
            if first not in classes or second not in classes:
                continue
            pair = (first, second)
            if was_direct:
                if pair not in direct_pairs:
                    # Confirm they still share at least one pairing identity.
                    first_decls = bodies.declarations_of(first)
                    second_decls = bodies.declarations_of(second)
                    if any(
                        ident in holders
                        and len(holders[ident]) <= limit
                        and first in holders[ident]
                        and second in holders[ident]
                        for name in first_decls.keys() & second_decls.keys()
                        for ident in first_decls[name].identities
                    ):
                        direct_pairs.add(pair)
            else:
                inh_pairs.add(pair)

        def _can_reuse_cached(
            first: str,
            second: str,
            rungs: tuple[ClassOverlapCandidate, ...],
        ) -> bool:
            if (
                first in dirty_symbols
                or second in dirty_symbols
                or (first, second) in axis_dirty_pairs
            ):
                return False
            first_decls = bodies.declarations_of(first)
            second_decls = bodies.declarations_of(second)
            for shared in rungs[0].overlap.shared:
                if (
                    shared.first_owner.symbol in dirty_symbols
                    or shared.second_owner.symbol in dirty_symbols
                ):
                    return False
                if owner_identities_changed:
                    owner1_decls = (
                        first_decls
                        if shared.first_owner.symbol == first
                        else bodies.declarations_of(shared.first_owner.symbol)
                    )
                    owner2_decls = (
                        second_decls
                        if shared.second_owner.symbol == second
                        else bodies.declarations_of(shared.second_owner.symbol)
                    )
                    d1 = owner1_decls.get(shared.name)
                    d2 = owner2_decls.get(shared.name)
                    if (d1 is not None and not owner_identities_changed.isdisjoint(d1.identities)) or (
                        d2 is not None and not owner_identities_changed.isdisjoint(d2.identities)
                    ):
                        return False
            for rung in rungs:
                if (
                    rung.owner.symbol in dirty_symbols
                    or rung.owner.symbol not in classes
                    or rung.restating.symbol in dirty_symbols
                    or rung.restating.symbol not in classes
                ):
                    return False
                if rung.co_restating is not None and (
                    rung.co_restating.symbol in dirty_symbols
                    or rung.co_restating.symbol not in classes
                ):
                    return False
                if rung.move is ExtractMixin or rung.owner_candidates:
                    if order_changed:
                        return False
                    if graph_changed_modules and (
                        _module_sees_graph_change(rung.owner.module_name)
                        or _module_sees_graph_change(rung.restating.module_name)
                        or (
                            rung.co_restating is not None
                            and _module_sees_graph_change(rung.co_restating.module_name)
                        )
                    ):
                        return False
                    for owner_cand in rung.owner_candidates:
                        if not dirty_symbols.isdisjoint(owner_cand.composers):
                            return False
                        if not dirty_symbols.isdisjoint(owner_cand.subject):
                            return False
                        if graph_changed_modules:
                            for sym in (*owner_cand.composers, *owner_cand.subject):
                                indexed = cur_index.class_for(sym)
                                if indexed is not None and _module_sees_graph_change(
                                    indexed.module_name
                                ):
                                    return False
            return True

        base_direct_set = {
            (first, second) for first, second, was_direct in base_pairs if was_direct
        }
        base_inh_set = {
            (first, second) for first, second, was_direct in base_pairs if not was_direct
        }
        candidates: list[ClassOverlapCandidate] = []
        judged: set[tuple[str, str]] = set()
        for first, second in sorted(direct_pairs):
            cached = base_by_pair.get((first, second))
            if cached is None:
                if (
                    (first, second) in base_direct_set
                    and first not in mro_changed
                    and second not in mro_changed
                    and (first, second) not in axis_dirty_pairs
                ):
                    continue
            elif cached[0] and _can_reuse_cached(first, second, cached[1]):
                candidates.extend(cached[1])
                judged.add((first, second))
                continue
            candidate = self.verdict(bodies.overlap(first, second))
            if candidate is not None:
                candidates.extend(candidate.rungs)
                judged.add((first, second))
        floor_declarations = REPORTED.declarations
        for first, second in sorted(inh_pairs - judged):
            cached = base_by_pair.get((first, second))
            if cached is None:
                if (
                    (first, second) in base_inh_set
                    and first not in mro_changed
                    and second not in mro_changed
                    and (first, second) not in axis_dirty_pairs
                ):
                    continue
            elif not cached[0] and _can_reuse_cached(first, second, cached[1]):
                candidates.extend(cached[1])
                continue
            first_resolved = self.resolved_declarations(first)
            second_resolved = self.resolved_declarations(second)
            shared_names = first_resolved.keys() & second_resolved.keys()
            if not shared_names:
                continue
            first_own = bodies.declarations_of(first)
            second_own = bodies.declarations_of(second)
            if (
                len(shared_names) < floor_declarations
                and not (first_own.keys() <= shared_names)
                and not (second_own.keys() <= shared_names)
            ):
                continue
            has_candidate_inherited = False
            for name in shared_names:
                first_owner = first_resolved[name].owner.symbol
                second_owner = second_resolved[name].owner.symbol
                if first_owner == second_owner:
                    continue
                if first_owner != first and second_owner != second:
                    continue
                if first_owner == first and second_owner == second:
                    continue
                has_candidate_inherited = True
                break
            if not has_candidate_inherited:
                continue
            overlap = bodies.overlap(first, second).with_inherited(
                first_resolved, second_resolved
            )
            if (
                overlap.inherited
                and (
                    overlap.first_is_contained
                    or overlap.second_is_contained
                    or len(overlap.shared) >= floor_declarations
                )
                and not self._restated_through_an_ancestor(overlap)
                and (candidate := self.verdict(overlap)) is not None
            ):
                candidates.extend(candidate.rungs)
        return tuple(candidates)

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

    def requested_as(
        self, first: str, second: str, move: type[ClassOverlapMove]
    ) -> ClassOverlapCandidate | None:
        """The verdict on one pair judged as `move`, the move a step makes."""

        return self.verdict(self.bodies.overlap(first, second), MoveRequest(move=move))

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
                yield from verdict.rungs

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
                candidates.extend(candidate.rungs)
                judged.add((overlap.first.symbol, overlap.second.symbol))
        floor_declarations = REPORTED.declarations
        for first, second in sorted(inheriting_pairs - judged):
            first_resolved = self.resolved_declarations(first)
            second_resolved = self.resolved_declarations(second)
            shared_names = first_resolved.keys() & second_resolved.keys()
            if not shared_names:
                continue
            first_own = self.bodies.declarations_of(first)
            second_own = self.bodies.declarations_of(second)
            if (
                len(shared_names) < floor_declarations
                and not (first_own.keys() <= shared_names)
                and not (second_own.keys() <= shared_names)
            ):
                continue
            has_candidate_inherited = False
            for name in shared_names:
                first_owner = first_resolved[name].owner.symbol
                second_owner = second_resolved[name].owner.symbol
                if first_owner == second_owner:
                    continue
                if first_owner != first and second_owner != second:
                    continue
                if first_owner == first and second_owner == second:
                    continue
                has_candidate_inherited = True
                break
            if not has_candidate_inherited:
                continue
            overlap = self.bodies.overlap(first, second).with_inherited(
                first_resolved, second_resolved
            )
            if (
                overlap.inherited
                and (
                    overlap.first_is_contained
                    or overlap.second_is_contained
                    or len(overlap.shared) >= floor_declarations
                )
                and not self._restated_through_an_ancestor(overlap)
                and (candidate := self.verdict(overlap)) is not None
            ):
                candidates.extend(candidate.rungs)
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

        resolved = self.resolved_declarations(symbol)
        return frozenset(
            name
            for name in block
            if (found := resolved.get(name)) is not None and found.declaration.kind.is_field
        )

    def _inheriting_pairs_of(self, symbols: Iterable[str]) -> frozenset[tuple[str, str]]:
        """Pairs where one of `symbols` inherits a declaration another class makes itself.

        The other is not the ancestor it inherits from, and neither is an
        ancestor of the other. Declarations more than `holder_limit` classes
        make do not pair, as for own declarations.
        """

        pairs: set[tuple[str, str]] = set()
        class_index = self.class_index
        holders_map = self.bodies.holders
        holder_limit = self.bodies.holder_limit
        for symbol in symbols:
            ancestors = class_index.ancestor_symbols(symbol)
            if not ancestors:
                continue
            family = {symbol, *ancestors, *class_index.descendant_symbols(symbol)}
            for resolved in self.resolved_declarations(symbol).values():
                if resolved.owner.symbol == symbol:
                    continue
                for identity in resolved.declaration.identities:
                    holders = holders_map.get(identity, ())
                    if len(holders) > holder_limit:
                        continue
                    for holder in holders:
                        if holder in family:
                            continue
                        pairs.add((holder, symbol) if holder < symbol else (symbol, holder))
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
        """The ladder over the whole block; when none of its rungs is a repair,
        that ladder ends with the one over the largest part of the block that is:
        first the names both classes declare the same way, where an owner can
        hoist one a class inherits; then those each class declares itself; then
        those of them that are not fields, which a plain mixin takes with no
        dataclass to agree on. The rest stays declared in each class (paper1
        Thm 4.8 on the part)."""

        ladder = self.ladder(overlap, floor)
        if ladder is None or ladder.rungs[-1].repairs:
            return ladder
        agreeing = overlap.agreeing_names
        parts = (
            overlap.matching_names,
            agreeing,
            frozenset(shared.name for shared in overlap.shared if shared.name in agreeing and not shared.kind.is_field),
        )
        tried: set[frozenset[str]] = {frozenset(shared.name for shared in overlap.shared)}
        for part in parts:
            if not part or part in tried:
                continue
            tried.add(part)
            # The whole block already passed the floor as one concept; the part
            # needs no weight of its own, but one name alone is no shared block.
            below = self.ladder(
                self.declaring(overlap, part).restricted(part),
                replace(
                    floor,
                    weight=NO_FLOOR.weight,
                    declarations=min(floor.declarations, PART_DECLARATIONS),
                ),
            )
            if below is None or not below.rungs[-1].repairs:
                continue
            lowest = ladder.rungs[-1]
            refusals = tuple(
                {
                    receipt.detail: receipt
                    for receipt in (
                        *lowest.refusals,
                        *(receipt for owner in lowest.owner_candidates for receipt in owner.refusals),
                    )
                }.values()
            )
            withheld = sorted(frozenset(shared.name for shared in overlap.shared) - part)
            note = Receipt(
                OverlapReceiptKind.STRONGER_MOVE_NEEDS,
                "the whole block would need: "
                + "; ".join(receipt.detail for receipt in refusals)
                + f", so the repair moves {', '.join(f'`{name}`' for name in sorted(part))} "
                + f"and each class keeps {', '.join(f'`{name}`' for name in withheld)}",
                tuple(location for receipt in refusals for location in receipt.evidence),
            )
            return ladder.ending_with(replace(below, receipts=(*below.receipts, note)))
        return ladder

    def declaring(self, overlap: ClassBodyOverlap, names: frozenset[str]) -> ClassBodyOverlap:
        """The overlap of the two classes that declare `names` themselves: this
        one, or, when one class inherits every one of them from one ancestor,
        the other class beside that ancestor, whose repair its heir inherits."""

        owners = frozenset(
            (shared.first_owner.symbol, shared.second_owner.symbol)
            for shared in overlap.shared
            if shared.name in names
        )
        if len(owners) != 1:
            return overlap
        ((first, second),) = owners
        if (first, second) == (overlap.first.symbol, overlap.second.symbol):
            return overlap
        return self.bodies.overlap(first, second)

    def ladder(
        self, overlap: ClassBodyOverlap, floor: ReportingFloor = REPORTED
    ) -> ClassOverlapCandidate | None:
        """The verdict on this block, with the extraction below a move only its
        own guards refuse."""

        if (
            not overlap.first_is_contained
            and not overlap.second_is_contained
            and len(overlap.shared) < floor.declarations
        ):
            return None
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
        judged = self._candidate(
            overlap,
            shape,
            floor.judged_repair(PairRepair(move, owner, restating, co_restating), first, second),
            extractable,
            members,
            shared_receipts,
            (*axis_receipts, *differing_receipts),
        )
        if judged.permitted or judged.move is ExtractMixin:
            return judged
        extraction = self._candidate(
            overlap,
            shape,
            PairRepair(ExtractMixin, first, second, None),
            extractable,
            members,
            shared_receipts,
            (*axis_receipts, *differing_receipts),
        )
        # paper1 Thm 4.8: when only the move itself refuses (it changes an
        # identity, keeps an owner whose construction, inherited copies or
        # ancestry the others lack), the implementations still agree, so the
        # verdict offers the rung below: share them on a mixin, keep identities.
        # A rung is offered only when it is a repair: one of its owners is
        # permitted, or permitted once built by keyword.
        own = frozenset((receipt.kind, receipt.detail) for receipt in extraction.refusals)
        if not own.isdisjoint(
            (receipt.kind, receipt.detail) for receipt in judged.refusals
        ) or not any(
            owner.permitted or owner.keyword_cure for owner in extraction.owner_candidates
        ):
            return judged
        return replace(
            judged,
            offered=replace(
                extraction,
                receipts=(
                    *extraction.receipts,
                    Receipt(
                        OverlapReceiptKind.STRONGER_MOVE_NEEDS,
                        f"`{judged.move.value}` would need: "
                        + "; ".join(receipt.detail for receipt in judged.refusals)
                        + ", so the pair shares a mixin and keeps both identities",
                        tuple(
                            location for receipt in judged.refusals for location in receipt.evidence
                        ),
                    ),
                ),
            ),
        )

    def _candidate(
        self,
        overlap: ClassBodyOverlap,
        shape: ClassOverlapShape,
        repair: PairRepair,
        extractable: tuple[SharedClassBodyDeclaration, ...],
        members: frozenset[str],
        shared_receipts: tuple[Receipt, ...],
        difference_receipts: tuple[Receipt, ...],
    ) -> ClassOverlapCandidate:
        """The verdict on one repair of the pair: its move, owner and receipts."""

        first, second = self.resolved(overlap.first), self.resolved(overlap.second)
        move, owner, restating, co_restating = repair
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
                *difference_receipts,
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
        if ScanCache.active() is not None:
            slots = _base_class_overlap_verdicts_slots()
            base_entry = slots.get(self._is_settled_projections) or next(
                iter(slots.values()), None
            )
            if base_entry is not None:
                base_verdicts = base_entry[0]
                if (
                    base_verdicts is not self
                    and "module_imports" in base_verdicts.__dict__
                ):
                    base_mi = base_verdicts.module_imports
                    base_by_mod = {
                        p.module_name: p for p in base_verdicts.projections
                    }
                    if len(self.projections) == len(base_by_mod) and all(
                        p.module_name in base_by_mod for p in self.projections
                    ):
                        changed = [
                            p
                            for p in self.projections
                            if (old_p := base_by_mod[p.module_name]) is not p
                            and old_p.import_aliases is not p.import_aliases
                            and old_p.import_aliases != p.import_aliases
                        ]
                        if not changed:
                            return base_mi
                        modules = frozenset(base_mi.imports)
                        new_imports = dict(base_mi.imports)
                        dirty: set[str] = set()
                        for p in changed:
                            mod = p.module_name
                            targets: set[str] = set()
                            for target in dict(p.import_aliases).values():
                                parts = target.split(".")
                                for length in range(1, len(parts) + 1):
                                    cand = ".".join(parts[:length])
                                    if cand in modules and cand != mod:
                                        targets.add(cand)
                            frozen = frozenset(targets)
                            if frozen != base_mi.imports[mod]:
                                new_imports[mod] = frozen
                                dirty.add(mod)
                        if not dirty:
                            return base_mi
                        graph = ModuleImportGraph(new_imports)
                        graph._reachable.update(
                            (k, v)
                            for k, v in base_mi._reachable.items()
                            if v.isdisjoint(dirty)
                        )
                        return graph
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
            field_order_read=self.field_order_reads.read_of,
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
