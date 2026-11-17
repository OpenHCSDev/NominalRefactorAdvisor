"""Where a shared block's one declaration could live: ranked owner candidates (#67).

When a move needs an owner decision, NRA does not decide it. It lists the
places the block could live, ranks them, and shows the facts behind each, so
that a picker can choose and cite those facts. The picker may be a person or
an agent. The candidates are:

- **shared base**: a scanned class both classes already descend from;
- **existing class**: a scanned class, outside both classes' families, that
  already declares part of the block identically, which both would compose;
- **existing classes**: two or more such classes, each composed by others,
  whose own declarations together cover the block. When such a set exists, a
  new mixin would be a third copy of declarations that already have owners,
  so the new mixin is refused;
- **new mixin**: a class NRA would create in one of the two classes' modules,
  or, where declaring it there closes an import cycle, in a module every
  composer already imports and that imports none of them, or else in a new
  module of their lowest common package.

**The score.** Each receipt has a fixed log-odds weight per unit, stated on its
kind below, times its count. A candidate's log-odds is the sum of its
receipts. Its confidence is the softmax of the log-odds over the candidates
no guard refuses; a refused candidate has confidence 0. The confidence is
uncalibrated: as with #70's concept score, the weights are set by hand, not
fitted, and are checked against labeled picks
(`tests/fixtures/owner_pick_labels.json`) whose Brier score and calibration
error are pinned. The score decomposes into the receipts that produced it.
Guards weigh nothing and only refuse.

**The cost** is what the move would do to the source: the statements it
rewrites, the imports it adds, and the import cycles those imports would
close. Closing a cycle is a guard.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from types import MappingProxyType
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import ClassVar, TypeVar

from nominal_refactor_advisor.audit.family import snake_case
from nominal_refactor_advisor.class_index import CompactIndexedClass, DataclassFrozenness
from nominal_refactor_advisor.json_reports import (
    SemanticRecord,
    json_report_field,
    json_report_property,
)
from nominal_refactor_advisor.receipts import (
    KeywordCuredReceipt,
    Receipt,
    ReceiptKind,
    ReorderingReceipt,
    ScoredByReceipts,
)
from nominal_refactor_advisor.models import SourceLocation
from nominal_refactor_advisor.needed_input import NeededInput


MINIMUM_SHARED_WEIGHT = 1.5
"""Declared bound on what is worth a verdict: the shared declarations' own receipts
must reach this weight (one method, a valued declaration and an annotation, or
three annotations). A lone shared field is not reported at all."""


class OwnerReceiptKind(ReceiptKind):
    """One kind of fact about a candidate owner, with its log-odds weight per unit."""

    HOLDS_DECLARATION = "holds_declaration", 0.5, False
    ANCESTOR_OF_BOTH = "ancestor_of_both", 1.0, False
    SERVES_CONSUMER = "serves_consumer", 0.25, False
    ALREADY_IMPORTED = "already_imported", 0.5, False
    IMPORT_ADDED = "import_added", -0.25, False
    OTHER_DESCENDANT = "other_descendant", -0.25, False
    ALSO_RESTATED_BY = "also_restated_by", 0.5, False
    NEW_CLASS = "new_class", -0.5, False
    BRINGS_DECLARATIONS = "brings_declarations", -0.25, False
    CYCLE_CREATED = "cycle_created", 0.0, True, NeededInput.SOURCE_CHANGE
    PLACED_BELOW = "placed_below", -0.25, False
    CONSTRUCTION_CHANGED = "construction_changed", 0.0, True, NeededInput.SOURCE_CHANGE
    FIELD_ORDER_CHANGED = "field_order_changed", 0.0, True, NeededInput.SOURCE_CHANGE
    FIELD_ORDER_READ = "field_order_read", 0.0, True, NeededInput.SOURCE_CHANGE
    KEYWORD_ONLY_COMPOSITION = "keyword_only_composition", -0.25, False
    POSITIONAL_CONSTRUCTION = "positional_construction", 0.0, True, NeededInput.SOURCE_CHANGE
    BASE_SERVES_OTHERS = "base_serves_others", 0.0, True, NeededInput.OWNER_PICK
    OWNER_NOT_CONTAINED = "owner_not_contained", 0.0, True, NeededInput.OWNER_PICK
    DIFFERENT_CONSTRUCTION = "different_construction", 0.0, True, NeededInput.SOURCE_CHANGE
    CONFLICTING_DECLARATION = "conflicting_declaration", 0.0, True, NeededInput.SOURCE_CHANGE
    COPIED_FROM = "copied_from", 0.0, True, NeededInput.OPERATION
    COPIED_HERE_ONLY = "copied_here_only", -0.25, False
    BUILT_WITHOUT_SOURCE = "built_without_source", 0.0, True, NeededInput.SOURCE_CHANGE
    COPIES_ONE_VALUE = "copies_one_value", 1.0, False
    SERVES_CALL_SITES = "serves_call_sites", 0.5, False
    NO_EXACT_SUBSTITUTION = "no_exact_substitution", 0.0, True, NeededInput.PROOF
    OWNED_BY_EXISTING_CLASSES = "owned_by_existing_classes", 0.0, True, NeededInput.OWNER_PICK
    PLAYS_SAME_ROLE = "plays_same_role", 1.0, False
    INHERITED_SHARE = "inherited_share", 0.0, True, NeededInput.OPERATION
    HOISTS_INHERITED = "hoists_inherited", -0.25, False
    ALSO_DECLARED_BY = "also_declared_by", -0.5, False
    FROZEN_DIFFERS = "frozen_differs", 0.0, True, NeededInput.SOURCE_CHANGE
    KEEPS_OWN_DECLARATION = "keeps_own_declaration", 0.0, False


SERVED_CONSUMER_CAP = 4
"""At most this many shared consumers, or other restating classes, count toward a
candidate's score."""


@dataclass(frozen=True)
class OwnerCost(SemanticRecord):
    """What choosing the candidate does to the source."""

    statements_rewritten: int
    imports_added: int
    cycles_created: int


@dataclass(frozen=True)
class OwnerCandidate(ScoredByReceipts, SemanticRecord, ABC):
    """One place the block could live, the facts for and against it, and its cost."""

    module_name: str
    receipts: tuple[Receipt, ...]
    cost: OwnerCost
    confidence: float = 0.0

    @property
    @abstractmethod
    def subject(self) -> tuple[str, ...]:
        """What this candidate is about, within its kind: a symbol, a module, or
        the symbols of a set of owners."""

    @json_report_property()
    def kind(self) -> str:
        """The candidate's class is its kind; this is its one spelling, from the
        class name without the family's `Owner` suffix."""

        return snake_case(type(self).__name__.removesuffix("Owner"))

    @json_report_property()
    def candidate_id(self) -> str:
        """The id a picks file names this candidate by."""

        return f"{self.kind}:{'+'.join(self.subject)}"

    def pick_refusals(self, pick: OwnerPick) -> Iterator[str]:
        """Why ``pick`` cannot choose this candidate; nothing when it can."""

        if not self.permitted and not self.keyword_cure:
            yield f"`{pick.owner}` needs: " + "; ".join(
                receipt.detail for receipt in self.refusals
            )
        if pick.because not in {receipt.kind.value for receipt in self.receipts}:
            yield (
                f"the pick cites `{pick.because or '(nothing)'}`, which `{pick.owner}` "
                "carries no receipt of"
            )

    @abstractmethod
    def describe(self) -> str:
        """The candidate in a few words."""

    @staticmethod
    def ranked_text(candidates: Iterable[OwnerCandidate]) -> str:
        """Ranked candidates in one line, each with its score and receipts."""

        return "; ".join(
            f"{candidate.confidence:.2f} {candidate.describe()} ("
            + ", ".join(
                f"{receipt.weight:+.2f} {receipt.kind.value}" for receipt in candidate.receipts
            )
            + ")"
            for candidate in candidates
        )


@dataclass(frozen=True, kw_only=True)
class BlockOwnerCandidate(OwnerCandidate, ABC):
    """A place for a block of class-body declarations several classes restate."""

    composers: tuple[str, ...]
    """The classes that would compose this owner: the two, and every other class
    that restates the whole block. A pick covers every verdict among them."""
    block: tuple[str, ...]
    """The block's declaration names."""

    keeps_both_copies: ClassVar[bool] = True
    """Whether both classes keep their own values of the block afterwards: true
    for every owner the two compose, false for an owner that removes a copy."""


@dataclass(frozen=True, kw_only=True)
class ScannedClassOwner(BlockOwnerCandidate, ABC):
    """An owner that is one scanned class, named by its symbol."""

    symbol: str = ""

    @property
    def subject(self) -> tuple[str, ...]:
        return (self.symbol,)


@dataclass(frozen=True, kw_only=True)
class SharedBaseOwner(ScannedClassOwner):
    """A scanned base both classes already descend from."""

    def describe(self) -> str:
        return f"shared base `{self.symbol}`"


@dataclass(frozen=True, kw_only=True)
class ExistingClassOwner(ScannedClassOwner):
    """A scanned class outside both families that already declares part of the block."""

    held: tuple[str, ...] = ()
    """The block names it already declares identically."""

    def describe(self) -> str:
        return f"existing class `{self.symbol}`"


@dataclass(frozen=True, kw_only=True)
class ComposeExistingOwner(BlockOwnerCandidate):
    """Scanned classes, each composed by others, whose own declarations cover the block.

    Both classes compose all of them and delete their copies; nothing is added
    to the owners.
    """

    symbols: tuple[str, ...] = ()
    """The owners in base order: the composers' leading fields come from the last."""

    @property
    def subject(self) -> tuple[str, ...]:
        return self.symbols

    def describe(self) -> str:
        return "existing classes " + " + ".join(f"`{symbol}`" for symbol in self.symbols)


@dataclass(frozen=True, kw_only=True)
class CopySourceOwner(BlockOwnerCandidate, ABC):
    """An owner for a class that copies the block off one value of another: the
    copy goes, so only the source keeps the block."""

    symbol: str = ""
    """The class whose values the copying class copied."""
    keeps_both_copies: ClassVar[bool] = False

    @property
    def subject(self) -> tuple[str, ...]:
        """The copying class."""
        return self.composers


@dataclass(frozen=True, kw_only=True)
class DeleteAndRedirectOwner(CopySourceOwner):
    """The copying class is deleted and its uses read the class it copies from:
    its declarations are a subset of the source's, so every read finds a field."""

    def describe(self) -> str:
        return f"delete the copy and use `{self.symbol}`"


@dataclass(frozen=True, kw_only=True)
class HoldSourceOwner(CopySourceOwner):
    """The copying class holds the instance it copied from instead of the copied
    block; its readers reach the block through it."""

    def describe(self) -> str:
        return f"hold a `{self.symbol}` instead of the copy"


@dataclass(frozen=True, kw_only=True)
class NewMixinOwner(BlockOwnerCandidate):
    """A mixin NRA would create in `module_name`, composed by both classes."""

    anchor: str
    """The composer whose declarations the mixin copies. Its module is the
    mixin's unless declaring it there closes an import cycle; then the mixin
    moves to a module every composer imports."""

    @property
    def subject(self) -> tuple[str, ...]:
        return (self.module_name,)

    def pick_refusals(self, pick: OwnerPick) -> Iterator[str]:
        yield from super().pick_refusals(pick)
        if not (pick.name or "").isidentifier():
            yield "a new mixin needs a class `name`"
        if not (pick.judgement or "").strip():
            yield (
                "a new mixin needs a `judgement`: what the class is, for its "
                "docstring"
            )

    def describe(self) -> str:
        return f"new mixin in `{self.module_name}`"


@dataclass(frozen=True, kw_only=True)
class NewModuleMixinOwner(NewMixinOwner):
    """A mixin NRA would create in a new module of package `module_name`, named
    for the mixin: no module every composer imports avoids an import cycle."""

    def describe(self) -> str:
        return f"new mixin in a new module of `{self.module_name}`"


@dataclass(frozen=True, kw_only=True)
class HoistingMixinOwner(NewMixinOwner):
    """A new mixin for the names both classes declare, after the names one of
    them inherits are hoisted out of the ancestor that declares them into a
    second new mixin, which that ancestor and the other class compose."""

    base: str
    """The ancestor that declares every inherited block name."""
    restater: str
    """The class of the two that declares those names itself."""
    hoisted: tuple[str, ...]
    """The inherited block names, in the ancestor's order."""
    hoist_composers: tuple[str, ...]
    """The classes that compose the hoisted mixin: the ancestor, the restater,
    and every other class that declares all the hoisted names itself."""

    def pick_refusals(self, pick: OwnerPick) -> Iterator[str]:
        yield from super().pick_refusals(pick)
        if not (pick.hoisted_name or "").isidentifier():
            yield f"hoisting {list(self.hoisted)} out of `{self.base}` needs a class `hoisted_name`"
        if not (pick.hoisted_judgement or "").strip():
            yield (
                f"hoisting {list(self.hoisted)} out of `{self.base}` needs a "
                "`hoisted_judgement`: what that class is, for its docstring"
            )

    def describe(self) -> str:
        return (
            f"new mixin in `{self.module_name}`, after hoisting {list(self.hoisted)} "
            f"out of `{self.base}`"
        )


@dataclass(frozen=True)
class OwnerPickOutcome(SemanticRecord):
    """What became of one extract-mixin finding under the picks."""

    finding: str
    outcome: str
    detail: str


@dataclass(frozen=True)
class OwnerPick(SemanticRecord):
    """One picker's choice of owner for one finding, with the receipt it rests on
    and, for a new mixin, the class name."""

    finding: str
    owner: str | None
    """The chosen `candidate_id`; None declines the finding."""
    because: str
    name: str | None = None
    judgement: str | None = None
    """What the owner is, in one sentence: a new mixin's docstring."""
    hoisted_name: str | None = None
    """For a hoisting owner, the class the inherited names are hoisted into."""
    hoisted_judgement: str | None = None
    """What that hoisted class is, in one sentence: its docstring."""

    def resolved(self, candidates: Sequence[OwnerCandidate]) -> OwnerPickResolution:
        """This pick against one finding's candidates: the one place a picks
        file's `owner` string is matched; everything after holds the object."""

        if self.owner is None:
            return DeclinedPick(self.because)
        candidate = next(
            (owner for owner in candidates if owner.candidate_id == self.owner), None
        )
        if candidate is None:
            return RejectedPick(f"`{self.owner}` is not a candidate of this finding")
        for refusal in candidate.pick_refusals(self):
            return RejectedPick(refusal)
        return PickedOwner(
            finding=self.finding,
            owner=self.owner,
            because=self.because,
            name=self.name,
            judgement=self.judgement,
            hoisted_name=self.hoisted_name,
            hoisted_judgement=self.hoisted_judgement,
            candidate=candidate,
        )


class OwnerPickResolution(ABC):
    """What one pick comes to against one finding's candidates."""

    @abstractmethod
    def outcome(self, finding: str) -> OwnerPickOutcome:
        """What this resolution reports for ``finding``."""

    def picked(self) -> tuple[PickedOwner, ...]:
        """The owner this resolution chooses, if it chooses one."""

        return ()


@dataclass(frozen=True)
class DeclinedPick(OwnerPickResolution):
    """A pick with no owner: the picker declines the finding."""

    because: str

    def outcome(self, finding: str) -> OwnerPickOutcome:
        return OwnerPickOutcome(finding, "declined", self.because)


@dataclass(frozen=True)
class RejectedPick(OwnerPickResolution):
    """A pick naming no candidate, one that needs more input, or one it cannot cite."""

    reason: str

    def outcome(self, finding: str) -> OwnerPickOutcome:
        return OwnerPickOutcome(finding, "rejected", self.reason)


@dataclass(frozen=True, kw_only=True)
class PickedOwner(OwnerPick, OwnerPickResolution):
    """A valid pick, holding the candidate it chose."""

    candidate: OwnerCandidate = json_report_field(included=False)

    def outcome(self, finding: str) -> OwnerPickOutcome:
        return OwnerPickOutcome(
            finding, "picked", f"{self.owner} because {self.because}"
        )

    def picked(self) -> tuple[PickedOwner, ...]:
        return (self,)


@dataclass(frozen=True)
class ModuleImportGraph:
    """Which scanned modules each scanned module imports."""

    imports: Mapping[str, frozenset[str]]
    _reachable: dict[str, frozenset[str]] = field(
        default_factory=dict, compare=False, repr=False
    )

    @classmethod
    def from_aliases(
        cls, aliases_by_module: Mapping[str, Iterable[str]]
    ) -> ModuleImportGraph:
        """Importing `a.b.c` runs `a/__init__` and `a/b/__init__` first, so each
        scanned package on the way is imported too, as the codemod's own graph
        counts it."""

        modules = frozenset(aliases_by_module)

        def scanned_modules(target: str) -> Iterator[str]:
            parts = target.split(".")
            for length in range(1, len(parts) + 1):
                candidate = ".".join(parts[:length])
                if candidate in modules:
                    yield candidate

        return cls(
            {
                module: frozenset(
                    imported
                    for target in targets
                    for imported in scanned_modules(target)
                    if imported != module
                )
                for module, targets in aliases_by_module.items()
            }
        )

    def imports_directly(self, importer: str, imported: str) -> bool:
        return importer == imported or imported in self.imports.get(importer, ())

    def reaches(self, start: str, goal: str) -> bool:
        return goal in self.reachable(start)

    def reachable(self, start: str) -> frozenset[str]:
        """Every module loading `start` loads, `start` included."""

        reachable = self._reachable.get(start)
        if reachable is None:
            seen = {start}
            pending = [start]
            while pending:
                for imported in self.imports.get(pending.pop(), ()):
                    if imported not in seen:
                        seen.add(imported)
                        pending.append(imported)
            reachable = self._reachable[start] = frozenset(seen)
        return reachable

    def module_of(self, qualified_name: str) -> str | None:
        """The scanned module declaring `qualified_name`: its longest scanned prefix."""

        parts = qualified_name.split(".")
        return next(
            (
                ".".join(parts[:length])
                for length in range(len(parts) - 1, 0, -1)
                if ".".join(parts[:length]) in self.imports
            ),
            None,
        )

    def can_bind_at(self, home: str, read: str, users: Iterable[str]) -> bool:
        """`home` can import a name from `read` with no cycle: `read` is `home`,
        or it neither reaches `home` nor is or reaches a module importing it."""

        return read == home or not (
            self.reaches(read, home)
            or any(read == user or self.reaches(read, user) for user in users if user != home)
        )

    def shared_lower_module(
        self,
        users: Iterable[str],
        reads: Iterable[str] = (),
        *,
        homes_among_users: Iterable[str] = (),
        above: Iterable[str] = (),
        modes: Mapping[str, object] = MappingProxyType({}),
        source: str = "",
        layer: frozenset[str] = frozenset(),
    ) -> LowerHome | None:
        """The module a declaration all of `users` import can live in without
        closing a cycle: one each of them already imports, directly or not, that
        imports none of them, in their lowest common package, and no package's
        `__init__`, which is the package's interface rather than a home. A user in
        `homes_among_users` may hold it too. A declaration moving out of `source`
        stays beside it: its home is a user that may hold it or a module of
        `source`'s package, never a module that is merely low enough. Every module in `reads`, which the
        moved code imports names from, must be importable there without a cycle,
        and neither the home nor any module in `reads` is or reaches a module in
        `above`, which will import it but need not yet.

        Of several, one of `layer`, the modules a composition root's service the
        declaration belongs to loads, first; then one keeping `source`'s
        annotation mode (`modes`), then one no user is, then the highest, the one
        no other such module imports, then by name; None when there is none."""

        importers = frozenset(users)
        allowed = frozenset(homes_among_users) & importers
        read_modules = frozenset(reads)
        readers = frozenset(above)
        if any(read in readers or self.reaches(read, reader) for read in read_modules for reader in readers):
            return None
        package = common_package(importers)
        within = package + "."
        qualifying = tuple(
            module
            for module in sorted(self.imports)
            if (module not in importers or module in allowed)
            and module.startswith(within)
            and (not source or module in allowed or common_package((module,)) == common_package((source,)))
            and not any(other.startswith(module + ".") for other in self.imports)
            and all(self.reaches(importer, module) for importer in importers if importer != module)
            and not any(self.reaches(module, importer) for importer in importers if importer != module)
            and all(self.can_bind_at(module, read, importers) for read in read_modules)
            and module not in readers
            and not any(self.reaches(module, reader) for reader in readers)
        )
        ranked = sorted(
            qualifying,
            key=lambda module: (
                bool(layer) and module not in layer,
                modes.get(module) != modes.get(source),
                module in importers,
                any(other != module and self.reaches(other, module) for other in qualifying),
                module,
            ),
        )
        return (
            LowerHome(ranked[0], modes.get(ranked[0]) == modes.get(source))
            if ranked
            else None
        )

    def pinning_reads(self, users: Iterable[str], reads: Iterable[str]) -> tuple[str, ...]:
        """The read modules no home below `users` can import without a cycle:
        each is a user, or reaches one."""

        importers = frozenset(users)
        return tuple(
            sorted(
                read
                for read in frozenset(reads)
                if read in importers or any(self.reaches(read, user) for user in importers)
            )
        )


@dataclass(frozen=True)
class LowerHome:
    """Where a moved declaration goes to avoid a cycle, and whether that module
    evaluates annotations as its source module does."""

    module: str
    keeps_annotation_mode: bool

def common_package(modules: Iterable[str]) -> str:
    """The deepest package that holds every one of `modules`; empty when they
    share none."""

    packages = [module.split(".")[:-1] for module in modules]
    if not packages:
        return ""
    shared = packages[0]
    for parts in packages[1:]:
        length = 0
        while length < min(len(shared), len(parts)) and shared[length] == parts[length]:
            length += 1
        shared = shared[:length]
    return ".".join(shared)


@dataclass(frozen=True)
class BlockOwnerQuestion:
    """Where should the block two classes share live?

    `holders_of` gives, for each block declaration, the scanned classes that
    declare it identically, own or inherited; `own_holders_of` only those that
    declare it themselves. An existing class is a candidate only when another
    class already composes it: a peer that merely restates the block is not its
    owner, and counts for a new mixin instead. `declares_differently` says
    whether a class declares a block name with a different answer.
    """

    first: CompactIndexedClass
    second: CompactIndexedClass
    block: tuple[str, ...]
    weights: Mapping[str, float]
    """Each block declaration's own receipt weight."""
    copies_of: Callable[[str], tuple[tuple[SourceLocation, tuple[str, ...]], ...]]
    """Constructions of a class with the fields each copies as `x.<same name>`."""
    holders_of: Callable[[str], frozenset[str]]
    own_holders_of: Callable[[str], frozenset[str]]
    declared_names: Callable[[str], frozenset[str]]
    required_fields: frozenset[str]
    block_fields: frozenset[str]
    """The block's field declarations: what a construction can copy."""
    """Block names that are dataclass fields without a default."""
    constructions_of: Callable[[str], tuple[SourceLocation, ...]]
    positional_constructions_of: Callable[[str], tuple[SourceLocation, ...]]
    same_construction: Callable[[CompactIndexedClass, CompactIndexedClass], bool]
    declares_differently: Callable[[str, str], bool]
    class_for: Callable[[str], CompactIndexedClass | None]
    ancestors_of: Callable[[str], tuple[str, ...]]
    descendants_of: Callable[[str], tuple[str, ...]]
    consumers_of: Callable[[CompactIndexedClass], frozenset[tuple[str, str]]]
    use_locations: Callable[[CompactIndexedClass], Mapping[tuple[str, str], SourceLocation]]
    imports: ModuleImportGraph
    field_order_read: Callable[[Iterable[str]], str | None]
    """Where class code reads the field order of one of these classes or a
    descendant (`FieldOrderReads.read_of`); none when nothing does."""
    EXISTING_CLASS_LIMIT: ClassVar[int] = 3
    SHARED_BASE_LIMIT: ClassVar[int] = 3
    COVERING_SET_LIMIT: ClassVar[int] = 3
    """At most this many existing classes make one covering set."""

    def candidates(self) -> tuple[OwnerCandidate, ...]:
        covering = self._covering_owners()
        composed = (self._covering_candidate(covering),) if covering else ()
        found: tuple[BlockOwnerCandidate, ...] = (
            *self._shared_bases(),
            *self._existing_classes(),
            *composed,
            *self._new_mixins(covering, any(owner.permitted for owner in composed)),
        )
        copy = self._copy()
        if copy is None:
            return ranked(found)
        copier, source, locations = copy
        copy_owner = self._copy_owner(copier, source, locations)
        refusal = (
            Receipt(
                OwnerReceiptKind.COPIED_FROM,
                f"`{copier.qualname}` is built from the block copied field by field off "
                f"one value: an owner both compose keeps both copies",
                locations[:4],
                count=len(locations),
            )
            if copy_owner.permitted
            else Receipt(
                OwnerReceiptKind.COPIED_HERE_ONLY,
                f"`{copier.qualname}` is built from the block copied off one value here, "
                "and from other values elsewhere: both keep their copy of the block",
                locations[:4],
                count=len(locations),
            )
        )
        return ranked(
            (
                *(
                    replace(candidate, receipts=(*candidate.receipts, refusal))
                    if candidate.keeps_both_copies
                    else candidate
                    for candidate in found
                ),
                copy_owner,
            )
        )

    def _copy(
        self,
    ) -> tuple[CompactIndexedClass, CompactIndexedClass, tuple[SourceLocation, ...]] | None:
        """The class of the two built by copying the whole block off one value,
        the other class, and where it is built so."""

        if not self.block_fields:
            return None
        for copier, source in (self.pair, self.pair[::-1]):
            locations = tuple(
                location
                for location, fields in self.copies_of(copier.symbol)
                if self.block_fields <= frozenset(fields)
            )
            if locations:
                return copier, source, locations
        return None

    def _copy_owner(
        self,
        copier: CompactIndexedClass,
        source: CompactIndexedClass,
        locations: tuple[SourceLocation, ...],
    ) -> BlockOwnerCandidate:
        """Delete the copy when it declares nothing the source lacks; otherwise
        hold the source instead of the copied block."""

        receipt = Receipt(
            OwnerReceiptKind.COPIES_ONE_VALUE,
            f"`{copier.qualname}` copies the block off one value field by field",
            locations[:4],
            count=len(locations),
        )
        cost = OwnerCost(
            statements_rewritten=len(self.block) + len(locations),
            imports_added=0,
            cycles_created=0,
        )
        if self.declared_names(copier.symbol) <= self.declared_names(source.symbol):
            return DeleteAndRedirectOwner(
                module_name=source.module_name,
                receipts=(receipt,),
                cost=cost,
                composers=(copier.symbol,),
                block=self.block,
                symbol=source.symbol,
            )
        copied = frozenset((location.file_path, location.line) for location in locations)
        elsewhere = tuple(
            location
            for location in self.constructions_of(copier.symbol)
            if (location.file_path, location.line) not in copied
        )
        return HoldSourceOwner(
            module_name=copier.module_name,
            receipts=(
                receipt,
                *(
                    (
                        Receipt(
                            OwnerReceiptKind.BUILT_WITHOUT_SOURCE,
                            f"`{copier.qualname}` is also built from other values: there is "
                            f"no `{source.qualname}` to hold there",
                            elsewhere[:4],
                            count=len(elsewhere),
                        ),
                    )
                    if elsewhere
                    else ()
                ),
            ),
            cost=cost,
            composers=(copier.symbol,),
            block=self.block,
            symbol=source.symbol,
        )

    @property
    def pair(self) -> tuple[CompactIndexedClass, CompactIndexedClass]:
        return (self.first, self.second)

    def _family(self) -> frozenset[str]:
        return frozenset(
            symbol
            for indexed_class in self.pair
            for symbol in (
                indexed_class.symbol,
                *self.ancestors_of(indexed_class.symbol),
                *self.descendants_of(indexed_class.symbol),
            )
        )

    def _shared_bases(self) -> Iterator[SharedBaseOwner]:
        common = [
            symbol
            for symbol in self.ancestors_of(self.first.symbol)
            if symbol in set(self.ancestors_of(self.second.symbol))
        ]
        for symbol in common[: self.SHARED_BASE_LIMIT]:
            base = self.class_for(symbol)
            if base is None:
                continue
            below = (
                set(self.descendants_of(symbol))
                - {self.first.symbol, self.second.symbol}
                - set(self.descendants_of(self.first.symbol))
                - set(self.descendants_of(self.second.symbol))
            )
            restaters = sorted(below & set(self._co_restaters()))
            others = sorted(below - set(restaters))
            held = self._held(symbol)
            receipts = [
                Receipt(
                    OwnerReceiptKind.ANCESTOR_OF_BOTH,
                    f"`{base.qualname}` is an ancestor of both",
                    (_location(base),),
                ),
                *self._holding_receipts(base, held),
                *self._consumer_receipts(base),
                *self._conflict_receipts(base),
                *self._restater_receipts(restaters),
                *self._construction_receipts(
                    base,
                    tuple(name for name in self.block if name not in held),
                    (symbol, *others),
                ),
            ]
            if others:
                receipts.append(
                    Receipt(
                        OwnerReceiptKind.BASE_SERVES_OTHERS,
                        f"`{base.qualname}` has descendants that do not restate the "
                        "block; a fold is proved only for a base whose every "
                        "descendant restates it",
                        (_location(base),),
                    )
                )
                receipts.append(
                    Receipt(
                        OwnerReceiptKind.OTHER_DESCENDANT,
                        f"{len(others)} other descendant(s) would gain the block: "
                        + ", ".join(f"`{other}`" for other in others[:4]),
                        tuple(
                            _location(indexed_class)
                            for other in others[:4]
                            if (indexed_class := self.class_for(other)) is not None
                        ),
                        count=len(others),
                    )
                )
            yield SharedBaseOwner(
                module_name=base.module_name,
                receipts=tuple(receipts),
                cost=OwnerCost(
                    statements_rewritten=len(self.block)
                    - len(held)
                    + (2 + len(restaters)) * len(self.block),
                    imports_added=0,
                    cycles_created=0,
                ),
                composers=self._composers(),
                block=self.block,
                symbol=symbol,
            )

    def _composers(self) -> tuple[str, ...]:
        return (self.first.symbol, self.second.symbol, *self._co_restaters())

    def _composer_modules(self) -> tuple[str, ...]:
        """Every composer's module, each of which imports the owner."""

        return tuple(
            dict.fromkeys(
                indexed_class.module_name
                for symbol in self._composers()
                if (indexed_class := self.class_for(symbol)) is not None
            )
        )

    def _held(self, symbol: str) -> tuple[str, ...]:
        return tuple(name for name in self.block if symbol in self.holders_of(name))

    def _existing_classes(self) -> Iterator[ExistingClassOwner]:
        family = self._family()
        counts: dict[str, int] = {}
        for name in self.block:
            for symbol in self.holders_of(name) - family:
                if self.descendants_of(symbol):
                    counts[symbol] = counts.get(symbol, 0) + 1
        chosen = sorted(counts, key=lambda symbol: (-counts[symbol], symbol))
        for symbol in chosen[: self.EXISTING_CLASS_LIMIT]:
            owner = self.class_for(symbol)
            if owner is None:
                continue
            held = self._held(symbol)
            receipts = [
                *self._holding_receipts(owner, held),
                *self._widened_receipts(owner, held),
                *self._brought_receipts(owner),
                *self._consumer_receipts(owner),
                *self._conflict_receipts(owner),
                *self._construction_receipts(
                    owner,
                    tuple(name for name in self.block if name not in held),
                    (
                        symbol,
                        *(
                            descendant
                            for descendant in self.descendants_of(symbol)
                            if descendant not in (self.first.symbol, self.second.symbol)
                        ),
                    ),
                ),
            ]
            receipts.extend(
                self._field_order_receipts(
                    owner,
                    (
                        *(owner.own_dataclass_field_names or ()),
                        *(
                            name
                            for name in self.first.own_dataclass_field_names or ()
                            if name in self.block and name not in held
                        ),
                    ),
                    keyword_only_route=False,
                    composers=self._composers(),
                )
            )
            if not self.same_construction(owner, self.first):
                receipts.append(
                    Receipt(
                        OwnerReceiptKind.DIFFERENT_CONSTRUCTION,
                        f"`{owner.qualname}` is created differently from "
                        f"`{self.first.qualname}` (dataclass declaration, metaclass "
                        "or class keywords): composing it changes construction",
                        (_location(owner), _location(self.first)),
                    )
                )
            imports, cycles, import_receipts = self._import_receipts(
                owner.module_name, (owner,)
            )
            yield ExistingClassOwner(
                module_name=owner.module_name,
                receipts=(*receipts, *import_receipts),
                cost=OwnerCost(
                    statements_rewritten=len(self.block) - len(held)
                    + 2
                    + 2 * len(self.block),
                    imports_added=imports,
                    cycles_created=cycles,
                ),
                composers=self._composers(),
                block=self.block,
                symbol=symbol,
                held=held,
            )

    def _covering_owners(self) -> tuple[CompactIndexedClass, ...]:
        """The fewest existing classes whose own declarations cover the block,
        or none.

        Each is outside both classes' families, composed by another class, and
        declares at least one block name itself; none covers the block alone
        (that is an existing-class owner). Between sets of one size, the one
        bringing the fewest other declarations wins, then the one more classes
        already compose, then the symbols.
        """

        family = self._family()
        owned: dict[str, frozenset[str]] = {}
        for name in self.block:
            for symbol in self.own_holders_of(name) - family:
                if self.descendants_of(symbol) and self.class_for(symbol) is not None:
                    owned[symbol] = owned.get(symbol, frozenset()) | {name}
        block = frozenset(self.block)
        members = sorted(symbol for symbol, names in owned.items() if names != block)
        for size in range(2, min(self.COVERING_SET_LIMIT, len(members)) + 1):
            covers = [
                combination
                for combination in combinations(members, size)
                if frozenset().union(*(owned[symbol] for symbol in combination)) == block
                and all(
                    other not in self.ancestors_of(symbol)
                    for symbol in combination
                    for other in combination
                )
            ]
            if covers:
                best = min(
                    covers,
                    key=lambda combination: (
                        len(self._brought_names(combination)),
                        -sum(len(self.descendants_of(symbol)) for symbol in combination),
                        combination,
                    ),
                )
                return tuple(
                    indexed_class
                    for symbol in self._base_order(best)
                    if (indexed_class := self.class_for(symbol)) is not None
                )
        return ()

    def _brought_names(self, symbols: Iterable[str]) -> frozenset[str]:
        return (
            frozenset().union(*(self.declared_names(symbol) for symbol in symbols))
            - set(self.block)
            - self.declared_names(self.first.symbol)
            - self.declared_names(self.second.symbol)
        )

    def _base_order(self, symbols: Sequence[str]) -> tuple[str, ...]:
        """Bases whose dataclass collection yields the block in `first`'s order.

        Collection walks the MRO in reverse, so the base holding the first
        leading field goes last.
        """

        fields = tuple(self.first.own_dataclass_field_names or ())

        def leading(symbol: str) -> int:
            positions = [
                fields.index(name)
                for name in self.block
                if name in fields and symbol in self.own_holders_of(name)
            ]
            return min(positions, default=len(fields))

        return tuple(sorted(symbols, key=lambda symbol: (-leading(symbol), symbol)))

    def _covering_candidate(
        self, owners: tuple[CompactIndexedClass, ...]
    ) -> ComposeExistingOwner:
        receipts: list[Receipt] = []
        imports_added = cycles_created = 0
        for owner in owners:
            held = tuple(
                name for name in self.block if owner.symbol in self.own_holders_of(name)
            )
            imports, cycles, import_receipts = self._import_receipts(
                owner.module_name, (owner,)
            )
            imports_added += imports
            cycles_created += cycles
            receipts.extend(
                (
                    *self._holding_receipts(owner, held),
                    *self._brought_receipts(owner),
                    *self._consumer_receipts(owner),
                    *self._conflict_receipts(owner),
                    *import_receipts,
                )
            )
            if not self.same_construction(owner, self.first):
                receipts.append(
                    Receipt(
                        OwnerReceiptKind.DIFFERENT_CONSTRUCTION,
                        f"`{owner.qualname}` is created differently from "
                        f"`{self.first.qualname}` (dataclass declaration, metaclass "
                        "or class keywords): composing it changes construction",
                        (_location(owner), _location(self.first)),
                    )
                )
        receipts.extend(
            self._field_order_receipts(
                owners[-1],
                tuple(
                    name
                    for owner in reversed(owners)
                    for name in owner.own_dataclass_field_names or ()
                ),
                keyword_only_route=False,
                composers=self._composers(),
            )
        )
        return ComposeExistingOwner(
            module_name=owners[0].module_name,
            receipts=tuple(receipts),
            cost=OwnerCost(
                statements_rewritten=(2 + len(self._co_restaters()))
                * (len(owners) + len(self.block)),
                imports_added=imports_added,
                cycles_created=cycles_created,
            ),
            composers=self._composers(),
            block=self.block,
            symbols=tuple(owner.symbol for owner in owners),
        )

    def _new_mixins(
        self, covering: tuple[CompactIndexedClass, ...], covering_composes: bool
    ) -> Iterator[NewMixinOwner]:
        declared = tuple(
            name
            for name in self.block
            if {self.first.symbol, self.second.symbol} <= self.own_holders_of(name)
        )
        inherited_share = (
            (
                Receipt(
                    OwnerReceiptKind.INHERITED_SHARE,
                    "a new mixin takes only what both classes declare themselves ("
                    + (", ".join(f"`{name}`" for name in declared) or "nothing")
                    + "), too little to compose: the rest is inherited, so the move "
                    "needs an operation that composes the inherited owner",
                    (),
                ),
            )
            if len(declared) < len(self.block)
            and sum(self.weights[name] for name in declared) < MINIMUM_SHARED_WEIGHT
            else ()
        )
        owned_elsewhere = (
            (
                Receipt(
                    OwnerReceiptKind.OWNED_BY_EXISTING_CLASSES,
                    "the block is already declared by "
                    + " + ".join(f"`{owner.qualname}`" for owner in covering)
                    + ", which other classes compose: a new mixin would be another copy",
                    tuple(_location(owner) for owner in covering),
                )
                if covering_composes
                else Receipt(
                    OwnerReceiptKind.ALSO_DECLARED_BY,
                    "the block is also declared by "
                    + " + ".join(f"`{owner.qualname}`" for owner in covering)
                    + ", which both classes cannot compose: identical declarations "
                    "there are evidence of one concept, not its owner",
                    tuple(_location(owner) for owner in covering),
                ),
            )
            if covering
            else ()
        )
        placed: set[str] = set()
        for anchor_module in dict.fromkeys(
            indexed_class.module_name for indexed_class in self.pair
        ):
            module_name, kind, imports, cycles, import_receipts = self._placement(
                anchor_module
            )
            if module_name in placed:
                continue
            placed.add(module_name)
            restaters = self._co_restaters()
            anchor = next(
                indexed_class
                for symbol in self._composers()
                if (indexed_class := self.class_for(symbol)) is not None
                and indexed_class.module_name == anchor_module
            )
            order_receipts = self._field_order_receipts(
                anchor,
                tuple(
                    name
                    for name in anchor.own_dataclass_field_names or ()
                    if name in self.block
                ),
                keyword_only_route=True,
                composers=self._composers(),
            )
            receipts = (
                Receipt(
                    OwnerReceiptKind.NEW_CLASS,
                    (
                        f"a new class in a new module of `{module_name}`"
                        if kind is NewModuleMixinOwner
                        else f"a new class in `{module_name}`"
                    )
                    + f" declares the {len(self.block)}-declaration block, "
                    f"copied from `{anchor.qualname}`",
                    (),
                ),
                *self._restater_receipts(restaters),
                *self._frozenness_receipts(
                    (self.first.symbol, self.second.symbol), self._whole_block_restaters(), self.block
                ),
                *import_receipts,
                *order_receipts,
                *owned_elsewhere,
            )
            cost = OwnerCost(
                statements_rewritten=1
                + len(self.block)
                + (2 + len(restaters)) * (1 + len(self.block)),
                imports_added=imports,
                cycles_created=cycles,
            )
            hoists = self._hoists(declared) if inherited_share else ()
            for base, restater, hoisted in hoists:
                yield HoistingMixinOwner(
                    module_name=module_name,
                    anchor=anchor.symbol,
                    receipts=(
                        *receipts,
                        *self._declared_part(declared).refusals_of_new_mixin(module_name),
                        Receipt(
                            OwnerReceiptKind.HOISTS_INHERITED,
                            f"`{base.qualname}` declares the inherited "
                            + ", ".join(f"`{name}`" for name in hoisted)
                            + f": they move into a new class `{base.qualname}` and "
                            f"`{restater.qualname}` compose",
                            (_location(base),),
                        ),
                        *self._frozenness_receipts(
                            (base.symbol, restater.symbol),
                            self._hoist_restaters((base.symbol, restater.symbol), hoisted),
                            hoisted,
                        ),
                        *self._field_order_receipts(
                            base, hoisted, keyword_only_route=True,
                            composers=self._hoist_composers(base, restater, hoisted),
                        ),
                    ),
                    cost=replace(
                        cost,
                        statements_rewritten=cost.statements_rewritten + 3 * (1 + len(hoisted)),
                    ),
                    composers=self._composers(),
                    block=self.block,
                    base=base.symbol,
                    restater=restater.symbol,
                    hoisted=hoisted,
                    hoist_composers=self._hoist_composers(base, restater, hoisted),
                )
            if not hoists:
                yield kind(
                    module_name=module_name,
                    anchor=anchor.symbol,
                    receipts=(*receipts, *inherited_share),
                    cost=cost,
                    composers=self._composers(),
                    block=self.block,
                )

    def _declared_part(self, declared: Sequence[str]) -> BlockOwnerQuestion:
        """The same question about the part of the block both classes declare: the
        block a hoist leaves for the new mixin."""

        kept = frozenset(declared)
        return replace(
            self,
            block=tuple(declared),
            block_fields=self.block_fields & kept,
            required_fields=self.required_fields & kept,
        )

    def refusals_of_new_mixin(self, module_name: str) -> tuple[Receipt, ...]:
        """What refuses a new mixin of this block in `module_name`."""

        return tuple(
            receipt
            for candidate in self.candidates()
            if type(candidate) is NewMixinOwner and candidate.module_name == module_name
            for receipt in candidate.refusals
        )

    def _hoists(
        self, declared: Sequence[str]
    ) -> tuple[tuple[CompactIndexedClass, CompactIndexedClass, tuple[str, ...]], ...]:
        """The one ancestor to hoist the inherited block names out of, if there is one.

        One of the two classes declares the whole block; the other declares only
        `declared` and inherits the rest from its nearest ancestor that declares
        all of it, which the first class does not descend from. Hoisting those
        names into a mixin that ancestor and the first class compose leaves the
        rest for the new mixin both classes compose.
        """

        inherited = tuple(name for name in self.block if name not in declared)
        if not declared or not inherited:
            return ()
        for heir, restater in (self.pair, self.pair[::-1]):
            if not all(restater.symbol in self.own_holders_of(name) for name in self.block):
                continue
            for symbol in self.ancestors_of(heir.symbol):
                base = self.class_for(symbol)
                if base is None or not all(symbol in self.own_holders_of(name) for name in inherited):
                    continue
                if symbol in self.ancestors_of(restater.symbol):
                    return ()
                ordered = tuple(
                    name for name in base.own_dataclass_field_names or () if name in inherited
                )
                return ((base, restater, (*ordered, *(name for name in inherited if name not in ordered))),)
        return ()

    def _field_order_receipts(
        self,
        owner: CompactIndexedClass,
        owner_fields: Sequence[str],
        *,
        keyword_only_route: bool,
        composers: Sequence[str],
    ) -> tuple[Receipt, ...]:
        """Composing an owner puts its fields ahead of each composer's own.

        A dataclass composer keeps its field order only when the owner's fields
        are its leading own fields, in the same order. Otherwise its generated
        `__init__` takes them in another order. A new mixin is created
        keyword-only, and each reordered composer becomes keyword-only too,
        so a call that binds by keyword binds the same values: that costs a
        receipt, and a construction of a reordered composer or a descendant
        that binds by position refuses, as does class code reading a reordered
        composer's field order (`field_order_read`). The re-proof also refuses
        `order=True`, an own `__init__` or `__match_args__`, positional class
        patterns and the class handed on as a value. An existing owner would have to become
        keyword-only for its other users too, which is not proved here, so a
        reorder refuses it. A field whose position is unknown leaves the order
        unproved.
        """

        moved = frozenset(owner_fields)
        reordered: list[str] = []
        unknown: list[str] = []
        for symbol in composers:
            composer = self.class_for(symbol)
            if composer is None:
                continue
            own = composer.own_dataclass_field_names
            if own is None:
                unknown.append(composer.qualname)
                continue
            kept = [name for name in own if name in moved]
            if tuple(own[: len(kept)]) != tuple(name for name in owner_fields if name in kept):
                reordered.append(symbol)
        reordered_names = [
            indexed_class.qualname
            for symbol in reordered
            if (indexed_class := self.class_for(symbol)) is not None
        ]
        receipts = []
        if reordered and keyword_only_route:
            receipts.append(
                ReorderingReceipt(
                    OwnerReceiptKind.KEYWORD_ONLY_COMPOSITION,
                    f"composing `{owner.qualname}` reorders the dataclass fields of "
                    + ", ".join(f"`{name}`" for name in reordered_names)
                    + ", so the owner and they become keyword-only; their repr "
                    "and field order change",
                    (_location(owner),),
                    count=len(reordered),
                    reordered=tuple(reordered),
                )
            )
            # A class whose code reads its field order (a table deriving its columns)
            # builds something else once reordered, by keyword or not.
            read = self.field_order_read(reordered)
            if read is not None:
                receipts.append(
                    Receipt(OwnerReceiptKind.FIELD_ORDER_READ, read, (_location(owner),))
                )
            # A keyword-only owner makes its fields keyword-only in every composer,
            # reordered or not, so a positional construction of any of them breaks.
            constructed = {
                member: locations
                for symbol in composers
                for member in (symbol, *self.descendants_of(symbol))
                if (locations := self.positional_constructions_of(member))
            }
            positional = tuple(location for locations in constructed.values() for location in locations)
            if positional:
                receipts.append(
                    KeywordCuredReceipt(
                        OwnerReceiptKind.POSITIONAL_CONSTRUCTION,
                        f"{len(positional)} construction(s) of a reordered composer or "
                        "a descendant bind fields by position",
                        positional[:4],
                        count=len(positional),
                        constructed=tuple(constructed),
                    )
                )
        elif reordered:
            receipts.append(
                Receipt(
                    OwnerReceiptKind.FIELD_ORDER_CHANGED,
                    f"composing `{owner.qualname}` reorders the dataclass fields of "
                    + ", ".join(f"`{name}`" for name in reordered_names)
                    + ": the block is not their leading fields in the owner's order",
                    (_location(owner),),
                    count=len(reordered),
                )
            )
        if unknown:
            receipts.append(
                Receipt(
                    OwnerReceiptKind.FIELD_ORDER_CHANGED,
                    "the field order of "
                    + ", ".join(f"`{name}`" for name in unknown)
                    + " is unknown, so composing leaves it unproved",
                    (_location(owner),),
                    count=len(unknown),
                )
            )
        return tuple(receipts)

    def _construction_receipts(
        self,
        owner: CompactIndexedClass,
        added: Sequence[str],
        affected: Sequence[str],
    ) -> tuple[Receipt, ...]:
        """Adding declarations to a class others build changes how they are built.

        A field with no default becomes a required argument of every affected
        class, so any place that constructs one breaks; and the added fields
        come after the owner's own, so any positional construction of an
        affected class binds other parameters. Either refuses until the
        operation proves otherwise. Only a dataclass owner's construction
        changes; a plain class's annotations are not parameters.
        """

        if not added or not owner.dataclass_declaration.generates_construction:
            return ()
        receipts: list[Receipt] = []
        required = [name for name in added if name in self.required_fields]
        built = tuple(
            location for symbol in affected for location in self.constructions_of(symbol)
        )
        if required and built:
            receipts.append(
                Receipt(
                    OwnerReceiptKind.CONSTRUCTION_CHANGED,
                    f"adding required field(s) {', '.join(f'`{name}`' for name in required)} "
                    f"to `{owner.qualname}` breaks {len(built)} construction(s) of it "
                    "or its descendants",
                    built[:4],
                    count=len(built),
                )
            )
        constructed = {
            symbol: locations
            for symbol in affected
            if (locations := self.positional_constructions_of(symbol))
        }
        positional = tuple(location for locations in constructed.values() for location in locations)
        if positional:
            receipts.append(
                KeywordCuredReceipt(
                    OwnerReceiptKind.CONSTRUCTION_CHANGED,
                    f"{len(positional)} construction(s) of `{owner.qualname}` or its "
                    "descendants bind fields by position; added fields would shift them",
                    positional[:4],
                    count=len(positional),
                    constructed=tuple(constructed),
                )
            )
        return tuple(receipts)

    def _widened_receipts(
        self, owner: CompactIndexedClass, held: Sequence[str]
    ) -> tuple[Receipt, ...]:
        """Classes that already compose the owner gain the declarations it lacks."""

        missing = [name for name in self.block if name not in held]
        composers = sorted(
            set(self.descendants_of(owner.symbol))
            - {self.first.symbol, self.second.symbol}
        )
        if not missing or not composers:
            return ()
        return (
            Receipt(
                OwnerReceiptKind.OTHER_DESCENDANT,
                f"adding {', '.join(f'`{name}`' for name in missing)} to "
                f"`{owner.qualname}` gives its {len(composers)} other descendant(s) "
                "those declarations: "
                + ", ".join(f"`{symbol}`" for symbol in composers[:4]),
                tuple(
                    _location(indexed_class)
                    for symbol in composers[:4]
                    if (indexed_class := self.class_for(symbol)) is not None
                ),
                count=len(composers),
            ),
        )

    def _brought_receipts(self, owner: CompactIndexedClass) -> tuple[Receipt, ...]:
        """Declarations composing the owner would give both classes that neither makes."""

        brought = sorted(self._brought_names((owner.symbol,)))
        if not brought:
            return ()
        return (
            Receipt(
                OwnerReceiptKind.BRINGS_DECLARATIONS,
                f"composing `{owner.qualname}` gives both classes {len(brought)} "
                "declaration(s) neither makes: "
                + ", ".join(f"`{name}`" for name in brought[:6]),
                (_location(owner),),
                count=len(brought),
            ),
            Receipt(
                OwnerReceiptKind.OWNER_NOT_CONTAINED,
                f"`{owner.qualname}` declares more than the block; composition is "
                "proved only for an owner whose whole body each composer restates",
                (_location(owner),),
            ),
        )

    def _hoist_composers(
        self,
        base: CompactIndexedClass,
        restater: CompactIndexedClass,
        hoisted: Sequence[str],
    ) -> tuple[str, ...]:
        """The ancestor and the restater, then every other class declaring all
        the hoisted names itself that can compose the same dataclass mixin: a
        hoisted mixin is composed like any new mixin, by each class that
        restates its block."""

        required = (base.symbol, restater.symbol)
        kept, _ = self._frozenness_split(required, self._hoist_restaters(required, hoisted), hoisted)
        return (*required, *kept)

    def _hoist_restaters(self, required: Sequence[str], hoisted: Sequence[str]) -> tuple[str, ...]:
        whole = frozenset.intersection(*(self.own_holders_of(name) for name in hoisted))
        return tuple(sorted(whole - set(required)))

    def _whole_block_restaters(self) -> tuple[str, ...]:
        """Other classes that declare the whole block identically themselves."""

        whole = frozenset.intersection(*(self.own_holders_of(name) for name in self.block))
        return tuple(sorted(whole - {self.first.symbol, self.second.symbol}))

    def _co_restaters(self) -> tuple[str, ...]:
        """The other classes declaring the whole block identically that can compose
        the dataclass owner the two classes compose."""

        kept, _ = self._frozenness_split(
            (self.first.symbol, self.second.symbol), self._whole_block_restaters(), self.block
        )
        return kept

    def _frozenness_of(self, symbol: str) -> DataclassFrozenness:
        indexed_class = self.class_for(symbol)
        if indexed_class is None:
            return DataclassFrozenness.UNKNOWN
        return indexed_class.dataclass_declaration.frozenness

    def _frozenness_split(
        self, required: Sequence[str], optional: Sequence[str], moved: Sequence[str]
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The optional composers that agree with the required ones on frozenness,
        and those left out. When the required ones disagree the owner refuses,
        so none is left out. A mixin of `moved` that takes no field is a plain
        class every frozenness composes, so none is left out either."""

        frozenness = tuple(self._frozenness_of(symbol) for symbol in required)
        if not self.block_fields & frozenset(moved) or not DataclassFrozenness.agree(frozenness):
            return tuple(optional), ()
        kept = tuple(
            symbol
            for symbol in optional
            if DataclassFrozenness.agree((*frozenness, self._frozenness_of(symbol)))
        )
        return kept, tuple(symbol for symbol in optional if symbol not in kept)

    def _frozenness_receipts(
        self, required: Sequence[str], optional: Sequence[str], moved: Sequence[str]
    ) -> tuple[Receipt, ...]:
        """A refusal when the classes that must compose one dataclass mixin differ
        in frozenness, and a note naming each optional composer left out. A mixin
        of `moved` that takes no field is no dataclass, so frozenness never refuses it."""

        _, left_out = self._frozenness_split(required, optional, moved)
        receipts: list[Receipt] = []
        if self.block_fields & frozenset(moved) and not DataclassFrozenness.agree(
            self._frozenness_of(symbol) for symbol in required
        ):
            receipts.append(
                Receipt(
                    OwnerReceiptKind.FROZEN_DIFFERS,
                    "Python refuses a frozen dataclass with a non-frozen dataclass base "
                    "and the reverse, so one mixin cannot serve "
                    + ", ".join(
                        f"`{self._qualname_of(symbol)}` ({self._frozenness_of(symbol)})"
                        for symbol in required
                    ),
                    tuple(
                        _location(indexed_class)
                        for symbol in required
                        if (indexed_class := self.class_for(symbol)) is not None
                    ),
                )
            )
        if left_out:
            receipts.append(
                Receipt(
                    OwnerReceiptKind.KEEPS_OWN_DECLARATION,
                    ", ".join(
                        f"`{self._qualname_of(symbol)}` ({self._frozenness_of(symbol)})"
                        for symbol in left_out
                    )
                    + " also declare(s) the block but cannot compose a dataclass mixin "
                    "of the other frozenness: it keeps its own declaration",
                    tuple(
                        _location(indexed_class)
                        for symbol in left_out
                        if (indexed_class := self.class_for(symbol)) is not None
                    ),
                    count=len(left_out),
                )
            )
        return tuple(receipts)

    def _qualname_of(self, symbol: str) -> str:
        indexed_class = self.class_for(symbol)
        return symbol if indexed_class is None else indexed_class.qualname

    def _restater_receipts(self, restaters: Sequence[str]) -> tuple[Receipt, ...]:
        if not restaters:
            return ()
        return (
            Receipt(
                OwnerReceiptKind.ALSO_RESTATED_BY,
                f"{len(restaters)} other class(es) restate the whole block and would "
                "compose it too: " + ", ".join(f"`{symbol}`" for symbol in restaters[:4]),
                tuple(
                    _location(indexed_class)
                    for symbol in restaters[:4]
                    if (indexed_class := self.class_for(symbol)) is not None
                ),
                count=min(len(restaters), SERVED_CONSUMER_CAP),
            ),
        )

    def _holding_receipts(
        self, owner: CompactIndexedClass, held: Sequence[str]
    ) -> tuple[Receipt, ...]:
        if not held:
            return ()
        return (
            Receipt(
                OwnerReceiptKind.HOLDS_DECLARATION,
                f"`{owner.qualname}` already declares {len(held)} of the "
                f"{len(self.block)} block declarations identically: "
                + ", ".join(f"`{name}`" for name in held),
                (_location(owner),),
                count=len(held),
            ),
        )

    def _consumer_receipts(self, owner: CompactIndexedClass) -> tuple[Receipt, ...]:
        served = sorted(
            self.consumers_of(owner)
            & (self.consumers_of(self.first) | self.consumers_of(self.second))
        )
        if not served:
            return ()
        return (
            Receipt(
                OwnerReceiptKind.SERVES_CONSUMER,
                f"{len(served)} consumer(s) already use `{owner.qualname}` and one "
                "of the classes: "
                + ", ".join(f"`{consumer}`" for _path, consumer in served[:4]),
                tuple(self.use_locations(owner)[key] for key in served[:4]),
                count=min(len(served), SERVED_CONSUMER_CAP),
            ),
        )

    def _conflict_receipts(self, owner: CompactIndexedClass) -> tuple[Receipt, ...]:
        conflicting = tuple(
            name for name in self.block if self.declares_differently(owner.symbol, name)
        )
        if not conflicting:
            return ()
        return (
            Receipt(
                OwnerReceiptKind.CONFLICTING_DECLARATION,
                f"`{owner.qualname}` already declares "
                + ", ".join(f"`{name}`" for name in conflicting)
                + " with a different answer",
                (_location(owner),),
                count=len(conflicting),
            ),
        )

    def _placement(
        self, anchor_module: str
    ) -> tuple[str, type[NewMixinOwner], int, int, tuple[Receipt, ...]]:
        """Where a new mixin copied from the composer in `anchor_module` is
        declared: that module, unless a composer importing it there closes a
        cycle. Then a module every composer already imports and that imports
        none of them; failing that, a new module of their lowest common package,
        which only the composers import; failing that, `anchor_module`, refused,
        with why no module qualifies. Returns the module, the candidate's kind,
        the imports added, the cycles closed, and their receipts."""

        imports, cycles, receipts = self._import_receipts(anchor_module, ())
        if not cycles:
            return anchor_module, NewMixinOwner, imports, cycles, receipts
        composer_modules = self._composer_modules()
        importers = ", ".join(f"`{module}`" for module in composer_modules)
        closing = next(
            receipt for receipt in receipts if receipt.kind is OwnerReceiptKind.CYCLE_CREATED
        )
        reads = self._block_reads(anchor_module)
        lower = self.imports.shared_lower_module(composer_modules, reads)
        if lower is not None:
            imports, cycles, receipts = self._import_receipts(lower.module, ())
            placed = Receipt(
                OwnerReceiptKind.PLACED_BELOW,
                f"not in `{anchor_module}`: {closing.detail}; `{lower.module}` is imported, "
                f"directly or not, by {importers} and imports none of them",
                (),
            )
            return lower.module, NewMixinOwner, imports, cycles, (placed, *receipts)
        package = common_package(composer_modules)
        pinned = self.imports.pinning_reads(composer_modules, reads)
        if pinned:
            unplaced = Receipt(
                OwnerReceiptKind.CYCLE_CREATED,
                f"not in `{anchor_module}`: {closing.detail}; and the block reads names from "
                + ", ".join(f"`{module}`" for module in pinned)
                + f", which no module below {importers} can import without a cycle",
                (),
            )
            return anchor_module, NewMixinOwner, imports, cycles, (*receipts, unplaced)
        if package:
            placed = Receipt(
                OwnerReceiptKind.PLACED_BELOW,
                f"not in `{anchor_module}`: {closing.detail}; no module that {importers} "
                f"all import avoids it, so a new module of `{package}`, which only "
                "they import, declares it",
                (),
            )
            added = Receipt(
                OwnerReceiptKind.IMPORT_ADDED,
                f"{importers} would import the new module of `{package}`",
                (),
                count=len(composer_modules),
            )
            return package, NewModuleMixinOwner, len(composer_modules), 0, (placed, added)
        unplaced = Receipt(
            OwnerReceiptKind.CYCLE_CREATED,
            f"no module that {importers} all import avoids the cycle, and they share no "
            "package to hold a new one",
            (),
        )
        return anchor_module, NewMixinOwner, imports, cycles, (*receipts, unplaced)

    def _block_reads(self, anchor_module: str) -> frozenset[str]:
        """The scanned modules the block's declarations, as the anchor in
        `anchor_module` writes them, import names from."""

        anchors = tuple(indexed for indexed in self.pair if indexed.module_name == anchor_module)
        return frozenset(
            module
            for anchor in anchors[:1]
            for declaration in anchor.body_declarations
            if declaration.name in self.block
            for _name, qualified in declaration.bindings
            if (module := self.imports.module_of(qualified)) is not None
        )

    def _import_receipts(
        self, module_name: str, owners: Sequence[CompactIndexedClass]
    ) -> tuple[int, int, tuple[Receipt, ...]]:
        importers = self._composer_modules()
        needed = tuple(
            importer
            for importer in importers
            if not self.imports.imports_directly(importer, module_name)
        )
        cycles = tuple(
            importer for importer in needed if self.imports.reaches(module_name, importer)
        )
        receipts: list[Receipt] = []
        already = tuple(importer for importer in importers if importer not in needed)
        if already and not needed:
            receipts.append(
                Receipt(
                    OwnerReceiptKind.ALREADY_IMPORTED,
                    f"both classes' modules already are or import `{module_name}`",
                    tuple(_location(owner) for owner in owners),
                )
            )
        if needed:
            receipts.append(
                Receipt(
                    OwnerReceiptKind.IMPORT_ADDED,
                    f"{', '.join(f'`{importer}`' for importer in needed)} would import "
                    f"`{module_name}`",
                    (),
                    count=len(needed),
                )
            )
        if cycles:
            receipts.append(
                Receipt(
                    OwnerReceiptKind.CYCLE_CREATED,
                    f"`{module_name}` already imports, directly or not, "
                    + ", ".join(f"`{importer}`" for importer in cycles)
                    + ": the new import closes a cycle",
                    (),
                    count=len(cycles),
                )
            )
        return len(needed), len(cycles), tuple(receipts)


CandidateT = TypeVar("CandidateT", bound=OwnerCandidate)


def ranked(candidates: Iterable[CandidateT]) -> tuple[CandidateT, ...]:
    """The candidates with confidence scores: a softmax over the permitted ones' log-odds, uncalibrated."""

    candidates = tuple(candidates)
    permitted = [candidate for candidate in candidates if candidate.permitted]
    top = max((candidate.log_odds for candidate in permitted), default=0.0)
    mass = sum(math.exp(candidate.log_odds - top) for candidate in permitted)
    scored = tuple(
        replace(
            candidate,
            confidence=round(math.exp(candidate.log_odds - top) / mass, 4)
            if candidate.permitted
            else 0.0,
        )
        for candidate in candidates
    )
    return tuple(
        sorted(
            scored,
            key=lambda candidate: (-candidate.confidence, -candidate.log_odds, candidate.candidate_id),
        )
    )


def _location(indexed_class: CompactIndexedClass) -> SourceLocation:
    return SourceLocation(indexed_class.file_path, indexed_class.line, indexed_class.qualname)


__all__ = (
    "BlockOwnerCandidate",
    "BlockOwnerQuestion",
    "ExistingClassOwner",
    "ComposeExistingOwner",
    "ModuleImportGraph",
    "HoldSourceOwner",
    "NewMixinOwner",
    "NewModuleMixinOwner",
    "DeleteAndRedirectOwner",
    "OwnerCandidate",
    "OwnerCost",
    "Receipt",
    "OwnerReceiptKind",
    "SharedBaseOwner",
    "ranked",
)
