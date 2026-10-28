"""Where a shared block's one declaration could live: ranked owner candidates (#67).

When a move needs an owner decision, NRA does not decide it. It lists the
places the block could live, ranks them, and shows the facts behind each, so
that a picker can choose and cite those facts. The picker may be a person or
an agent. The candidates are:

- **shared base**: a scanned class both classes already descend from;
- **existing class**: a scanned class, outside both classes' families, that
  already declares part of the block identically, which both would compose;
- **new mixin**: a class NRA would create in one of the two classes' modules.

**The score.** Each receipt has a fixed log-odds weight per unit, stated on its
kind below, times its count. A candidate's log-odds is the sum of its
receipts. Its probability is the softmax of the log-odds over the candidates
no guard refuses; a refused candidate has probability 0. As with #70, the
weights are set by hand rather than fitted, and the score decomposes into the
receipts that produced it. Guards weigh nothing and only refuse.

**The cost** is what the move would do to the source: the statements it
rewrites, the imports it adds, and the import cycles those imports would
close. Closing a cycle is a guard.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import ClassVar

from .class_index import CompactIndexedClass
from .json_reports import SemanticRecord
from .receipts import Receipt, ReceiptKind, ScoredByReceipts
from .models import SourceLocation


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
    CYCLE_CREATED = "cycle_created", 0.0, True
    CONFLICTING_DECLARATION = "conflicting_declaration", 0.0, True


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

    candidate_id: str
    module_name: str
    receipts: tuple[Receipt, ...]
    cost: OwnerCost
    probability: float = 0.0
    kind: str = field(init=False, default="")

    KIND: ClassVar[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", type(self).KIND)

    @abstractmethod
    def describe(self) -> str:
        """The candidate in a few words."""


@dataclass(frozen=True)
class SharedBaseOwner(OwnerCandidate):
    """A scanned base both classes already descend from."""

    symbol: str = ""
    KIND: ClassVar[str] = "shared_base"

    def describe(self) -> str:
        return f"shared base `{self.symbol}`"


@dataclass(frozen=True)
class ExistingClassOwner(OwnerCandidate):
    """A scanned class outside both families that already declares part of the block."""

    symbol: str = ""
    KIND: ClassVar[str] = "existing_class"

    def describe(self) -> str:
        return f"existing class `{self.symbol}`"


@dataclass(frozen=True)
class NewMixinOwner(OwnerCandidate):
    """A mixin NRA would create in `module_name`, composed by both classes."""

    KIND: ClassVar[str] = "new_mixin"

    def describe(self) -> str:
        return f"new mixin in `{self.module_name}`"


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
        modules = frozenset(aliases_by_module)

        def scanned_module(target: str) -> str | None:
            parts = target.split(".")
            for length in range(len(parts), 0, -1):
                candidate = ".".join(parts[:length])
                if candidate in modules:
                    return candidate
            return None

        return cls(
            {
                module: frozenset(
                    imported
                    for target in targets
                    if (imported := scanned_module(target)) is not None
                    and imported != module
                )
                for module, targets in aliases_by_module.items()
            }
        )

    def imports_directly(self, importer: str, imported: str) -> bool:
        return importer == imported or imported in self.imports.get(importer, ())

    def reaches(self, start: str, goal: str) -> bool:
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
        return goal in reachable


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
    holders_of: Callable[[str], frozenset[str]]
    own_holders_of: Callable[[str], frozenset[str]]
    declared_names: Callable[[str], frozenset[str]]
    declares_differently: Callable[[str, str], bool]
    class_for: Callable[[str], CompactIndexedClass | None]
    ancestors_of: Callable[[str], tuple[str, ...]]
    descendants_of: Callable[[str], tuple[str, ...]]
    consumers_of: Callable[[CompactIndexedClass], frozenset[tuple[str, str]]]
    use_locations: Callable[[CompactIndexedClass], Mapping[tuple[str, str], SourceLocation]]
    imports: ModuleImportGraph
    EXISTING_CLASS_LIMIT: ClassVar[int] = 3
    SHARED_BASE_LIMIT: ClassVar[int] = 3

    def candidates(self) -> tuple[OwnerCandidate, ...]:
        found = (
            *self._shared_bases(),
            *self._existing_classes(),
            *self._new_mixins(),
        )
        return ranked(found)

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
            ]
            if others:
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
                candidate_id=f"shared_base:{symbol}",
                module_name=base.module_name,
                receipts=tuple(receipts),
                cost=OwnerCost(
                    statements_rewritten=len(self.block)
                    - len(held)
                    + (2 + len(restaters)) * len(self.block),
                    imports_added=0,
                    cycles_created=0,
                ),
                symbol=symbol,
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
            ]
            imports, cycles, import_receipts = self._import_receipts(
                owner.module_name, (owner,)
            )
            yield ExistingClassOwner(
                candidate_id=f"existing_class:{symbol}",
                module_name=owner.module_name,
                receipts=(*receipts, *import_receipts),
                cost=OwnerCost(
                    statements_rewritten=len(self.block) - len(held)
                    + 2
                    + 2 * len(self.block),
                    imports_added=imports,
                    cycles_created=cycles,
                ),
                symbol=symbol,
            )

    def _new_mixins(self) -> Iterator[NewMixinOwner]:
        for module_name in dict.fromkeys(
            indexed_class.module_name for indexed_class in self.pair
        ):
            imports, cycles, import_receipts = self._import_receipts(module_name, ())
            restaters = self._co_restaters()
            yield NewMixinOwner(
                candidate_id=f"new_mixin:{module_name}",
                module_name=module_name,
                receipts=(
                    Receipt(
                        OwnerReceiptKind.NEW_CLASS,
                        f"a new class in `{module_name}` declares the "
                        f"{len(self.block)}-declaration block",
                        (),
                    ),
                    *self._restater_receipts(restaters),
                    *import_receipts,
                ),
                cost=OwnerCost(
                    statements_rewritten=1
                    + len(self.block)
                    + (2 + len(restaters)) * (1 + len(self.block)),
                    imports_added=imports,
                    cycles_created=cycles,
                ),
            )

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

        brought = sorted(
            self.declared_names(owner.symbol)
            - set(self.block)
            - self.declared_names(self.first.symbol)
            - self.declared_names(self.second.symbol)
        )
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
        )

    def _co_restaters(self) -> tuple[str, ...]:
        """Other classes that declare the whole block identically themselves."""

        whole = frozenset.intersection(*(self.own_holders_of(name) for name in self.block))
        return tuple(sorted(whole - {self.first.symbol, self.second.symbol}))

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

    def _import_receipts(
        self, module_name: str, owners: Sequence[CompactIndexedClass]
    ) -> tuple[int, int, tuple[Receipt, ...]]:
        importers = tuple(
            dict.fromkeys(indexed_class.module_name for indexed_class in self.pair)
        )
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


def ranked(candidates: Iterable[OwnerCandidate]) -> tuple[OwnerCandidate, ...]:
    """The candidates with probabilities: a softmax over the permitted ones' log-odds."""

    candidates = tuple(candidates)
    permitted = [candidate for candidate in candidates if candidate.permitted]
    top = max((candidate.log_odds for candidate in permitted), default=0.0)
    mass = sum(math.exp(candidate.log_odds - top) for candidate in permitted)
    scored = tuple(
        replace(
            candidate,
            probability=round(math.exp(candidate.log_odds - top) / mass, 4)
            if candidate.permitted
            else 0.0,
        )
        for candidate in candidates
    )
    return tuple(
        sorted(
            scored,
            key=lambda candidate: (-candidate.probability, -candidate.log_odds, candidate.candidate_id),
        )
    )


def _location(indexed_class: CompactIndexedClass) -> SourceLocation:
    return SourceLocation(indexed_class.file_path, indexed_class.line, indexed_class.qualname)


__all__ = (
    "BlockOwnerQuestion",
    "ExistingClassOwner",
    "ModuleImportGraph",
    "NewMixinOwner",
    "OwnerCandidate",
    "OwnerCost",
    "Receipt",
    "OwnerReceiptKind",
    "SharedBaseOwner",
    "ranked",
)
