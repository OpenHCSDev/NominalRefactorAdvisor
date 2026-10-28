"""What inheritance derives: the (class, method) pairs nobody wrote.

A class that inherits a method instead of defining it gets that implementation
from its parent's one declaration. Remove the parent's method and each such
pair has to be written by hand: the ablation counts them. A pair counts only
when a call whose receiver the type checker resolves to a scanned class reaches
it, so a fat root nobody calls through earns nothing. The report names how
many calls have no such receiver, the coverage this leaves out.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from itertools import chain

from .ast_tools import ParsedModule
from .class_index import CompactClassFamilyIndex
from .json_reports import SemanticRecord, json_report_property
from .probe_owner import ReceiverSpan, ScannedClasses
from .type_checker_session import TypeCheckerSession


@dataclass(frozen=True)
class InheritedMethodOwner(SemanticRecord):
    """One class whose methods are inherited by classes typed calls reach."""

    owner: str
    derived_pairs: int
    members: int
    consumers: int


@dataclass(frozen=True)
class InheritanceAmortization(SemanticRecord):
    """Pairs inheritance derives that typed calls reach, by the class that declares them."""

    scanned_line_count: int
    type_checker: str
    attribute_calls: int
    receiver_typed_calls: int
    calls_reaching_inherited: int
    owners: tuple[InheritedMethodOwner, ...]

    @json_report_property()
    def derived_pairs(self) -> int:
        """(class, method) pairs that would be hand-written without their parent."""
        return sum(owner.derived_pairs for owner in self.owners)

    @json_report_property()
    def derived_pairs_per_kloc(self) -> float:
        return 1000 * self.derived_pairs / max(self.scanned_line_count, 1)


@dataclass(frozen=True)
class InheritedMethods:
    """Which declaring class answers a method a class does not define itself."""

    index: CompactClassFamilyIndex

    @cached_property
    def own_by_symbol(self) -> dict[str, frozenset[str]]:
        """Each class's own non-dunder methods."""
        return {
            symbol: frozenset(
                name
                for name in declared.method_names
                if not (name.startswith("__") and name.endswith("__"))
            )
            for symbol, declared in self.index.classes_by_symbol.items()
        }

    def own(self, symbol: str) -> frozenset[str]:
        return self.own_by_symbol.get(symbol, frozenset())

    def owner_of(self, symbol: str, method: str) -> str | None:
        if method in self.own(symbol):
            return None
        return next(
            (
                ancestor
                for ancestor in self.index.ancestor_symbols(symbol)
                if method in self.own(ancestor)
            ),
            None,
        )

    def reached(self, symbol: str) -> tuple[str, ...]:
        return (symbol, *self.index.descendant_symbols(symbol))


@dataclass(frozen=True)
class MethodCall:
    """One ``receiver.method(...)`` call: where its receiver is, and which function makes it."""

    span: ReceiverSpan
    method: str
    consumer: str

    @classmethod
    def in_module(cls, module: ParsedModule) -> Iterator[MethodCall]:
        index = module.syntax_index
        for node_index, call in index.indexed_nodes_of_type(ast.Call):
            match call.func:
                case ast.Attribute(value=receiver, attr=method):
                    yield cls(
                        ReceiverSpan.of(module.file_path, receiver),
                        method,
                        f"{module.module_name}.{index.lexical_symbol(node_index)}",
                    )


def inheritance_amortization(
    modules: Iterable[ParsedModule],
    index: CompactClassFamilyIndex,
    session: TypeCheckerSession,
) -> InheritanceAmortization:
    """Count the inherited pairs that calls with resolved receivers reach across ``modules``."""

    modules = tuple(modules)
    calls = tuple(chain.from_iterable(MethodCall.in_module(module) for module in modules))
    answers = session.receiver_types(call.span for call in calls)
    classes = ScannedClasses(index.classes_by_symbol)
    methods = InheritedMethods(index)
    pairs: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    consumers: defaultdict[str, set[str]] = defaultdict(set)
    typed = reaching = 0
    for call in calls:
        symbols = answers[call.span].owner(classes).class_symbols()
        typed += bool(symbols)
        reached = False
        for member in chain.from_iterable(methods.reached(symbol) for symbol in symbols):
            owner = methods.owner_of(member, call.method)
            if owner is not None:
                pairs[owner].add((member, call.method))
                consumers[owner].add(call.consumer)
                reached = True
        reaching += reached
    return InheritanceAmortization(
        scanned_line_count=sum(len(module.source.splitlines()) for module in modules),
        type_checker=session.status,
        attribute_calls=len(calls),
        receiver_typed_calls=typed,
        calls_reaching_inherited=reaching,
        owners=tuple(
            sorted(
                (
                    InheritedMethodOwner(
                        owner=owner,
                        derived_pairs=len(found),
                        members=len({member for member, _ in found}),
                        consumers=len(consumers[owner]),
                    )
                    for owner, found in pairs.items()
                ),
                key=lambda item: (-item.derived_pairs, item.owner),
            )
        ),
    )
