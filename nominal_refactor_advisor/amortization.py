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
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from pathlib import Path

from nra_introspection_trace import IntrospectionTrace

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


@dataclass(frozen=True)
class GenericIntrospector(SemanticRecord):
    """A function a recorded run saw introspecting two or more classes."""

    consumer: str
    classes: int
    fields: int


@dataclass(frozen=True)
class IntrospectionAmortization(SemanticRecord):
    """Field uses generic introspection derives: without the declaration-driven
    consumer, each (field, consumer) pair is a hand-written statement."""

    trace: str
    consumers: int
    classes: int
    generic: tuple[GenericIntrospector, ...]

    @json_report_property()
    def derived_field_uses(self) -> int:
        return sum(item.fields for item in self.generic)


class IntrospectionSource(ABC):
    """Where the run-time introspection term comes from."""

    @abstractmethod
    def amortization(self) -> IntrospectionAmortization:
        raise NotImplementedError

    @staticmethod
    def from_argument(path: Path | None) -> IntrospectionSource:
        """The CLI option decoded once: a recorded trace, or none given."""
        match path:
            case Path():
                return RecordedTrace(path)
        return NoTrace()


@dataclass(frozen=True)
class NoTrace(IntrospectionSource):
    """No run was recorded, so the term is unmeasured, not zero."""

    def amortization(self) -> IntrospectionAmortization:
        return IntrospectionAmortization(
            trace="not measured: record a run with `pytest -p nra_introspection_trace` "
            "and pass --introspection-trace",
            consumers=0,
            classes=0,
            generic=(),
        )


@dataclass(frozen=True)
class RecordedTrace(IntrospectionSource):
    """A trace file written by the nra_introspection_trace pytest plugin."""

    path: Path

    def amortization(self) -> IntrospectionAmortization:
        trace = IntrospectionTrace.read(self.path)
        fields = {count.introspected: count.fields for count in trace.field_counts}
        by_consumer: defaultdict[str, set[str]] = defaultdict(set)
        for use in trace.uses:
            by_consumer[use.consumer].add(use.introspected)
        return IntrospectionAmortization(
            trace=f"recorded in {self.path} for {', '.join(trace.prefixes)}",
            consumers=len(by_consumer),
            classes=len({use.introspected for use in trace.uses}),
            generic=tuple(
                sorted(
                    (
                        GenericIntrospector(
                            consumer,
                            len(introspected),
                            sum(fields.get(name, 0) for name in introspected),
                        )
                        for consumer, introspected in by_consumer.items()
                        if len(introspected) >= 2
                    ),
                    key=lambda item: (-item.fields, item.consumer),
                )
            ),
        )
