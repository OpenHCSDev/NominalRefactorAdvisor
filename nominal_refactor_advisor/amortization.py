"""What inheritance derives: the (class, method) pairs nobody wrote.

A class that inherits a method instead of defining it gets that implementation
from its parent's one declaration. Remove the parent's method and each such
pair has to be written by hand: the ablation counts them. A pair counts only
when a call whose receiver type is known reaches it (``self``/``cls`` inside a
class, or a parameter annotated with a scanned class), so a fat root nobody
calls through earns nothing. Untyped receivers are not resolved here; the
report names how many calls that leaves out.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property

from .ast_tools import ParsedModule
from .class_index import CompactClassFamilyIndex
from .json_reports import SemanticRecord, json_report_property
from .observation_families import TypeArgumentResolver


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


@dataclass
class TypedReceiverWalk(ast.NodeVisitor):
    """One module's method calls on receivers whose class is declared in scope."""

    module: ParsedModule
    resolver: TypeArgumentResolver
    methods: InheritedMethods
    pairs: defaultdict[str, set[tuple[str, str]]]
    consumers: defaultdict[str, set[str]]
    attribute_calls: int = 0
    receiver_typed_calls: int = 0
    calls_reaching_inherited: int = 0
    qualname: tuple[str, ...] = ()
    class_symbol: str | None = None
    receivers: dict[str, str] = field(default_factory=dict)

    def scoped(self, node: ast.AST, **scope: object) -> None:
        saved = {name: getattr(self, name) for name in scope}
        for name, value in scope.items():
            setattr(self, name, value)
        self.generic_visit(node)
        for name, value in saved.items():
            setattr(self, name, value)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualname = (*self.qualname, node.name)
        self.scoped(
            node,
            qualname=qualname,
            class_symbol=self.methods.index.symbol_for(
                file_path=self.module.file_path, qualname=".".join(qualname)
            ),
            receivers={},
        )

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        receivers = dict(self.receivers)
        arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        for position, argument in enumerate(arguments):
            if argument.annotation is not None:
                name = self.resolver.qualified_name(argument.annotation, frozenset())
                if self.methods.index.class_for(name or "") is not None:
                    receivers[argument.arg] = name
            elif position == 0 and self.binds_instance(node):
                receivers[argument.arg] = self.class_symbol
        self.scoped(
            node,
            qualname=(*self.qualname, node.name),
            class_symbol=None,
            receivers=receivers,
        )

    visit_AsyncFunctionDef = visit_FunctionDef

    def binds_instance(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Whether the first parameter is the class or its instance."""
        return self.class_symbol is not None and not any(
            isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
            for decorator in node.decorator_list
        )

    def visit_Call(self, node: ast.Call) -> None:
        match node.func:
            case ast.Attribute(value=receiver, attr=method):
                self.attribute_calls += 1
                match receiver:
                    case ast.Name(id=name) if name in self.receivers:
                        self.receiver_typed_calls += 1
                        self.reach(self.receivers[name], method)
        self.generic_visit(node)

    def reach(self, symbol: str, method: str) -> None:
        consumer = ".".join((self.module.module_name, *self.qualname))
        reached = False
        for member in self.methods.reached(symbol):
            owner = self.methods.owner_of(member, method)
            if owner is not None:
                self.pairs[owner].add((member, method))
                self.consumers[owner].add(consumer)
                reached = True
        self.calls_reaching_inherited += reached


def inheritance_amortization(
    modules: Iterable[ParsedModule],
    index: CompactClassFamilyIndex,
    resolvers: dict[str, TypeArgumentResolver],
) -> InheritanceAmortization:
    """Count the inherited pairs typed calls reach across ``modules``."""

    methods = InheritedMethods(index)
    pairs: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    consumers: defaultdict[str, set[str]] = defaultdict(set)
    walks = []
    for module in modules:
        walk = TypedReceiverWalk(
            module, resolvers[module.file_path], methods, pairs, consumers
        )
        walk.visit(module.module)
        walks.append(walk)
    return InheritanceAmortization(
        scanned_line_count=sum(len(walk.module.source.splitlines()) for walk in walks),
        attribute_calls=sum(walk.attribute_calls for walk in walks),
        receiver_typed_calls=sum(walk.receiver_typed_calls for walk in walks),
        calls_reaching_inherited=sum(walk.calls_reaching_inherited for walk in walks),
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
