"""What a receiver expression is declared to be, from annotations alone.

A receiver is the object an attribute is read from. Its declared type comes
from the declarations the scan holds and nothing else: a parameter's
annotation, the method's own class, a member's annotation, a property's or a
function's declared return, the element type of a declared collection, and a
local bound from any of these. The type checker is never run and a name is
never matched to a class by spelling.

Typing happens in two halves. Per module, `ReceiverPathBuilder` turns the
receiver expression into a `ReceiverPath`: a root (an annotation, the method's
class, a module-level reference, a call of one) and the steps taken from it
(attribute, method call, iteration, subscript). The path is AST-free, so a
module projection stores it. Across the scan, `DeclaredReceiverTyping`
evaluates the path over the class index: members are looked up through the
declared MRO (`CompactClassFamilyIndex.declared_member`), returns through the
declared return annotations the class projections carry. The result is a
`DeclaredType`; anything not declared is an `UndeclaredType` naming why.

A class the scan does not hold is read from the installed typeshed stub that
declares it (`StubDeclarations`): `ast.AnnAssign.value` is the stub's
`expr | None`. The stub modules a scan reaches are indexed apart from the
scan's, declaration-only; a name the scan binds is always the scan's.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import importlib.util
import types
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache, cached_property
from typing import ClassVar

from .ast_field_decoders import declared_annotations, variadic_parameters
from .ast_tools import ParsedModule, collect_family_items
from .class_index import (
    ClassMemberAnswersABC,
    CompactAnnotatedFunctionReturn,
    CompactAnnotatedMethod,
    CompactAnnotatedProperty,
    CompactClassFamilyIndex,
    CompactClassFamilyIndexBuilder,
    CompactClassMemberDeclaration,
    CompactClassReferenceResolver,
    CompactIndexedClass,
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
    CompactUnannotatedCallable,
    native_identity,
)
from .typeshed_stubs import StubRoot, installed_stub_root

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


# --- declared types ----------------------------------------------------------


class ContainerClass(ABC):
    """The class a declared collection is an instance of, as its declaration
    names it: never an abstract base it only reaches by `register()`."""

    def class_symbols(self) -> tuple[str, ...]:
        return ()

    def joined(self, other: ContainerClass) -> ContainerClass:
        return self if self == other else UNNAMED_CONTAINER


@dataclass(frozen=True)
class NamedContainer(ContainerClass):
    """The container's own class, by the qualified name of the class object."""

    qualified_name: str

    def class_symbols(self) -> tuple[str, ...]:
        return (self.qualified_name,)


@dataclass(frozen=True)
class UnnamedContainer(ContainerClass):
    """No declaration names the container's class (a view, `sorted(...)`, a join
    of two container classes)."""


UNNAMED_CONTAINER = UnnamedContainer()


def container_of(value: object) -> ContainerClass:
    """The class an object constructs: a class constructs itself, a typing alias
    its origin class; a function declares no class here."""

    match value:
        case type() as constructed:
            return NamedContainer(native_identity(constructed))
    match getattr(value, "__origin__", None):
        case type() as origin:
            return NamedContainer(native_identity(origin))
    return UNNAMED_CONTAINER


def container_named(qualified: str) -> ContainerClass:
    """The container class an annotation head's qualified name binds, read off
    the module that defines it; a module not installed names no class."""

    module, _, name = qualified.rpartition(".")
    if importlib.util.find_spec(module) is None:
        return UNNAMED_CONTAINER
    return container_of(vars(importlib.import_module(module)).get(name))


class DeclaredType(ABC):
    """What a receiver is declared to be; each operation answers the type it yields."""

    def attribute(self, typing: DeclaredReceiverTyping, name: str) -> DeclaredType:
        return UndeclaredType(f"{self.spelled()} has no declared attribute `{name}`")

    def called(self, typing: DeclaredReceiverTyping, name: str) -> DeclaredType:
        return UndeclaredType(f"{self.spelled()} has no declared method `{name}`")

    def element(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return UndeclaredType(f"{self.spelled()} is not a declared collection")

    def indexed(self, position: int) -> DeclaredType:
        return UndeclaredType(f"{self.spelled()} is not a declared sequence")

    def subscripted(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return UndeclaredType(f"{self.spelled()} is not a declared collection")

    def sliced(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return UndeclaredType(f"{self.spelled()} is not a declared sequence")

    def class_symbols(self) -> tuple[str, ...]:
        """The scanned classes a value of this type is an instance of: none unless a class."""
        return ()

    def class_object(self) -> DeclaredType:
        """`type[T]` for this T: the class object, when T is one scanned class."""
        return UndeclaredType(f"`type[...]` of {self.spelled()} is no scanned class object")

    def why_no_class(self) -> str:
        """Why a value of this type is no instance of a scanned class."""
        return f"the receiver is declared as {self.spelled()}, no scanned class instance"

    @abstractmethod
    def spelled(self) -> str:
        raise NotImplementedError

    # A union or rebinding joins two declared types. Double dispatch: each
    # type answers how it joins every other kind, by default as a refusal.

    @abstractmethod
    def join(self, other: DeclaredType, typing: DeclaredReceiverTyping) -> DeclaredType:
        raise NotImplementedError

    def join_instance(
        self, other: DeclaredInstance, typing: DeclaredReceiverTyping
    ) -> DeclaredType:
        return self.heterogeneous(other)

    def join_collection(
        self, other: DeclaredCollection, typing: DeclaredReceiverTyping
    ) -> DeclaredType:
        return self.heterogeneous(other)

    def join_tuple(
        self, other: DeclaredTuple, typing: DeclaredReceiverTyping
    ) -> DeclaredType:
        return self.heterogeneous(other)

    def join_mapping(
        self, other: DeclaredMapping, typing: DeclaredReceiverTyping
    ) -> DeclaredType:
        return self.heterogeneous(other)

    def join_class_object(
        self, other: DeclaredClassObject, typing: DeclaredReceiverTyping
    ) -> DeclaredType:
        return self.heterogeneous(other)

    def join_none(self, other: DeclaredNone) -> DeclaredType:
        return self

    def join_undeclared(self, other: UndeclaredType) -> DeclaredType:
        return other

    def join_foreign(self, other: DeclaredForeignInstance) -> DeclaredType:
        return self.heterogeneous(other)

    def heterogeneous(self, other: DeclaredType) -> DeclaredType:
        return UndeclaredType(
            f"a union of {self.spelled()} and {other.spelled()} is no one family"
        )


class DeclaredScannedClasses(DeclaredType):
    """Scanned classes of one family, as instances or as class objects: two of
    the same kind join when the scan gives their classes one common family."""

    symbols: tuple[str, ...]

    def class_symbols(self) -> tuple[str, ...]:
        return self.symbols

    @abstractmethod
    def with_symbols(self, symbols: tuple[str, ...]) -> DeclaredType:
        raise NotImplementedError

    def family_joined(
        self, other: DeclaredScannedClasses, typing: DeclaredReceiverTyping
    ) -> DeclaredType:
        symbols = tuple(sorted({*self.symbols, *other.symbols}))
        if typing.one_family(symbols):
            return self.with_symbols(symbols)
        return self.heterogeneous(other)


@dataclass(frozen=True)
class DeclaredInstance(DeclaredScannedClasses):
    """An instance of one of these scanned classes, all of one family."""

    symbols: tuple[str, ...]

    def attribute(self, typing: DeclaredReceiverTyping, name: str) -> DeclaredType:
        return typing.joined(
            typing.member_read(symbol, name, self) for symbol in self.symbols
        )

    def called(self, typing: DeclaredReceiverTyping, name: str) -> DeclaredType:
        return typing.joined(
            typing.method_call(symbol, name, self) for symbol in self.symbols
        )

    def with_symbols(self, symbols: tuple[str, ...]) -> DeclaredType:
        return DeclaredInstance(symbols)

    def class_object(self) -> DeclaredType:
        return DeclaredClassObject(self.symbols)

    def spelled(self) -> str:
        return " | ".join(f"`{symbol}`" for symbol in self.symbols)

    def join(self, other: DeclaredType, typing: DeclaredReceiverTyping) -> DeclaredType:
        return other.join_instance(self, typing)

    def join_instance(self, other: DeclaredInstance, typing: DeclaredReceiverTyping) -> DeclaredType:
        return self.family_joined(other, typing)


@dataclass(frozen=True)
class DeclaredForeignInstance(DeclaredType):
    """An instance of a class neither the scan nor an installed stub declares,
    by its qualified name.

    Its members are declared nowhere NRA reads, so nothing past it is typed;
    but it is declared, so it is no class of the scan's.
    """

    qualified_names: tuple[str, ...]

    def class_symbols(self) -> tuple[str, ...]:
        return self.qualified_names

    def spelled(self) -> str:
        return " | ".join(f"`{name}`" for name in self.qualified_names)

    def join(self, other: DeclaredType, typing: DeclaredReceiverTyping) -> DeclaredType:
        return other.join_foreign(self)

    def join_foreign(self, other: DeclaredForeignInstance) -> DeclaredType:
        """Classes outside the scan make no family claim: their union is just declared."""

        return DeclaredForeignInstance(
            tuple(sorted({*self.qualified_names, *other.qualified_names}))
        )


@dataclass(frozen=True)
class DeclaredClassObject(DeclaredScannedClasses):
    """Scanned classes themselves, not instances: calling their methods passes
    the class as `cls`. Their `class_symbols` are the classes they denote, so a
    class-level answer read off one is that class's."""

    symbols: tuple[str, ...]

    def attribute(self, typing: DeclaredReceiverTyping, name: str) -> DeclaredType:
        return typing.joined(
            typing.member_read(symbol, name, DeclaredInstance((symbol,)))
            for symbol in self.symbols
        )

    def called(self, typing: DeclaredReceiverTyping, name: str) -> DeclaredType:
        return typing.joined(
            typing.method_call(symbol, name, DeclaredInstance((symbol,)))
            for symbol in self.symbols
        )

    def element(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        """Iterating an enum class yields its members."""
        return typing.joined(typing.enum_members(symbol) for symbol in self.symbols)

    def with_symbols(self, symbols: tuple[str, ...]) -> DeclaredType:
        return DeclaredClassObject(symbols)

    def spelled(self) -> str:
        return " | ".join(f"the class `{symbol}`" for symbol in self.symbols)

    def join(self, other: DeclaredType, typing: DeclaredReceiverTyping) -> DeclaredType:
        return other.join_class_object(self, typing)

    def join_class_object(
        self, other: DeclaredClassObject, typing: DeclaredReceiverTyping
    ) -> DeclaredType:
        return self.family_joined(other, typing)


@dataclass(frozen=True)
class DeclaredCollection(DeclaredType):
    """An iterable whose every element is declared as one type."""

    item: DeclaredType
    container: ContainerClass

    def element(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return self.item

    def indexed(self, position: int) -> DeclaredType:
        return self.item

    def subscripted(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return self.item

    def sliced(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return self

    def class_symbols(self) -> tuple[str, ...]:
        return self.container.class_symbols()

    def spelled(self) -> str:
        return f"a collection of {self.item.spelled()}"

    def join(self, other: DeclaredType, typing: DeclaredReceiverTyping) -> DeclaredType:
        return other.join_collection(self, typing)

    def join_collection(self, other: DeclaredCollection, typing: DeclaredReceiverTyping) -> DeclaredType:
        return DeclaredCollection(
            self.item.join(other.item, typing), self.container.joined(other.container)
        )


@dataclass(frozen=True)
class DeclaredTuple(DeclaredType):
    """`tuple[A, B, ...]` with one declared type per position."""

    items: tuple[DeclaredType, ...]

    def element(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return typing.joined(self.items)

    def indexed(self, position: int) -> DeclaredType:
        if -len(self.items) <= position < len(self.items):
            return self.items[position]
        return UndeclaredType(f"{self.spelled()} has no position {position}")

    def subscripted(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return typing.joined(self.items)

    def sliced(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return DeclaredCollection(typing.joined(self.items), TUPLE_CONTAINER)

    def class_symbols(self) -> tuple[str, ...]:
        return TUPLE_CONTAINER.class_symbols()

    def spelled(self) -> str:
        return f"tuple[{', '.join(item.spelled() for item in self.items)}]"

    def join(self, other: DeclaredType, typing: DeclaredReceiverTyping) -> DeclaredType:
        return other.join_tuple(self, typing)

    def join_tuple(self, other: DeclaredTuple, typing: DeclaredReceiverTyping) -> DeclaredType:
        if len(self.items) != len(other.items):
            return self.heterogeneous(other)
        return DeclaredTuple(
            tuple(
                mine.join(theirs, typing)
                for mine, theirs in zip(self.items, other.items, strict=True)
            )
        )


@dataclass(frozen=True)
class DeclaredMapping(DeclaredType):
    """A mapping with declared key and value types."""

    key: DeclaredType
    value: DeclaredType
    container: ContainerClass

    def element(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return self.key

    def subscripted(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return self.value

    def called(self, typing: DeclaredReceiverTyping, name: str) -> DeclaredType:
        for view in MappingMethod.__subclasses__():
            if name in view.names:
                return view.of(self)
        return super().called(typing, name)

    def class_symbols(self) -> tuple[str, ...]:
        return self.container.class_symbols()

    def spelled(self) -> str:
        return f"a mapping of {self.key.spelled()} to {self.value.spelled()}"

    def join(self, other: DeclaredType, typing: DeclaredReceiverTyping) -> DeclaredType:
        return other.join_mapping(self, typing)

    def join_mapping(self, other: DeclaredMapping, typing: DeclaredReceiverTyping) -> DeclaredType:
        return DeclaredMapping(
            self.key.join(other.key, typing),
            self.value.join(other.value, typing),
            self.container.joined(other.container),
        )


class MappingMethod(ABC):
    """A `Mapping` method by its names, and the type it returns."""

    names: ClassVar[frozenset[str]]

    @staticmethod
    @abstractmethod
    def of(mapping: DeclaredMapping) -> DeclaredType:
        raise NotImplementedError


class MappingKeys(MappingMethod):
    names = frozenset(("keys",))

    @staticmethod
    def of(mapping: DeclaredMapping) -> DeclaredType:
        return DeclaredCollection(mapping.key, UNNAMED_CONTAINER)


class MappingValues(MappingMethod):
    names = frozenset(("values",))

    @staticmethod
    def of(mapping: DeclaredMapping) -> DeclaredType:
        return DeclaredCollection(mapping.value, UNNAMED_CONTAINER)


class MappingItems(MappingMethod):
    names = frozenset(("items",))

    @staticmethod
    def of(mapping: DeclaredMapping) -> DeclaredType:
        return DeclaredCollection(
            DeclaredTuple((mapping.key, mapping.value)), UNNAMED_CONTAINER
        )


class MappingValueLookup(MappingMethod):
    """`get`, `pop`, `setdefault`: a value, or the absent arm a receiver read excludes."""

    names = frozenset(("get", "pop", "setdefault"))

    @staticmethod
    def of(mapping: DeclaredMapping) -> DeclaredType:
        return mapping.value


@dataclass(frozen=True)
class DeclaredNone(DeclaredType):
    """`None`: the absent arm of an `X | None`, which reading a receiver excludes."""

    def spelled(self) -> str:
        return "`None`"

    def join(self, other: DeclaredType, typing: DeclaredReceiverTyping) -> DeclaredType:
        return other.join_none(self)

    def join_instance(self, other: DeclaredInstance, typing: DeclaredReceiverTyping) -> DeclaredType:
        return other

    def join_collection(self, other: DeclaredCollection, typing: DeclaredReceiverTyping) -> DeclaredType:
        return other

    def join_tuple(self, other: DeclaredTuple, typing: DeclaredReceiverTyping) -> DeclaredType:
        return other

    def join_mapping(self, other: DeclaredMapping, typing: DeclaredReceiverTyping) -> DeclaredType:
        return other

    def join_class_object(
        self, other: DeclaredClassObject, typing: DeclaredReceiverTyping
    ) -> DeclaredType:
        return other

    def join_none(self, other: DeclaredNone) -> DeclaredType:
        return self

    def join_foreign(self, other: DeclaredForeignInstance) -> DeclaredType:
        return other


@dataclass(frozen=True)
class UndeclaredType(DeclaredType):
    """Nothing the scan declares fixes the type; the reason names which step failed."""

    reason: str

    def attribute(self, typing: DeclaredReceiverTyping, name: str) -> DeclaredType:
        return self

    def called(self, typing: DeclaredReceiverTyping, name: str) -> DeclaredType:
        return self

    def element(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return self

    def indexed(self, position: int) -> DeclaredType:
        return self

    def subscripted(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return self

    def sliced(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return self

    def spelled(self) -> str:
        return "an undeclared type"

    def why_no_class(self) -> str:
        return self.reason

    def join(self, other: DeclaredType, typing: DeclaredReceiverTyping) -> DeclaredType:
        return other.join_undeclared(self)

    def join_instance(self, other: DeclaredInstance, typing: DeclaredReceiverTyping) -> DeclaredType:
        return self

    def join_collection(self, other: DeclaredCollection, typing: DeclaredReceiverTyping) -> DeclaredType:
        return self

    def join_tuple(self, other: DeclaredTuple, typing: DeclaredReceiverTyping) -> DeclaredType:
        return self

    def join_mapping(self, other: DeclaredMapping, typing: DeclaredReceiverTyping) -> DeclaredType:
        return self

    def join_class_object(
        self, other: DeclaredClassObject, typing: DeclaredReceiverTyping
    ) -> DeclaredType:
        return self

    def join_none(self, other: DeclaredNone) -> DeclaredType:
        return self

    def join_undeclared(self, other: UndeclaredType) -> DeclaredType:
        return self

    def join_foreign(self, other: DeclaredForeignInstance) -> DeclaredType:
        return self


# --- annotation heads ----------------------------------------------------------


class AnnotationHead(ABC):
    """A typing or builtin subscript head, by the qualified names that spell it,
    and the type its subscript declares."""

    spellings: ClassVar[frozenset[str]]

    def __init__(self, qualified: str) -> None:
        self.qualified = qualified

    @cached_property
    def container(self) -> ContainerClass:
        """The class the head's spelling binds, when it declares a collection."""

        return container_named(self.qualified)

    @abstractmethod
    def declared(
        self,
        items: Sequence[ast.expr],
        typed: Sequence[DeclaredType],
        typing: DeclaredReceiverTyping,
        named: Callable[[], DeclaredType],
    ) -> DeclaredType:
        """`named` is the head read as a plain class annotation."""
        raise NotImplementedError

    @classmethod
    def spelled_as(cls, qualified: str) -> AnnotationHead:
        """The head a qualified name spells; a scanned generic class names itself."""

        for head in cls.__subclasses__():
            if qualified in head.spellings:
                return head(qualified)
        return ClassHead(qualified)


def _typing_names(*names: str) -> frozenset[str]:
    return frozenset(
        f"{module}.{name}" for module in ("typing", "typing_extensions") for name in names
    )


class OptionalHead(AnnotationHead):
    spellings = _typing_names("Optional")

    def declared(self, items: Sequence[ast.expr], typed: Sequence[DeclaredType], typing: DeclaredReceiverTyping, named: Callable[[], DeclaredType]) -> DeclaredType:
        return typed[0].join(DeclaredNone(), typing)


class UnionHead(AnnotationHead):
    spellings = _typing_names("Union")

    def declared(self, items: Sequence[ast.expr], typed: Sequence[DeclaredType], typing: DeclaredReceiverTyping, named: Callable[[], DeclaredType]) -> DeclaredType:
        return typing.joined(typed)


class QualifierHead(AnnotationHead):
    """`ClassVar[T]`, `Final[T]`, `Annotated[T, ...]`: a value of T."""

    spellings = _typing_names("ClassVar", "Final", "Annotated")

    def declared(self, items: Sequence[ast.expr], typed: Sequence[DeclaredType], typing: DeclaredReceiverTyping, named: Callable[[], DeclaredType]) -> DeclaredType:
        return typed[0]


class ClassObjectHead(AnnotationHead):
    spellings = frozenset(("builtins.type", "typing.Type"))

    def declared(self, items: Sequence[ast.expr], typed: Sequence[DeclaredType], typing: DeclaredReceiverTyping, named: Callable[[], DeclaredType]) -> DeclaredType:
        return typed[0].class_object()


class HomogeneousHead(AnnotationHead):
    """A collection whose every element is its first argument."""

    spellings = frozenset(
        (
            *(f"builtins.{name}" for name in ("list", "set", "frozenset")),
            *_typing_names(
                "List",
                "Set",
                "FrozenSet",
                "Sequence",
                "MutableSequence",
                "Iterable",
                "Iterator",
                "Collection",
                "AbstractSet",
                "MutableSet",
                "Reversible",
                "Deque",
                "Generator",
            ),
            *(
                f"collections.abc.{name}"
                for name in (
                    "Sequence",
                    "MutableSequence",
                    "Iterable",
                    "Iterator",
                    "Collection",
                    "Set",
                    "MutableSet",
                    "Reversible",
                    "Generator",
                )
            ),
            "collections.deque",
        )
    )

    def declared(self, items: Sequence[ast.expr], typed: Sequence[DeclaredType], typing: DeclaredReceiverTyping, named: Callable[[], DeclaredType]) -> DeclaredType:
        return DeclaredCollection(typed[0], self.container)


class TupleHead(AnnotationHead):
    """`tuple[A, ...]` is homogeneous; `tuple[A, B]` positional."""

    spellings = frozenset(("builtins.tuple", "typing.Tuple"))

    def declared(self, items: Sequence[ast.expr], typed: Sequence[DeclaredType], typing: DeclaredReceiverTyping, named: Callable[[], DeclaredType]) -> DeclaredType:
        match items:
            case [_, ast.Constant(value=builtins.Ellipsis)]:
                return DeclaredCollection(typed[0], self.container)
        return DeclaredTuple(tuple(typed))


class MappingHead(AnnotationHead):
    spellings = frozenset(
        (
            "builtins.dict",
            *_typing_names("Dict", "Mapping", "MutableMapping", "DefaultDict", "OrderedDict"),
            "collections.abc.Mapping",
            "collections.abc.MutableMapping",
            "collections.defaultdict",
            "collections.OrderedDict",
            "types.MappingProxyType",
        )
    )

    def declared(self, items: Sequence[ast.expr], typed: Sequence[DeclaredType], typing: DeclaredReceiverTyping, named: Callable[[], DeclaredType]) -> DeclaredType:
        if len(typed) != 2:
            return UndeclaredType("a mapping annotation takes a key and a value type")
        return DeclaredMapping(typed[0], typed[1], self.container)


class ClassHead(AnnotationHead):
    """A subscripted class (`Family[Key]`): an instance of the class itself."""

    spellings = frozenset()

    def declared(self, items: Sequence[ast.expr], typed: Sequence[DeclaredType], typing: DeclaredReceiverTyping, named: Callable[[], DeclaredType]) -> DeclaredType:
        return named()


SELF_SPELLINGS = _typing_names("Self")
ANY_SPELLINGS = _typing_names("Any")


# --- evaluation ---------------------------------------------------------------------


@dataclass(frozen=True)
class DeclaredReceiverTyping:
    """Evaluates receiver paths over the scan's class index and declared returns."""

    class_index: CompactClassFamilyIndex
    resolver: CompactClassReferenceResolver
    aliases_by_module: Mapping[str, Mapping[str, str]]
    definitions_by_module: Mapping[str, frozenset[str]]
    function_returns: Mapping[str, tuple[CompactAnnotatedFunctionReturn, ...]]
    type_variables_by_module: Mapping[str, tuple[str, ...]]
    declarations: StubDeclarations

    @classmethod
    def from_projections(
        cls,
        projections: Sequence[CompactModuleClassProjection],
        class_index: CompactClassFamilyIndex,
    ) -> DeclaredReceiverTyping:
        """The scan's typing; a class it does not hold is read from the installed stubs."""

        return cls.declared_over(
            projections,
            class_index,
            StubDeclarations(
                installed_stub_root(),
                frozenset(
                    projection.module_name.partition(".")[0] for projection in projections
                ),
            ),
        )

    @classmethod
    def declared_over(
        cls,
        projections: Sequence[CompactModuleClassProjection],
        class_index: CompactClassFamilyIndex,
        declarations: StubDeclarations,
    ) -> DeclaredReceiverTyping:
        returns: dict[str, list[CompactAnnotatedFunctionReturn]] = {}
        for projection in projections:
            for declared in projection.annotated_function_returns:
                returns.setdefault(
                    f"{projection.module_name}.{declared.name}", []
                ).append(declared)
        return cls(
            class_index=class_index,
            resolver=CompactClassReferenceResolver.from_index(projections, class_index),
            aliases_by_module={
                projection.module_name: dict(projection.import_aliases)
                for projection in projections
            },
            definitions_by_module={
                projection.module_name: frozenset(
                    name for name, _line in projection.top_level_definitions
                )
                for projection in projections
            },
            function_returns={name: tuple(found) for name, found in returns.items()},
            type_variables_by_module={
                projection.module_name: projection.type_variable_names
                for projection in projections
            },
            declarations=declarations,
        )

    # members and calls ---------------------------------------------------

    def declaring(self, symbol: str) -> DeclaredReceiverTyping:
        """The typing whose index holds `symbol`: this one, else the stubs'.

        `chased` yields only symbols one of the two holds."""

        if symbol in self.class_index.classes_by_symbol:
            return self
        return self.declarations.typing

    def member_read(
        self, symbol: str, name: str, receiver: DeclaredInstance
    ) -> DeclaredType:
        typing = self.declaring(symbol)
        return typing.class_index.declared_member(symbol, name).answer(
            _MemberRead(typing, symbol, name, receiver)
        )

    def method_call(
        self, symbol: str, name: str, receiver: DeclaredInstance
    ) -> DeclaredType:
        typing = self.declaring(symbol)
        return typing.class_index.declared_member(symbol, name).answer(
            _MethodCall(typing, symbol, name, receiver)
        )

    def function_result(self, function_symbol: str) -> DeclaredType:
        module = function_symbol.rpartition(".")[0]
        declared = self.function_returns.get(function_symbol, ())
        if len(declared) != 1:
            return UndeclaredType(
                f"`{function_symbol}` declares no single return annotation"
            )
        return self.annotation_source_type(
            module, declared[0].return_annotation_expression, OUTSIDE_METHOD
        )

    @cached_property
    def type_variables(self) -> frozenset[str]:
        return frozenset(
            f"{module}.{name}"
            for module, names in self.type_variables_by_module.items()
            for name in names
        )

    def type_parameters(self, symbol: str) -> tuple[str, ...]:
        """A generic class's type variables, qualified, in declaration order.

        `class C[T]:` declares them; otherwise a `Generic[...]` or
        `Protocol[...]` base lists them; otherwise they are the type variables
        its bases' arguments name, in order.
        """

        declaration = self.class_index.class_for(symbol)
        if declaration is None:
            return ()
        module = declaration.module_name
        if declaration.type_parameter_names:
            return tuple(f"{module}.{name}" for name in declaration.type_parameter_names)
        named: list[str] = []
        for reference, arguments in zip(
            declaration.base_references, declaration.base_argument_sources, strict=False
        ):
            variables = [
                qualified
                for argument in arguments
                if (parts := _reference_parts(ast.parse(argument, mode="eval").body))
                and (qualified := self.qualified(module, parts)) in self.type_variables
            ]
            if reference.qualified_name in GENERIC_DECLARERS:
                return tuple(variables)
            named.extend(variable for variable in variables if variable not in named)
        return tuple(named)

    def generic_arguments(self, symbol: str, owner_symbol: str) -> Mapping[str, DeclaredType]:
        """The types `symbol`'s bases bind `owner_symbol`'s type variables to.

        Walks the declared bases from `symbol` up to the owner, evaluating each
        subscripted base's arguments where the subscript is written, under the
        bindings already made below it.
        """

        pending: list[tuple[str, Mapping[str, DeclaredType]]] = [(symbol, {})]
        seen: set[str] = set()
        for _step in range(len(self.class_index.classes_by_symbol) + 1):
            if not pending:
                break
            current, bound = pending.pop(0)
            if current == owner_symbol:
                return bound
            if current in seen:
                continue
            seen.add(current)
            declaration = self.class_index.class_for(current)
            if declaration is None:
                continue
            scope = AnnotationScope(DeclaredInstance((symbol,)), bound)
            for reference, arguments in zip(
                declaration.base_references, declaration.base_argument_sources, strict=False
            ):
                def bindings(
                    parameters: tuple[str, ...],
                    arguments: tuple[str, ...] = arguments,
                    module: str = declaration.module_name,
                    scope: AnnotationScope = scope,
                ) -> Mapping[str, DeclaredType]:
                    if len(parameters) != len(arguments):
                        return {}
                    return {
                        parameter: self.annotation_source_type(module, argument, scope)
                        for parameter, argument in zip(parameters, arguments, strict=True)
                    }

                self.chased(reference.qualified_name).pend_base(pending, bindings, self)
        return {}

    def enum_members(self, symbol: str) -> DeclaredType:
        declaration = self.declaring(symbol).class_index.class_for(symbol)
        if declaration is not None and declaration.direct_enum_member_names:
            return DeclaredInstance((symbol,))
        return UndeclaredType(f"the class `{symbol}` is no enum, not a declared collection")

    def class_symbol(self, module_name: str, parts: tuple[str, ...]) -> ScannedClassLookup:
        """The scanned class a module-level name chain names, through re-exports.

        An import of `pkg.Name` where `pkg` itself imports `Name` from
        `pkg._impl` names `pkg._impl.Name`: each hop follows a binding the
        importing module declares, never a spelling.
        """

        match self.resolver.symbol_for(
            module_name=module_name,
            reference_parts=parts,
            allow_unique_unqualified=False,
        ):
            case str() as symbol:
                return ScannedClass(symbol)
        return self.chased(self.qualified(module_name, parts))

    def chased(self, qualified: str) -> ScannedClassLookup:
        """The class a qualified name binds: the scan's when the scan binds it,
        else the class an installed stub declares by it."""

        return self.bound(qualified).declared_by(self.declarations)

    def bound(self, qualified: str) -> ScannedClassLookup:
        """The class this typing's own index binds a qualified name to,
        following re-exporting imports."""

        spelled = qualified
        for _hop in range(REEXPORT_HOPS):
            if qualified in self.class_index.classes_by_symbol:
                return ScannedClass(qualified)
            owner, _, name = qualified.rpartition(".")
            following = self.aliases_by_module.get(owner, {}).get(name, qualified)
            if following == qualified:
                break
            qualified = following
        return UnscannedName(spelled)

    def unscanned(self, qualified: str) -> DeclaredType:
        """A name no scanned or stub class binds: undeclared inside the scanned
        packages and the stub modules, a declared foreign class outside them."""

        if qualified.partition(".")[0] in self.scanned_roots:
            return UndeclaredType(f"`{qualified}` names no scanned class")
        return self.declarations.unbound(qualified)

    # references -------------------------------------------------------------

    def reference(self, module_name: str, parts: tuple[str, ...]) -> DeclaredType:
        """A module-level name chain: a scanned class, as the class object."""

        return self.class_symbol(module_name, parts).class_object(self)

    def called_reference(
        self, module_name: str, parts: tuple[str, ...]
    ) -> DeclaredType:
        """`C(...)` constructs a C; `f(...)` is f's declared return; `C.m(...)` is m's."""

        return self.class_symbol(module_name, parts).constructed(self, module_name, parts)

    def called_unscanned(
        self, module_name: str, parts: tuple[str, ...]
    ) -> DeclaredType:
        """A call of a name no scanned class binds: a declared function's return,
        or a method of the reference before it."""

        function = self.qualified(module_name, parts)
        if function in self.function_returns:
            return self.function_result(function)
        if len(parts) > 1:
            return self.reference(module_name, parts[:-1]).called(self, parts[-1])
        return UndeclaredType(
            f"`{'.'.join(parts)}(...)` calls no scanned class or annotated function"
        )

    def qualified(self, module_name: str, parts: tuple[str, ...]) -> str:
        """The qualified name a module-level name chain spells."""

        head, *rest = parts
        aliases = self.aliases_by_module.get(module_name, {})
        if head in aliases:
            return ".".join((aliases[head], *rest))
        if head in self.definitions_by_module.get(module_name, frozenset()) or not hasattr(
            builtins, head
        ):
            return ".".join((module_name, *parts))
        return ".".join(("builtins", *parts))

    @cached_property
    def scanned_roots(self) -> frozenset[str]:
        return frozenset(module.partition(".")[0] for module in self.aliases_by_module)

    # annotations ---------------------------------------------------------------

    def annotation_source_type(
        self, module_name: str, source: str, scope: AnnotationScope
    ) -> DeclaredType:
        try:
            expression = ast.parse(source, mode="eval").body
        except SyntaxError:
            return UndeclaredType(f"annotation `{source}` does not parse")
        return self.annotation_type(module_name, expression, scope)

    def annotation_type(
        self, module_name: str, annotation: ast.expr, scope: AnnotationScope
    ) -> DeclaredType:
        match annotation:
            case ast.Constant(value=str() as source):
                return self.annotation_source_type(module_name, source, scope)
            case ast.Constant(value=None):
                return DeclaredNone()
            case ast.BinOp(left=left, op=ast.BitOr(), right=right):
                return self.annotation_type(module_name, left, scope).join(
                    self.annotation_type(module_name, right, scope), self
                )
            case ast.Subscript(value=head, slice=arguments):
                return self.subscripted_type(module_name, head, arguments, scope)
            case ast.Name() | ast.Attribute():
                return self.named_type(module_name, annotation, scope)
        return UndeclaredType(f"annotation `{ast.unparse(annotation)}` names no type")

    def named_type(
        self,
        module_name: str,
        annotation: ast.Name | ast.Attribute,
        scope: AnnotationScope,
    ) -> DeclaredType:
        parts = _reference_parts(annotation)
        if not parts:
            return UndeclaredType(f"annotation `{ast.unparse(annotation)}` names no type")
        qualified = self.qualified(module_name, parts)
        if qualified in scope.parameters:
            return scope.parameters[qualified]
        if qualified in SELF_SPELLINGS:
            return scope.self_type
        if qualified in ANY_SPELLINGS:
            return UndeclaredType("`Any` declares no type")
        return self.class_symbol(module_name, parts).instance(self)

    def subscripted_type(
        self,
        module_name: str,
        head: ast.expr,
        arguments: ast.expr,
        scope: AnnotationScope,
    ) -> DeclaredType:
        items = list(arguments.elts) if isinstance(arguments, ast.Tuple) else [arguments]
        parts = _reference_parts(head)
        if not parts:
            return UndeclaredType(f"annotation `{ast.unparse(head)}[...]` names no type")
        typed = [self.annotation_type(module_name, item, scope) for item in items]
        assert isinstance(head, ast.Name | ast.Attribute)
        return AnnotationHead.spelled_as(self.qualified(module_name, parts)).declared(
            items, typed, self, lambda: self.named_type(module_name, head, scope)
        )

    # joins -----------------------------------------------------------------------

    def joined(self, types: Iterator[DeclaredType] | Sequence[DeclaredType]) -> DeclaredType:
        found = list(types)
        if not found:
            return UndeclaredType("nothing declares a type")
        result, *rest = found
        for other in rest:
            result = result.join(other, self)
        return result

    def one_family(self, symbols: tuple[str, ...]) -> bool:
        """Whether the scan gives these classes one common class family."""

        return bool(
            set.intersection(
                *(
                    {symbol, *self.declaring(symbol).class_index.ancestor_symbols(symbol)}
                    for symbol in symbols
                )
            )
        )


class ScannedClassLookup(ABC):
    """Which scanned class a name chain binds, if any; each use asks it for the
    type it declares."""

    @abstractmethod
    def instance(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        """The name used as an annotation."""

    @abstractmethod
    def class_object(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        """The name read as a value."""

    @abstractmethod
    def constructed(
        self, typing: DeclaredReceiverTyping, module_name: str, parts: tuple[str, ...]
    ) -> DeclaredType:
        """The name called."""

    def pend_base(
        self,
        pending: list[tuple[str, Mapping[str, DeclaredType]]],
        bindings: Callable[[tuple[str, ...]], Mapping[str, DeclaredType]],
        typing: DeclaredReceiverTyping,
    ) -> None:
        """Queue this base for a generic-argument walk: a base outside the scan
        binds nothing the walk can follow."""

    @abstractmethod
    def declared_by(self, declarations: StubDeclarations) -> ScannedClassLookup:
        """This lookup, or the stub class the declarations hold where the scan binds none."""


@dataclass(frozen=True)
class ScannedClass(ScannedClassLookup):
    symbol: str

    def instance(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return DeclaredInstance((self.symbol,))

    def class_object(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return DeclaredClassObject((self.symbol,))

    def constructed(
        self, typing: DeclaredReceiverTyping, module_name: str, parts: tuple[str, ...]
    ) -> DeclaredType:
        return DeclaredInstance((self.symbol,))

    def pend_base(
        self,
        pending: list[tuple[str, Mapping[str, DeclaredType]]],
        bindings: Callable[[tuple[str, ...]], Mapping[str, DeclaredType]],
        typing: DeclaredReceiverTyping,
    ) -> None:
        pending.append((self.symbol, bindings(typing.type_parameters(self.symbol))))

    def declared_by(self, declarations: StubDeclarations) -> ScannedClassLookup:
        return self


@dataclass(frozen=True)
class UnscannedName(ScannedClassLookup):
    qualified: str

    def instance(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return typing.unscanned(self.qualified)

    def class_object(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return typing.unscanned(self.qualified)

    def constructed(
        self, typing: DeclaredReceiverTyping, module_name: str, parts: tuple[str, ...]
    ) -> DeclaredType:
        return typing.called_unscanned(module_name, parts)

    def declared_by(self, declarations: StubDeclarations) -> ScannedClassLookup:
        return declarations.chased(self.qualified)


@dataclass
class StubDeclarations:
    """The installed stub modules a scan's names reach, indexed apart from the scan.

    A stub module is loaded when a name the scan does not bind lives in it,
    with the stub modules its classes' bases live in, so each stub class's
    declared MRO is as closed as the stubs make it. A top-level package the
    scan holds is never read from a stub: the scan's own module wins. Stub
    classes are read here and nowhere else: they are never scanned modules,
    so no detector or codemod sees them.
    """

    root: StubRoot
    shadowed_roots: frozenset[str]
    projections: dict[str, CompactModuleClassProjection] = field(default_factory=dict)
    built: list[DeclaredReceiverTyping] = field(default_factory=list)

    def chased(self, qualified: str) -> ScannedClassLookup:
        if self.reaches(qualified):
            return self.typing.bound(qualified)
        return UnscannedName(qualified)

    def unbound(self, qualified: str) -> DeclaredType:
        """A name neither the scan nor a stub binds to a class."""

        missing = self.root.missing(qualified)
        if missing is not None:
            return UndeclaredType(missing)
        stub = self.declaring_stub(qualified)
        if stub is not None and stub.module_name != qualified:
            return UndeclaredType(
                f"`{qualified}` names no class the `{stub.module_name}` stub declares"
            )
        return DeclaredForeignInstance((qualified,))

    def declaring_stub(self, qualified: str) -> ParsedModule | None:
        if qualified.partition(".")[0] in self.shadowed_roots:
            return None
        return self.root.declaring_module(qualified)

    def reaches(self, qualified: str) -> bool:
        """Load the stub declaring `qualified`, and its classes' bases' stubs; whether one does."""

        stub = self.declaring_stub(qualified)
        if stub is None:
            return False
        pending = [stub]
        for module in iter(pending):
            if module.module_name in self.projections:
                continue
            projection = stub_projection(module)
            self.projections[module.module_name] = projection
            self.built.clear()
            pending.extend(
                base_stub
                for indexed_class in projection.classes
                for reference in indexed_class.base_references
                if (base_stub := self.declaring_stub(reference.qualified_name)) is not None
                and base_stub.module_name not in self.projections
            )
        return True

    @property
    def typing(self) -> DeclaredReceiverTyping:
        """The typing over the stub modules loaded so far, rebuilt when one is added."""

        if not self.built:
            projections = tuple(self.projections.values())
            self.built.append(
                DeclaredReceiverTyping.declared_over(
                    projections, CompactClassFamilyIndexBuilder(projections).build(), self
                )
            )
        return self.built[0]


@cache
def stub_projection(stub: ParsedModule) -> CompactModuleClassProjection:
    """A stub module's class declarations, projected once per process."""

    (projection,) = collect_family_items(stub, CompactModuleClassProjectionFamily)
    return projection


REEXPORT_HOPS = 8
GENERIC_DECLARERS = frozenset(
    f"{module}.{name}"
    for module in ("typing", "typing_extensions")
    for name in ("Generic", "Protocol")
)
@dataclass(frozen=True)
class AnnotationScope:
    """What an annotation's free type names mean where it is read.

    `self_type` is what `Self` names; `parameters` maps a generic class's type
    variables, by qualified name, to the types a subclass's base arguments
    bind them to.
    """

    self_type: DeclaredType
    parameters: Mapping[str, DeclaredType] = field(default_factory=dict)


OUTSIDE_METHOD = AnnotationScope(UndeclaredType("`Self` outside a method names no class"))


@dataclass(frozen=True)
class _MemberRead(ClassMemberAnswersABC[DeclaredType]):
    """Reading `receiver.name`: an assignment's annotation or a property's return."""

    typing: DeclaredReceiverTyping
    symbol: str
    name: str
    receiver: DeclaredInstance

    def assigned(self, owner: CompactIndexedClass, declaration: CompactClassMemberDeclaration) -> DeclaredType:
        if self.name in owner.direct_enum_member_names:
            return DeclaredInstance((owner.symbol,))
        match declaration.annotation_expression:
            case str() as source:
                return self.typing.annotation_source_type(
                    owner.module_name, source, self.scope(owner)
                )
        return UndeclaredType(
            f"`{owner.symbol}.{self.name}` is assigned without an annotation"
        )

    def scope(self, owner: CompactIndexedClass) -> AnnotationScope:
        """`Self` is the receiver; the owner's type variables are what the receiver's bases bind."""

        return AnnotationScope(
            self.receiver, self.typing.generic_arguments(self.symbol, owner.symbol)
        )

    def annotated_property(self, owner: CompactIndexedClass, declaration: CompactAnnotatedProperty) -> DeclaredType:
        return self.typing.annotation_source_type(
            owner.module_name, declaration.return_annotation_expression, self.scope(owner)
        )

    def annotated_method(self, owner: CompactIndexedClass, declaration: CompactAnnotatedMethod) -> DeclaredType:
        return UndeclaredType(f"`{self.name}` read uncalled is a bound method")

    def unannotated_callable(self, owner: CompactIndexedClass, declaration: CompactUnannotatedCallable) -> DeclaredType:
        return UndeclaredType(
            f"`{owner.symbol}.{self.name}` declares no return annotation"
        )

    def missing(self) -> DeclaredType:
        return UndeclaredType(f"no class in `{self.symbol}`'s MRO declares `{self.name}`")

    def ambiguous(self) -> DeclaredType:
        return UndeclaredType(f"`{self.name}` is bound more than once on `{self.symbol}`")

    def open(self) -> DeclaredType:
        return UndeclaredType(f"`{self.symbol}`'s MRO is open before `{self.name}`")


@dataclass(frozen=True)
class _MethodCall(_MemberRead):
    """Calling `receiver.name(...)`: the method's declared return, `Self` the receiver."""

    def assigned(self, owner: CompactIndexedClass, declaration: CompactClassMemberDeclaration) -> DeclaredType:
        return UndeclaredType(f"`{self.name}` is assigned, not a declared method")

    def annotated_property(self, owner: CompactIndexedClass, declaration: CompactAnnotatedProperty) -> DeclaredType:
        return UndeclaredType(f"`{self.name}` is a property; calling its value")

    def annotated_method(self, owner: CompactIndexedClass, declaration: CompactAnnotatedMethod) -> DeclaredType:
        return self.typing.annotation_source_type(
            owner.module_name, declaration.return_annotation_expression, self.scope(owner)
        )


def _reference_parts(node: ast.expr) -> tuple[str, ...]:
    match node:
        case ast.Name(id=name):
            return (name,)
        case ast.Attribute(value=value, attr=attribute):
            parent = _reference_parts(value)
            return (*parent, attribute) if parent else ()
    return ()


# --- paths --------------------------------------------------------------------------


class ReceiverRoot(ABC):
    """Where a receiver path starts."""

    @abstractmethod
    def declared(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        raise NotImplementedError

    def attribute_of(self, path: ReceiverPath, name: str) -> ReceiverPath:
        """`P.name`, for a path P starting here."""
        return path.then(AttributeStep(name))

    def method_call_of(self, path: ReceiverPath, name: str) -> ReceiverPath:
        """`P.name(...)`, for a path P starting here."""
        return path.then(CallStep(name))

    def call_of(
        self,
        path: ReceiverPath,
        builder: ReceiverPathBuilder,
        arguments: Sequence[ast.expr],
        pending: Pending,
    ) -> ReceiverPath:
        """`P(...)`, for a path P starting here: only a declared callable has a type."""
        return unresolved("the call's callee is not a declared callable")


@dataclass(frozen=True)
class AnnotatedRoot(ReceiverRoot):
    """A value declared by an annotation, read in its module."""

    module_name: str
    annotation: str

    def declared(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return typing.annotation_source_type(
            self.module_name, self.annotation, OUTSIDE_METHOD
        )


@dataclass(frozen=True)
class MethodAnnotatedRoot(AnnotatedRoot):
    """An annotation inside a method, where `Self` is the method's class."""

    class_symbol: str

    def declared(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return typing.annotation_source_type(
            self.module_name,
            self.annotation,
            AnnotationScope(DeclaredInstance((self.class_symbol,))),
        )


@dataclass(frozen=True)
class OwnInstanceRoot(ReceiverRoot):
    """A method's own first parameter, or `cls(...)` in a classmethod: an instance of the class."""

    class_symbol: str

    def declared(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return DeclaredInstance((self.class_symbol,))


@dataclass(frozen=True)
class OwnClassRoot(ReceiverRoot):
    """A classmethod's first parameter: its class object."""

    class_symbol: str

    def declared(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return DeclaredClassObject((self.class_symbol,))

    def call_of(self, path: ReceiverPath, builder: ReceiverPathBuilder, arguments: Sequence[ast.expr], pending: Pending) -> ReceiverPath:
        if path.steps:
            return super().call_of(path, builder, arguments, pending)
        return ReceiverPath(OwnInstanceRoot(self.class_symbol))


@dataclass(frozen=True)
class ReferenceRoot(ReceiverRoot):
    """A module-level or builtin name chain."""

    module_name: str
    parts: tuple[str, ...]

    def declared(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return typing.reference(self.module_name, self.parts)

    def attribute_of(self, path: ReceiverPath, name: str) -> ReceiverPath:
        if path.steps:
            return super().attribute_of(path, name)
        return ReceiverPath(ReferenceRoot(self.module_name, (*self.parts, name)))

    def method_call_of(self, path: ReceiverPath, name: str) -> ReceiverPath:
        if path.steps:
            return super().method_call_of(path, name)
        return ReceiverPath(CalledReferenceRoot(self.module_name, (*self.parts, name)))

    def call_of(self, path: ReceiverPath, builder: ReceiverPathBuilder, arguments: Sequence[ast.expr], pending: Pending) -> ReceiverPath:
        if path.steps:
            return super().call_of(path, builder, arguments, pending)
        return builder.module_call(self.parts, arguments, pending)


@dataclass(frozen=True)
class CalledReferenceRoot(ReceiverRoot):
    """A call of a module-level name chain."""

    module_name: str
    parts: tuple[str, ...]

    def declared(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return typing.called_reference(self.module_name, self.parts)


@dataclass(frozen=True)
class AgreeingRoot(ReceiverRoot):
    """A local bound more than once: the join of every binding's declared type."""

    name: str
    bindings: tuple[ReceiverPath, ...]

    def declared(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        """Its bindings join as a union does; a binding to `None` is the absent
        arm, which reading the receiver excludes."""

        return typing.joined(binding.declared(typing) for binding in self.bindings)


@dataclass(frozen=True)
class NoneRoot(ReceiverRoot):
    """The literal `None`."""

    def declared(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return DeclaredNone()


@dataclass(frozen=True)
class UnionRoot(ReceiverRoot):
    """`A if test else B`: either arm, joined as a union is."""

    arms: tuple[ReceiverPath, ...]

    def declared(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return typing.joined(arm.declared(typing) for arm in self.arms)


@dataclass(frozen=True)
class TupleDisplayRoot(ReceiverRoot):
    """`(a, b)`: a tuple with each position's own type."""

    items: tuple[ReceiverPath, ...]

    def declared(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return DeclaredTuple(tuple(item.declared(typing) for item in self.items))


@dataclass(frozen=True)
class CollectionDisplayRoot(ReceiverRoot):
    """`[a, b]`, `{a, b}`, or a comprehension: a collection of its elements' joined type."""

    elements: tuple[ReceiverPath, ...]
    container: ContainerClass

    def declared(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return DeclaredCollection(
            typing.joined(element.declared(typing) for element in self.elements),
            self.container,
        )


@dataclass(frozen=True)
class LiteralRoot(ReceiverRoot):
    """A literal: an instance of its builtin class."""

    qualified_name: str

    def declared(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return typing.chased(self.qualified_name).instance(typing)


@dataclass(frozen=True)
class MappingDisplayRoot(ReceiverRoot):
    """A dict comprehension: a mapping of its key's type to its value's."""

    key: ReceiverPath
    value: ReceiverPath

    def declared(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return DeclaredMapping(
            self.key.declared(typing), self.value.declared(typing), DICT_CONTAINER
        )


@dataclass(frozen=True)
class UnresolvedRoot(ReceiverRoot):
    """No declaration fixes the receiver's root; the reason says why."""

    reason: str

    def declared(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        return UndeclaredType(self.reason)


class ReceiverStep(ABC):
    """One step from a receiver path's value to the next."""

    @abstractmethod
    def applied(
        self, declared: DeclaredType, typing: DeclaredReceiverTyping
    ) -> DeclaredType:
        raise NotImplementedError


@dataclass(frozen=True)
class AttributeStep(ReceiverStep):
    name: str

    def applied(self, declared: DeclaredType, typing: DeclaredReceiverTyping) -> DeclaredType:
        return declared.attribute(typing, self.name)


@dataclass(frozen=True)
class CallStep(ReceiverStep):
    name: str

    def applied(self, declared: DeclaredType, typing: DeclaredReceiverTyping) -> DeclaredType:
        return declared.called(typing, self.name)


@dataclass(frozen=True)
class ElementStep(ReceiverStep):
    """Iterating the value: `for x in E` binds an element of E."""

    def applied(self, declared: DeclaredType, typing: DeclaredReceiverTyping) -> DeclaredType:
        return declared.element(typing)


@dataclass(frozen=True)
class IndexStep(ReceiverStep):
    """`E[k]` with a constant integer k."""

    position: int

    def applied(self, declared: DeclaredType, typing: DeclaredReceiverTyping) -> DeclaredType:
        return declared.indexed(self.position)


@dataclass(frozen=True)
class SubscriptStep(ReceiverStep):
    """`E[k]` with any other key: a collection's element or a mapping's value."""

    def applied(self, declared: DeclaredType, typing: DeclaredReceiverTyping) -> DeclaredType:
        return declared.subscripted(typing)


@dataclass(frozen=True)
class SliceStep(ReceiverStep):
    """`E[i:j]`: a sequence of E's elements."""

    def applied(self, declared: DeclaredType, typing: DeclaredReceiverTyping) -> DeclaredType:
        return declared.sliced(typing)


@dataclass(frozen=True)
class CollectedStep(ReceiverStep):
    """`sorted(E)`, `list(E)`, `tuple(E)`, `set(E)`, `frozenset(E)`, `reversed(E)`,
    `iter(E)`: E's elements, in the class the call constructs."""

    container: ContainerClass

    def applied(self, declared: DeclaredType, typing: DeclaredReceiverTyping) -> DeclaredType:
        return DeclaredCollection(declared.element(typing), self.container)


@dataclass(frozen=True)
class EnumeratedStep(ReceiverStep):
    """`enumerate(E)`: pairs of a count and an element of E."""

    def applied(self, declared: DeclaredType, typing: DeclaredReceiverTyping) -> DeclaredType:
        return DeclaredCollection(
            DeclaredTuple((UndeclaredType("an enumerate count"), declared.element(typing))),
            container_of(enumerate),
        )


@dataclass(frozen=True)
class ReceiverPath:
    """A receiver's root and the steps taken from it, stored without AST."""

    root: ReceiverRoot
    steps: tuple[ReceiverStep, ...] = ()

    def then(self, step: ReceiverStep) -> ReceiverPath:
        return ReceiverPath(self.root, (*self.steps, step))

    def attribute(self, name: str) -> ReceiverPath:
        return self.root.attribute_of(self, name)

    def method_call(self, name: str) -> ReceiverPath:
        return self.root.method_call_of(self, name)

    def call(
        self,
        builder: ReceiverPathBuilder,
        arguments: Sequence[ast.expr],
        pending: Pending,
    ) -> ReceiverPath:
        return self.root.call_of(self, builder, arguments, pending)

    def declared(self, typing: DeclaredReceiverTyping) -> DeclaredType:
        declared = self.root.declared(typing)
        for step in self.steps:
            declared = step.applied(declared, typing)
        return declared


def unresolved(reason: str) -> ReceiverPath:
    return ReceiverPath(UnresolvedRoot(reason))


# --- building paths from a module ------------------------------------------------------


Pending = frozenset[tuple[int, str]]
PathThunk = Callable[[], ReceiverPath]
TUPLE_CONTAINER = container_of(tuple)
DICT_CONTAINER = container_of(dict)
DISPLAY_CLASSES: Mapping[type[ast.expr], type] = {
    ast.List: list,
    ast.ListComp: list,
    ast.Set: set,
    ast.SetComp: set,
    ast.Tuple: tuple,
    ast.GeneratorExp: types.GeneratorType,
}
"""The class each display or comprehension constructs."""
COLLECTING_BUILTINS = frozenset(
    ("sorted", "list", "tuple", "set", "frozenset", "reversed")
)
ENUMERATE_BUILTIN = "enumerate"
ELEMENT_BUILTINS = frozenset(("next",))
CAST_FUNCTIONS = frozenset(("typing.cast", "typing_extensions.cast"))


@dataclass
class ReceiverPathBuilder:
    """Builds receiver paths for the expressions of one module.

    A local is typed by its bindings in its own scope: a parameter's
    annotation (a method's first parameter is its class), an assignment's
    value, an annotated assignment's annotation, a loop or comprehension
    target's iterable element, destructured by position. A local bound more
    than once is declared only when every binding declares the same type;
    any binding this typing cannot read (augmented assignment, `with`,
    `except`, a match capture, an import, a nested definition, `del`,
    `global`/`nonlocal`) leaves it unresolved, with the reason.
    """

    module_name: str
    module: ast.Module
    aliases: Mapping[str, str]
    _bindings: dict[tuple[int, str], ReceiverPath] = field(default_factory=dict)

    @cached_property
    def parents(self) -> dict[int, ast.AST]:
        return {
            id(child): parent
            for parent in ast.walk(self.module)
            for child in ast.iter_child_nodes(parent)
        }

    @cached_property
    def module_names(self) -> frozenset[str]:
        """Names the module binds at top level: defs, classes, assignments, imports."""

        names: set[str] = set(self.aliases)
        for statement in self.module.body:
            match statement:
                case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(
                    name=name
                ) | ast.ClassDef(name=name):
                    names.add(name)
                case ast.Assign(targets=targets):
                    names.update(_target_names(targets))
                case ast.AnnAssign(target=ast.Name(id=name)):
                    names.add(name)
        return frozenset(names)

    @cached_property
    def class_qualnames(self) -> dict[int, str]:
        """Each class statement's dotted qualname within the module."""

        found: dict[int, str] = {}

        def walk(node: ast.AST, prefix: str) -> None:
            for child in ast.iter_child_nodes(node):
                match child:
                    case ast.ClassDef(name=name):
                        found[id(child)] = f"{prefix}{name}"
                        walk(child, f"{prefix}{name}.")
                    case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name):
                        walk(child, f"{prefix}{name}.<locals>.")
                    case _:
                        walk(child, prefix)

        walk(self.module, "")
        return found

    # expressions -------------------------------------------------------------

    def path_of(self, expression: ast.expr) -> ReceiverPath:
        if id(expression) not in self.parents:
            return unresolved(f"`{ast.unparse(expression)}` is not in this module's tree")
        return self._path(expression, frozenset())

    def _path(self, expression: ast.expr, pending: Pending) -> ReceiverPath:
        match expression:
            case ast.Name(id=name):
                return self._name(expression, name, pending)
            case ast.Attribute(value=value, attr=attribute):
                return self._path(value, pending).attribute(attribute)
            case ast.Call(func=ast.Attribute(value=value, attr=method)):
                return self._path(value, pending).method_call(method)
            case ast.Call(func=func, args=arguments):
                return self._path(func, pending).call(self, arguments, pending)
            case ast.Subscript(value=value, slice=key):
                base = self._path(value, pending)
                match key:
                    case ast.Constant(value=bool()):
                        return base.then(SubscriptStep())
                    case ast.Constant(value=int() as position):
                        return base.then(IndexStep(position))
                    case ast.UnaryOp(
                        op=ast.USub(), operand=ast.Constant(value=int() as position)
                    ):
                        return base.then(IndexStep(-position))
                    case ast.Slice():
                        return base.then(SliceStep())
                return base.then(SubscriptStep())
            case ast.Constant(value=None):
                return ReceiverPath(NoneRoot())
            case ast.Constant(value=value):
                return ReceiverPath(LiteralRoot(f"builtins.{type(value).__name__}"))
            case ast.JoinedStr():
                return ReceiverPath(LiteralRoot("builtins.str"))
            case ast.DictComp(key=key, value=value):
                return ReceiverPath(
                    MappingDisplayRoot(self._path(key, pending), self._path(value, pending))
                )
            case ast.IfExp(body=body, orelse=orelse):
                return ReceiverPath(
                    UnionRoot((self._path(body, pending), self._path(orelse, pending)))
                )
            case ast.Tuple(elts=items) if not any(
                isinstance(item, ast.Starred) for item in items
            ):
                return ReceiverPath(
                    TupleDisplayRoot(tuple(self._path(item, pending) for item in items))
                )
            case ast.List(elts=items) | ast.Set(elts=items) | ast.Tuple(elts=items) if items:
                return ReceiverPath(
                    CollectionDisplayRoot(
                        tuple(self._display_element(item, pending) for item in items),
                        container_of(DISPLAY_CLASSES[type(expression)]),
                    )
                )
            case ast.ListComp(elt=element) | ast.SetComp(elt=element) | ast.GeneratorExp(
                elt=element
            ):
                return ReceiverPath(
                    CollectionDisplayRoot(
                        (self._path(element, pending),),
                        container_of(DISPLAY_CLASSES[type(expression)]),
                    )
                )
        return unresolved(
            f"`{ast.unparse(expression)}` is a {type(expression).__name__} expression"
        )

    def _display_element(self, item: ast.expr, pending: Pending) -> ReceiverPath:
        """A display item's element type: a starred iterable contributes its elements."""

        match item:
            case ast.Starred(value=iterable):
                return self._path(iterable, pending).then(ElementStep())
        return self._path(item, pending)

    def module_call(
        self,
        parts: tuple[str, ...],
        arguments: Sequence[ast.expr],
        pending: Pending,
    ) -> ReceiverPath:
        """A call of a module-level or builtin name."""

        (name,) = parts
        if name not in self.module_names and name in COLLECTING_BUILTINS:
            if len(arguments) != 1:
                return unresolved(f"`{name}(...)` takes no single iterable")
            return self._path(arguments[0], pending).then(
                CollectedStep(container_of(vars(builtins)[name]))
            )
        if name not in self.module_names and name in ELEMENT_BUILTINS:
            match arguments:
                case [iterable] | [iterable, ast.Constant(value=None)]:
                    return self._path(iterable, pending).then(ElementStep())
            return unresolved(f"`{name}(...)` has a default of another type")
        if name not in self.module_names and name == "iter" and len(arguments) == 1:
            return self._path(arguments[0], pending).then(
                CollectedStep(container_of(vars(builtins)[name]))
            )
        if name not in self.module_names and name == ENUMERATE_BUILTIN:
            if not arguments:
                return unresolved("`enumerate()` has no iterable")
            return self._path(arguments[0], pending).then(EnumeratedStep())
        if self.aliases.get(name) in CAST_FUNCTIONS and len(arguments) == 2:
            return ReceiverPath(AnnotatedRoot(self.module_name, ast.unparse(arguments[0])))
        return ReceiverPath(CalledReferenceRoot(self.module_name, parts))

    # names -----------------------------------------------------------------

    def scopes(self, node: ast.AST) -> Iterator[ast.AST]:
        """The scopes `node` is evaluated in, innermost first, ending at the module.

        A comprehension's first iterable, a def's decorators, defaults and
        annotations, and a class's bases are evaluated in the enclosing scope;
        a class body is no scope for the functions inside it.
        """

        trail: list[ast.AST] = [node]
        current = self.parents.get(id(node))
        for _depth in range(len(self.parents) + 1):
            if current is None:
                break
            child = trail[-1]
            match current:
                case ast.ListComp() | ast.SetComp() | ast.DictComp() | ast.GeneratorExp():
                    first = current.generators[0]
                    grandchild = trail[-2] if len(trail) > 1 else child
                    if not (child is first and grandchild is first.iter):
                        yield current
                case ast.Lambda(body=body) if child is body:
                    yield current
                case ast.FunctionDef(body=body) | ast.AsyncFunctionDef(body=body) if any(
                    statement is child for statement in body
                ):
                    yield current
                case ast.ClassDef(body=body) if any(statement is child for statement in body):
                    yield current
            trail.append(current)
            current = self.parents.get(id(current))
        yield self.module

    def _name(self, node: ast.Name, name: str, pending: Pending) -> ReceiverPath:
        for scope in self.scopes(node):
            match scope:
                case ast.Module():
                    return ReceiverPath(ReferenceRoot(self.module_name, (name,)))
                case ast.Lambda(args=arguments):
                    if name in _parameter_names(arguments):
                        return unresolved(f"`{name}` is a lambda parameter")
                case ast.ListComp() | ast.SetComp() | ast.DictComp() | ast.GeneratorExp():
                    for generator in scope.generators:
                        found = self._target_path(
                            generator.target,
                            name,
                            _element_thunk(self, generator.iter, pending),
                        )
                        if found:
                            return found[0]
                case ast.FunctionDef() | ast.AsyncFunctionDef():
                    if name in _declared_global_or_nonlocal(scope):
                        return unresolved(f"`{name}` is declared global or nonlocal")
                    if name in _parameter_names(scope.args) or name in _bound_names(scope):
                        return self._function_binding(scope, name, pending)
        return unresolved(f"`{name}` is bound by no scope")

    def _function_binding(
        self, function: FunctionNode, name: str, pending: Pending
    ) -> ReceiverPath:
        key = (id(function), name)
        if key in pending:
            return unresolved(f"`{name}` is rebound from its own value")
        if key in self._bindings:
            return self._bindings[key]
        inner = pending | {key}
        paths = [
            *self._parameter_paths(function, name),
            *(
                path
                for node in _own_nodes(function)
                for path in self._binding_paths(function, node, name, inner)
            ),
        ]
        match paths:
            case []:
                path = unresolved(f"`{name}` has no binding this typing reads")
            case [only]:
                path = only
            case _:
                path = ReceiverPath(AgreeingRoot(name, tuple(paths)))
        if not pending:
            self._bindings[key] = path
        return path

    def _parameter_paths(self, function: FunctionNode, name: str) -> list[ReceiverPath]:
        arguments = function.args
        placement = self._placement(function)
        for index, parameter in enumerate(
            (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
        ):
            if parameter.arg != name:
                continue
            if index == 0:
                for own in placement.first_parameter(function):
                    return [own]
            return [
                self._annotated(function, annotation)
                for annotation in declared_annotations(parameter)
            ] or [unresolved(f"`{name}` declares no annotation")]
        if any(parameter.arg == name for parameter in variadic_parameters(arguments)):
            return [unresolved(f"`{name}` collects variadic arguments")]
        return []

    def _annotated(self, function: FunctionNode, annotation: ast.expr) -> ReceiverPath:
        return ReceiverPath(
            self._placement(function).annotated(self.module_name, ast.unparse(annotation))
        )

    def _placement(self, function: FunctionNode) -> FunctionPlacement:
        match self.parents.get(id(function)):
            case ast.ClassDef() as owner:
                return MethodPlacement(
                    f"{self.module_name}.{self.class_qualnames[id(owner)]}"
                )
        return FreeFunctionPlacement()

    def _binding_paths(
        self, function: FunctionNode, node: ast.AST, name: str, pending: Pending
    ) -> Iterator[ReceiverPath]:
        match node:
            case ast.Assign(targets=targets, value=value):
                for target in targets:
                    yield from self._target_path(
                        target, name, _value_thunk(self, value, pending)
                    )
            case ast.AnnAssign(target=ast.Name(id=bound), annotation=annotation) if (
                bound == name
            ):
                yield self._annotated(function, annotation)
            case ast.NamedExpr(target=ast.Name(id=bound), value=value) if bound == name:
                yield self._path(value, pending)
            case ast.For(target=target, iter=iterable) | ast.AsyncFor(
                target=target, iter=iterable
            ):
                yield from self._target_path(
                    target, name, _element_thunk(self, iterable, pending)
                )
            case ast.AugAssign(target=ast.Name(id=bound)) if bound == name:
                yield unresolved(f"`{name}` is rebound by augmented assignment")
            case ast.withitem(optional_vars=ast.expr() as target) if name in _target_names(
                [target]
            ):
                yield unresolved(f"`{name}` is bound by a `with` statement")
            case ast.ExceptHandler(name=str() as bound) if bound == name:
                yield unresolved(f"`{name}` is bound by an `except` clause")
            case ast.Import(names=aliases) | ast.ImportFrom(names=aliases) if any(
                (alias.asname or alias.name).partition(".")[0] == name for alias in aliases
            ):
                yield unresolved(f"`{name}` is bound by an import")
            case ast.FunctionDef(name=bound) | ast.AsyncFunctionDef(
                name=bound
            ) | ast.ClassDef(name=bound) if bound == name:
                yield unresolved(f"`{name}` is bound by a nested definition")
            case ast.Delete(targets=targets) if name in _target_names(targets):
                yield unresolved(f"`{name}` is deleted in its function")
            case ast.MatchAs(name=str() as bound) | ast.MatchStar(
                name=str() as bound
            ) if bound == name:
                yield unresolved(f"`{name}` is bound by a match pattern")
            case ast.MatchMapping(rest=str() as bound) if bound == name:
                yield unresolved(f"`{name}` is bound by a match pattern")

    def _target_path(
        self, target: ast.expr, name: str, value: PathThunk
    ) -> list[ReceiverPath]:
        """The path a binding target gives `name`, through tuple destructuring."""

        match target:
            case ast.Name(id=bound):
                return [value()] if bound == name else []
            case ast.Tuple(elts=items) | ast.List(elts=items):
                for index, item in enumerate(items):
                    match item:
                        case ast.Starred(value=starred):
                            if name in _target_names([starred]):
                                return [unresolved(f"`{name}` is a starred target")]
                            continue
                    found = self._target_path(item, name, _index_thunk(value, index))
                    if found and any(isinstance(other, ast.Starred) for other in items[:index]):
                        return [unresolved(f"`{name}` follows a starred target")]
                    if found:
                        return found
        return []


class FunctionPlacement(ABC):
    """Where a function is defined: what its first parameter and `Self` are."""

    @abstractmethod
    def first_parameter(self, function: FunctionNode) -> Iterator[ReceiverPath]:
        """The first parameter's path when the placement fixes it, else nothing."""
        raise NotImplementedError

    @abstractmethod
    def annotated(self, module_name: str, annotation: str) -> ReceiverRoot:
        raise NotImplementedError


@dataclass(frozen=True)
class MethodPlacement(FunctionPlacement):
    """A def in a class body: its first parameter is the instance, or the class for a classmethod."""

    class_symbol: str

    def first_parameter(self, function: FunctionNode) -> Iterator[ReceiverPath]:
        decorators = {_decorator_name(decorator) for decorator in function.decorator_list}
        if "staticmethod" in decorators:
            return
        if "classmethod" in decorators:
            yield ReceiverPath(OwnClassRoot(self.class_symbol))
            return
        yield ReceiverPath(OwnInstanceRoot(self.class_symbol))

    def annotated(self, module_name: str, annotation: str) -> ReceiverRoot:
        return MethodAnnotatedRoot(module_name, annotation, self.class_symbol)


class FreeFunctionPlacement(FunctionPlacement):
    """A def outside a class body: no parameter is fixed and `Self` names nothing."""

    def first_parameter(self, function: FunctionNode) -> Iterator[ReceiverPath]:
        return iter(())

    def annotated(self, module_name: str, annotation: str) -> ReceiverRoot:
        return AnnotatedRoot(module_name, annotation)


def _value_thunk(
    builder: ReceiverPathBuilder, value: ast.expr, pending: Pending
) -> PathThunk:
    return lambda: builder._path(value, pending)


def _element_thunk(
    builder: ReceiverPathBuilder, iterable: ast.expr, pending: Pending
) -> PathThunk:
    return lambda: builder._path(iterable, pending).then(ElementStep())


def _index_thunk(value: PathThunk, index: int) -> PathThunk:
    return lambda: value().then(IndexStep(index))


def _parameter_names(arguments: ast.arguments) -> frozenset[str]:
    return frozenset(
        parameter.arg
        for parameter in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            *variadic_parameters(arguments),
        )
    )


def _own_nodes(function: FunctionNode) -> Iterator[ast.AST]:
    """Nodes of a function's own scope: not inside nested defs, classes, lambdas or
    comprehensions, except a comprehension's first iterable and its walruses, which
    belong to the function."""

    stack: list[ast.AST] = list(function.body)
    for _visit in range(sum(1 for _ in ast.walk(function))):
        if not stack:
            break
        node = stack.pop()
        yield node
        match node:
            case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef():
                stack.extend(node.decorator_list)
            case ast.Lambda():
                pass
            case ast.ListComp() | ast.SetComp() | ast.DictComp() | ast.GeneratorExp():
                stack.extend(
                    walrus for walrus in ast.walk(node) if isinstance(walrus, ast.NamedExpr)
                )
                stack.append(node.generators[0].iter)
            case _:
                stack.extend(ast.iter_child_nodes(node))


def _bound_names(function: FunctionNode) -> frozenset[str]:
    names: set[str] = set()
    for node in _own_nodes(function):
        match node:
            case ast.Name(id=name, ctx=ast.Store() | ast.Del()):
                names.add(name)
            case ast.ExceptHandler(name=str() as name):
                names.add(name)
            case ast.Import(names=aliases) | ast.ImportFrom(names=aliases):
                names.update((alias.asname or alias.name).partition(".")[0] for alias in aliases)
            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.ClassDef(
                name=name
            ):
                names.add(name)
            case ast.MatchAs(name=str() as name) | ast.MatchStar(name=str() as name):
                names.add(name)
            case ast.MatchMapping(rest=str() as name):
                names.add(name)
    return frozenset(names)


def _declared_global_or_nonlocal(function: FunctionNode) -> frozenset[str]:
    return frozenset(
        name
        for node in _own_nodes(function)
        if isinstance(node, ast.Global | ast.Nonlocal)
        for name in node.names
    )


def _target_names(targets: Sequence[ast.expr]) -> frozenset[str]:
    return frozenset(
        node.id
        for target in targets
        for node in ast.walk(target)
        if isinstance(node, ast.Name)
    )


def _decorator_name(decorator: ast.expr) -> str:
    match decorator:
        case ast.Name(id=name):
            return name
        case ast.Attribute(attr=name):
            return name
        case ast.Call(func=func):
            return _decorator_name(func)
    return ast.unparse(decorator)


__all__ = (
    "AnnotatedRoot",
    "DeclaredClassObject",
    "DeclaredCollection",
    "DeclaredInstance",
    "DeclaredMapping",
    "DeclaredNone",
    "DeclaredReceiverTyping",
    "DeclaredTuple",
    "DeclaredType",
    "ReceiverPath",
    "ReceiverPathBuilder",
    "UndeclaredType",
)
