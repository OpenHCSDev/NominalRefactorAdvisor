"""Native C3 derivation from source-proved, inert class declarations."""

from __future__ import annotations

import builtins
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache
from typing import (
    Generic,
    TYPE_CHECKING,
    TypeVar,
    cast,
)

from nominal_refactor_advisor.native_declarations import QualifiedDeclaration

if TYPE_CHECKING:
    from nominal_refactor_advisor.class_index import CompactIndexedClass


class NativeMroBase(StrEnum):
    """Standard terminal bases whose native construction has no repository hooks."""

    OBJECT = "builtins.object", object
    ABSTRACT_ROOT = "abc.ABC", ABC

    python_type: type

    def __new__(cls, qualified_name: str, python_type: type) -> NativeMroBase:
        member = str.__new__(cls, qualified_name)
        member._value_ = qualified_name
        member.python_type = python_type
        return member

    @classmethod
    def for_qualified_name(cls, qualified_name: str) -> NativeMroBase | None:
        return _NATIVE_MRO_BASE_BY_NAME.get(qualified_name)

    @classmethod
    def for_python_type(cls, python_type: type) -> NativeMroBase | None:
        for member in cls:
            if python_type is member.python_type:
                return member
        return None


def _pure_builtin_types() -> tuple[type, ...]:
    """Every pure-C `builtins` class (excluding `object`) whose entire MRO stays
    in `builtins` with metaclass `type`: Python exposes each class's exact C3
    bases and own `vars()` without running repository code."""

    seen: dict[type, None] = {}
    for value in vars(builtins).values():
        if (
            isinstance(value, type)
            and value is not object
            and value.__module__ == "builtins"
            and type(value) is type
            and all(
                ancestor.__module__ == "builtins" and type(ancestor) is type
                for ancestor in value.__mro__
            )
        ):
            seen[value] = None
    return tuple(seen)


NAMESPACE_READ_NATIVES: tuple[type, ...] = (Generic, *_pure_builtin_types())
"""Natively implemented terminal bases with no `.py` source to read: each one's
own namespace (`vars()`) names every member it binds, and its bases stay inside
`NAMESPACE_READ_NATIVES` and `object`. `Generic`'s creation hook reads just the
type parameters its subclass's `__orig_bases__` name, which a base declaring
none leaves as they were."""

_NATIVE_MRO_BASE_BY_NAME: dict[str, NativeMroBase] = {
    member.value: member for member in NativeMroBase
}
_NATIVE_MRO_BASE_BY_TYPE: dict[type, NativeMroBase] = {
    member.python_type: member for member in NativeMroBase
}
_NAMESPACE_READ_NATIVE_BY_NAME: dict[str, type] = {
    f"{native.__module__}.{native.__qualname__}": native
    for native in NAMESPACE_READ_NATIVES
}
_NATIVE_ROOT_TYPE_BY_NAME: dict[str, type] = {
    **{name: member.python_type for name, member in _NATIVE_MRO_BASE_BY_NAME.items()},
    **_NAMESPACE_READ_NATIVE_BY_NAME,
}


def namespace_read_native(qualified_name: str) -> type | None:
    return _NAMESPACE_READ_NATIVE_BY_NAME.get(qualified_name)


@dataclass(frozen=True)
class NativeStandIn:
    """The inert MRO entry standing for a namespace-read native, one per native:
    a real `Generic` base would run its creation hook, which refuses a plain base.
    A lookup learns nothing past it for the names the native binds."""

    entry: type
    names: frozenset[str]

    @staticmethod
    @cache
    def for_qualified_name(qualified_name: str) -> NativeStandIn | None:
        native = _NAMESPACE_READ_NATIVE_BY_NAME.get(qualified_name)
        if native is None:
            return None
        stand_in_bases = tuple(
            base_stand_in.entry
            for base in native.__bases__
            if base is not object
            and (
                base_stand_in := NativeStandIn.for_qualified_name(
                    f"{base.__module__}.{base.__qualname__}"
                )
            )
            is not None
        )
        return NativeStandIn(
            type(qualified_name, stand_in_bases, {}), native_declared_names(native)
        )

    @staticmethod
    def stops(entry: type, member_name: str) -> bool | None:
        """Whether a lookup of `member_name` stops at `entry`, or None when `entry`
        stands for no native."""

        stand_in = NativeStandIn.for_qualified_name(entry.__name__)
        if stand_in is None or stand_in.entry is not entry:
            return None
        return member_name in stand_in.names


def native_root_type(qualified_name: str) -> type | None:
    """The native class a terminal base names: a `NativeMroBase` root or a
    namespace-read native."""

    return _NATIVE_ROOT_TYPE_BY_NAME.get(qualified_name)


def native_declared_names(native: type) -> frozenset[str]:
    """What a native's own namespace binds besides what type creation gives
    every class (`__module__`, `__doc__`, `__dict__`, `__weakref__`, ...)."""

    return frozenset(vars(native)) - frozenset(vars(type("Plain", (), {})))


MroDeclarationT = TypeVar("MroDeclarationT", bound=QualifiedDeclaration)


class DeclarationMroType(ABCMeta, Generic[MroDeclarationT]):
    """An inert class carrying its source declaration; Python owns its C3 order."""

    declaration: MroDeclarationT

    @classmethod
    def from_declaration(
        cls, declaration: MroDeclarationT, bases: tuple[type, ...]
    ) -> DeclarationMroType[MroDeclarationT]:
        return cls(declaration.qualified_name, bases, {"declaration": declaration})

    @property
    def declarations(self) -> tuple[MroDeclarationT, ...]:
        return self.declarations_before(lambda entry: False)

    def declarations_before(
        self, stops: Callable[[type], bool]
    ) -> tuple[MroDeclarationT, ...]:
        """The declarations along this MRO up to the first entry `stops` marks."""

        found: list[MroDeclarationT] = []
        for owner in self.__mro__:
            if stops(owner):
                break
            if isinstance(owner, DeclarationMroType):
                found.append(owner.declaration)
        return tuple(found)


@dataclass(frozen=True)
class ClassMroResolution(ABC):
    class_symbol: str

    @property
    @abstractmethod
    def mro_type(self) -> DeclarationMroType[CompactIndexedClass] | None:
        raise NotImplementedError

    @abstractmethod
    def known_prefix(
        self, authority: ClassMroAuthority, symbol: str
    ) -> tuple[CompactIndexedClass, ...]:
        """The declarations of ``symbol``'s MRO known before any opaque base."""


@dataclass(frozen=True)
class ResolvedClassMro(ClassMroResolution):
    declaration_type: DeclarationMroType[CompactIndexedClass]

    @property
    def mro_type(self) -> DeclarationMroType[CompactIndexedClass]:
        return self.declaration_type

    def known_prefix(
        self, authority: ClassMroAuthority, symbol: str
    ) -> tuple[CompactIndexedClass, ...]:
        return self.declaration_type.declarations


@dataclass(frozen=True)
class OpenClassMro(ClassMroResolution, ABC):
    """A class whose MRO stays open; each subclass is why. ``class_symbol`` is the class
    where it opened: the requested class or one of its ancestors."""

    @property
    def mro_type(self) -> None:
        return None

    def known_prefix(
        self, authority: ClassMroAuthority, symbol: str
    ) -> tuple[CompactIndexedClass, ...]:
        return authority.single_base_prefix(symbol)


class MissingDeclaration(OpenClassMro):
    """No declaration of the class is indexed."""


class UnresolvedBases(OpenClassMro):
    """A base reference names no indexed class."""


class DynamicClassDeclaration(OpenClassMro):
    """Class creation runs code: dynamic bases, keywords, a decorator or a creation hook."""


class HierarchyViolation(OpenClassMro):
    """The hierarchy itself admits no MRO, so no walk reads declarations past it."""

    def known_prefix(
        self, authority: ClassMroAuthority, symbol: str
    ) -> tuple[CompactIndexedClass, ...]:
        return ()


class CyclicHierarchy(HierarchyViolation):
    """A class is among its own ancestors."""


class InconsistentHierarchy(HierarchyViolation):
    """C3 linearization finds no consistent order."""


class MroBasePolicy(ABC):
    """What a hierarchy needs proved before Python's C3 orders it."""

    @abstractmethod
    def creation_is_static(self, declaration: CompactIndexedClass) -> bool:
        raise NotImplementedError

    def creation_may_keep_order(self, declaration: CompactIndexedClass) -> bool:
        """Whether creation that runs code could still be proved to keep the order
        C3 gives the declared bases: only code a proof reads, never a computed
        base or a decorator that may replace the class."""
        return False

    def creation_keeps_order(
        self, declaration: CompactIndexedClass, ancestry: tuple[CompactIndexedClass, ...]
    ) -> bool:
        """Whether the code creating `declaration`, whose declared order is
        `ancestry`, is proved to pass its bases on unchanged."""
        return False

    def foreign_base(self, qualified_name: str) -> type | None:
        """The type standing for a base the scan does not declare, or None when it
        has none: a native root itself, a namespace-read native by its stand-in."""
        native = NativeMroBase.for_qualified_name(qualified_name)
        if native is not None:
            return native.python_type
        stand_in = NativeStandIn.for_qualified_name(qualified_name)
        return None if stand_in is None else stand_in.entry

    def is_opaque(self, entry: type) -> bool:
        """Whether an MRO entry stands for a base whose members nothing declares."""
        return False

    def stops_lookup(self, entry: type, member_name: str) -> bool:
        """Whether a lookup of `member_name` learns nothing past this entry."""
        native = NativeStandIn.stops(entry, member_name)
        return self.is_opaque(entry) if native is None else native


class RuntimeProofBases(MroBasePolicy):
    """The MRO Python builds at runtime: no scanned class runs code at creation,
    and every base is known. A namespace-read native stands in inert: its own
    order is itself then `object`, so C3 places the stand-in where it places the
    native, and its creation hook binds no member a lookup reads."""

    def creation_is_static(self, declaration: CompactIndexedClass) -> bool:
        return (
            declaration.mro_bases_are_static
            and not declaration.class_keyword_names
            and declaration.class_decorators_are_promotion_safe
            and not declaration.has_class_creation_hook
        )


@dataclass(frozen=True)
class ProvedCreationBases(RuntimeProofBases):
    """The runtime order, also for a class whose creation runs code a proof reads.

    Python builds the order from the bases a metaclass's `__new__` hands to
    `type.__new__`, so creation code that passes them on unchanged, defines no
    `mro` and assigns no `__bases__` leaves the order C3 gives the declared
    bases. `keeps_order` is that proof's answer (`class_index.ClassCreationOrderProof`).
    """

    keeps_order: Callable[[CompactIndexedClass, tuple[CompactIndexedClass, ...]], bool]

    def creation_may_keep_order(self, declaration: CompactIndexedClass) -> bool:
        return declaration.mro_bases_are_static and declaration.class_decorators_are_promotion_safe

    def creation_keeps_order(
        self, declaration: CompactIndexedClass, ancestry: tuple[CompactIndexedClass, ...]
    ) -> bool:
        return self.keeps_order(declaration, ancestry)


@dataclass
class DeclaredBases(MroBasePolicy):
    """The MRO the declarations state, for reading declared member types.

    Class creation hooks, metaclasses and subscripted bases (`Base[T]`) do not
    change which class declares a member, so they do not open the order. A
    base the scan does not declare is read from its installed source when
    `installed` finds it and its own bases resolve the same way; any other
    stands in as an opaque placeholder: its members are unknown, so a lookup
    that reaches it learns nothing past it.
    """

    installed: Callable[[str], CompactIndexedClass | None] = lambda qualified_name: None
    """The class an installed module outside the scan declares under a name."""
    _placeholders: dict[str, type] = field(default_factory=dict)
    _installed: dict[str, type | None] = field(default_factory=dict)

    def creation_is_static(self, declaration: CompactIndexedClass) -> bool:
        return True

    def foreign_base(self, qualified_name: str) -> type | None:
        return (
            super().foreign_base(qualified_name)
            or self.installed_base(qualified_name)
            or self._placeholders.setdefault(qualified_name, type(qualified_name, (), {}))
        )

    def installed_base(self, qualified_name: str) -> type | None:
        """The installed class, as an MRO entry carrying its declaration, once its
        whole ancestry resolves; a cycle or an unknown base leaves it opaque."""

        if qualified_name in self._installed:
            return self._installed[qualified_name]
        self._installed[qualified_name] = None
        declaration = self.installed(qualified_name)
        if declaration is None or not declaration.base_references_are_complete:
            return None
        bases = tuple(
            self.foreign_base(reference.qualified_name) for reference in declaration.base_references
        )
        if any(self.is_opaque(base) for base in bases if base is not None) or None in bases:
            return None
        try:
            entry = DeclarationMroType.from_declaration(declaration, cast(tuple[type, ...], bases))
        except TypeError:
            return None
        self._installed[qualified_name] = entry
        return entry

    def is_opaque(self, entry: type) -> bool:
        return self._placeholders.get(entry.__name__) is entry


@dataclass(frozen=True)
class ClassMroAuthority:
    """Lazily derive each required hierarchy from the authoritative class graph."""

    classes_by_symbol: Mapping[str, CompactIndexedClass]
    bases: MroBasePolicy = field(default_factory=RuntimeProofBases)
    _resolutions: dict[str, ClassMroResolution] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )


    def resolve(self, symbol: str) -> ClassMroResolution:
        return self._resolve(symbol, frozenset())

    def _resolve(self, symbol: str, pending: frozenset[str]) -> ClassMroResolution:
        if symbol in self._resolutions:
            return self._resolutions[symbol]
        if symbol in pending:
            return CyclicHierarchy(symbol)
        resolution = self._derive(symbol, pending | {symbol})
        self._resolutions[symbol] = resolution
        return resolution

    def lookup_prefix(self, symbol: str) -> tuple[CompactIndexedClass, ...]:
        """Known declaration prefix before an opaque or branching base.

        A single direct base guarantees the next declaration even when a later
        ancestor prevents closing the full C3 order. Never guess past that point.
        This does not strengthen an OpenClassMro into a runtime proof.
        """
        return self.resolve(symbol).known_prefix(self, symbol)

    def single_base_prefix(self, symbol: str) -> tuple[CompactIndexedClass, ...]:
        """The declarations reached from ``symbol`` through single static bases."""
        prefix = []
        pending = set()
        while symbol not in pending:
            pending.add(symbol)
            declaration = self.classes_by_symbol.get(symbol)
            if declaration is None:
                break
            prefix.append(declaration)
            if not self.bases.creation_is_static(declaration):
                break
            if not declaration.base_references_are_complete:
                break
            if (
                declaration.direct_base_count != 1
                or len(declaration.resolved_base_symbols) != 1
            ):
                break
            symbol = declaration.resolved_base_symbols[0]
        return tuple(prefix)


    def _derive(self, symbol: str, pending: frozenset[str]) -> ClassMroResolution:
        declaration = self.classes_by_symbol.get(symbol)
        if declaration is None:
            return MissingDeclaration(symbol)
        static = self.bases.creation_is_static(declaration)
        if not static and not self.bases.creation_may_keep_order(declaration):
            return DynamicClassDeclaration(symbol)
        native_bases = tuple(
            (
                None
                if reference.qualified_name in self.classes_by_symbol
                else self.bases.foreign_base(reference.qualified_name)
            )
            for reference in declaration.base_references
        )
        if (
            not declaration.base_references_are_complete
            or len(declaration.resolved_base_symbols)
            != sum(base is None for base in native_bases)
        ):
            return UnresolvedBases(symbol)
        if declaration.base_bindings.unbound:
            return UnresolvedBases(symbol)
        source_bases = iter(declaration.resolved_base_symbols)
        bases = []
        for native_base in native_bases:
            if native_base is not None:
                bases.append(native_base)
                continue
            base_symbol = next(source_bases)
            resolution = self._resolve(base_symbol, pending)
            if resolution.mro_type is None:
                return resolution
            bases.append(resolution.mro_type)
        for qualified_name in declaration.implicit_base_names:
            implicit = self.bases.foreign_base(qualified_name)
            if implicit is None:
                return UnresolvedBases(symbol)
            bases.append(implicit)
        try:
            projected = DeclarationMroType.from_declaration(declaration, tuple(bases))
        except TypeError:
            return InconsistentHierarchy(symbol)
        if not static and not self.bases.creation_keeps_order(declaration, projected.declarations):
            return DynamicClassDeclaration(symbol)
        return ResolvedClassMro(symbol, projected)
