"""Native C3 derivation from source-proved, inert class declarations."""

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import (
    Generic,
    TYPE_CHECKING,
    TypeVar,
)

from .native_declarations import QualifiedDeclaration

if TYPE_CHECKING:
    from .class_index import CompactIndexedClass


class UnboundBaseReference(ValueError):
    """A base reference no declaration-time binding roots."""


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
        return next((member for member in cls if member.value == qualified_name), None)

    @classmethod
    def for_python_type(cls, python_type: type) -> NativeMroBase | None:
        return next(
            (member for member in cls if member.python_type is python_type), None
        )


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
        return tuple(
            owner.declaration
            for owner in self.__mro__
            if isinstance(owner, DeclarationMroType)
        )


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


@dataclass(frozen=True)
class ClassMroAuthority:
    """Lazily derive each required hierarchy from the authoritative class graph."""

    classes_by_symbol: Mapping[str, CompactIndexedClass]
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
            if not self._creation_is_static(declaration):
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

    @staticmethod
    def _creation_is_static(declaration: CompactIndexedClass) -> bool:
        return (
            declaration.mro_bases_are_static
            and not declaration.class_keyword_names
            and declaration.class_decorators_are_promotion_safe
            and not declaration.has_class_creation_hook
        )

    def _derive(self, symbol: str, pending: frozenset[str]) -> ClassMroResolution:
        declaration = self.classes_by_symbol.get(symbol)
        if declaration is None:
            return MissingDeclaration(symbol)
        if not self._creation_is_static(declaration):
            return DynamicClassDeclaration(symbol)
        native_bases = tuple(
            (
                None
                if reference.qualified_name in self.classes_by_symbol
                else NativeMroBase.for_qualified_name(reference.qualified_name)
            )
            for reference in declaration.base_references
        )
        if (
            not declaration.base_references_are_complete
            or len(declaration.resolved_base_symbols)
            != sum(base is None for base in native_bases)
        ):
            return UnresolvedBases(symbol)
        try:
            for reference in declaration.base_references:
                reference.require_bound()
        except UnboundBaseReference:
            return UnresolvedBases(symbol)
        source_bases = iter(declaration.resolved_base_symbols)
        bases = []
        for native_base in native_bases:
            if native_base is not None:
                bases.append(native_base.python_type)
                continue
            base_symbol = next(source_bases)
            resolution = self._resolve(base_symbol, pending)
            if resolution.mro_type is None:
                return resolution
            bases.append(resolution.mro_type)
        try:
            projected = DeclarationMroType.from_declaration(declaration, tuple(bases))
        except TypeError:
            return InconsistentHierarchy(symbol)
        return ResolvedClassMro(symbol, projected)
