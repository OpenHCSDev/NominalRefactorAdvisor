"""Native C3 derivation from source-proved, inert class declarations."""

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .class_index import CompactIndexedClass


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


class DeclarationMroType(ABCMeta):
    """An inert class carrying its source declaration; Python owns its C3 order."""

    declaration: CompactIndexedClass

    @classmethod
    def from_declaration(
        cls, declaration: CompactIndexedClass, bases: tuple[type, ...]
    ) -> DeclarationMroType:
        return cls(declaration.symbol, bases, {"declaration": declaration})

    @property
    def declarations(self) -> tuple[CompactIndexedClass, ...]:
        return tuple(
            owner.declaration
            for owner in self.__mro__
            if isinstance(owner, DeclarationMroType)
        )


class ClassMroViolation(StrEnum):
    MISSING_DECLARATION = "missing_declaration"
    UNRESOLVED_BASES = "unresolved_bases"
    DYNAMIC_CLASS_DECLARATION = "dynamic_class_declaration"
    CYCLIC_HIERARCHY = "cyclic_hierarchy"
    INCONSISTENT_HIERARCHY = "inconsistent_hierarchy"


@dataclass(frozen=True)
class ClassMroResolution(ABC):
    class_symbol: str

    @property
    @abstractmethod
    def mro_type(self) -> DeclarationMroType | None:
        raise NotImplementedError


@dataclass(frozen=True)
class ResolvedClassMro(ClassMroResolution):
    declaration_type: DeclarationMroType

    @property
    def mro_type(self) -> DeclarationMroType:
        return self.declaration_type


@dataclass(frozen=True)
class OpenClassMro(ClassMroResolution):
    violation: ClassMroViolation

    @property
    def mro_type(self) -> None:
        return None


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
            return OpenClassMro(symbol, ClassMroViolation.CYCLIC_HIERARCHY)
        resolution = self._derive(symbol, pending | {symbol})
        self._resolutions[symbol] = resolution
        return resolution

    def _derive(self, symbol: str, pending: frozenset[str]) -> ClassMroResolution:
        declaration = self.classes_by_symbol.get(symbol)
        if declaration is None:
            return OpenClassMro(symbol, ClassMroViolation.MISSING_DECLARATION)
        if (
            not declaration.mro_bases_are_static
            or declaration.class_keyword_names
            or not declaration.class_decorators_are_promotion_safe
            or declaration.has_class_creation_hook
        ):
            return OpenClassMro(symbol, ClassMroViolation.DYNAMIC_CLASS_DECLARATION)
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
            or any(
                reference.root_binding is None
                for reference in declaration.base_references
            )
        ):
            return OpenClassMro(symbol, ClassMroViolation.UNRESOLVED_BASES)
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
            return OpenClassMro(symbol, ClassMroViolation.INCONSISTENT_HIERARCHY)
        return ResolvedClassMro(symbol, projected)
