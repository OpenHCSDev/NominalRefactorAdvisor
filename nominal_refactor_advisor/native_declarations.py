"""Identity and source derived from a native Python declaration."""

from __future__ import annotations

import ast
import __future__
import builtins
import types
import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import (
    dataclass,
    field,
)
from enum import StrEnum
from functools import cached_property, lru_cache
import inspect
from textwrap import dedent
from types import (
    FunctionType,
    ModuleType,
)
from typing import (
    ClassVar,
    Self,
    TYPE_CHECKING,
    TypeAlias,
    cast,
    get_args,
)

from .semantic_match import loaded_concrete_nominal_descendants


class DataclassRuntimeDeclaration(StrEnum):
    """Standard-library dataclass declarations with nominal qualified identity."""

    DATACLASS = (dataclasses.dataclass, True, False)
    FIELD = (dataclasses.field, False, True)

    declaration: Callable[..., object]

    def __new__(
        cls,
        declaration: Callable[..., object],
        is_dataclass_decorator: bool,
        is_field_factory: bool,
    ) -> Self:
        name = declaration.__name__
        member = str.__new__(cls, name)
        member._value_ = name
        member.declaration = declaration
        member._is_dataclass_decorator = is_dataclass_decorator
        member._is_field_factory = is_field_factory
        return member

    @cached_property
    def native_declaration(self) -> NativeDeclaration:
        return NativeDeclaration(self.declaration)

    @property
    def qualified_name(self) -> str:
        return self.native_declaration.qualified_name

    @property
    def is_dataclass_decorator(self) -> bool:
        return self._is_dataclass_decorator

    @property
    def is_field_factory(self) -> bool:
        return self._is_field_factory

    def matches(self, qualified_name: str | None) -> bool:
        return qualified_name == self.qualified_name

    def matches_reference_name(self, reference_name: str | None) -> bool:
        return reference_name in (self.value, self.qualified_name)

    @classmethod
    def for_qualified_name(cls, qualified_name: str | None) -> Self | None:
        return next(
            (member for member in cls if member.matches(qualified_name)),
            None,
        )

    @classmethod
    def for_reference_name(cls, reference_name: str | None) -> Self | None:
        return next(
            (member for member in cls if member.matches_reference_name(reference_name)),
            None,
        )

    @classmethod
    def dataclass_decorator_for_name(cls, reference_name: str | None) -> Self | None:
        """Resolve a standard dataclass decorator from either source spelling."""

        return next(
            (
                member
                for member in cls
                if member.is_dataclass_decorator
                and member.matches_reference_name(reference_name)
            ),
            None,
        )


if TYPE_CHECKING:
    from .captured_reference import CapturedReferenceResolution


NativeScalar: TypeAlias = str | int | bool | None
NativeDictionaryKey: TypeAlias = NativeScalar | type
NativeConstantAtom: TypeAlias = NativeScalar | types.EllipsisType


class NativeConstantContentsABC(ABC):
    """Exact primitive, ellipsis and tuple contents, independent of object identity.

    Comparison consumes only declared exact atoms and exact tuples. It never
    invokes equality, iteration or conversion on an unadmitted object.
    """

    @staticmethod
    def supports_scalar(value: object) -> bool:
        return any(type(value) is declaration for declaration in get_args(NativeScalar))

    def _native_constant_value(self) -> object:
        raise ValueError("Native constant contents remain unproved")

    @staticmethod
    def _require_same_constant(left: object, right: object) -> None:
        pending = [(left, right)]
        examined: set[tuple[int, int]] = set()
        while pending:
            actual, expected = pending.pop()
            pair = id(actual), id(expected)
            if pair in examined:
                continue
            examined.add(pair)
            if type(actual) is not type(expected):
                raise ValueError("Native constant types differ")
            if type(actual) is tuple:
                if len(actual) != len(expected):
                    raise ValueError("Native constant tuple lengths differ")
                pending.extend(zip(actual, expected, strict=True))
                continue
            if not any(
                type(actual) is declaration
                for declaration in get_args(NativeConstantAtom)
            ):
                raise ValueError("Native constant requires admitted exact contents")
            if actual != expected:
                raise ValueError("Native constant contents differ")

    @classmethod
    def supports_constant(cls, value: object) -> bool:
        try:
            cls._require_same_constant(value, value)
        except ValueError:
            return False
        return True

    def require_constant_contents(self, expected: object) -> None:
        self._require_same_constant(self._native_constant_value(), expected)

    def require_same_native_constant(self, other: NativeConstantContentsABC) -> None:
        self.require_constant_contents(other._native_constant_value())


@dataclass(frozen=True, eq=False)
class NativeParameterDefault(NativeConstantContentsABC):
    """One observed native default association, not source syntax or an effect proof.

    Each query reads current function storage. Values retain original identity;
    no cache, annotations, signature override, body execution or equality callback
    participates in collecting them.
    """

    parameter_name: str
    value: object = field(repr=False)

    def _native_constant_value(self) -> object:
        return self.value

    @classmethod
    def from_function(cls, function: FunctionType) -> tuple[Self, ...]:
        if type(function) is not FunctionType:
            raise ValueError("Native defaults require an exact function")
        code = function.__code__
        positional_storage = function.__defaults__
        keyword_storage = function.__kwdefaults__
        positional = (
            ()
            if positional_storage is None
            else tuple.__getitem__(positional_storage, slice(None))
        )
        keyword_items = (
            () if keyword_storage is None else tuple(dict.items(keyword_storage))
        )
        if any(type(name) is not str for name, _ in keyword_items):
            raise ValueError("Native keyword default keys require exact string storage")
        names = code.co_varnames[: code.co_argcount]
        keyword_names = frozenset(
            code.co_varnames[
                code.co_argcount : code.co_argcount + code.co_kwonlyargcount
            ]
        )
        return tuple(
            cls(name, value)
            for name, value in zip(
                names[max(0, len(names) - len(positional)) :],
                positional[max(0, len(positional) - len(names)) :],
                strict=True,
            )
        ) + tuple(
            cls(name, value) for name, value in keyword_items if name in keyword_names
        )


class NativeScalarValueABC(NativeConstantContentsABC):
    """Exact primitive contents, independently of analyzed object identity.

    The declared domain excludes subclasses, mutable values and compound or
    floating-point values whose observations need separate native laws.
    Type-only evidence supplies no contents. Queries never coerce values or
    execute their equality, hashing, conversion or descriptor protocols.
    """

    def _native_constant_value(self) -> object:
        return self.require_native_scalar()

    def _native_scalar_value(self) -> object:
        raise ValueError("Native scalar contents remain unproved")

    def require_native_scalar(self) -> NativeScalar:
        value = self._native_scalar_value()
        if not self.supports_scalar(value):
            raise ValueError("Native scalar requires an admitted exact primitive")
        return cast(NativeScalar, value)

    def require_native_text(self) -> str:
        value = self.require_native_scalar()
        if type(value) is not str:
            raise ValueError("Native text requires an exact Unicode value")
        return cast(str, value)


class QualifiedDeclaration(ABC):
    """A declaration with a qualified source name, independent of representation."""

    @property
    @abstractmethod
    def qualified_name(self) -> str:
        raise NotImplementedError


class ClassNamespaceDeclaration(QualifiedDeclaration):
    """Names whose class-level binding must be accounted for in member lookup."""

    @property
    @abstractmethod
    def member_binding_names(self) -> frozenset[str]:
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class NativeLanguageFeature(QualifiedDeclaration):
    """An actual Python language-feature export, not source-name authentication."""

    declaration: __future__._Feature
    module: ClassVar[ModuleType] = __future__

    @cached_property
    def export_name(self) -> str:
        namespace = vars(self.module)
        matches = tuple(
            name
            for name in self.module.all_feature_names
            if namespace[name] is self.declaration
        )
        if len(matches) != 1:
            raise ValueError("Native language feature has no unique declared export")
        return matches[0]

    @property
    def qualified_name(self) -> str:
        return f"{self.module.__name__}.{self.export_name}"

    @property
    def import_source(self) -> str:
        return f"from {self.module.__name__} import {self.export_name}\n"

    def imported_by(self, module: ast.Module) -> bool:
        """Select syntax policy only; runtime import still requires source admission."""
        return any(
            isinstance(statement, ast.ImportFrom)
            and statement.level == 0
            and statement.module == self.module.__name__
            and any(alias.name == self.export_name for alias in statement.names)
            for statement in module.body
        )


@dataclass(frozen=True, eq=False)
class NativeDeclaration(QualifiedDeclaration):
    """Keep native identity and lazily inspected source on one declaration."""

    declaration: type | Callable[..., object]

    def __hash__(self) -> int:
        return id(self.declaration)

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self.declaration is cast(NativeDeclaration, other).declaration

    @property
    def qualified_name(self) -> str:
        module = self.declaration.__module__
        qualname = self.declaration.__qualname__
        if type(module) is not str or type(qualname) is not str:
            raise ValueError("Native declaration qualification requires exact text")
        return f"{module}.{qualname}"

    @property
    @lru_cache(maxsize=None)
    def node(self) -> ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef:
        try:
            source = inspect.getsource(self.declaration)
        except (OSError, TypeError) as error:
            raise ValueError("Native declaration has no inspectable source") from error
        statements = ast.parse(dedent(source)).body
        if len(statements) != 1 or not isinstance(
            statements[0], (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            raise ValueError("Native source does not identify one declaration")
        return statements[0]

    def require_source_matches(self, node: ast.AST) -> None:
        if ast.dump(self.node, include_attributes=False) != ast.dump(
            node, include_attributes=False
        ):
            raise ValueError(
                f"Source does not match native declaration {self.qualified_name!r}"
            )


class NativeDeclarationFamily(ABC):
    """Select a native protocol by captured identity and nominal refinement."""

    native_declarations: ClassVar[tuple[NativeDeclaration, ...]]

    @classmethod
    def select_from_capture(cls, captured: CapturedReferenceResolution) -> type[Self]:
        authorities = loaded_concrete_nominal_descendants(cls)
        native = captured.require_native(
            tuple(
                declaration
                for authority in authorities
                for declaration in authority.native_declarations
            )
        )
        matches = tuple(
            authority
            for authority in authorities
            if native in authority.native_declarations
        )
        refinements = tuple(
            authority
            for authority in matches
            if not any(authority in other.__mro__[1:] for other in matches)
        )
        if len(refinements) != 1:
            raise ValueError(
                "Native operation has no unique most-specific declaration-owned protocol"
            )
        return refinements[0]


@dataclass(frozen=True, eq=False)
class NativeTypeDeclaration(NativeDeclaration):
    """An actual native type serialized through its standard-library export.

    Some native types advertise a builtins name that is not exported there.
    Export selection is derived by object identity; no type-name table is kept.
    This transport relation does not identify an analyzed-program object.
    """

    declaration: type
    native_modules: ClassVar[tuple[ModuleType, ...]] = (builtins, types)

    @property
    @lru_cache(maxsize=None)
    def qualified_name(self) -> str:
        exports = tuple(
            f"{module.__name__}.{name}"
            for module in self.native_modules
            for name, value in vars(module).items()
            if value is self.declaration
        )
        if not exports:
            raise ValueError("Native type has no proved standard-library export")
        return min(exports)

    @classmethod
    def from_qualified_name(cls, qualified_name: str) -> NativeTypeDeclaration:
        module_name, _, name = qualified_name.rpartition(".")
        module = next(
            module for module in cls.native_modules if module.__name__ == module_name
        )
        declaration = vars(module)[name]
        if type(declaration) is not type:
            raise ValueError("Native export does not identify an exact type")
        return cls(declaration)

    def __reduce__(self) -> tuple[Callable[[str], NativeTypeDeclaration], tuple[str]]:
        return type(self).from_qualified_name, (self.qualified_name,)
