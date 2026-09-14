"""Nominal authority for standard-library enum declaration families."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum, Flag, IntEnum, IntFlag, StrEnum
from functools import cached_property
from typing import Callable, Self

from .value_expression import LiteralExpressionEffects

QualifiedReferenceResolver = Callable[[ast.expr, frozenset[str]], str | None]


class PythonEnumBaseKind(StrEnum):
    """Exact standard-library enum bases with member-owned native behavior."""

    ENUM = "enum", Enum
    FLAG = "flag", Flag
    INT_ENUM = "int_enum", IntEnum
    INT_FLAG = "int_flag", IntFlag
    STR_ENUM = "str_enum", StrEnum

    native_type: type[Enum]

    def __new__(cls, value: str, native_type: type[Enum]) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.native_type = native_type
        return member

    @property
    def qualified_name(self) -> str:
        return f"{self.native_type.__module__}.{self.native_type.__qualname__}"

    @classmethod
    def for_qualified_name(cls, qualified_name: str) -> Self:
        try:
            return next(
                member for member in cls if member.qualified_name == qualified_name
            )
        except StopIteration as error:
            raise ValueError(
                f"{qualified_name!r} is not an exact standard-library enum base"
            ) from error


class PythonEnumMemberValueABC(ABC):
    """One closed source value used by standard Enum class construction."""

    @abstractmethod
    def native_value(
        self,
        base: PythonEnumBaseKind,
        name: str,
        count: int,
        previous_values: tuple[object, ...],
        values_by_name: Mapping[str, object],
    ) -> object:
        raise NotImplementedError


@dataclass(frozen=True)
class PythonEnumLiteralMemberValue(PythonEnumMemberValueABC):
    """An exact inert literal member value."""

    value: Hashable

    def native_value(
        self,
        base: PythonEnumBaseKind,
        name: str,
        count: int,
        previous_values: tuple[object, ...],
        values_by_name: Mapping[str, object],
    ) -> Hashable:
        del base, name, count, previous_values, values_by_name
        return self.value


class PythonEnumAutoMemberValue(PythonEnumMemberValueABC):
    """A standard ``enum.auto`` call under the exact inherited generator."""

    def native_value(
        self,
        base: PythonEnumBaseKind,
        name: str,
        count: int,
        previous_values: tuple[object, ...],
        values_by_name: Mapping[str, object],
    ) -> object:
        del values_by_name
        return base.native_type._generate_next_value_(
            name,
            1,
            count,
            list(previous_values),
        )


@dataclass(frozen=True)
class PythonEnumAliasMemberValue(PythonEnumMemberValueABC):
    """A direct reference to one preceding member in the same class body."""

    member_name: str

    def native_value(
        self,
        base: PythonEnumBaseKind,
        name: str,
        count: int,
        previous_values: tuple[object, ...],
        values_by_name: Mapping[str, object],
    ) -> object:
        del base, name, count, previous_values
        try:
            return values_by_name[self.member_name]
        except KeyError as error:
            raise ValueError(
                f"Enum alias {self.member_name!r} has no preceding member"
            ) from error


@dataclass(frozen=True)
class PythonEnumMemberDeclaration:
    """One ordered enum-member declaration and its closed value proof."""

    name: str
    value: PythonEnumMemberValueABC


@dataclass(frozen=True)
class PythonEnumDeclarationAuthority:
    """Native mapping-key equality derived from one closed standard Enum declaration."""

    identity: str
    base: PythonEnumBaseKind
    members: tuple[PythonEnumMemberDeclaration, ...]

    @cached_property
    def native_members_by_name(self) -> Mapping[str, Enum]:
        values_by_name: dict[str, object] = {}
        previous_values: list[object] = []
        for count, member in enumerate(self.members):
            value = member.value.native_value(
                self.base,
                member.name,
                count,
                tuple(previous_values),
                values_by_name,
            )
            values_by_name[member.name] = value
            previous_values.append(value)
        enum_type = self.base.native_type(
            f"_NRA_{self.identity.replace('.', '_')}",
            values_by_name,
        )
        members = enum_type.__members__
        if tuple(members) != tuple(member.name for member in self.members):
            raise ValueError("Enum construction did not preserve declared member names")
        for member in members.values():
            hash(member)
        return members

    def require_mapping_key(self, member_name: str) -> Hashable:
        try:
            return self.native_members_by_name[member_name]
        except KeyError as error:
            raise ValueError(
                f"Enum declaration has no member {member_name!r}"
            ) from error


@dataclass(frozen=True)
class PythonEnumBaseAuthority:
    """Recognize enum bases from their resolved or source-level names."""

    base_names: frozenset[str] = frozenset(
        member.native_type.__name__ for member in PythonEnumBaseKind
    )
    inherited_member_names: frozenset[str] = frozenset(
        member_name
        for enum_type in (Enum, Flag, IntEnum, IntFlag, StrEnum)
        for ancestor_type in enum_type.__mro__
        for member_name in vars(ancestor_type)
    )

    def matches(self, base_name: str | None) -> bool:
        return (
            base_name is not None
            and base_name.rsplit(".", maxsplit=1)[-1] in self.base_names
        )

    def matches_any(self, base_names: Iterable[str | None]) -> bool:
        return any(self.matches(base_name) for base_name in base_names)

    def matches_qualified(self, qualified_name: str | None) -> bool:
        """Recognize only declarations resolved to the standard enum module."""

        if qualified_name is None:
            return False
        module_name, separator, base_name = qualified_name.rpartition(".")
        return (
            separator == "." and module_name == "enum" and base_name in self.base_names
        )

    def require_base(self, qualified_name: str) -> PythonEnumBaseKind:
        """Select the exact standard-library base through its nominal declaration."""

        return PythonEnumBaseKind.for_qualified_name(qualified_name)

    def declaration_from_node(
        self,
        node: ast.ClassDef,
        *,
        identity: str,
        qualified_reference: QualifiedReferenceResolver,
    ) -> PythonEnumDeclarationAuthority:
        """Derive closed key equality from one exact standard Enum class body."""

        if node.decorator_list or node.keywords or len(node.bases) != 1:
            raise ValueError("Enum key proof requires one undecorated direct base")
        qualified_base = qualified_reference(node.bases[0], frozenset())
        if qualified_base is None:
            raise ValueError("Enum base binding remains unresolved")
        base = self.require_base(qualified_base)
        declarations: list[PythonEnumMemberDeclaration] = []
        declared_names: set[str] = set()
        for statement in node.body:
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and type(statement.value.value) is str
            ):
                continue
            if isinstance(statement, ast.AnnAssign) and statement.value is None:
                continue
            targets: tuple[ast.expr, ...]
            value: ast.expr | None
            if isinstance(statement, ast.AnnAssign):
                targets, value = (statement.target,), statement.value
            elif isinstance(statement, ast.Assign):
                targets, value = tuple(statement.targets), statement.value
            else:
                raise ValueError("Enum key proof requires a closed member-only body")
            if (
                value is None
                or len(targets) != 1
                or not isinstance(targets[0], ast.Name)
            ):
                raise ValueError("Enum member binding remains ambiguous")
            name = targets[0].id
            if name.startswith("_") or not self.permits_new_member(name):
                raise ValueError(
                    "Enum key proof rejects configuration and hook bindings"
                )
            member_value: PythonEnumMemberValueABC
            try:
                member_value = PythonEnumLiteralMemberValue(
                    LiteralExpressionEffects(value).hashable_value
                )
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                if isinstance(value, ast.Name) and value.id in declared_names:
                    member_value = PythonEnumAliasMemberValue(value.id)
                elif (
                    isinstance(value, ast.Call)
                    and not value.args
                    and not value.keywords
                    and qualified_reference(value.func, frozenset(declared_names))
                    == "enum.auto"
                ):
                    member_value = PythonEnumAutoMemberValue()
                else:
                    raise ValueError("Enum member value semantics remain open")
            declarations.append(PythonEnumMemberDeclaration(name, member_value))
            declared_names.add(name)
        if len(declarations) < 2:
            raise ValueError("Enum key proof requires at least two declared members")
        return PythonEnumDeclarationAuthority(identity, base, tuple(declarations))

    def permits_new_member(self, member_name: str) -> bool:
        """Reject additions that would replace standard enum behavior."""

        return member_name not in self.inherited_member_names

    def declared_member_names(
        self,
        bindings: Iterable[tuple[str, bool]],
    ) -> tuple[str, ...]:
        """Derive runtime enum members from direct named value bindings."""

        return tuple(
            sorted(
                name
                for name, has_value in bindings
                if has_value
                if not name.startswith("_")
                if self.permits_new_member(name)
            )
        )


PYTHON_ENUM_BASE_AUTHORITY = PythonEnumBaseAuthority()
