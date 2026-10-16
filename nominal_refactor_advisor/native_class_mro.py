"""Inert native C3 carriers for already loaded class declarations."""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
)

from dataclasses import dataclass
from functools import lru_cache
from types import FunctionType
from typing import (
    ClassVar,
    Generic,
    cast,
)

from .class_mro import DeclarationMroType, NativeMroBase
from .native_declarations import ClassNamespaceDeclaration, NativeDeclaration


@dataclass(frozen=True, eq=False)
class NativeClassMroDeclaration(NativeDeclaration, ClassNamespaceDeclaration):
    """Project actual native storage and C3, without executing class hooks."""

    native_mro: ClassVar[Callable[[type], tuple[type, ...]]] = vars(type)[
        "__mro__"
    ].__get__
    native_bases: ClassVar[Callable[[type], tuple[type, ...]]] = vars(type)[
        "__bases__"
    ].__get__
    _native_namespace: ClassVar[Callable[[type], Mapping[object, object]]] = vars(type)[
        "__dict__"
    ].__get__
    _native_module: ClassVar[Callable[[type], object]] = vars(type)[
        "__module__"
    ].__get__
    _native_qualname: ClassVar[Callable[[type], str]] = vars(type)[
        "__qualname__"
    ].__get__

    declaration: type

    def require_type_preparation(self) -> None:
        """Require ordinary lookup selecting the actual native type prepare descriptor."""
        # Exact type here is the metaclass of the selected metaclass. Custom
        # metameta lookup and descriptor precedence are outside this first law.
        if type(self.declaration) is not type or not any(
            owner is type for owner in self.native_mro(self.declaration)
        ):
            raise ValueError(
                "Native preparation requires ordinary metaclass attribute lookup"
            )
        name = type.__prepare__.__name__
        expected = self.stored_namespace(type)[name]
        owner = self.member_owner(name)
        if owner is None or self.stored_namespace(owner)[name] is not expected:
            raise ValueError(
                "Native preparation does not select the exact type descriptor"
            )

    def python_constructor(self, *, start_after: type | None = None) -> FunctionType:
        """Select current Python __new__ through actual ordinary metaclass lookup.

        A constructor's implicit class cell belongs to the selected MRO owner,
        not necessarily the invoked metaclass. This supplies current operand
        provenance, not super lookup, invocation or constructor effects.
        """
        if type(self.declaration) is not type:
            raise ValueError("Native metaclass invocation has unproved metameta hooks")
        owner = self.member_owner("__new__", start_after=start_after)
        if owner is None:
            raise ValueError("Native metaclass has no selected constructor")
        descriptor = self.stored_namespace(owner)["__new__"]
        if (
            type(descriptor) is not staticmethod
            or type(descriptor.__func__) is not FunctionType
        ):
            raise ValueError("Native metaclass has no exact Python constructor source")
        function = descriptor.__func__
        freevars = function.__code__.co_freevars
        if freevars:
            if freevars != ("__class__",):
                raise ValueError("Native constructor closure roles remain unproved")
            closure = function.__closure__
            if closure is None or len(closure) != 1:
                raise ValueError("Native constructor has no exact class-cell binding")
            try:
                declaration = closure[0].cell_contents
            except ValueError as error:
                raise ValueError("Native constructor class cell is empty") from error
            if declaration is not owner:
                raise ValueError(
                    "Native constructor class cell differs from its selected MRO owner"
                )
        return function

    @classmethod
    def stored_namespace(cls, declaration: type) -> Mapping[str, object]:
        namespace = cls._native_namespace(declaration)
        # Even a raw class dictionary can hold non-string keys with active
        # equality hooks. Validate by iteration before any name lookup.
        if any(type(name) is not str for name in namespace):
            raise ValueError("Native class namespace has unproved member keys")
        return cast(Mapping[str, object], namespace)

    @property
    def qualified_name(self) -> str:
        self.stored_namespace(self.declaration)
        module = self._native_module(self.declaration)
        qualname = self._native_qualname(self.declaration)
        if any(type(part) is not str for part in (module, qualname)):
            raise ValueError("Native class qualification is not exact native text")
        return f"{module}.{qualname}"

    @property
    def member_binding_names(self) -> frozenset[str]:
        return frozenset(self.stored_namespace(self.declaration))

    def member_owner(
        self, name: str, *, start_after: type | None = None
    ) -> type | None:
        """Select current stored C3 members; descriptor execution remains unproved."""
        if type(name) is not str:
            raise ValueError("Lookup requires an exact native member name")
        mro = self.native_mro(self.declaration)
        if start_after is not None:
            starts = tuple(
                index for index, owner in enumerate(mro) if owner is start_after
            )
            if len(starts) != 1:
                raise ValueError("Native MRO lookup start owner is absent or ambiguous")
            mro = mro[starts[0] + 1 :]
        return next(
            (owner for owner in mro if name in self.stored_namespace(owner)),
            None,
        )

    def require_generic_origin(self) -> None:
        """Require the original Generic lookup, not binding or cached results."""
        metaclass = NativeClassMroDeclaration(type(self.declaration))
        if metaclass.member_owner("__getitem__") is not None:
            raise ValueError("Metaclass subscription has no proved generic origin")
        getter = type.__getattribute__
        getter_owner = metaclass.member_owner(getter.__name__)
        if (
            getter_owner is None
            or self.stored_namespace(getter_owner)[getter.__name__] is not getter
            or metaclass.member_owner("__class_getitem__") is not None
        ):
            raise ValueError("Metaclass subscription lookup remains unproved")
        if self.member_owner("__class_getitem__") is not Generic:
            raise ValueError("Custom class subscription has no proved generic origin")

    @lru_cache(maxsize=None)
    def _projected_mro(self, bases: tuple[type, ...]) -> type:
        """Reuse only inert carrier creation, keyed by its actual projected bases."""
        return DeclarationMroType.from_declaration(self, bases)

    @property
    def mro_type(self) -> type:
        # A valid C3 order places every base after its children. Walking that
        # order backwards validates shared ancestors once per observation.
        actual_mro = self.native_mro(self.declaration)
        if not actual_mro or actual_mro[0] is not self.declaration:
            raise ValueError("Native class ancestry excludes its original root")
        projections: dict[NativeClassMroDeclaration, type] = {}
        for original in reversed(actual_mro):
            native = NativeClassMroDeclaration(original)
            terminal = NativeMroBase.for_python_type(original)
            if terminal is not None:
                projections[native] = terminal.python_type
                continue
            native_bases = tuple(
                NativeClassMroDeclaration(base) for base in self.native_bases(original)
            )
            if any(base not in projections for base in native_bases):
                raise ValueError("Native class ancestry excludes its original bases")
            bases = tuple(projections[base] for base in native_bases)
            projected = native._projected_mro(bases)
            actual = self.native_mro(original)
            projected_declarations = tuple(
                (
                    owner.declaration.declaration
                    if isinstance(owner, DeclarationMroType)
                    else owner
                )
                for owner in self.native_mro(projected)
            )
            if len(actual) != len(projected_declarations) or any(
                left is not right for left, right in zip(actual, projected_declarations)
            ):
                raise ValueError("Native class has no matching C3 projection")
            projections[native] = projected
        return projections[self]
