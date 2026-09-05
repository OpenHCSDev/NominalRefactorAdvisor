"""Nominal lexical lookup and ordered class namespace control-flow state."""

from __future__ import annotations

import ast

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Self

from .lexical_bindings import (
    LEXICAL_SCOPE_BINDING_AUTHORITY,
    ScopeBindingCollector,
)


class LexicalNameResolution(StrEnum):
    """Whether a read is owned locally, externally, or remains path-dependent."""

    INTERNAL = ("internal", True)
    EXTERNAL = ("external", False)
    UNPROVED = ("unproved", None)

    def __new__(
        cls,
        value: str,
        is_internal: bool | None,
    ) -> LexicalNameResolution:
        member = str.__new__(cls, value)
        member._value_ = value
        member._is_internal = is_internal
        return member

    @property
    def is_external_candidate(self) -> bool:
        return self._is_internal is not True

    def require_known(self, name: str) -> None:
        if self._is_internal is None:
            raise ValueError(f"Unproved class namespace binding for {name!r}")

    @classmethod
    def join(
        cls, resolutions: Iterable[LexicalNameResolution]
    ) -> LexicalNameResolution:
        alternatives = frozenset(resolutions)
        if len(alternatives) == 1:
            return next(iter(alternatives))
        return cls.UNPROVED


class LexicalScopeABC(ABC):
    """A scope owns whether and how it participates in lexical lookup."""

    declarations: ScopeBindingProjection
    requires_class_namespace_visibility: ClassVar[bool] = False
    hides_enclosing_class_namespace: ClassVar[bool] = True

    def resolve_name(
        self,
        name: str,
        *,
        class_namespace_visible: bool,
    ) -> LexicalNameResolution | None:
        if self.requires_class_namespace_visibility and not class_namespace_visible:
            return None
        if name in self.declarations.global_names:
            return LexicalNameResolution.EXTERNAL
        return self.local_resolution_for(name)

    @abstractmethod
    def local_resolution_for(self, name: str) -> LexicalNameResolution | None:
        """Return an owned resolution, or continue through the enclosing scopes."""

        raise NotImplementedError

    @property
    def execution_namespace(self) -> ClassNamespaceScope | None:
        """Only class scopes change lookup ownership as statements execute."""

        return None


@dataclass(frozen=True)
class ScopeBindingProjection:
    """Compile-time name ownership shared by function and class scopes."""

    local_names: frozenset[str]
    global_names: frozenset[str]
    nonlocal_names: frozenset[str]

    @classmethod
    def from_nodes(
        cls,
        nodes: Iterable[ast.AST],
        argument_names: Iterable[str] = (),
    ) -> Self:
        collector = ScopeBindingCollector()
        for node in nodes:
            collector.visit(node)
        local_names = collector.bound_names.union(argument_names)
        return cls(
            local_names=frozenset(
                local_names - collector.global_names - collector.nonlocal_names
            ),
            global_names=frozenset(collector.global_names),
            nonlocal_names=frozenset(collector.nonlocal_names),
        )


@dataclass(frozen=True)
class FunctionBindingProjection(ScopeBindingProjection, LexicalScopeABC):
    """Function lookup is governed by compile-time name ownership."""

    @property
    def declarations(self) -> ScopeBindingProjection:
        return self

    def local_resolution_for(self, name: str) -> LexicalNameResolution | None:
        return LexicalNameResolution.INTERNAL if name in self.local_names else None

    @classmethod
    def from_function(
        cls,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
    ) -> FunctionBindingProjection:
        nodes = (node.body,) if isinstance(node, ast.Lambda) else node.body
        return cls.from_nodes(
            nodes, (LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(node))
        )


class TypeParameterScope(FunctionBindingProjection):
    """PEP 695 annotation scopes retain access to their enclosing class namespace."""

    hides_enclosing_class_namespace = False


@dataclass
class ClassNamespaceScope(LexicalScopeABC):
    """One ordered class namespace; alternatives join without choosing a path."""

    declarations: ScopeBindingProjection
    bindings: dict[str, LexicalNameResolution] = field(
        default_factory=lambda: dict.fromkeys(
            ("__module__", "__qualname__"), LexicalNameResolution.INTERNAL
        )
    )
    unproved_execution_names: frozenset[str] = frozenset()
    requires_class_namespace_visibility = True

    @property
    def execution_namespace(self) -> ClassNamespaceScope:
        return self

    def local_resolution_for(self, name: str) -> LexicalNameResolution | None:
        resolution = self.resolution_for(name)
        if (
            resolution is LexicalNameResolution.EXTERNAL
            and name not in self.declarations.local_names
        ):
            return None
        return resolution

    def resolution_for(self, name: str) -> LexicalNameResolution:
        if name in self.unproved_execution_names:
            return LexicalNameResolution.UNPROVED
        return self.bindings.get(name, LexicalNameResolution.EXTERNAL)

    @contextmanager
    def unproved_execution(self, names: Iterable[str]) -> Iterator[None]:
        """Keep affected reads open while traversing an unproved execution region."""

        affected = frozenset(names).intersection(self.declarations.local_names)
        previous = self.unproved_execution_names
        self.unproved_execution_names = previous | affected
        try:
            yield
        finally:
            self.unproved_execution_names = previous
            self.record(affected, LexicalNameResolution.UNPROVED)

    def record(self, names: Iterable[str], resolution: LexicalNameResolution) -> None:
        for name in names:
            if name in self.declarations.local_names:
                self.bindings[name] = resolution

    def join(self, alternatives: Iterable[dict[str, LexicalNameResolution]]) -> None:
        snapshots = tuple(alternatives)
        self.bindings = {
            name: LexicalNameResolution.join(
                snapshot.get(name, LexicalNameResolution.EXTERNAL)
                for snapshot in snapshots
            )
            for name in set().union(*snapshots)
        }
