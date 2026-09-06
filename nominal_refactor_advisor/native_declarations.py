"""Identity and source derived from a native Python declaration."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import inspect
from textwrap import dedent
from typing import cast


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
        return f"{self.declaration.__module__}.{self.declaration.__qualname__}"

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
