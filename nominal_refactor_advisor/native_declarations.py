"""Identity and source derived from a native Python declaration."""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
import inspect
from textwrap import dedent


@dataclass(frozen=True)
class NativeDeclaration:
    """Keep native identity and lazily inspected source on one declaration."""

    declaration: type | Callable[..., object]

    @property
    def qualified_name(self) -> str:
        return f"{self.declaration.__module__}.{self.declaration.__qualname__}"

    @cached_property
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
